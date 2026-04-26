"""Entry point for `python -m zapme.src`.

Wires the runtime components together and runs the inference loop:
camera → pose → features → buffer → classifier → debounce → gate,
all under a heartbeat watchdog that defaults the EMS line to off on
any fault.

Two backends are supported behind the same loop:

- `--backend lgpio` (Pi): real GPIO via `lgpio`. Requires the package
  to be installed and the user to have access to `/dev/gpiochip*`.
- `--backend fake` (default on non-Linux): in-memory `FakeGate` that
  logs transitions. Lets the entire loop be exercised on a developer
  machine without hardware.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from zapme.src.model.classifier import SlouchClassifier
from zapme.src.model.vision import PoseEstimator
from zapme.src.runtime.feature_buffer import FeatureBuffer
from zapme.src.runtime.gate import FakeGate, Gate, LgpioGate
from zapme.src.runtime.loop import (
    Debouncer,
    DebouncerConfig,
    EscalationConfig,
    Loop,
    Pulser,
    PulserConfig,
    WarningEscalator,
    open_camera,
)
from zapme.src.runtime.watchdog import Watchdog
from zapme.src.utils.inout import FakeSpeaker, FileSpeaker, Speaker

DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DEFAULT_CLIPS = ("warning_1", "warning_2", "zap")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the runtime entry point.

    Returns:
        Parsed argparse namespace.

    Preconditions:
        - `sys.argv` is set as expected for a CLI entry point.

    Postconditions:
        - Returned namespace exposes camera / model / gate / debouncer
          / watchdog parameters.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--repo", type=str, default="Ultralytics/YOLO11",
        help="Hugging Face repo holding the YOLO pose checkpoint.",
    )
    parser.add_argument(
        "--filename", type=str, default="yolo11n-pose.pt",
        help="YOLO weights filename within the HF repo.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=320,
        help="YOLO input image size (square). Lower = faster inference. "
             "Default 320 is ~4x the FPS of YOLO's standard 640. Bump "
             "to 640 if you need more keypoint precision and your "
             "hardware can keep up.",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("models/slouch_cnn.onnx"),
        help="Trained slouch classifier ONNX path. If absent, the "
             "placeholder rule is used so the loop still runs.",
    )

    parser.add_argument(
        "--backend", choices=("fake", "lgpio"), default=None,
        help="Gate backend. Defaults to 'lgpio' on Linux aarch64 (Pi), "
             "'fake' elsewhere.",
    )
    parser.add_argument(
        "--gate-pin", type=int, default=17,
        help="BCM GPIO pin for LgpioGate.",
    )
    parser.add_argument(
        "--gate-active-low", action="store_true",
        help="Treat low as the active level (for inverted-logic gating).",
    )

    parser.add_argument("--on-threshold", type=float, default=0.8)
    parser.add_argument("--off-threshold", type=float, default=0.4)
    parser.add_argument("--debounce-window", type=int, default=20)
    parser.add_argument("--min-on-fraction", type=float, default=0.6)

    parser.add_argument(
        "--cooldown", type=float, default=15.0,
        help="Seconds the gate is locked low after a pulse fires. "
             "Each slouch detection produces a single one-frame pulse, "
             "then the gate is forced low for this many seconds — gives "
             "the user time to remove the EMS pad if anything malfunctions. "
             "Set to 0 to disable (not recommended).",
    )

    parser.add_argument(
        "--warn-to-warn", type=float, default=5.0,
        help="Seconds of sustained slouch between warning 1 and warning 2.",
    )
    parser.add_argument(
        "--warn-to-fire", type=float, default=5.0,
        help="Seconds of sustained slouch between warning 2 and the EMS pulse.",
    )
    parser.add_argument(
        "--fire-to-warn", type=float, default=5.0,
        help="Seconds of silence after a FIRE before the warning cycle "
             "restarts. Warnings cycle (unlike the zap, which is "
             "one-shot-per-slouch) so an uncorrected user keeps getting "
             "nagged. The pulser cooldown still gates the physical zap.",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable spoken warnings — uses an in-memory FakeSpeaker so the "
             "loop still runs the warning state machine but no sound plays. "
             "Useful for benchtop testing.",
    )
    parser.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR,
        help="Directory containing voice clip WAV files. The runtime "
             "looks for warning_1.wav / warning_2.wav / zap.wav in here.",
    )

    parser.add_argument(
        "--watchdog-timeout", type=float, default=3.0,
        help="Seconds without a heartbeat before the watchdog forces the gate low. "
             "Default of 3.0s comfortably accommodates a single slow inference "
             "iteration (YOLO + classifier + camera read) on a Pi while still "
             "catching real hangs — anything stuck for >3s is a real fault.",
    )
    parser.add_argument(
        "--watchdog-check-interval", type=float, default=0.1,
        help="How often the watchdog thread polls.",
    )

    parser.add_argument(
        "--log-interval", type=float, default=1.0,
        help="Minimum seconds between per-frame log lines.",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
    )
    return parser.parse_args()


def _resolve_backend(requested: str | None) -> str:
    """Pick the gate backend, defaulting based on platform.

    Args:
        requested: User-provided `--backend` value, or `None` for auto.

    Returns:
        Either `"lgpio"` or `"fake"`.

    Preconditions:
        - None.

    Postconditions:
        - Always returns a string; never raises for unknown platforms.
    """
    if requested is not None:
        return requested
    if sys.platform.startswith("linux"):
        return "lgpio"
    return "fake"


def _build_gate(backend: str, pin: int, active_high: bool) -> Gate:
    """Construct the requested gate backend.

    Args:
        backend: Either `"lgpio"` or `"fake"`.
        pin: BCM GPIO pin (used only by `lgpio`).
        active_high: Active level (used only by `lgpio`).

    Returns:
        A constructed `Gate` in the off state.

    Raises:
        ValueError: For unknown backends.
        ImportError: If `lgpio` is requested but not installed.

    Preconditions:
        - `backend` was resolved by `_resolve_backend`.

    Postconditions:
        - The gate is constructed and the line is low.
    """
    if backend == "lgpio":
        return LgpioGate(pin=pin, active_high=active_high)
    if backend == "fake":
        return FakeGate()
    raise ValueError(f"Unknown gate backend: {backend!r}")


def main() -> int:
    """Boot the zapme runtime loop.

    Returns:
        `0` on a clean exit; `1` if the loop terminated because the
        watchdog tripped (a hint to systemd / the launcher to restart).

    Preconditions:
        - Process is running on a host with the requested camera and
          gate backend available.

    Postconditions:
        - The gate has been driven low before return, regardless of
          how the loop exited (clean exit, exception, signal, or
          watchdog trip).
    """
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("zapme")

    backend = _resolve_backend(args.backend)
    log.info("Gate backend: %s", backend)

    estimator = PoseEstimator(
        repo_id=args.repo,
        filename=args.filename,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
    )

    weights_path = args.weights if args.weights.exists() else None
    if weights_path is None:
        log.warning(
            "Classifier weights not found at %s; using placeholder rule.",
            args.weights,
        )
    classifier = SlouchClassifier(weights_path=weights_path)
    buffer = FeatureBuffer(config=classifier.config)

    debouncer = Debouncer(
        config=DebouncerConfig(
            window_size=args.debounce_window,
            on_threshold=args.on_threshold,
            off_threshold=args.off_threshold,
            min_on_fraction=args.min_on_fraction,
        )
    )
    escalator = WarningEscalator(
        config=EscalationConfig(
            warn_to_warn_s=args.warn_to_warn,
            warn_to_fire_s=args.warn_to_fire,
            fire_to_warn_s=args.fire_to_warn,
        )
    )
    pulser = Pulser(config=PulserConfig(cooldown_s=args.cooldown))
    if args.cooldown <= 0:
        log.warning(
            "Cooldown disabled — every rising-edge slouch will fire a pulse. "
            "This is unsafe for body-worn EMS; reconsider before plugging in."
        )

    speaker: Speaker
    if args.no_audio:
        speaker = FakeSpeaker()
        log.info("Audio disabled (--no-audio); warnings will not play out loud.")
    else:
        clip_map = {name: args.assets_dir / f"{name}.wav" for name in DEFAULT_CLIPS}
        speaker = FileSpeaker(clips=clip_map, logger=log.getChild("speaker"))
        log.info("Audio assets dir: %s", args.assets_dir)

    gate = _build_gate(
        backend=backend,
        pin=args.gate_pin,
        active_high=not args.gate_active_low,
    )

    watchdog = Watchdog(
        gate=gate,
        timeout_s=args.watchdog_timeout,
        check_interval_s=args.watchdog_check_interval,
        logger=log.getChild("watchdog"),
    )

    try:
        camera = open_camera(args.camera)
    except RuntimeError as exc:
        log.error(str(exc))
        gate.close()
        return 1

    def _signal_handler(signum: int, _frame: object) -> None:
        log.warning("Received signal %d; stopping watchdog and closing gate.", signum)
        watchdog.stop()
        camera.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    loop = Loop(
        camera=camera,
        estimator=estimator,
        classifier=classifier,
        buffer=buffer,
        debouncer=debouncer,
        escalator=escalator,
        speaker=speaker,
        pulser=pulser,
        gate=gate,
        watchdog=watchdog,
        log_interval_s=args.log_interval,
        logger=log.getChild("loop"),
    )

    try:
        return loop.run()
    finally:
        camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
