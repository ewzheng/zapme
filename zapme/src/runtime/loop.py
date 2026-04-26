"""End-to-end inference loop: camera → pose → features → buffer → classifier → debounce → gate.

This is the thing you run on the Pi as `python -m zapme.src`. It owns
the lifecycle of the camera and the watchdog-protected gate, and
exposes a single `Loop.run()` entry point that returns when the
operator presses `q` (in the optional preview window) or the watchdog
trips.

Composition order matters for the safety contract in `.llm/llm.MD`:

1. The gate is constructed *first*, in its off state.
2. The watchdog wraps the gate and starts its monitor thread *before*
   any frame inference runs, so even an exception during model
   warm-up leaves the line low.
3. Only after both safety pieces are wired does the loop pull frames.

Hysteresis-based debouncing lives in this module too (`Debouncer`).
It's small enough not to deserve its own file, and it's only
meaningful in the context of the loop's `slouch_prob` stream.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from zapme.src.model.classifier import SlouchClassifier
from zapme.src.model.features import compute_slouch_features
from zapme.src.model.vision import PoseEstimator
from zapme.src.runtime.feature_buffer import FeatureBuffer
from zapme.src.runtime.gate import Gate
from zapme.src.runtime.watchdog import Watchdog


@dataclass(frozen=True)
class DebouncerConfig:
    """Hysteresis parameters for the gate decision.

    Attributes:
        window_size: Number of recent `slouch_prob` values to consider
            when deciding whether to flip the gate.
        on_threshold: Probability a sample must exceed to count as a
            "slouch" vote. Going from off→on requires at least
            `min_on_fraction` of the window to be at or above this.
        off_threshold: Mean probability across the window must drop
            below this to flip on→off. Strictly below `on_threshold`
            to give true hysteresis (no oscillation around a single
            threshold).
        min_on_fraction: Fraction of window samples that must be at or
            above `on_threshold` to trigger off→on.
    """

    window_size: int = 20
    on_threshold: float = 0.8
    off_threshold: float = 0.4
    min_on_fraction: float = 0.6


class Debouncer:
    """Hysteresis-based decision over a stream of slouch probabilities.

    Two thresholds and a fraction-of-window vote, designed so the gate
    does not chatter around a single decision boundary:

    - **off → on** when at least `min_on_fraction` of the most recent
      `window_size` probabilities are at or above `on_threshold`.
    - **on → off** when the mean of the most recent `window_size`
      probabilities drops below `off_threshold`.

    Any other state holds the previous decision. Probabilities reported
    as `NaN` (which the upstream pipeline avoids, but which can happen
    in pathological corner cases) are treated as `0.0` for safety.
    """

    def __init__(self, config: DebouncerConfig | None = None) -> None:
        """Construct an initially-off debouncer.

        Args:
            config: Hysteresis parameters. Defaults to a 20-sample
                window with on=0.7 / off=0.4 / on-fraction=0.6 — about
                two seconds of context at 10 FPS.

        Raises:
            ValueError: If thresholds are out of `[0, 1]` or
              `off_threshold >= on_threshold` (no hysteresis margin).

        Preconditions:
            - `config.window_size >= 1`.

        Postconditions:
            - `is_active()` returns `False`.
            - The window is empty.
        """
        cfg = config or DebouncerConfig()
        if not (0.0 <= cfg.off_threshold < cfg.on_threshold <= 1.0):
            raise ValueError(
                f"need 0 <= off_threshold ({cfg.off_threshold}) < "
                f"on_threshold ({cfg.on_threshold}) <= 1"
            )
        if not (0.0 < cfg.min_on_fraction <= 1.0):
            raise ValueError(
                f"min_on_fraction must be in (0, 1], got {cfg.min_on_fraction}"
            )
        if cfg.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {cfg.window_size}")
        self._cfg = cfg
        self._window: deque[float] = deque(maxlen=cfg.window_size)
        self._active = False

    def update(self, prob: float) -> bool:
        """Append a new probability and return the resulting gate decision.

        Args:
            prob: Latest slouch probability in `[0, 1]`. `NaN` is
                clamped to `0.0` for safety (don't zap on a degenerate
                input).

        Returns:
            `True` if the gate should be active after this sample,
            `False` otherwise.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - The internal window has the new sample appended; the
              oldest sample is dropped if the window was full.
            - The active state is updated according to the hysteresis
              rule and persists across calls.
        """
        if not np.isfinite(prob):
            prob = 0.0
        prob = float(max(0.0, min(1.0, prob)))
        self._window.append(prob)
        if not self._window:
            return self._active

        if not self._active:
            high = sum(1 for p in self._window if p >= self._cfg.on_threshold)
            if high / len(self._window) >= self._cfg.min_on_fraction:
                self._active = True
        else:
            mean_prob = sum(self._window) / len(self._window)
            if mean_prob < self._cfg.off_threshold:
                self._active = False
        return self._active

    def is_active(self) -> bool:
        """Return the current gate decision.

        Returns:
            `True` if the gate should be active.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return self._active

    def reset(self) -> None:
        """Clear the window and force the decision back to inactive.

        Useful when the runtime resumes after a fault and the recent
        history is no longer representative.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - The window is empty.
            - `is_active()` returns `False`.
        """
        self._window.clear()
        self._active = False


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a webcam by OpenCV index.

    Args:
        index: OpenCV camera index. `0` is the first camera the
            platform reports (usually the built-in laptop webcam).

    Returns:
        An opened `cv2.VideoCapture`. The caller owns it and must
        `.release()` when done.

    Raises:
        RuntimeError: If the camera cannot be opened.

    Preconditions:
        - A webcam exists at `index`.

    Postconditions:
        - The returned capture has `isOpened()` true.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")
    return cap


class Loop:
    """End-to-end runtime loop wiring camera through to the watchdog-protected gate.

    Owns the lifetime of the watchdog (and therefore the gate) for the
    duration of `run()`. Every iteration:

    1. Reads a frame from the camera.
    2. Runs YOLO pose to extract keypoints.
    3. Computes geometric features.
    4. Pushes them into the rolling buffer.
    5. Runs the classifier on the current window.
    6. Feeds the probability through the debouncer.
    7. Drives the gate via `watchdog.set_active`.
    8. Heartbeats the watchdog.

    On any exception or watchdog trip, the gate is left low before
    `run()` returns.
    """

    def __init__(
        self,
        camera: cv2.VideoCapture,
        estimator: PoseEstimator,
        classifier: SlouchClassifier,
        buffer: FeatureBuffer,
        debouncer: Debouncer,
        gate: Gate,
        watchdog: Watchdog,
        log_interval_s: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """Wire all the runtime components together.

        Args:
            camera: Opened `cv2.VideoCapture`. The loop does not open
                or release it — caller owns the lifecycle.
            estimator: Pose estimator (e.g. `PoseEstimator(...)`).
            classifier: Slouch classifier; can be the placeholder rule
                or a trained ONNX-backed instance.
            buffer: Feature buffer sized to match the classifier's
                window. Reset on entry to `run()` so prior state does
                not leak in.
            debouncer: Hysteresis debouncer over `slouch_prob`.
            gate: Owned by `watchdog`; passed in here so that the loop
                can also enforce gate-off in its `finally` block as a
                belt-and-suspenders safety measure.
            watchdog: Heartbeat-based fail-safe wrapping `gate`.
                `start()` is called inside `run()`; `stop()` is called
                in the `finally`.
            log_interval_s: Minimum seconds between per-frame stdout
                log lines. Set to `0` to log every frame.
            logger: Logger for runtime events. Defaults to a
                module-scoped logger.

        Preconditions:
            - All injected components are constructed and ready.
            - `buffer.config.num_features` and `buffer.config.window_size`
              match what the `classifier` expects.

        Postconditions:
            - The loop is wired but not yet running. Call `run()` to
              start.
        """
        self._camera = camera
        self._estimator = estimator
        self._classifier = classifier
        self._buffer = buffer
        self._debouncer = debouncer
        self._gate = gate
        self._watchdog = watchdog
        self._log_interval_s = log_interval_s
        self._logger = logger or logging.getLogger(__name__)

    def run(self) -> int:
        """Run the loop until the camera fails or the watchdog trips.

        Returns:
            `0` on a clean exit, `1` if the loop terminated because
            the watchdog tripped (a hint to the wrapping process /
            systemd that a restart is appropriate).

        Preconditions:
            - All injected components are healthy.

        Postconditions:
            - The watchdog has been stopped and the gate is low.
            - The buffer and debouncer have been reset so a follow-up
              `run()` call does not see stale state.
        """
        self._buffer.reset()
        self._debouncer.reset()
        last_log = 0.0
        try:
            self._watchdog.start()
            while not self._watchdog.is_tripped():
                ok, frame = self._camera.read()
                if not ok:
                    self._logger.error("Camera read failed; exiting loop")
                    break

                pose = self._estimator.infer(frame)
                features = compute_slouch_features(pose) if pose is not None else None
                self._buffer.push(features)
                window = self._buffer.as_window()
                prob = self._classifier.predict(window)
                should_be_active = self._debouncer.update(prob)
                self._watchdog.set_active(should_be_active)
                self._watchdog.heartbeat()

                now = time.perf_counter()
                if self._log_interval_s == 0 or now - last_log >= self._log_interval_s:
                    last_log = now
                    self._logger.info(
                        "prob=%.2f gate=%s buffer_full=%s",
                        prob,
                        "ON" if should_be_active else "off",
                        self._buffer.is_full(),
                    )
        finally:
            self._watchdog.stop()
            try:
                self._gate.close()
            except Exception:
                self._logger.exception("Gate.close raised in Loop.run finally")
            self._buffer.reset()
            self._debouncer.reset()

        return 1 if self._watchdog.is_tripped() else 0
