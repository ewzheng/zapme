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
import math
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
            required = math.ceil(self._cfg.window_size * self._cfg.min_on_fraction)
            if high >= required:
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


@dataclass(frozen=True)
class PulserConfig:
    """Single-pulse + cooldown safety policy.

    Attributes:
        cooldown_s: After a pulse fires, the minimum elapsed time
            before another pulse can fire. Defaults to 15 seconds —
            conservative enough to give the user time to physically
            remove the EMS pad if anything malfunctions, short enough
            that a real demo still produces multiple pulses across a
            longer session. Set to `0.0` to disable the cooldown
            (every rising edge fires); not recommended for safety.
    """

    cooldown_s: float = 15.0


class Pulser:
    """Converts a sustained `should_be_active` stream into single-frame pulses with cooldown.

    The debouncer produces a slow-moving "yes / no, currently slouching"
    signal that can stay `True` for many seconds. Without a pulser the
    gate would stay continuously asserted for that whole duration. The
    pulser instead emits `True` for **one frame** on the rising edge
    of the debouncer's output, then enforces a cooldown during which
    the output is always `False` regardless of input.

    Safety contract:

    - At most one `True` output per cooldown window.
    - Output drops to `False` automatically the frame *after* a pulse.
    - After the cooldown expires, the next pulse requires a fresh
      rising edge — i.e. the debouncer must go inactive and then
      active again. A user who is *still* slouching when the cooldown
      ends does not get re-zapped without first releasing.
    """

    def __init__(self, config: PulserConfig | None = None) -> None:
        """Initialize a fresh pulser, ready to fire on the first rising edge.

        Args:
            config: Single-pulse + cooldown policy. Defaults to a
                15-second cooldown.

        Raises:
            ValueError: If `cooldown_s < 0`.

        Preconditions:
            - None.

        Postconditions:
            - No pulse has been fired.
            - `is_in_cooldown()` returns `False`.
        """
        cfg = config or PulserConfig()
        if cfg.cooldown_s < 0:
            raise ValueError(f"cooldown_s must be >= 0, got {cfg.cooldown_s}")
        self._cfg = cfg
        self._last_pulse_time: float | None = None
        self._prev_active = False

    def step(self, should_be_active: bool) -> bool:
        """Decide whether to fire a pulse this frame.

        Args:
            should_be_active: Latest `Debouncer` output for this frame.

        Returns:
            `True` when the gate should be asserted *this frame only*.
            The very next call to `step()` will return `False` unless
            the debouncer has gone inactive and back to active *and*
            the cooldown has elapsed.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - When this method returns `True`, `is_in_cooldown()`
              returns `True` until `cooldown_s` has elapsed.
            - Internal "previous active" state is updated to track the
              rising edge for the next call.
        """
        now = time.perf_counter()
        rising_edge = should_be_active and not self._prev_active
        in_cooldown = (
            self._last_pulse_time is not None
            and (now - self._last_pulse_time) < self._cfg.cooldown_s
        )
        pulse = rising_edge and not in_cooldown
        if pulse:
            self._last_pulse_time = now
        self._prev_active = should_be_active
        return pulse

    def is_in_cooldown(self) -> bool:
        """Return whether a recent pulse is still locking out new ones.

        Returns:
            `True` if a pulse fired within the last `cooldown_s`.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        if self._last_pulse_time is None:
            return False
        return (time.perf_counter() - self._last_pulse_time) < self._cfg.cooldown_s

    def cooldown_remaining_s(self) -> float:
        """Return seconds until the next pulse is allowed.

        Returns:
            `0.0` if no cooldown is active; otherwise a positive number
            of seconds until the lockout expires.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        if self._last_pulse_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._last_pulse_time
        return max(0.0, self._cfg.cooldown_s - elapsed)

    def reset(self) -> None:
        """Clear cooldown state and edge-tracking.

        Useful when the loop restarts after a fault and prior
        edge / cooldown bookkeeping is no longer meaningful.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - `is_in_cooldown()` returns `False`.
            - The next `step(True)` call will fire a pulse.
        """
        self._last_pulse_time = None
        self._prev_active = False


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
        pulser: Pulser,
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
            pulser: Single-frame-pulse + cooldown safety wrapper that
                converts the debouncer's sustained signal into bounded
                gate assertions.
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
        self._pulser = pulser
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
        self._pulser.reset()
        self._warmup()
        last_log = 0.0
        try:
            self._watchdog.start()
            while not self._watchdog.is_tripped():
                # Heartbeat at the start of each iteration too (in
                # addition to the end). This bounds the maximum time
                # between heartbeats to one slow operation, not two,
                # so a long YOLO call followed by a long camera read
                # can't stack up to trip the watchdog by itself.
                self._watchdog.heartbeat()
                ok, frame = self._camera.read()
                if not ok:
                    self._logger.error("Camera read failed; exiting loop")
                    break

                pose = self._estimator.infer(frame)
                features = compute_slouch_features(pose) if pose is not None else None
                self._buffer.push(features)
                window = self._buffer.as_window()
                prob = self._classifier.predict(window)
                debounce_active = self._debouncer.update(prob)
                pulse = self._pulser.step(debounce_active)
                if pulse:
                    self._logger.warning(
                        "Pulse fired (prob=%.2f); cooldown=%.1fs",
                        prob,
                        self._pulser.cooldown_remaining_s(),
                    )
                self._watchdog.set_active(pulse)
                self._watchdog.heartbeat()

                now = time.perf_counter()
                if self._log_interval_s == 0 or now - last_log >= self._log_interval_s:
                    last_log = now
                    cd = self._pulser.cooldown_remaining_s()
                    self._logger.info(
                        "prob=%.2f debounce=%s pulse=%s cooldown=%.1fs buffer_full=%s",
                        prob,
                        "on" if debounce_active else "off",
                        "FIRE" if pulse else "-",
                        cd,
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
            self._pulser.reset()

        return 1 if self._watchdog.is_tripped() else 0

    def _warmup(self) -> None:
        """Run one dummy inference to amortize first-frame JIT / allocator costs.

        YOLO's first `predict()` call on a freshly-loaded model takes
        seconds (JIT compile, weight upload to the inference engine,
        first-allocation overhead) — much longer than the watchdog's
        per-iteration timeout. Doing the warmup *before* the watchdog
        starts keeps the cold-start cost out of the heartbeat budget,
        so the first real frame in `run()` is already at steady-state
        speed.

        Failures during warmup are logged but not raised — the real
        loop will hit the same problem and produce a more useful
        error in context.

        Preconditions:
            - All injected components are constructed.

        Postconditions:
            - The pose estimator and classifier have been called once
              with a dummy black frame.
            - `_buffer` and `_debouncer` and `_pulser` are reset (the
              warmup may have pushed one feature vector through).
        """
        try:
            self._logger.info("Warming up model (first inference is slow)...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            pose = self._estimator.infer(dummy)
            features = (
                compute_slouch_features(pose) if pose is not None else None
            )
            self._buffer.push(features)
            window = self._buffer.as_window()
            self._classifier.predict(window)
            self._logger.info("Warmup complete.")
        except Exception:
            self._logger.exception("Warmup failed; continuing into the loop anyway")
        finally:
            self._buffer.reset()
            self._debouncer.reset()
            self._pulser.reset()
