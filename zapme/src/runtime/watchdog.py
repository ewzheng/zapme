"""Heartbeat-based fail-safe for the EMS gate.

The runtime loop calls `heartbeat()` on every iteration. A daemon
thread monitors the elapsed time since the last heartbeat; if it
exceeds `timeout_s`, the watchdog drives the gate low and latches a
"tripped" flag the loop can poll to know it should exit.

This enforces the safety constraint in `.llm/llm.MD`: the EMS line
must never be left high after a process hang, deadlock, or unhandled
exception. The watchdog runs independently of the main loop's
correctness — even if the loop is wedged inside a blocking call (e.g.
camera read), the watchdog still releases the gate.

The watchdog *owns* the gate. Callers drive it via `set_active()` so
every write goes through a single lock, preventing the main loop and
the watchdog thread from racing on the underlying GPIO.
"""

from __future__ import annotations

import logging
import threading
import time

from zapme.src.runtime.gate import Gate


class Watchdog:
    """Owns a `Gate` and forces it low when the main loop stops feeding it.

    Lifecycle:

    1. Construct with the gate and a timeout.
    2. Call `start()` to spawn the monitor thread.
    3. Drive the gate through `set_active(bool)` and call `heartbeat()`
       once per loop iteration.
    4. Poll `is_tripped()` to detect timeouts and exit cleanly.
    5. Call `stop()` (or use the context-manager protocol) to tear
       everything down. The gate is always driven low on stop.

    Construction does not start the monitor thread; nothing happens
    automatically until `start()` is called. This lets the caller wire
    up exception handling around the start/stop pair.
    """

    def __init__(
        self,
        gate: Gate,
        timeout_s: float = 3.0,
        check_interval_s: float = 0.1,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the watchdog around a gate.

        Args:
            gate: Gate to drive on timeout. The watchdog takes
                ownership for the duration of `start()`/`stop()`.
            timeout_s: Maximum allowed elapsed time between
                heartbeats before the watchdog trips.
            check_interval_s: How often the monitor thread wakes to
                check the elapsed time. Smaller is more responsive at
                the cost of a tiny bit of CPU; larger is the opposite.
                Should be much smaller than `timeout_s`.
            logger: Optional logger for trip events. Defaults to a
                module-scoped logger so the runtime gets a record of
                why the loop exited.

        Preconditions:
            - `gate` is freshly constructed and not yet `close()`-d.
            - `timeout_s > 0` and `0 < check_interval_s < timeout_s`.

        Postconditions:
            - The watchdog is constructed but the monitor thread is
              not yet running. `start()` must be called separately.
            - `is_tripped()` returns `False`.
        """
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        if check_interval_s <= 0 or check_interval_s >= timeout_s:
            raise ValueError(
                f"check_interval_s must be in (0, {timeout_s}), got {check_interval_s}"
            )
        self._gate = gate
        self._timeout_s = timeout_s
        self._check_interval_s = check_interval_s
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tripped = threading.Event()
        self._last_heartbeat = time.perf_counter()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the monitor thread.

        Idempotent: a second call does nothing.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - A daemon thread is running and will trip the gate if
              `heartbeat()` is not called within `timeout_s`.
            - `_last_heartbeat` is reset to the current time so the
              watchdog does not trip on a slow startup.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self.heartbeat()
        self._thread = threading.Thread(
            target=self._monitor, daemon=True, name="zapme-watchdog"
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 2.0) -> None:
        """Stop the monitor thread and drive the gate low.

        Idempotent. Always drives the gate low, even if joining the
        thread fails or times out — releasing the EMS comes first.

        Args:
            join_timeout_s: Maximum time to wait for the monitor
                thread to exit cleanly.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - The monitor thread has exited (or `join_timeout_s` has
              elapsed).
            - `gate.set(False)` has been called under the lock.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_s)
        with self._lock:
            try:
                self._gate.set(False)
            except Exception:
                self._logger.exception("Gate.set(False) raised during watchdog stop")

    def heartbeat(self) -> None:
        """Reset the heartbeat clock to the current time.

        The main loop calls this once per iteration. Cheap by design:
        a single timestamp write.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - The next watchdog check sees a fresh timestamp.
        """
        self._last_heartbeat = time.perf_counter()

    def is_tripped(self) -> bool:
        """Return whether the watchdog has tripped.

        Returns:
            `True` once the watchdog has detected a heartbeat timeout
            and forced the gate low. Stays `True` afterward; the loop
            should exit.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return self._tripped.is_set()

    def set_active(self, active: bool) -> None:
        """Drive the gate through the watchdog's lock.

        After the watchdog has tripped, requests to assert the gate
        (`active=True`) are silently dropped — the safe state is
        sticky. Requests to release (`active=False`) are always
        honored, since releasing is always safe.

        Args:
            active: Desired gate state.

        Preconditions:
            - `start()` has been called.

        Postconditions:
            - When not tripped, the gate's commanded state matches
              `active`.
            - When tripped, the gate stays low regardless of `active`.
        """
        with self._lock:
            if self._tripped.is_set() and active:
                return
            try:
                self._gate.set(active)
            except Exception:
                self._logger.exception("Gate.set raised inside watchdog.set_active")

    def __enter__(self) -> "Watchdog":
        """Start the monitor thread on context entry.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - The monitor thread is running.
        """
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop the monitor thread and drive the gate low on context exit.

        Preconditions:
            - `__enter__` completed.

        Postconditions:
            - `stop()` has been called.
        """
        self.stop()

    def _monitor(self) -> None:
        """Monitor thread loop: trip the gate on heartbeat timeout.

        Runs until `stop_event` is set. After tripping, keeps running
        and re-asserts gate-low on every check interval — this defends
        against any stray code that might still hold a gate reference
        and try to flip it.

        Preconditions:
            - Called only as the target of the daemon thread.

        Postconditions:
            - Exits when `stop_event` is set.
            - On any heartbeat-timeout, `_tripped` is set and the gate
              is driven low under the lock.
        """
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._check_interval_s):
                break
            elapsed = time.perf_counter() - self._last_heartbeat
            if elapsed > self._timeout_s:
                already_tripped = self._tripped.is_set()
                with self._lock:
                    try:
                        self._gate.set(False)
                    except Exception:
                        self._logger.exception("Gate.set(False) raised inside watchdog")
                    self._tripped.set()
                if not already_tripped:
                    self._logger.error(
                        "Watchdog tripped: %.3fs since last heartbeat (timeout=%.3fs)",
                        elapsed,
                        self._timeout_s,
                    )
