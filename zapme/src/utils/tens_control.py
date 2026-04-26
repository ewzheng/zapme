"""High-level API for the TENS / EMS unit relay.

Wraps any `Gate` implementation with TENS-specific timing helpers
(`pulse`, `hold`) so hardware bring-up scripts and one-off operator
commands don't have to repeat the same `set(True) / sleep / set(False)
/ try-finally` boilerplate.

`TensController` itself implements the `Gate` interface (via
composition over the inner gate), so it can be substituted anywhere a
`Gate` is expected — including the runtime loop. The runtime doesn't
*need* the blocking helpers (it sends per-frame pulses through the
`Pulser`), but using `TensController` instead of bare `LgpioGate` lets
the same object surface both code paths.

Run as a module for a quick hardware test:

    python -m zapme.src.utils.tens_control --pulse 0.2
    python -m zapme.src.utils.tens_control --hold 5.0

Use `--backend fake` (default off-Pi) to dry-run the API on a dev
machine without GPIO hardware.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from types import TracebackType

from zapme.src.runtime.gate import FakeGate, Gate, LgpioGate

DEFAULT_TENS_PIN: int = 17
DEFAULT_PULSE_S: float = 0.2


class TensController(Gate):
    """`Gate` wrapper exposing TENS-specific pulse / hold helpers.

    Constructed with any `Gate` (real or fake). The wrapper implements
    the full `Gate` interface by delegating to the inner gate, so the
    runtime can take a `TensController` anywhere a `Gate` is expected.
    Adds `pulse(duration_s)` and `hold(duration_s)` for one-shot
    blocking timed assertions, which is what hardware bring-up and
    operator test scripts want.

    All `Gate` safety guarantees carry over unchanged: low at
    construction, low on close, low even when `pulse()` or `hold()`
    raises mid-sleep.
    """

    def __init__(self, gate: Gate) -> None:
        """Wrap an existing `Gate` instance.

        Args:
            gate: Already-constructed `Gate`. The controller assumes
                ownership for the duration of its lifetime — the
                caller should not also drive the underlying gate.

        Preconditions:
            - `gate` has been constructed and is in the off state.

        Postconditions:
            - The controller is ready to accept `set` / `pulse` /
              `hold` / `close` calls.
        """
        self._gate = gate

    @classmethod
    def open_pi(
        cls,
        pin: int = DEFAULT_TENS_PIN,
        active_high: bool = False,
    ) -> "TensController":
        """Open a controller backed by a real `LgpioGate` on the Pi.

        Args:
            pin: BCM GPIO pin wired to the TENS relay's trigger input.
                Defaults to the same pin the original team script used.
            active_high: `True` when the wiring drives the relay on
                with a HIGH level. Default `False` matches the
                deployment's active-LOW opto-isolated relay (line LOW
                = relay ON, line HIGH at boot = safely de-energized).

        Returns:
            A `TensController` ready to drive real hardware.

        Raises:
            ImportError: If `lgpio` is not installed.
            lgpio.error: If the chip / pin cannot be claimed.

        Preconditions:
            - The current process has access to `/dev/gpiochip*`.

        Postconditions:
            - The line is low (inactive level given `active_high`).
        """
        return cls(gate=LgpioGate(pin=pin, active_high=active_high))

    @classmethod
    def open_fake(cls) -> "TensController":
        """Open a controller backed by an in-memory `FakeGate`.

        Useful for unit tests and dev-machine bring-up without GPIO.

        Returns:
            A `TensController` whose underlying gate is a `FakeGate`.
            The fake's `transitions` log is accessible via
            `controller.gate.transitions`.

        Preconditions:
            - None.

        Postconditions:
            - The (in-memory) line is off.
        """
        return cls(gate=FakeGate())

    @property
    def gate(self) -> Gate:
        """Expose the underlying `Gate`.

        Returns:
            The wrapped gate instance. Useful for tests that need to
            introspect `FakeGate.transitions` or for callers that
            need to check `isinstance(...)` for a specific backend.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return self._gate

    def set(self, active: bool) -> None:
        """Drive the gate directly (delegates to the inner `Gate`).

        Args:
            active: Requested gate state.

        Preconditions:
            - `close()` has not been called.

        Postconditions:
            - The inner gate's commanded state matches `active`.
        """
        self._gate.set(active)

    def is_active(self) -> bool:
        """Return the inner gate's last commanded state.

        Returns:
            `True` if the gate is currently asserted.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return self._gate.is_active()

    def close(self) -> None:
        """Drive the gate low and release the inner gate's resources.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - The inner gate has been closed; the line is low.
        """
        self._gate.close()

    def pulse(self, duration_s: float = DEFAULT_PULSE_S) -> None:
        """Assert the gate for `duration_s` seconds, then release.

        Blocks the calling thread for `duration_s`. Always releases
        the gate before returning, including on `KeyboardInterrupt`
        or any other exception raised mid-sleep — releasing is the
        safe direction so we want it on every exit path.

        Args:
            duration_s: Pulse length in seconds. Must be positive.

        Raises:
            ValueError: If `duration_s <= 0`.

        Preconditions:
            - `close()` has not been called.

        Postconditions:
            - The gate is low when this returns.
            - The gate has been low or transitioning back to low for
              roughly `duration_s` of the call's wall time (within
              `time.sleep` precision).
        """
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        try:
            self._gate.set(True)
            time.sleep(duration_s)
        finally:
            self._gate.set(False)

    def hold(self, duration_s: float) -> None:
        """Alias for `pulse(duration_s)` for readability.

        Use `hold()` when the intent is "keep the gate on for X
        seconds" rather than "send a quick stim pulse." Behavior is
        identical to `pulse()`; the distinction is just for the
        reader of the call site.

        Args:
            duration_s: Hold length in seconds. Must be positive.

        Raises:
            ValueError: If `duration_s <= 0`.

        Preconditions:
            - `close()` has not been called.

        Postconditions:
            - As for `pulse()`.
        """
        self.pulse(duration_s)

    def __enter__(self) -> "TensController":
        """Return `self` for the `with` statement.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Always close the controller on context exit.

        Preconditions:
            - `__enter__` completed.

        Postconditions:
            - `close()` has been called regardless of exception state.
            - This method does not suppress exceptions.
        """
        self.close()


def _build_controller(backend: str, pin: int, active_high: bool) -> TensController:
    """Build a `TensController` for the requested backend.

    Args:
        backend: Either `"lgpio"` or `"fake"`.
        pin: BCM pin (used only by `"lgpio"`).
        active_high: Active level (used only by `"lgpio"`).

    Returns:
        A constructed `TensController` with the line low.

    Raises:
        ValueError: For unknown backends.
        ImportError: If `lgpio` is requested but not installed.

    Preconditions:
        - `backend` is one of the supported values.

    Postconditions:
        - The returned controller's gate is low.
    """
    if backend == "lgpio":
        return TensController.open_pi(pin=pin, active_high=active_high)
    if backend == "fake":
        return TensController.open_fake()
    raise ValueError(f"Unknown backend: {backend!r}")


def _resolve_default_backend() -> str:
    """Pick a sensible default backend for the host platform.

    Returns:
        `"lgpio"` on Linux (assumed to be the Pi); `"fake"` everywhere
        else (Windows / macOS dev machines).

    Preconditions:
        - None.

    Postconditions:
        - Always returns a valid backend name.
    """
    return "lgpio" if sys.platform.startswith("linux") else "fake"


def main(argv: list[str] | None = None) -> int:
    """CLI for hardware bring-up testing of the TENS relay.

    Examples:
        # 200ms pulse on the default pin:
        python -m zapme.src.utils.tens_control --pulse 0.2

        # Hold the relay on for 5 seconds:
        python -m zapme.src.utils.tens_control --hold 5

        # Dry-run the API on a dev machine (no hardware):
        python -m zapme.src.utils.tens_control --backend fake --pulse 0.2

    Args:
        argv: Optional override for `sys.argv[1:]`. Defaults to the
            real CLI args. Useful for tests.

    Returns:
        `0` on success, `1` on usage / hardware errors, `130` if
        interrupted by `KeyboardInterrupt`.

    Preconditions:
        - `sys.argv` is set as expected for a CLI entry point (when
          `argv` is `None`).

    Postconditions:
        - Any opened gate has been closed; the line is low on return.
    """
    parser = argparse.ArgumentParser(description="TENS relay hardware test.")
    parser.add_argument(
        "--backend", choices=("lgpio", "fake"), default=_resolve_default_backend(),
    )
    parser.add_argument("--pin", type=int, default=DEFAULT_TENS_PIN)
    parser.add_argument(
        "--active-high", action="store_true",
        help="Override the default active-LOW polarity. Pass this only "
             "if your relay asserts on a HIGH line level — the default "
             "matches the active-LOW opto-relay in the deployed box.",
    )
    parser.add_argument(
        "--pulse", type=float, default=None,
        help="Fire a single pulse for this many seconds.",
    )
    parser.add_argument(
        "--hold", type=float, default=None,
        help="Hold the gate on for this many seconds (alias for --pulse).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("tens")

    if args.pulse is None and args.hold is None:
        log.error("Specify --pulse N or --hold N (seconds).")
        return 1
    if args.pulse is not None and args.hold is not None:
        log.error("--pulse and --hold are mutually exclusive.")
        return 1
    duration = args.pulse if args.pulse is not None else args.hold

    log.info(
        "Opening TENS on BCM %d (active_%s, backend=%s).",
        args.pin,
        "high" if args.active_high else "low",
        args.backend,
    )
    try:
        with _build_controller(
            backend=args.backend,
            pin=args.pin,
            active_high=args.active_high,
        ) as ctrl:
            log.info("Pulsing for %.2fs", duration)
            ctrl.pulse(duration)
            log.info("Done; gate is low.")
    except KeyboardInterrupt:
        log.warning("Interrupted; gate forced low.")
        return 130
    except ImportError as exc:
        log.error("lgpio not available: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
