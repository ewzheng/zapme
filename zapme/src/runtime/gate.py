"""Abstract gate to the EMS actuator's trigger line.

The runtime loop's only legal path to driving the EMS unit is through
a `Gate` instance. Two implementations live in this module:

- `LgpioGate` — real Raspberry Pi GPIO line driven via the `lgpio`
  library. The shipping path on the Pi.
- `FakeGate` — in-memory implementation that records transitions for
  tests and Windows-side development. Lets the entire runtime loop be
  exercised end-to-end without hardware.

All implementations honor the safety contract spelled out in
`.llm/llm.MD`:

- The line is **off (low)** at construction time, before any other code
  has a chance to touch it.
- `close()` always returns the line to off, even on its way out.
- The context-manager protocol is the recommended usage pattern so
  exceptions and early returns can't leave the line stuck high.

`lgpio` is imported lazily inside `LgpioGate.__init__` so this module
itself stays importable on Windows / macOS dev machines that don't
have it installed. Constructing `LgpioGate` without `lgpio` available
raises `ImportError`, which is what we want — fail loudly at the gate
boundary, not at top-level import.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any


@dataclass(frozen=True)
class GateTransition:
    """A single off↔on transition recorded by `FakeGate`.

    Attributes:
        timestamp: `time.perf_counter()` value at the moment of the
            transition. Useful for tests that need to assert debounce
            timing in addition to the transition itself.
        new_state: `True` if the line went off→on, `False` if on→off.
    """

    timestamp: float
    new_state: bool


class Gate(abc.ABC):
    """Single-pin abstraction over an actuator's gate line.

    Every implementation must:

    - Construct in the **off** state (line low) before returning from
      `__init__`. Callers should never need to call `set(False)`
      defensively right after construction.
    - Provide `set(active)` to drive the line high or low.
    - Provide `is_active()` for cheap state queries (no I/O required).
    - Provide `close()` that returns the line to off and releases any
      OS-level resources (file handles, GPIO claims).
    - Support the context-manager protocol so `with Gate(...) as g:`
      closes cleanly even on exceptions.
    """

    @abc.abstractmethod
    def set(self, active: bool) -> None:
        """Drive the gate line.

        Args:
            active: `True` to assert the line (energize EMS),
                `False` to release it.

        Preconditions:
            - The gate has not yet been `close()`-d.

        Postconditions:
            - The hardware (or in-memory) line reflects the requested
              state.
            - `is_active()` returns `active` afterward.
        """

    @abc.abstractmethod
    def is_active(self) -> bool:
        """Return the gate's last commanded state.

        Returns:
            `True` if the line is currently asserted, `False` otherwise.

        Preconditions:
            - The gate has been constructed.

        Postconditions:
            - No I/O is performed; the cost is a field read.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Drive the line low and release any OS resources.

        Idempotent: calling twice is safe. Any exception raised by the
        underlying release path is suppressed *after* the line has been
        driven low — we would rather leak a file handle than leave the
        EMS energized.

        Preconditions:
            - The gate has been constructed.

        Postconditions:
            - The line is low.
            - `is_active()` returns `False`.
            - Subsequent `set(...)` calls have undefined behavior; the
              caller should construct a fresh gate after `close()`.
        """

    def __enter__(self) -> Gate:
        """Return `self` for the `with` statement.

        Preconditions:
            - `__init__` has completed successfully.

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
        """Always `close()` the gate on context exit.

        Args:
            exc_type: Exception class if leaving the `with` block via
                an exception, else `None`.
            exc_val: Exception instance, or `None`.
            exc_tb: Traceback, or `None`.

        Preconditions:
            - `__enter__` completed.

        Postconditions:
            - `close()` has been called regardless of exception state.
            - This method does not suppress exceptions.
        """
        self.close()


class FakeGate(Gate):
    """In-memory `Gate` for tests and dev machines without GPIO.

    Records each off↔on transition (deduplicated against the current
    state, so two consecutive `set(True)` calls produce one transition).
    The total number of `set` calls is also tracked separately, which
    is occasionally useful for spotting code paths that thrash the
    gate without ever changing its state.
    """

    def __init__(self) -> None:
        """Initialize a fresh, off-state fake gate.

        Preconditions:
            - None.

        Postconditions:
            - `is_active()` returns `False`.
            - `transitions` is an empty list.
            - `set_call_count` is `0`.
        """
        self._active = False
        self.transitions: list[GateTransition] = []
        self.set_call_count = 0

    def set(self, active: bool) -> None:
        """Update the in-memory state and record any transition.

        Args:
            active: Requested new state.

        Preconditions:
            - `close()` has not been called.

        Postconditions:
            - `is_active()` returns `bool(active)`.
            - If the state changed, one new entry is appended to
              `transitions` with the current `time.perf_counter()`.
            - `set_call_count` is incremented regardless of state change.
        """
        new_state = bool(active)
        self.set_call_count += 1
        if new_state != self._active:
            self._active = new_state
            self.transitions.append(
                GateTransition(timestamp=time.perf_counter(), new_state=new_state)
            )

    def is_active(self) -> bool:
        """Return the last commanded state.

        Returns:
            Current in-memory state as a bool.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return self._active

    def close(self) -> None:
        """Drive the in-memory line low (recording the transition if any).

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - `is_active()` returns `False`.
            - If the gate was on, a final off-transition is recorded.
        """
        if self._active:
            self.set(False)


class LgpioGate(Gate):
    """`Gate` implementation backed by a real Raspberry Pi GPIO line.

    Imports `lgpio` lazily so this module can be imported on developer
    machines without the library installed. Constructing this class
    without `lgpio` available raises `ImportError`, which is correct
    behavior — the caller asked for hardware and there is none.

    Designed for the Pi-side EMS gating path: a single output pin that
    drives a transistor / optoisolator gating the EMS unit. The Pi
    never sources the EMS current itself.
    """

    def __init__(self, pin: int, active_high: bool = True, chip: int = 0) -> None:
        """Claim the requested GPIO pin and drive it low.

        Args:
            pin: BCM pin number to drive.
            active_high: When `True` (the default), `set(True)` writes
                `1` to the line. When `False`, `set(True)` writes `0`
                — useful when the gating circuit is wired with inverted
                logic (e.g., a PNP transistor).
            chip: GPIO chip index. `0` is the only chip on a Pi 4 (the
                BCM2711). Pi 5 has multiple chips; the one carrying the
                40-pin header is also `0`.

        Raises:
            ImportError: If the `lgpio` package is not installed.
            lgpio.error: If the chip cannot be opened or the pin
                cannot be claimed (e.g., already in use by another
                process).

        Preconditions:
            - `pin` is a valid BCM pin number on the target chip.
            - The current user has permission to access `/dev/gpiochip*`
              (members of the `gpio` group on Raspberry Pi OS).

        Postconditions:
            - The pin is claimed as an output and driven to its
              inactive level (low if `active_high`, else high).
            - `is_active()` returns `False`.
        """
        import lgpio

        self._lgpio = lgpio
        self._pin = pin
        self._active_high = active_high
        self._handle = lgpio.gpiochip_open(chip)
        inactive_level = 0 if active_high else 1
        try:
            lgpio.gpio_claim_output(self._handle, pin, inactive_level)
        except Exception:
            lgpio.gpiochip_close(self._handle)
            raise
        self._active = False
        self._closed = False

    def set(self, active: bool) -> None:
        """Write the requested state to the GPIO line.

        Args:
            active: `True` to assert the line (subject to
                `active_high`), `False` to release it.

        Raises:
            RuntimeError: If `close()` has already been called.
            lgpio.error: If the underlying write fails.

        Preconditions:
            - `__init__` completed and `close()` has not been called.

        Postconditions:
            - The hardware line is at the requested level.
            - `is_active()` returns `bool(active)`.
        """
        if self._closed:
            raise RuntimeError("set() called on a closed LgpioGate")
        new_state = bool(active)
        level = (1 if new_state else 0) if self._active_high else (0 if new_state else 1)
        self._lgpio.gpio_write(self._handle, self._pin, level)
        self._active = new_state

    def is_active(self) -> bool:
        """Return the last commanded state.

        Returns:
            `True` if the gate is currently asserted in software's view.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No I/O is performed.
        """
        return self._active

    def close(self) -> None:
        """Drive the line to its inactive level and release the chip handle.

        Idempotent. The line is taken low *before* the handle is freed,
        so even if the close path raises, the EMS is already off.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - The hardware line is at its inactive level.
            - The GPIO chip handle is freed (best-effort; a release
              failure is suppressed after the line is safe).
            - Subsequent `set(...)` calls raise `RuntimeError`.
        """
        if self._closed:
            return
        try:
            inactive_level = 0 if self._active_high else 1
            try:
                self._lgpio.gpio_write(self._handle, self._pin, inactive_level)
            finally:
                self._active = False
        finally:
            try:
                self._lgpio.gpiochip_close(self._handle)
            except Exception:
                pass
            self._closed = True
