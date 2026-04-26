"""Tests for `utils.tens_control.TensController`.

Uses `FakeGate` underneath so the entire API can be exercised on a
dev machine without GPIO. The blocking `pulse` / `hold` semantics
are verified against wall-clock timing with deliberately short
durations.
"""

from __future__ import annotations

import time

import pytest

from zapme.src.runtime.gate import FakeGate, Gate, GateTransition
from zapme.src.utils.tens_control import (
    DEFAULT_PULSE_S,
    DEFAULT_TENS_PIN,
    TensController,
    main,
)


def _new_controller() -> TensController:
    """Build a fake-backed controller, ready to use."""
    return TensController.open_fake()


def test_controller_implements_gate_interface() -> None:
    """`TensController` is substitutable anywhere a `Gate` is expected."""
    assert isinstance(_new_controller(), Gate)


def test_starts_off() -> None:
    """A freshly-constructed controller has its inner gate low."""
    ctrl = _new_controller()
    assert ctrl.is_active() is False


def test_set_delegates_to_inner_gate() -> None:
    """`set(True)` flips the inner gate and is reflected in `is_active()`."""
    ctrl = _new_controller()
    ctrl.set(True)
    assert ctrl.is_active() is True
    ctrl.set(False)
    assert ctrl.is_active() is False


def test_close_drives_inner_gate_low() -> None:
    """`close()` propagates to the inner gate."""
    ctrl = _new_controller()
    ctrl.set(True)
    ctrl.close()
    assert ctrl.is_active() is False


def test_pulse_records_on_then_off_transitions() -> None:
    """`pulse()` produces an off→on transition followed by on→off."""
    ctrl = _new_controller()
    fake = ctrl.gate
    assert isinstance(fake, FakeGate)
    ctrl.pulse(0.05)
    states = [t.new_state for t in fake.transitions]
    assert states == [True, False]


def test_pulse_duration_blocks_for_at_least_requested_time() -> None:
    """`pulse()` blocks the caller for approximately the requested duration."""
    ctrl = _new_controller()
    start = time.perf_counter()
    ctrl.pulse(0.1)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.09, f"pulse returned too quickly: {elapsed:.3f}s"
    assert elapsed < 0.5, f"pulse blocked too long: {elapsed:.3f}s"


def test_pulse_releases_gate_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """If `time.sleep` raises mid-pulse, the gate is still released.

    The most safety-critical guarantee for this API: even when
    interrupted, the EMS line must come back down before control
    returns to the caller.
    """
    ctrl = _new_controller()

    def _raise(_d: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("zapme.src.utils.tens_control.time.sleep", _raise)
    with pytest.raises(KeyboardInterrupt):
        ctrl.pulse(0.5)
    assert ctrl.is_active() is False


def test_hold_is_pulse_alias() -> None:
    """`hold()` produces the same on→off transition pattern as `pulse()`."""
    ctrl = _new_controller()
    fake = ctrl.gate
    assert isinstance(fake, FakeGate)
    ctrl.hold(0.05)
    states = [t.new_state for t in fake.transitions]
    assert states == [True, False]


def test_pulse_zero_or_negative_rejects() -> None:
    """A non-positive pulse duration rejects construction of the call."""
    ctrl = _new_controller()
    with pytest.raises(ValueError, match="duration_s"):
        ctrl.pulse(0.0)
    with pytest.raises(ValueError, match="duration_s"):
        ctrl.pulse(-1.0)
    assert ctrl.is_active() is False


def test_context_manager_closes_on_exit() -> None:
    """The `with` block leaves the inner gate low after a clean exit."""
    with TensController.open_fake() as ctrl:
        ctrl.set(True)
        assert ctrl.is_active() is True
    assert ctrl.is_active() is False


def test_context_manager_closes_on_exception() -> None:
    """The `with` block leaves the inner gate low even on exception."""
    ctrl = TensController.open_fake()
    with pytest.raises(ValueError, match="boom"):
        with ctrl:
            ctrl.set(True)
            assert ctrl.is_active() is True
            raise ValueError("boom")
    assert ctrl.is_active() is False


def test_default_pin_matches_team_script() -> None:
    """Default pin stays at 17 so existing wiring keeps working."""
    assert DEFAULT_TENS_PIN == 17


def test_main_cli_pulse_with_fake_backend() -> None:
    """The CLI runs end-to-end with the fake backend and returns 0."""
    rc = main(["--backend", "fake", "--pulse", "0.05"])
    assert rc == 0


def test_main_cli_requires_action() -> None:
    """The CLI rejects invocations without `--pulse` or `--hold`."""
    rc = main(["--backend", "fake"])
    assert rc == 1


def test_main_cli_rejects_both_pulse_and_hold() -> None:
    """`--pulse` and `--hold` are mutually exclusive."""
    rc = main(["--backend", "fake", "--pulse", "0.05", "--hold", "0.05"])
    assert rc == 1
