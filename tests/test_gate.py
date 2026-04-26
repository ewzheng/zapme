"""Tests for `runtime.gate` — the safety-critical EMS line abstraction.

`FakeGate` is the only implementation testable on a dev machine
without GPIO hardware; `LgpioGate` requires a Pi, so its tests live
on-device. The contract every `Gate` must honor is enforced here
against `FakeGate`, but the same assertions would hold for
`LgpioGate`.
"""

from __future__ import annotations

import pytest

from zapme.src.runtime.gate import FakeGate, Gate, GateTransition


def test_fake_gate_starts_off() -> None:
    """A freshly-constructed gate is in the off state with no transitions."""
    g = FakeGate()
    assert g.is_active() is False
    assert g.transitions == []
    assert g.set_call_count == 0


def test_fake_gate_redundant_off_is_noop() -> None:
    """`set(False)` on an already-off gate records no transition."""
    g = FakeGate()
    g.set(False)
    assert g.is_active() is False
    assert g.transitions == []
    assert g.set_call_count == 1


def test_fake_gate_off_to_on_records_transition() -> None:
    """An off→on flip records exactly one transition with the new state."""
    g = FakeGate()
    g.set(True)
    assert g.is_active() is True
    assert len(g.transitions) == 1
    assert g.transitions[0].new_state is True


def test_fake_gate_consecutive_on_records_one_transition() -> None:
    """Repeated `set(True)` calls are deduplicated against the current state."""
    g = FakeGate()
    g.set(True)
    g.set(True)
    g.set(True)
    assert g.is_active() is True
    assert len(g.transitions) == 1
    assert g.set_call_count == 3


def test_fake_gate_on_off_records_two_transitions() -> None:
    """Each direction change adds a transition with the right `new_state`."""
    g = FakeGate()
    g.set(True)
    g.set(False)
    assert g.is_active() is False
    assert len(g.transitions) == 2
    assert [t.new_state for t in g.transitions] == [True, False]


def test_fake_gate_close_drives_low() -> None:
    """`close()` returns the line to off and records the final transition."""
    g = FakeGate()
    g.set(True)
    g.close()
    assert g.is_active() is False
    assert g.transitions[-1].new_state is False


def test_fake_gate_close_idempotent() -> None:
    """Calling `close()` twice does not double-record an off transition."""
    g = FakeGate()
    g.set(True)
    g.close()
    n_transitions = len(g.transitions)
    g.close()
    assert len(g.transitions) == n_transitions
    assert g.is_active() is False


def test_fake_gate_context_manager_closes_on_exit() -> None:
    """`with` block exits cleanly and leaves the line low."""
    with FakeGate() as g:
        g.set(True)
        assert g.is_active() is True
    assert g.is_active() is False
    assert g.transitions[-1].new_state is False


def test_fake_gate_context_manager_closes_on_exception() -> None:
    """`with` block leaves the line low even when exiting via an exception.

    This is the most safety-critical guarantee: an EMS gate left high
    after an exception is the failure mode we never want.
    """
    g = FakeGate()
    with pytest.raises(ValueError, match="boom"):
        with g:
            g.set(True)
            assert g.is_active() is True
            raise ValueError("boom")
    assert g.is_active() is False
    assert g.transitions[-1].new_state is False


def test_fake_gate_implements_gate_interface() -> None:
    """`FakeGate` is a `Gate` so it can be substituted in tests."""
    assert isinstance(FakeGate(), Gate)


def test_gate_transition_dataclass_is_frozen() -> None:
    """`GateTransition` is immutable so callers can't tamper with the log."""
    t = GateTransition(timestamp=1.0, new_state=True)
    with pytest.raises(Exception):
        t.timestamp = 2.0  # type: ignore[misc]
