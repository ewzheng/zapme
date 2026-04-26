"""Tests for the `Pulser` single-frame-pulse + cooldown safety policy.

The pulser is the last decision stage before the watchdog-protected
gate. Its safety contract is the most important one in the runtime:
no matter what the debouncer says, the gate must never stay asserted
for more than one frame, and once a pulse fires no further pulse can
occur until both the cooldown elapses *and* the debouncer goes
inactive then active again.
"""

from __future__ import annotations

import time

import pytest

from zapme.src.runtime.loop import Pulser, PulserConfig


def _make(cooldown: float = 0.2) -> Pulser:
    """Build a pulser with a short cooldown for fast test execution."""
    return Pulser(PulserConfig(cooldown_s=cooldown))


def test_starts_idle() -> None:
    """A fresh pulser is not in cooldown and has no pending state."""
    p = _make()
    assert p.is_in_cooldown() is False
    assert p.cooldown_remaining_s() == 0.0


def test_fires_on_first_rising_edge() -> None:
    """The first `True` after construction fires a pulse."""
    p = _make()
    assert p.step(True) is True


def test_returns_false_for_inactive_input() -> None:
    """An inactive input never fires a pulse."""
    p = _make()
    assert p.step(False) is False
    assert p.is_in_cooldown() is False


def test_fires_only_once_for_sustained_active() -> None:
    """A continuously-active input fires exactly one pulse, then stays low."""
    p = _make()
    assert p.step(True) is True
    for _ in range(10):
        assert p.step(True) is False


def test_cooldown_blocks_subsequent_pulses() -> None:
    """Within the cooldown window, no rising edge fires."""
    p = _make(cooldown=0.5)
    assert p.step(True) is True
    p.step(False)
    p.step(True)
    p.step(False)
    p.step(True)
    assert p.is_in_cooldown() is True


def test_cooldown_remaining_decreases() -> None:
    """`cooldown_remaining_s()` decreases monotonically until it hits zero."""
    p = _make(cooldown=0.3)
    p.step(True)
    first = p.cooldown_remaining_s()
    time.sleep(0.1)
    second = p.cooldown_remaining_s()
    assert first > second > 0.0


def test_can_fire_again_after_cooldown_with_fresh_edge() -> None:
    """After cooldown expires *and* the debouncer dips low + back high, fire again."""
    p = _make(cooldown=0.15)
    assert p.step(True) is True
    time.sleep(0.2)
    assert p.is_in_cooldown() is False
    p.step(False)
    assert p.step(True) is True


def test_no_re_fire_if_debouncer_stayed_active_through_cooldown() -> None:
    """If `step(True)` is fed continuously through the cooldown, no second pulse fires.

    This is the "give the user time to release" safety behavior. After
    cooldown ends, the user must let go (debouncer goes inactive) and
    re-engage (debouncer goes active again) before another pulse can
    fire — they don't get re-zapped just because they're still
    slouching.
    """
    p = _make(cooldown=0.15)
    assert p.step(True) is True
    end = time.perf_counter() + 0.3
    second_pulse_seen = False
    while time.perf_counter() < end:
        if p.step(True):
            second_pulse_seen = True
        time.sleep(0.01)
    assert second_pulse_seen is False
    assert p.is_in_cooldown() is False


def test_falling_edge_does_not_fire() -> None:
    """`True → False` transitions never fire a pulse."""
    p = _make()
    p.step(True)
    assert p.step(False) is False
    assert p.step(False) is False


def test_reset_clears_cooldown_and_edge_state() -> None:
    """`reset()` allows an immediate pulse on the next active input."""
    p = _make(cooldown=10.0)
    assert p.step(True) is True
    assert p.is_in_cooldown() is True
    p.reset()
    assert p.is_in_cooldown() is False
    assert p.step(True) is True


def test_zero_cooldown_disables_lockout() -> None:
    """`cooldown_s=0` allows back-to-back pulses on consecutive rising edges."""
    p = Pulser(PulserConfig(cooldown_s=0.0))
    assert p.step(True) is True
    assert p.step(False) is False
    assert p.step(True) is True
    assert p.step(False) is False
    assert p.step(True) is True


def test_negative_cooldown_rejected() -> None:
    """Negative `cooldown_s` rejects construction."""
    with pytest.raises(ValueError, match="cooldown_s"):
        Pulser(PulserConfig(cooldown_s=-1.0))
