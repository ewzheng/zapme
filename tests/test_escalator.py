"""Tests for `runtime.loop.WarningEscalator`.

Verifies the two-warnings-before-zap ladder, idle reset on release,
and that `FIRE` is sticky-once (only emitted on the rising edge of
the FIRED state).
"""

from __future__ import annotations

import time

import pytest

from zapme.src.runtime.loop import (
    EscalationAction,
    EscalationConfig,
    WarningEscalator,
)


def _make(
    warn_to_warn: float = 0.1,
    warn_to_fire: float = 0.1,
    fire_to_warn: float = 0.1,
) -> WarningEscalator:
    """Build an escalator with deliberately short delays for fast tests."""
    return WarningEscalator(
        EscalationConfig(
            warn_to_warn_s=warn_to_warn,
            warn_to_fire_s=warn_to_fire,
            fire_to_warn_s=fire_to_warn,
        )
    )


def test_idle_returns_none() -> None:
    """A fresh escalator with `False` input never escalates."""
    e = _make()
    for _ in range(20):
        assert e.step(False) == EscalationAction.NONE


def test_first_active_emits_warn_1() -> None:
    """The very first `step(True)` issues warning 1."""
    e = _make()
    assert e.step(True) == EscalationAction.WARN_1


def test_second_step_within_delay_returns_none() -> None:
    """Repeated `step(True)` within the delay window does not re-warn."""
    e = _make(warn_to_warn=0.5)
    assert e.step(True) == EscalationAction.WARN_1
    for _ in range(5):
        assert e.step(True) == EscalationAction.NONE


def test_warn_2_after_delay() -> None:
    """After `warn_to_warn_s` of sustained activity, warning 2 fires."""
    e = _make(warn_to_warn=0.1)
    assert e.step(True) == EscalationAction.WARN_1
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.WARN_2


def test_fire_after_full_ladder() -> None:
    """After warning 2 + the second delay, FIRE is emitted."""
    e = _make(warn_to_warn=0.1, warn_to_fire=0.1)
    assert e.step(True) == EscalationAction.WARN_1
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.WARN_2
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.FIRE


def test_warnings_cycle_after_fire() -> None:
    """Sustained slouch past FIRE re-enters the ladder after fire_to_warn_s.

    The warnings are *not* sticky like the zap. An uncorrected user
    keeps getting WARN_1 → WARN_2 → FIRE again, separated by the
    configured delays.
    """
    e = _make(warn_to_warn=0.1, warn_to_fire=0.1, fire_to_warn=0.1)
    # First cycle
    assert e.step(True) == EscalationAction.WARN_1
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.WARN_2
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.FIRE
    # During the post-fire silence, no actions emit
    assert e.step(True) == EscalationAction.NONE
    # After fire_to_warn_s elapses, the cycle restarts with WARN_1
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.WARN_1
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.WARN_2
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.FIRE


def test_post_fire_silence_emits_none() -> None:
    """During the fire_to_warn window, no actions are emitted."""
    e = _make(warn_to_warn=0.05, warn_to_fire=0.05, fire_to_warn=0.5)
    e.step(True)
    time.sleep(0.07)
    e.step(True)
    time.sleep(0.07)
    assert e.step(True) == EscalationAction.FIRE
    # Even with sustained activity, post-fire is silent until elapsed
    for _ in range(10):
        assert e.step(True) == EscalationAction.NONE
        time.sleep(0.01)


def test_release_resets_to_idle() -> None:
    """Going inactive at any point resets the ladder."""
    e = _make(warn_to_warn=0.1, warn_to_fire=0.1)
    assert e.step(True) == EscalationAction.WARN_1
    time.sleep(0.15)
    assert e.step(True) == EscalationAction.WARN_2
    e.step(False)
    # Next active should start over from warning 1
    assert e.step(True) == EscalationAction.WARN_1


def test_release_during_post_fire_resets() -> None:
    """Releasing after FIRE skips the post-fire wait and starts the ladder fresh."""
    e = _make(warn_to_warn=0.05, warn_to_fire=0.05, fire_to_warn=10.0)
    e.step(True)
    time.sleep(0.07)
    e.step(True)
    time.sleep(0.07)
    e.step(True)  # FIRE
    # Without release, post-fire silence would last 10s. With release, immediate reset.
    e.step(False)
    assert e.step(True) == EscalationAction.WARN_1


def test_reset_method_returns_to_idle() -> None:
    """`reset()` clears state regardless of where the escalator is."""
    e = _make()
    e.step(True)
    e.reset()
    assert e.step(True) == EscalationAction.WARN_1


def test_negative_delay_rejected() -> None:
    """Non-positive delays reject construction (any of the three)."""
    with pytest.raises(ValueError, match="delays"):
        WarningEscalator(EscalationConfig(warn_to_warn_s=0.0))
    with pytest.raises(ValueError, match="delays"):
        WarningEscalator(EscalationConfig(warn_to_fire_s=-1.0))
    with pytest.raises(ValueError, match="delays"):
        WarningEscalator(EscalationConfig(fire_to_warn_s=0.0))
