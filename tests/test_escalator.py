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
    release_grace: float = 0.0,
) -> WarningEscalator:
    """Build an escalator with deliberately short delays for fast tests.

    `release_grace` defaults to `0.0` so existing tests retain the
    pre-grace "release immediately resets to IDLE" semantics. Tests
    of the grace behavior pass an explicit non-zero value.
    """
    return WarningEscalator(
        EscalationConfig(
            warn_to_warn_s=warn_to_warn,
            warn_to_fire_s=warn_to_fire,
            fire_to_warn_s=fire_to_warn,
            release_grace_s=release_grace,
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


def test_audio_busy_pauses_warn_to_warn_countdown() -> None:
    """While audio_busy=True the warn-to-warn countdown freezes."""
    e = _make(warn_to_warn=0.1, warn_to_fire=0.1)
    assert e.step(True) == EscalationAction.WARN_1
    # Sleep enough to normally trigger WARN_2…
    time.sleep(0.15)
    # …but report the speaker as busy. No advance, no emit.
    for _ in range(5):
        assert e.step(True, audio_busy=True) == EscalationAction.NONE
    # Audio finishes; first non-busy step still inside warn_to_warn
    # window (which restarted at the last busy call) so still NONE.
    assert e.step(True, audio_busy=False) == EscalationAction.NONE
    # After the full warn_to_warn elapses post-audio, WARN_2 fires.
    time.sleep(0.15)
    assert e.step(True, audio_busy=False) == EscalationAction.WARN_2


def test_audio_busy_in_idle_does_not_emit_warn_1() -> None:
    """A re-engagement during leftover audio does not blast WARN_1."""
    e = _make()
    # Audio still playing from a previous cycle, debounce just went on.
    assert e.step(True, audio_busy=True) == EscalationAction.NONE
    # Audio ends; the next step starts the ladder fresh.
    assert e.step(True, audio_busy=False) == EscalationAction.WARN_1


def test_audio_busy_does_not_block_release_reset() -> None:
    """Releasing while audio_busy still resets the escalator to IDLE."""
    e = _make(warn_to_warn=0.1)
    assert e.step(True) == EscalationAction.WARN_1
    e.step(False, audio_busy=True)  # release should still reset
    # Next active step starts a new ladder from WARN_1.
    assert e.step(True) == EscalationAction.WARN_1


def test_release_within_grace_preserves_state() -> None:
    """A brief release within `release_grace_s` keeps the ladder."""
    e = _make(warn_to_warn=0.2, release_grace=1.0)
    assert e.step(True) == EscalationAction.WARN_1
    # Release for 50ms (well within 1s grace).
    e.step(False)
    time.sleep(0.05)
    e.step(False)
    # Re-engage. State should still be WAIT_WARN_2; not WARN_1 again.
    assert e.step(True) == EscalationAction.NONE
    # After warn_to_warn elapses (timer was paused during release),
    # WARN_2 fires.
    time.sleep(0.25)
    assert e.step(True) == EscalationAction.WARN_2


def test_release_beyond_grace_resets_to_idle() -> None:
    """A release longer than `release_grace_s` does fully reset."""
    e = _make(warn_to_warn=0.2, release_grace=0.05)
    assert e.step(True) == EscalationAction.WARN_1
    # Release past the grace window.
    e.step(False)
    time.sleep(0.1)
    e.step(False)  # second tick now exceeds grace → reset
    # Re-engage. State is IDLE → WARN_1 fires fresh.
    assert e.step(True) == EscalationAction.WARN_1


def test_release_grace_pauses_timer() -> None:
    """Time spent released doesn't count toward `warn_to_warn_s`."""
    e = _make(warn_to_warn=0.2, release_grace=2.0)
    assert e.step(True) == EscalationAction.WARN_1
    # Spend 100ms slouched (half the warn_to_warn window).
    time.sleep(0.1)
    # Release for 200ms — without the pause this would still be
    # under 0.2s of "active time", so WARN_2 should NOT fire yet.
    e.step(False)
    time.sleep(0.2)
    # Re-engage. Active time is still 100ms; need another 100ms.
    assert e.step(True) == EscalationAction.NONE
    time.sleep(0.15)
    # Now total active time exceeds warn_to_warn → WARN_2.
    assert e.step(True) == EscalationAction.WARN_2


def test_negative_grace_rejected() -> None:
    """Negative `release_grace_s` rejects construction."""
    with pytest.raises(ValueError, match="release_grace_s"):
        WarningEscalator(EscalationConfig(release_grace_s=-1.0))


def test_negative_delay_rejected() -> None:
    """Non-positive delays reject construction (any of the three)."""
    with pytest.raises(ValueError, match="delays"):
        WarningEscalator(EscalationConfig(warn_to_warn_s=0.0))
    with pytest.raises(ValueError, match="delays"):
        WarningEscalator(EscalationConfig(warn_to_fire_s=-1.0))
    with pytest.raises(ValueError, match="delays"):
        WarningEscalator(EscalationConfig(fire_to_warn_s=0.0))
