"""Tests for the gate-decision `Debouncer` in `runtime.loop`.

The debouncer is what stops a single noisy frame from triggering an
EMS pulse. Its hysteresis is configured by two thresholds (`on` and
`off`) and a minimum fraction of recent samples that must clear the
on-threshold to flip the gate on.
"""

from __future__ import annotations

import math

import pytest

from zapme.src.runtime.loop import Debouncer, DebouncerConfig


def _make(
    window: int = 10,
    on: float = 0.7,
    off: float = 0.4,
    fraction: float = 0.6,
) -> Debouncer:
    """Build a debouncer with explicit, easily-overridable parameters."""
    return Debouncer(
        DebouncerConfig(
            window_size=window,
            on_threshold=on,
            off_threshold=off,
            min_on_fraction=fraction,
        )
    )


def test_starts_inactive() -> None:
    """A fresh debouncer never flips on without input."""
    d = _make()
    assert d.is_active() is False


def test_low_probs_never_trigger() -> None:
    """A long stream of low probabilities leaves the gate off."""
    d = _make()
    for _ in range(50):
        assert d.update(0.1) is False
    assert d.is_active() is False


def test_sustained_high_eventually_triggers() -> None:
    """Enough samples above `on_threshold` flip the gate on."""
    d = _make(window=10, on=0.7, fraction=0.6)
    triggered = False
    for _ in range(20):
        if d.update(0.9):
            triggered = True
            break
    assert triggered, "sustained high probability should have flipped the gate on"


def test_below_on_fraction_does_not_trigger() -> None:
    """5/10 above on-threshold (below 60%) keeps the gate off."""
    d = _make(window=10, on=0.7, fraction=0.6)
    pattern = [0.9, 0.1] * 5  # exactly 50% above
    for p in pattern:
        d.update(p)
    assert d.is_active() is False


def test_off_threshold_releases_active_gate() -> None:
    """Once active, mean probability dropping below `off_threshold` releases it."""
    d = _make(window=10, on=0.7, off=0.4, fraction=0.6)
    for _ in range(15):
        d.update(0.9)
    assert d.is_active() is True
    for _ in range(15):
        d.update(0.05)
    assert d.is_active() is False


def test_hysteresis_no_chatter_in_middle_band() -> None:
    """Probabilities between `off` and `on` neither trigger nor release.

    The middle band is the whole point of two thresholds; if the gate
    flapped here the user would feel a jittering EMS line.
    """
    d = _make(window=10, on=0.7, off=0.4, fraction=0.6)
    for _ in range(30):
        d.update(0.55)
    assert d.is_active() is False
    for _ in range(15):
        d.update(0.9)
    assert d.is_active() is True
    for _ in range(30):
        d.update(0.55)
    assert d.is_active() is True, "mean above off_threshold should keep gate on"


def test_nan_treated_as_zero() -> None:
    """NaN inputs are clamped to 0 — never accidentally trigger the gate."""
    d = _make()
    for _ in range(50):
        result = d.update(float("nan"))
    assert result is False
    assert d.is_active() is False


def test_input_clamped_to_unit_range() -> None:
    """Out-of-range inputs are clamped to `[0, 1]`."""
    d = _make(window=5, on=0.7, fraction=0.6)
    for _ in range(10):
        d.update(5.0)
    assert d.is_active() is True
    for _ in range(10):
        d.update(-2.0)
    assert d.is_active() is False


def test_reset_returns_to_initial_state() -> None:
    """`reset()` clears the window and forces the decision back to off."""
    d = _make()
    for _ in range(15):
        d.update(0.9)
    assert d.is_active() is True
    d.reset()
    assert d.is_active() is False
    d.update(0.9)
    assert d.is_active() is False


def test_invalid_config_off_above_on_rejects() -> None:
    """Off-threshold at or above on-threshold rejects construction.

    No hysteresis margin → no point pretending the debouncer works.
    """
    with pytest.raises(ValueError, match="on_threshold"):
        _make(on=0.5, off=0.6)
    with pytest.raises(ValueError, match="on_threshold"):
        _make(on=0.5, off=0.5)


def test_invalid_config_thresholds_outside_unit_range() -> None:
    """Thresholds outside `[0, 1]` reject construction."""
    with pytest.raises(ValueError):
        _make(on=1.5)
    with pytest.raises(ValueError):
        _make(off=-0.1)


def test_invalid_config_zero_window_rejects() -> None:
    """A non-positive `window_size` rejects construction."""
    with pytest.raises(ValueError):
        _make(window=0)


def test_invalid_min_on_fraction_rejects() -> None:
    """`min_on_fraction` must lie in `(0, 1]`."""
    with pytest.raises(ValueError):
        Debouncer(DebouncerConfig(min_on_fraction=0.0))
    with pytest.raises(ValueError):
        Debouncer(DebouncerConfig(min_on_fraction=1.5))
