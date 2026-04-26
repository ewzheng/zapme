"""Tests for the `Watchdog` heartbeat fail-safe.

These tests use real time and a daemon thread, so they're slightly
slower than pure-logic tests. Timeouts are kept tiny (0.2s and below)
to keep the suite fast; on a heavily-loaded CI machine they could
flake, in which case bumping each timeout by a constant factor is the
expected fix.
"""

from __future__ import annotations

import time

import pytest

from zapme.src.runtime.gate import FakeGate
from zapme.src.runtime.watchdog import Watchdog


def test_does_not_trip_while_heartbeating() -> None:
    """A loop that calls `heartbeat()` faster than the timeout never trips."""
    gate = FakeGate()
    wd = Watchdog(gate, timeout_s=0.2, check_interval_s=0.02)
    try:
        wd.start()
        end = time.perf_counter() + 0.3
        while time.perf_counter() < end:
            wd.heartbeat()
            time.sleep(0.01)
        assert wd.is_tripped() is False
    finally:
        wd.stop()


def test_trips_after_timeout_without_heartbeat() -> None:
    """Stopping heartbeats causes the watchdog to trip within ~`timeout_s`."""
    gate = FakeGate()
    wd = Watchdog(gate, timeout_s=0.15, check_interval_s=0.02)
    try:
        wd.start()
        time.sleep(0.4)
        assert wd.is_tripped() is True
    finally:
        wd.stop()


def test_trip_drives_gate_low() -> None:
    """A tripped watchdog forces the gate off, even if it was on."""
    gate = FakeGate()
    wd = Watchdog(gate, timeout_s=0.15, check_interval_s=0.02)
    try:
        wd.start()
        wd.set_active(True)
        assert gate.is_active() is True
        time.sleep(0.4)
        assert wd.is_tripped() is True
        assert gate.is_active() is False
    finally:
        wd.stop()


def test_set_active_true_ignored_after_trip() -> None:
    """Once tripped, the watchdog refuses to re-assert the gate.

    The safe state is sticky. This is the most important guarantee:
    nothing should be able to revive an EMS line that the watchdog
    has already shut down.
    """
    gate = FakeGate()
    wd = Watchdog(gate, timeout_s=0.15, check_interval_s=0.02)
    try:
        wd.start()
        time.sleep(0.4)
        assert wd.is_tripped() is True
        wd.set_active(True)
        assert gate.is_active() is False
    finally:
        wd.stop()


def test_set_active_false_always_honored() -> None:
    """Releasing the gate is always safe and always honored."""
    gate = FakeGate()
    wd = Watchdog(gate, timeout_s=0.5, check_interval_s=0.05)
    try:
        wd.start()
        wd.heartbeat()
        wd.set_active(True)
        assert gate.is_active() is True
        wd.set_active(False)
        assert gate.is_active() is False
    finally:
        wd.stop()


def test_stop_drives_gate_low_and_is_idempotent() -> None:
    """`stop()` always leaves the gate low and is safe to call twice."""
    gate = FakeGate()
    wd = Watchdog(gate, timeout_s=0.5, check_interval_s=0.05)
    wd.start()
    wd.heartbeat()
    wd.set_active(True)
    wd.stop()
    assert gate.is_active() is False
    wd.stop()
    assert gate.is_active() is False


def test_context_manager_starts_and_stops() -> None:
    """`with` block starts the monitor on entry, stops on exit."""
    gate = FakeGate()
    with Watchdog(gate, timeout_s=0.5, check_interval_s=0.05) as wd:
        wd.heartbeat()
        wd.set_active(True)
        assert gate.is_active() is True
    assert gate.is_active() is False


def test_context_manager_closes_on_exception() -> None:
    """An exception inside the `with` block still leaves the gate low."""
    gate = FakeGate()
    with pytest.raises(RuntimeError, match="boom"):
        with Watchdog(gate, timeout_s=0.5, check_interval_s=0.05) as wd:
            wd.heartbeat()
            wd.set_active(True)
            assert gate.is_active() is True
            raise RuntimeError("boom")
    assert gate.is_active() is False


def test_invalid_config_rejects() -> None:
    """Bad timeout / check_interval combinations reject construction."""
    gate = FakeGate()
    with pytest.raises(ValueError):
        Watchdog(gate, timeout_s=0.0)
    with pytest.raises(ValueError):
        Watchdog(gate, timeout_s=0.1, check_interval_s=0.0)
    with pytest.raises(ValueError):
        Watchdog(gate, timeout_s=0.1, check_interval_s=0.2)
