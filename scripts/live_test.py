"""Live end-to-end test on real hardware. Drives a real EMS pulse.

This is the script you run on the Pi when you actually want to
exercise *every* part of the pipeline — camera capture, YOLO pose,
the trained classifier, the debouncer, the warning escalator, the
audio clips, the pulser, the watchdog, and a real GPIO assertion
into the EMS relay.

It's a thin wrapper over `python -m zapme.src` that:

1. Prints a loud "LIVE" banner so you don't confuse it with `dry_run`.
2. Fires a brief relay-sanity pulse *before* the loop starts, so you
   know the GPIO line is wired up before you trust it to fire on a
   slouch detection. Skip with `--no-self-test`.
3. Forces `--backend lgpio` regardless of what the host platform
   would default to.
4. Uses **shortened warning intervals** by default
   (`--warn-to-warn 2 --warn-to-fire 2 --fire-to-warn 5`) so a live
   test cycle takes ~5 seconds of slouching instead of 30. Override
   any of those by passing them explicitly — your values win.

Any other argument the regular entry point accepts is forwarded
through (camera index, model weights, thresholds, etc.).

Usage:

    python scripts/live_test.py
    python scripts/live_test.py --no-self-test
    python scripts/live_test.py --pulse-test 0.1
    python scripts/live_test.py --warn-to-warn 5 --warn-to-fire 5

Run it under systemd via `zapme.service` for the unattended
deployment scenario; this script is for the operator-in-the-loop
hardware bring-up.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Self-bootstrap so the script works regardless of whether
# `pip install -e .` succeeded. Adds the repo root (parent of
# `scripts/`) to sys.path so `from zapme...` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zapme.src.__main__ import main
from zapme.src.utils.tens_control import DEFAULT_TENS_PIN, TensController

LIVE_BANNER = """
================================================================
  ZAPME LIVE TEST — REAL EMS HARDWARE WILL BE DRIVEN.
  Each "Pulse fired" log line below is an actual GPIO assertion.
  Confirm the EMS unit is at a safe intensity before continuing.
  Press Ctrl-C at any time to drop the line and exit.
================================================================
"""

DEFAULT_SELF_TEST_PULSE_S = 0.05

# Faster cadence than the production defaults so an operator can
# iterate on slouch detection without sitting through 30s ladders.
LIVE_TEST_OVERRIDES: dict[str, str] = {
    "--warn-to-warn": "2",
    "--warn-to-fire": "2",
    "--fire-to-warn": "5",
}


def _has_arg(argv: list[str], name: str) -> bool:
    """Return True if `name` (with or without `=value`) is present in argv.

    Args:
        argv: Forwarded argument list (no script name).
        name: Long-option flag including the leading dashes.

    Returns:
        Whether the user already supplied this flag.

    Preconditions:
        - `name` starts with `--`.

    Postconditions:
        - No state change.
    """
    prefix = f"{name}="
    return any(a == name or a.startswith(prefix) for a in argv)


def _strip_backend_args(argv: list[str]) -> list[str]:
    """Remove any user-supplied `--backend X` so we can force `lgpio`.

    Args:
        argv: Forwarded argument list (no script name).

    Returns:
        A copy of `argv` with both `--backend X` and `--backend=X`
        removed.

    Preconditions:
        - `argv` may contain at most one `--backend` (argparse would
          reject more anyway).

    Postconditions:
        - The returned list contains no `--backend*` token.
    """
    out: list[str] = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == "--backend":
            skip_next = True
            continue
        if a.startswith("--backend="):
            continue
        out.append(a)
    return out


def _self_test(pin: int, active_high: bool, duration_s: float) -> int:
    """Fire one short pulse to confirm the GPIO line / relay is alive.

    Args:
        pin: BCM GPIO pin wired to the EMS relay trigger.
        active_high: Pin polarity. `True` = HIGH asserts the relay.
        duration_s: How long to hold the line asserted.

    Returns:
        `0` on success, `1` if `lgpio` could not be opened.

    Preconditions:
        - The current process has access to `/dev/gpiochip*`.

    Postconditions:
        - The line has been held high for `duration_s` and then
          driven low again, regardless of exception state.
    """
    log = logging.getLogger("live_test.selftest")
    log.info(
        "Self-test: pulsing BCM %d for %.3fs (active_%s).",
        pin, duration_s, "high" if active_high else "low",
    )
    try:
        with TensController.open_pi(pin=pin, active_high=active_high) as ctrl:
            ctrl.pulse(duration_s)
    except ImportError as exc:
        log.error("lgpio not available: %s — install it before running live.", exc)
        return 1
    except Exception:
        log.exception("Self-test failed; not entering the loop.")
        return 1
    log.info("Self-test OK — relay clicked. Entering full pipeline.")
    return 0


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split out `live_test`-only flags from the rest, which gets forwarded.

    Args:
        argv: `sys.argv[1:]`-style argument list.

    Returns:
        A pair of `(wrapper_namespace, forwarded_argv)`. The wrapper
        namespace carries the self-test settings; `forwarded_argv` is
        everything else, ready to be patched into `sys.argv` for the
        main entry point.

    Preconditions:
        - None.

    Postconditions:
        - The returned forwarded list contains no `--no-self-test` /
          `--pulse-test` tokens.
    """
    parser = argparse.ArgumentParser(
        add_help=False, description="zapme live test wrapper",
    )
    parser.add_argument(
        "--no-self-test", action="store_true",
        help="Skip the relay sanity pulse before entering the loop.",
    )
    parser.add_argument(
        "--pulse-test", type=float, default=DEFAULT_SELF_TEST_PULSE_S,
        help=(
            f"Self-test pulse duration in seconds. Default {DEFAULT_SELF_TEST_PULSE_S}s "
            "(short, just to confirm the relay clicks)."
        ),
    )
    return parser.parse_known_args(argv)


def cli() -> int:
    """Run the full pipeline on real hardware after a relay self-test.

    Returns:
        `0` on clean exit, `1` on watchdog trip / camera failure /
        self-test failure.

    Preconditions:
        - Running on the Pi with `lgpio` installed and GPIO access.
        - EMS unit is connected at a safe intensity.

    Postconditions:
        - The GPIO line is low on return regardless of how the loop
          exited (clean exit, exception, signal, watchdog).
    """
    print(LIVE_BANNER)

    wrapper_args, forwarded = parse_wrapper_args(sys.argv[1:])

    # Pull the gate pin / polarity from forwarded argv so the
    # self-test uses the same wiring as the loop will.
    pin_parser = argparse.ArgumentParser(add_help=False)
    pin_parser.add_argument("--gate-pin", type=int, default=DEFAULT_TENS_PIN)
    pin_parser.add_argument("--gate-active-high", action="store_true")
    pin_args, _ = pin_parser.parse_known_args(forwarded)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not wrapper_args.no_self_test:
        rc = _self_test(
            pin=pin_args.gate_pin,
            active_high=pin_args.gate_active_high,
            duration_s=wrapper_args.pulse_test,
        )
        if rc != 0:
            return rc

    # Strip the user's --backend (if any) and force lgpio.
    forwarded = _strip_backend_args(forwarded)

    # Inject faster-cadence defaults only where the user hasn't set
    # them explicitly.
    for flag, value in LIVE_TEST_OVERRIDES.items():
        if not _has_arg(forwarded, flag):
            forwarded.extend([flag, value])

    sys.argv = [sys.argv[0], "--backend", "lgpio", *forwarded]
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
