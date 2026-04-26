"""Dry-run the full zapme pipeline without driving any real hardware.

Identical end-to-end pipeline as `python -m zapme.src`, except the
gate backend is forced to `FakeGate` so no GPIO line is ever
asserted. Every component the deployment uses runs for real:

- Real camera frames (via OpenCV)
- Real YOLO pose inference
- Real geometric features
- Real classifier (placeholder rule or trained ONNX, your choice)
- Real debouncer + pulser
- Real watchdog
- ...and a fake gate that just logs "I would have pulsed" instead
  of actually pulsing.

Useful for:

- Verifying the trained model + camera setup before committing to
  hardware deployment.
- Demoing the system on a laptop with no Pi attached.
- Tuning `--on-threshold`, `--cooldown`, etc. interactively without
  needing the EMS hardware nearby.

Usage:

    python scripts/dry_run.py
    python scripts/dry_run.py --camera 1
    python scripts/dry_run.py --weights models/slouch_cnn.onnx --cooldown 5
    python scripts/dry_run.py --log-interval 0.2 --log-level DEBUG

Any argument the regular `python -m zapme.src` accepts is forwarded
through, *except* `--backend` which is always forced to `fake`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Self-bootstrap so the script works regardless of whether
# `pip install -e .` succeeded. Adds the repo root (parent of
# `scripts/`) to sys.path so `from zapme...` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zapme.src.__main__ import main

DRY_RUN_BANNER = """
================================================================
  ZAPME DRY RUN — no real hardware is being driven.
  Every "Pulse fired" log line below is a SIMULATED zap.
  Press Ctrl-C to quit cleanly.
================================================================
"""


def _strip_backend_args(argv: list[str]) -> list[str]:
    """Remove any user-supplied `--backend X` from the argv passthrough.

    Args:
        argv: Argument list from the dry-run invocation
            (`sys.argv[1:]`-style; no script name).

    Returns:
        A copy of `argv` with both `--backend X` (two-word form) and
        `--backend=X` (single-token form) removed. The caller appends
        a hard-coded `--backend fake` afterward.

    Preconditions:
        - `argv` may contain at most one occurrence of `--backend`
          (argparse would reject more anyway).

    Postconditions:
        - The returned list is independent of `argv` (a fresh list).
        - No `--backend*` argument remains.
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


def cli() -> int:
    """Run the zapme entry point with `--backend fake` forced.

    Returns:
        Whatever `zapme.src.main()` returns: `0` on clean exit, `1`
        if the watchdog tripped or the camera failed.

    Preconditions:
        - `sys.argv` is set as expected for a CLI entry point.

    Postconditions:
        - The fake gate has been closed; `sys.argv` may have been
          rewritten in place to inject `--backend fake`.
    """
    print(DRY_RUN_BANNER)
    forwarded = _strip_backend_args(sys.argv[1:])
    sys.argv = [sys.argv[0], "--backend", "fake", *forwarded]
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
