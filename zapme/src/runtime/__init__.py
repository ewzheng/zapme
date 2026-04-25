"""Hardware I/O and control loop for the Raspberry Pi 4 deployment.

This subpackage owns everything Pi-specific: camera capture, GPIO setup,
MMIO writes to `/dev/gpiomem`, the EMS gating policy, the main inference
loop, and the watchdog that returns the gate to its safe state on
failure.

The EMS trigger line defaults to *off* and must be explicitly enabled.
Any code path that can assert the line is paired with a watchdog that
releases it on missing frames, model errors, exceptions, or process exit.
"""
