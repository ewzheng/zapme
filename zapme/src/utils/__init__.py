"""Cross-cutting utilities shared across `model` and `runtime`.

This subpackage holds helpers that have no hardware dependency and no
model dependency on their own — config loading, logging setup, timing,
small data structures, and similar glue code.

Anything that imports `RPi.GPIO`, `picamera2`, `mmap`, or other
Pi-specific libraries belongs in `zapme.src.runtime`, not here. Anything
that loads or invokes the posture model belongs in `zapme.src.model`.
"""
