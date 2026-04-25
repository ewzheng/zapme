"""Entry point for `python -m zapme.src`.

Boots the runtime loop: initializes the GPIO gate in its safe (off) state,
opens the camera, loads the posture model, and hands control to the main
loop. All hardware-specific behavior lives in `zapme.src.runtime`.
"""

from __future__ import annotations


def main() -> None:
    """Boot the zapme runtime loop.

    Preconditions:
        - Process is running on the deployment target (Raspberry Pi 4) with
          camera and GPIO access available, OR a fake backend has been
          configured for development.

    Postconditions:
        - The GPIO gate is left in its safe (off) state on return, including
          when the loop exits via exception or signal.
    """
    raise NotImplementedError("runtime loop not implemented yet")


if __name__ == "__main__":
    main()
