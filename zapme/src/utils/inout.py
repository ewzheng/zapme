"""Audio output for spoken warnings and effect cues.

Plays pre-recorded clips by name (`"warning_1"`, `"warning_2"`,
`"zap"`, etc.) through the system's audio output. Used by the
runtime to escalate verbal warnings before the EMS gate ever fires.

Three implementations:

- `FileSpeaker` — plays WAV files via the platform's native audio
  command (`aplay` on Linux, `afplay` on macOS, PowerShell's
  `Media.SoundPlayer` on Windows). Non-blocking: spawns the audio
  subprocess and returns immediately.
- `FakeSpeaker` — records play calls in memory; for tests and
  silent dry-runs.
- `Speaker` — abstract base class so the runtime takes any
  implementation.

Missing clip files are logged and skipped, never raised. The runtime
must keep ticking even if the audio stack misbehaves.
"""

from __future__ import annotations

import abc
import logging
import subprocess
import sys
import threading
from pathlib import Path
from types import TracebackType


class Speaker(abc.ABC):
    """Abstract speaker: plays named clips through some output device.

    Implementations must:

    - Provide `play(clip_name)` that **does not block** the caller.
      The runtime loop calls `play` in its hot path; a blocking
      implementation would stall pose inference and trip the watchdog.
    - Provide `close()` that releases any background resources
      (subprocesses, threads) and is safe to call multiple times.
    - Support the context-manager protocol for `with` lifecycle.
    """

    @abc.abstractmethod
    def play(self, clip_name: str) -> None:
        """Play a named clip in the background.

        Args:
            clip_name: Logical clip identifier (e.g. `"warning_1"`).
                Implementations decide how to map it to a file or
                synthesizer call.

        Preconditions:
            - `close()` has not been called.

        Postconditions:
            - The audio is playing or queued; this method has returned
              without waiting for playback.
            - Unknown / missing clips log a warning and are silent.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release background resources and stop any ongoing playback.

        Idempotent.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - Any subprocesses / threads have been signalled to stop.
            - Subsequent `play()` calls have undefined behavior; the
              caller should construct a fresh speaker after `close()`.
        """

    def __enter__(self) -> "Speaker":
        """Return self for the `with` statement.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Always close on context exit.

        Preconditions:
            - `__enter__` completed.

        Postconditions:
            - `close()` has been called.
        """
        self.close()


class FakeSpeaker(Speaker):
    """In-memory `Speaker` for tests and `--no-audio` dry-runs.

    Records the sequence of clip names passed to `play()` so tests
    can assert the right warnings fired in the right order.
    """

    def __init__(self) -> None:
        """Initialize an empty play log.

        Preconditions:
            - None.

        Postconditions:
            - `played` is an empty list.
        """
        self.played: list[str] = []

    def play(self, clip_name: str) -> None:
        """Append `clip_name` to the play log.

        Args:
            clip_name: Logical clip identifier.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - `played[-1] == clip_name`.
            - `len(played)` increases by 1.
        """
        self.played.append(clip_name)

    def close(self) -> None:
        """No-op; nothing to release.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - No state change.
        """
        return


class FileSpeaker(Speaker):
    """Plays WAV files via the platform's native audio command.

    Spawns the audio command as a subprocess so playback runs in the
    background without blocking the runtime loop. Tracks spawned
    processes so `close()` can clean them up.

    Missing clip files and missing platform audio commands log a
    warning and silently skip — the runtime keeps running, just
    without sound.
    """

    def __init__(
        self,
        clips: dict[str, Path],
        logger: logging.Logger | None = None,
    ) -> None:
        """Build a speaker around a clip-name → file mapping.

        Args:
            clips: Map from logical clip name (e.g. `"warning_1"`)
                to the WAV file path that should play. Missing files
                are tolerated at `play()` time, not at construction.
            logger: Optional logger for diagnostic messages.

        Preconditions:
            - All values in `clips` are `Path`-like.

        Postconditions:
            - The speaker is ready to accept `play()` calls.
            - No subprocesses have been spawned yet.
        """
        self._clips = {name: Path(p) for name, p in clips.items()}
        self._logger = logger or logging.getLogger(__name__)
        self._procs: list[subprocess.Popen[bytes]] = []
        self._lock = threading.Lock()

    def play(self, clip_name: str) -> None:
        """Spawn the platform audio command on the requested clip.

        Args:
            clip_name: Logical clip identifier; must be a key in the
                `clips` mapping passed to `__init__`.

        Preconditions:
            - `close()` has not been called.

        Postconditions:
            - On success: a background subprocess is running and the
              method has returned without blocking.
            - On any failure (unknown clip, missing file, missing
              audio command, exception spawning subprocess): a warning
              is logged and the method returns silently.
        """
        path = self._clips.get(clip_name)
        if path is None:
            self._logger.warning("Speaker: unknown clip name '%s'", clip_name)
            return
        if not path.exists():
            self._logger.warning("Speaker: clip file missing: %s", path)
            return
        cmd = self._command_for(path)
        if cmd is None:
            self._logger.warning(
                "Speaker: no audio command available for platform %s", sys.platform
            )
            return
        try:
            with self._lock:
                self._procs = [p for p in self._procs if p.poll() is None]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                self._procs.append(proc)
        except FileNotFoundError as exc:
            self._logger.warning(
                "Speaker: audio binary not installed (%s); install it or use --no-audio",
                exc,
            )
        except Exception:
            self._logger.exception("Speaker: failed to play %s", path)

    @staticmethod
    def _command_for(path: Path) -> list[str] | None:
        """Pick the right OS command to play `path` non-blocking.

        Args:
            path: WAV file to play.

        Returns:
            A subprocess argv list, or `None` if the host platform is
            unsupported.

        Preconditions:
            - `path.exists()`.

        Postconditions:
            - For the returned command, calling `subprocess.Popen(cmd)`
              starts playback in the background.
        """
        if sys.platform.startswith("linux"):
            return ["aplay", "-q", str(path)]
        if sys.platform == "darwin":
            return ["afplay", str(path)]
        if sys.platform == "win32":
            return [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(New-Object Media.SoundPlayer '{path}').PlaySync()",
            ]
        return None

    def close(self) -> None:
        """Terminate any still-running playback subprocesses.

        Idempotent. Errors during termination are suppressed — the
        runtime is exiting anyway.

        Preconditions:
            - `__init__` completed.

        Postconditions:
            - All tracked subprocesses have been signalled to terminate.
            - The internal process list is cleared.
        """
        with self._lock:
            for proc in self._procs:
                try:
                    proc.terminate()
                except Exception:
                    pass
            self._procs.clear()
