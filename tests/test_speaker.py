"""Tests for `utils.inout` Speaker classes.

`FileSpeaker` requires a real audio stack and is not unit-tested
here — it's exercised on the Pi at deployment time. The contract
every `Speaker` must honor is verified against `FakeSpeaker`.
"""

from __future__ import annotations

import pytest

from zapme.src.utils.inout import FakeSpeaker, FileSpeaker, Speaker


def test_fake_speaker_starts_with_empty_log() -> None:
    """A fresh fake speaker has played nothing."""
    s = FakeSpeaker()
    assert s.played == []


def test_fake_speaker_records_each_play_in_order() -> None:
    """`play(name)` appends to the log; order is preserved."""
    s = FakeSpeaker()
    s.play("warning_1")
    s.play("warning_2")
    s.play("zap")
    assert s.played == ["warning_1", "warning_2", "zap"]


def test_fake_speaker_records_duplicates() -> None:
    """Repeated plays of the same clip each get their own log entry."""
    s = FakeSpeaker()
    s.play("warning_1")
    s.play("warning_1")
    assert s.played == ["warning_1", "warning_1"]


def test_fake_speaker_close_is_noop() -> None:
    """Closing the fake doesn't drop the play log."""
    s = FakeSpeaker()
    s.play("warning_1")
    s.close()
    assert s.played == ["warning_1"]


def test_fake_speaker_context_manager() -> None:
    """`with FakeSpeaker()` works and closes cleanly."""
    with FakeSpeaker() as s:
        s.play("warning_1")
    assert s.played == ["warning_1"]


def test_fake_speaker_context_manager_closes_on_exception() -> None:
    """An exception inside `with` still calls close (no audio side effects)."""
    s = FakeSpeaker()
    with pytest.raises(RuntimeError, match="boom"):
        with s:
            s.play("warning_1")
            raise RuntimeError("boom")
    assert s.played == ["warning_1"]


def test_fake_speaker_play_sequence_appends_in_order() -> None:
    """`play_sequence` records all clips in the given order."""
    s = FakeSpeaker()
    s.play_sequence(["zapwarn", "zapscream"])
    assert s.played == ["zapwarn", "zapscream"]


def test_fake_speaker_play_sequence_empty_is_noop() -> None:
    """Empty sequence does not touch the play log."""
    s = FakeSpeaker()
    s.play_sequence([])
    assert s.played == []


def test_fake_speaker_play_sequence_after_play() -> None:
    """A `play()` followed by `play_sequence()` yields the combined order."""
    s = FakeSpeaker()
    s.play("firstwarn")
    s.play_sequence(["zapwarn", "zapscream"])
    assert s.played == ["firstwarn", "zapwarn", "zapscream"]


def test_fake_speaker_is_busy_always_false() -> None:
    """In-memory fake never has playback in flight."""
    s = FakeSpeaker()
    assert s.is_busy() is False
    s.play("firstwarn")
    assert s.is_busy() is False
    s.play_sequence(["zapwarn", "zapscream"])
    assert s.is_busy() is False


def test_implements_speaker_interface() -> None:
    """Both implementations are substitutable as `Speaker`."""
    assert isinstance(FakeSpeaker(), Speaker)
    assert issubclass(FileSpeaker, Speaker)


def test_file_speaker_unknown_clip_does_not_raise(tmp_path) -> None:
    """Asking `FileSpeaker` for an unknown clip logs a warning, never raises."""
    spk = FileSpeaker(clips={"hello": tmp_path / "hello.wav"})
    spk.play("not_in_map")
    spk.close()


def test_file_speaker_missing_file_does_not_raise(tmp_path) -> None:
    """A configured clip whose file is missing logs a warning, never raises."""
    spk = FileSpeaker(clips={"hello": tmp_path / "missing.wav"})
    spk.play("hello")
    spk.close()


def test_pick_path_single_path_returns_same_path(tmp_path) -> None:
    """A non-list value is returned verbatim."""
    p = tmp_path / "only.mp3"
    assert FileSpeaker._pick_path(p) is p


def test_pick_path_list_uses_random_choice(tmp_path) -> None:
    """A list is sampled via the supplied RNG."""
    import random as _rand
    paths = [tmp_path / f"v{i}.mp3" for i in range(5)]
    rng = _rand.Random(42)
    seen = {FileSpeaker._pick_path(paths, rng=rng) for _ in range(50)}
    # All five variants should appear with seed 42 over 50 picks.
    assert seen == set(paths)


def test_file_speaker_normalizes_single_item_list(tmp_path) -> None:
    """A list of one path collapses to that path internally."""
    p = tmp_path / "only.mp3"
    spk = FileSpeaker(clips={"clip": [p]})
    assert spk._clips["clip"] == p
    spk.close()


def test_file_speaker_keeps_multi_variant_list(tmp_path) -> None:
    """A list of two+ paths is preserved as a list (for randomization)."""
    paths = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    spk = FileSpeaker(clips={"clip": paths})
    assert spk._clips["clip"] == paths
    spk.close()
