"""Tests for `model.features` and `model._common.clean_feature_window`.

Targets the pure-function logic that's central to model correctness:

- The `MLP_FEATURES` ordering invariant (drift here silently corrupts
  the trained model's input mapping).
- `compute_slouch_features` returning `None` when the shoulder anchor
  is unusable, vs. populating optional fields with `None` when only
  certain keypoints are missing.
- `SlouchFeatures.as_vector()` shape, dtype, and column ordering.
- `clean_feature_window` ffill/bfill/zerofill behavior in both
  time-major and feature-major orientations.
"""

from __future__ import annotations

import numpy as np
import pytest

from zapme.src.model._common import clean_feature_window
from zapme.src.model.features import (
    EAR_DROP_INDEX,
    LEFT_EAR,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_SHOULDER,
    MLP_FEATURES,
    NOSE,
    NUM_FEATURES,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    SlouchFeatures,
    compute_slouch_features,
)
from zapme.src.model.vision import Pose


def _blank_keypoints() -> np.ndarray:
    """Return a `(17, 3)` keypoint array with everything at zero confidence."""
    return np.zeros((17, 3), dtype=np.float32)


def _upright_pose(shoulder_y: float = 200.0) -> Pose:
    """Build a synthetic pose roughly representing an upright sitter.

    Both shoulders, both ears, both eyes, and the nose are placed at
    confident, anatomically-plausible positions. Hips are intentionally
    left at zero confidence (typical at a desk).
    """
    kp = _blank_keypoints()
    kp[LEFT_SHOULDER] = [80.0, shoulder_y, 0.95]
    kp[RIGHT_SHOULDER] = [200.0, shoulder_y, 0.95]
    kp[LEFT_EAR] = [110.0, shoulder_y - 80.0, 0.9]
    kp[RIGHT_EAR] = [170.0, shoulder_y - 80.0, 0.9]
    kp[LEFT_EYE] = [120.0, shoulder_y - 90.0, 0.9]
    kp[RIGHT_EYE] = [160.0, shoulder_y - 90.0, 0.9]
    kp[NOSE] = [140.0, shoulder_y - 75.0, 0.9]
    return Pose(keypoints=kp, bbox=np.zeros(4, dtype=np.float32), score=0.9)


def test_num_features_matches_tuple_length() -> None:
    """`NUM_FEATURES` is derived from `MLP_FEATURES` and stays in sync."""
    assert NUM_FEATURES == len(MLP_FEATURES)


def test_ear_drop_index_points_at_ear_drop() -> None:
    """The shared `EAR_DROP_INDEX` constant matches the tuple position."""
    assert MLP_FEATURES[EAR_DROP_INDEX] == "ear_drop"


def test_compute_returns_none_when_shoulders_low_conf() -> None:
    """No usable shoulders → no anchor → caller gets `None`."""
    kp = _blank_keypoints()
    pose = Pose(keypoints=kp, bbox=np.zeros(4, dtype=np.float32), score=0.9)
    assert compute_slouch_features(pose) is None


def test_compute_returns_none_when_shoulders_too_close() -> None:
    """Shoulder distance below `SHOULDER_WIDTH_MIN_PX` is rejected."""
    kp = _blank_keypoints()
    kp[LEFT_SHOULDER] = [100.0, 100.0, 0.95]
    kp[RIGHT_SHOULDER] = [110.0, 100.0, 0.95]
    pose = Pose(keypoints=kp, bbox=np.zeros(4, dtype=np.float32), score=0.9)
    assert compute_slouch_features(pose) is None


def test_compute_returns_features_for_upright_pose() -> None:
    """A confident upright pose produces a `SlouchFeatures` with finite anchors."""
    feats = compute_slouch_features(_upright_pose())
    assert feats is not None
    assert np.isfinite(feats.shoulder_width_px)
    assert np.isfinite(feats.shoulder_tilt_deg)
    assert feats.ear_drop is not None and feats.ear_drop < 0
    assert feats.nose_drop is not None and feats.nose_drop < 0


def test_optional_features_are_none_when_inputs_unreliable() -> None:
    """Features that need the nose / ears return `None` when those are unconfident."""
    kp = _blank_keypoints()
    kp[LEFT_SHOULDER] = [80.0, 200.0, 0.95]
    kp[RIGHT_SHOULDER] = [200.0, 200.0, 0.95]
    pose = Pose(keypoints=kp, bbox=np.zeros(4, dtype=np.float32), score=0.9)
    feats = compute_slouch_features(pose)
    assert feats is not None
    assert feats.ear_drop is None
    assert feats.nose_drop is None
    assert feats.ear_forward is None
    assert feats.head_pitch is None


def test_as_vector_shape_and_dtype() -> None:
    """`as_vector()` returns the expected shape and dtype."""
    feats = compute_slouch_features(_upright_pose())
    assert feats is not None
    vec = feats.as_vector()
    assert vec.shape == (NUM_FEATURES,)
    assert vec.dtype == np.float32


def test_as_vector_ordering_matches_mlp_features() -> None:
    """`as_vector()[i]` is the value of the field named `MLP_FEATURES[i]`.

    This is the invariant the trained model depends on. If anyone ever
    reorders `MLP_FEATURES` without updating `as_vector()` (or vice
    versa), this test must fail.
    """
    feats = compute_slouch_features(_upright_pose())
    assert feats is not None
    vec = feats.as_vector()
    for i, name in enumerate(MLP_FEATURES):
        attr = getattr(feats, name)
        expected = float("nan") if attr is None else float(attr)
        if np.isnan(expected):
            assert np.isnan(vec[i]), f"{name}: expected NaN, got {vec[i]}"
        else:
            assert vec[i] == pytest.approx(expected), f"{name}: mismatch"


def test_as_vector_emits_nan_for_none_optional_fields() -> None:
    """Optional fields that are `None` become `NaN` in the output vector."""
    kp = _blank_keypoints()
    kp[LEFT_SHOULDER] = [80.0, 200.0, 0.95]
    kp[RIGHT_SHOULDER] = [200.0, 200.0, 0.95]
    pose = Pose(keypoints=kp, bbox=np.zeros(4, dtype=np.float32), score=0.9)
    feats = compute_slouch_features(pose)
    assert feats is not None
    vec = feats.as_vector()
    ear_drop_idx = MLP_FEATURES.index("ear_drop")
    nose_drop_idx = MLP_FEATURES.index("nose_drop")
    assert np.isnan(vec[ear_drop_idx])
    assert np.isnan(vec[nose_drop_idx])


def test_clean_feature_window_no_nan_passthrough() -> None:
    """A NaN-free input is returned unchanged in value (a fresh copy)."""
    w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = clean_feature_window(w, time_axis=0)
    assert np.array_equal(out, w)
    assert out is not w


def test_clean_feature_window_forward_fill_time_major() -> None:
    """Trailing NaN is filled from the most recent non-NaN, per column."""
    w = np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]], dtype=np.float32)
    out = clean_feature_window(w, time_axis=0)
    assert np.array_equal(out, np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 4.0]]))


def test_clean_feature_window_backward_fill_time_major() -> None:
    """Leading NaN is filled from the next non-NaN value, per column."""
    w = np.array([[np.nan, np.nan], [np.nan, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = clean_feature_window(w, time_axis=0)
    assert np.array_equal(out, np.array([[5.0, 4.0], [5.0, 4.0], [5.0, 6.0]]))


def test_clean_feature_window_all_nan_column_zero_filled() -> None:
    """A column that is NaN everywhere is replaced with zeros."""
    w = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=np.float32)
    out = clean_feature_window(w, time_axis=0)
    assert out[0, 0] == 0.0
    assert out[1, 0] == 0.0
    assert out[0, 1] == 1.0
    assert out[1, 1] == 2.0


def test_clean_feature_window_feature_major_matches_transposed_time_major() -> None:
    """Both orientations produce the same result, modulo transpose.

    This is what lets the training pipeline (time-major input) and the
    runtime feature buffer (feature-major input) share a single cleaner.
    """
    w_time = np.array(
        [[1.0, np.nan, 3.0], [np.nan, 2.0, 4.0], [5.0, np.nan, np.nan]],
        dtype=np.float32,
    )
    cleaned_time = clean_feature_window(w_time, time_axis=0)
    cleaned_feat = clean_feature_window(w_time.T, time_axis=1)
    assert np.array_equal(cleaned_feat, cleaned_time.T)


def test_clean_feature_window_does_not_mutate_input() -> None:
    """The cleaner returns a fresh array; the input remains NaN-bearing."""
    w = np.array([[1.0, np.nan]], dtype=np.float32)
    snapshot = w.copy()
    _ = clean_feature_window(w, time_axis=0)
    assert np.array_equal(np.isnan(w), np.isnan(snapshot))


def test_clean_feature_window_rejects_invalid_axis() -> None:
    """Asking for a non-existent axis is loud, not silently wrong."""
    w = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="time_axis"):
        clean_feature_window(w, time_axis=2)
