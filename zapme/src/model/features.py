"""Geometric feature engineering on top of raw pose keypoints.

The slouch classifier does not consume raw `(x, y, confidence)` keypoints
directly. It consumes a small vector of *scale-invariant* derived
features, normalized by inter-shoulder distance so they generalize
across users sitting at different distances from the camera.

This module is pure NumPy and contains no model code. It is safe to
import on the Pi alongside `vision.py`, and is unit-testable on any
machine without a camera or GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from zapme.src.model.vision import Pose

NOSE = 0
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

KEYPOINT_CONF_MIN = 0.3
SHOULDER_WIDTH_MIN_PX = 20.0

MLP_FEATURES: tuple[str, ...] = (
    "ear_drop",
    "nose_drop",
    "ear_forward",
    "shoulder_tilt_deg",
    "ear_visibility_asymmetry",
)
NUM_FEATURES: int = len(MLP_FEATURES)
EAR_DROP_INDEX: int = MLP_FEATURES.index("ear_drop")


@dataclass(frozen=True)
class SlouchFeatures:
    """Scale-invariant geometric summary of a single `Pose`.

    Vertical / forward distances are signed in image coordinates: y grows
    *downward*, so `ear_drop < 0` means the ears sit above the shoulder
    line (an upright posture). All distance-style fields are divided by
    `shoulder_width_px`, so a value of `0.5` always means "half a
    shoulder-width," regardless of how close the user is to the camera.

    Attributes:
        shoulder_width_px: Raw inter-shoulder distance in pixels. Provided
            for reference and as a sanity-check; not intended as an MLP
            input feature itself.
        ear_drop: `(ear_mid_y - shoulder_mid_y) / shoulder_width`, or
            `None` if neither ear cleared the confidence threshold.
            Negative = ears above shoulders (upright). Closer to zero or
            positive = head dropped (likely slouching).
        nose_drop: `(nose_y - shoulder_mid_y) / shoulder_width`, or
            `None` if the nose was not confidently detected. Same sign
            convention as `ear_drop`; useful as a fallback when ears are
            occluded.
        ear_forward: `(ear_mid_x - shoulder_mid_x) / shoulder_width`, or
            `None` if neither ear was confident. Signed horizontal offset
            of the head from the shoulders' midline. The MLP can learn
            its own asymmetry sign per camera setup.
        shoulder_tilt_deg: Angle of the shoulder line relative to
            horizontal, in degrees. `0` is level; non-zero indicates a
            shoulder slump or lean.
        ear_visibility_asymmetry: `left_ear_conf - right_ear_conf`. A
            proxy for head rotation: large magnitude implies the user is
            turned to one side.
    """

    shoulder_width_px: float
    ear_drop: float | None
    nose_drop: float | None
    ear_forward: float | None
    shoulder_tilt_deg: float
    ear_visibility_asymmetry: float

    def as_vector(self) -> np.ndarray:
        """Return the classifier-input vector as a 1D `float32` array.

        Only the fields listed in `MLP_FEATURES` are included; the raw
        `shoulder_width_px` is intentionally omitted because it carries
        absolute scale (not posture information) and would just teach the
        classifier to overfit to camera distance. Optional fields whose
        values are `None` become `NaN` so downstream consumers can decide
        how to handle missing data (e.g. `nanmean`, carry-forward, or a
        learned imputation).

        Returns:
            A `(NUM_FEATURES,)` `float32` array, ordered to match
            `MLP_FEATURES` element-wise.

        Preconditions:
            - `self` was constructed via `compute_slouch_features`.

        Postconditions:
            - Returned array is freshly allocated; safe for the caller to
              mutate or stack.
            - Order matches `MLP_FEATURES` exactly so column indices stay
              stable across the codebase.
        """
        values = [getattr(self, name) for name in MLP_FEATURES]
        return np.array(
            [np.nan if v is None else v for v in values],
            dtype=np.float32,
        )


def compute_slouch_features(pose: Pose) -> SlouchFeatures | None:
    """Derive scale-invariant slouch features from a single pose.

    Args:
        pose: A pose with keypoints in COCO order (`(17, 3)`).

    Returns:
        The computed `SlouchFeatures`, or `None` when the shoulders are
        too unreliable to anchor the geometry (either shoulder confidence
        below threshold or inter-shoulder distance below
        `SHOULDER_WIDTH_MIN_PX`). When shoulders are usable but other
        keypoints (ears, nose) are not, the affected fields are returned
        as `None` rather than failing the whole computation.

    Preconditions:
        - `pose.keypoints` is shaped `(17, 3)` in COCO order.

    Postconditions:
        - `pose` is not mutated.
        - When the result is non-`None`, `shoulder_width_px` and
          `shoulder_tilt_deg` are always finite.
    """
    kp = pose.keypoints
    l_sho_x, l_sho_y, l_sho_c = kp[LEFT_SHOULDER]
    r_sho_x, r_sho_y, r_sho_c = kp[RIGHT_SHOULDER]

    if l_sho_c < KEYPOINT_CONF_MIN or r_sho_c < KEYPOINT_CONF_MIN:
        return None

    dx = r_sho_x - l_sho_x
    dy = r_sho_y - l_sho_y
    shoulder_width = float(math.hypot(dx, dy))
    if shoulder_width < SHOULDER_WIDTH_MIN_PX:
        return None

    sho_mid_x = (l_sho_x + r_sho_x) / 2.0
    sho_mid_y = (l_sho_y + r_sho_y) / 2.0

    if dx < 0:
        dx, dy = -dx, -dy
    shoulder_tilt_deg = float(math.degrees(math.atan2(dy, dx)))

    l_ear_x, l_ear_y, l_ear_c = kp[LEFT_EAR]
    r_ear_x, r_ear_y, r_ear_c = kp[RIGHT_EAR]
    ear_visibility_asymmetry = float(l_ear_c - r_ear_c)

    l_ear_ok = l_ear_c >= KEYPOINT_CONF_MIN
    r_ear_ok = r_ear_c >= KEYPOINT_CONF_MIN
    if l_ear_ok and r_ear_ok:
        ear_x = (l_ear_x + r_ear_x) / 2.0
        ear_y = (l_ear_y + r_ear_y) / 2.0
    elif l_ear_ok:
        ear_x, ear_y = l_ear_x, l_ear_y
    elif r_ear_ok:
        ear_x, ear_y = r_ear_x, r_ear_y
    else:
        ear_x = ear_y = None

    if ear_x is not None:
        ear_drop: float | None = float((ear_y - sho_mid_y) / shoulder_width)
        ear_forward: float | None = float((ear_x - sho_mid_x) / shoulder_width)
    else:
        ear_drop = None
        ear_forward = None

    _, nose_y, nose_c = kp[NOSE]
    if nose_c >= KEYPOINT_CONF_MIN:
        nose_drop: float | None = float((nose_y - sho_mid_y) / shoulder_width)
    else:
        nose_drop = None

    return SlouchFeatures(
        shoulder_width_px=shoulder_width,
        ear_drop=ear_drop,
        nose_drop=nose_drop,
        ear_forward=ear_forward,
        shoulder_tilt_deg=shoulder_tilt_deg,
        ear_visibility_asymmetry=ear_visibility_asymmetry,
    )
