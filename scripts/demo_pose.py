"""Live webcam demo: overlay YOLO11n-pose keypoints on the camera feed.

Run from the repo root:

    python scripts/demo_pose.py

Press `q` in the preview window to quit. The first run downloads the
model from Hugging Face (~12 MB) and caches it under `~/.cache/huggingface`.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from zapme.src.model.features import SlouchFeatures, compute_slouch_features
from zapme.src.model.vision import Pose, PoseEstimator

PRINT_INTERVAL_S = 0.5

COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

KEYPOINT_COLOR = (0, 255, 0)
SKELETON_COLOR = (255, 200, 0)
BBOX_COLOR = (0, 200, 255)


def draw_pose(frame: np.ndarray, pose: Pose, kp_threshold: float = 0.3) -> None:
    """Overlay a `Pose` onto `frame` in place.

    Args:
        frame: `(H, W, 3)` `uint8` BGR array; mutated.
        pose: Pose to render.
        kp_threshold: Per-keypoint confidence below which a keypoint is not
            drawn (and any skeleton edge that touches it is skipped).

    Preconditions:
        - `frame` is a writable `(H, W, 3)` `uint8` BGR array.
        - `pose.keypoints` is shaped `(17, 3)` (COCO format).

    Postconditions:
        - `frame` has bbox, skeleton, and keypoint markers drawn in place.
    """
    x1, y1, x2, y2 = pose.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)

    visible = pose.keypoints[:, 2] >= kp_threshold

    for a, b in COCO_SKELETON:
        if visible[a] and visible[b]:
            pa = tuple(pose.keypoints[a, :2].astype(int))
            pb = tuple(pose.keypoints[b, :2].astype(int))
            cv2.line(frame, pa, pb, SKELETON_COLOR, 2)

    for idx, (x, y, c) in enumerate(pose.keypoints):
        if c >= kp_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)


def _fmt(value: float | None, spec: str) -> str:
    """Format an optional float, rendering `None` as a fixed-width placeholder.

    Args:
        value: Value to format, or `None`.
        spec: Format spec applied when `value` is not `None` (e.g. `"+.2f"`).

    Returns:
        Formatted string, or `"  n/a"` when `value` is `None`.

    Preconditions:
        - `spec` is a valid Python format-spec for `float`.

    Postconditions:
        - Returned string has consistent width across calls with the same
          `spec`, so column-aligned output stays aligned.
    """
    return f"{value:{spec}}" if value is not None else "  n/a"


def format_pose_summary(pose: Pose, features: SlouchFeatures | None) -> str:
    """Build a one-line summary of slouch-relevant signals for one frame.

    Combines the raw shoulder / ear confidences with the derived
    geometric features so the user can correlate detection quality with
    posture geometry while watching live output.

    Args:
        pose: Pose to summarize.
        features: Pre-computed features for `pose`, or `None` if shoulders
            were too unreliable to anchor the geometry.

    Returns:
        A compact human-readable single-line string.

    Preconditions:
        - `pose.keypoints` is shaped `(17, 3)` in COCO order.

    Postconditions:
        - Returned string is single-line and safe to print directly.
    """
    c = pose.keypoints[:, 2]
    head = (
        f"det={pose.score:.2f} sho={c[5]:.2f}/{c[6]:.2f} ears={c[3]:.2f}/{c[4]:.2f}"
    )
    if features is None:
        return f"{head} | (shoulders unreliable, no geometry)"
    geom = (
        f"drop_ear={_fmt(features.ear_drop, '+.2f')} "
        f"drop_nose={_fmt(features.nose_drop, '+.2f')} "
        f"fwd={_fmt(features.ear_forward, '+.2f')} "
        f"tilt={features.shoulder_tilt_deg:+.1f}deg"
    )
    return f"{head} | {geom}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo.

    Returns:
        Parsed argparse namespace.

    Preconditions:
        - `sys.argv` is populated as expected for a CLI entry point.

    Postconditions:
        - Returned namespace exposes `camera`, `repo`, `filename`, `conf`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--repo", type=str, default="Ultralytics/YOLO11",
        help="Hugging Face repo holding the pose checkpoint.",
    )
    parser.add_argument(
        "--filename", type=str, default="yolo11n-pose.pt",
        help="Weights filename within the HF repo.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the live webcam demo loop.

    Returns:
        `0` on clean exit, `1` if the camera could not be opened.

    Preconditions:
        - A webcam is available at the requested camera index.
        - Network access is available on first run for the model download.

    Postconditions:
        - The capture device is released and all OpenCV windows are closed
          before return.
    """
    args = parse_args()

    estimator = PoseEstimator(
        repo_id=args.repo, filename=args.filename, confidence_threshold=args.conf
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera}", file=sys.stderr)
        return 1

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.", file=sys.stderr)
                break

            pose = estimator.infer(frame)
            if pose is not None:
                draw_pose(frame, pose)

            now = time.perf_counter()
            if now - last_print >= PRINT_INTERVAL_S:
                if pose is None:
                    print("(no detection)")
                else:
                    features = compute_slouch_features(pose)
                    print(format_pose_summary(pose, features))
                last_print = now

            cv2.imshow("zapme — pose demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
