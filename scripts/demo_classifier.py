"""End-to-end live demo: webcam → YOLO pose → features → buffer → classifier.

Same camera + pose preview as `demo_pose.py`, but augmented with the
temporal buffer and the slouch classifier. Prints the per-frame
slouch probability alongside the geometric features so you can watch
the model react in real time.

Defaults to the placeholder rule (no weights file). Once training has
produced an ONNX checkpoint, point at it:

    python scripts/demo_classifier.py --weights models/slouch_cnn.onnx

Hotkey: `q` to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from zapme.src.model.classifier import ClassifierConfig, SlouchClassifier
from zapme.src.model.features import SlouchFeatures, compute_slouch_features
from zapme.src.model.vision import Pose, PoseEstimator
from zapme.src.runtime.feature_buffer import FeatureBuffer

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
PROB_TEXT_COLOR = (255, 255, 255)


def draw_pose(frame: np.ndarray, pose: Pose, kp_threshold: float = 0.3) -> None:
    """Overlay a `Pose` onto `frame` in place.

    Args:
        frame: `(H, W, 3)` `uint8` BGR array; mutated.
        pose: Pose to render.
        kp_threshold: Per-keypoint confidence below which a keypoint is
            not drawn (and any skeleton edge that touches it is skipped).

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

    for _, (x, y, c) in enumerate(pose.keypoints):
        if c >= kp_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)


def draw_probability(frame: np.ndarray, prob: float, buffer_full: bool) -> None:
    """Overlay the live slouch probability onto `frame`.

    Args:
        frame: `(H, W, 3)` `uint8` BGR array; mutated.
        prob: Slouch probability in `[0, 1]`.
        buffer_full: `True` once the rolling buffer has accumulated a
            full window. While `False`, the probability is printed but
            tagged as "warming up" so the operator knows the value is
            based on padded NaNs.

    Preconditions:
        - `frame` is a writable `(H, W, 3)` `uint8` BGR array.
        - `prob` is finite.

    Postconditions:
        - `frame` has a status banner drawn along the top with the
          current slouch probability.
    """
    label = f"slouch_prob = {prob:.2f}"
    if not buffer_full:
        label += "  (warming up)"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PROB_TEXT_COLOR, 2)


def _fmt(value: float | None, spec: str) -> str:
    """Format an optional float, rendering `None` as a fixed-width placeholder.

    Args:
        value: Value to format, or `None`.
        spec: Format spec applied when `value` is not `None`.

    Returns:
        Formatted string, or `"  n/a"` when `value` is `None`.

    Preconditions:
        - `spec` is a valid Python format-spec for `float`.

    Postconditions:
        - Returned width is consistent across `None` / non-`None` calls.
    """
    return f"{value:{spec}}" if value is not None else "  n/a"


def format_summary(
    pose: Pose,
    features: SlouchFeatures | None,
    prob: float,
    buffer_full: bool,
) -> str:
    """Build a one-line per-frame summary.

    Args:
        pose: Pose for the current frame.
        features: Computed features for `pose`, or `None` when shoulder
            geometry was unreliable.
        prob: Latest slouch probability in `[0, 1]`.
        buffer_full: Whether the classifier window has been fully filled.

    Returns:
        A single-line, terminal-safe string.

    Preconditions:
        - `pose.keypoints` is shaped `(17, 3)` in COCO order.
        - `prob` is finite.

    Postconditions:
        - Returned string is single-line.
    """
    c = pose.keypoints[:, 2]
    head = (
        f"prob={prob:.2f}{'!' if prob >= 0.5 else ' '}"
        f" det={pose.score:.2f} sho={c[5]:.2f}/{c[6]:.2f} ears={c[3]:.2f}/{c[4]:.2f}"
    )
    if not buffer_full:
        head = f"[warmup] {head}"
    if features is None:
        return f"{head} | (shoulders unreliable)"
    geom = (
        f"drop_ear={_fmt(features.ear_drop, '+.2f')} "
        f"drop_nose={_fmt(features.nose_drop, '+.2f')} "
        f"fwd={_fmt(features.ear_forward, '+.2f')} "
        f"tilt={features.shoulder_tilt_deg:+.1f}deg"
    )
    return f"{head} | {geom}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.

    Preconditions:
        - `sys.argv` is set as expected for a CLI entry point.

    Postconditions:
        - Returned namespace exposes `camera`, `repo`, `filename`,
          `conf`, `weights`.
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
        help="YOLO detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help=(
            "Path to the trained classifier .onnx file. When omitted, "
            "the placeholder rule is used so the pipeline runs end-to-end "
            "before training is finished."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the live end-to-end demo loop.

    Returns:
        `0` on clean exit, `1` if the camera could not be opened.

    Preconditions:
        - A webcam is available at the requested camera index.
        - Network access is available on first run for the YOLO download.

    Postconditions:
        - Capture device released and OpenCV windows closed before return.
    """
    args = parse_args()

    estimator = PoseEstimator(
        repo_id=args.repo, filename=args.filename, confidence_threshold=args.conf
    )
    classifier = SlouchClassifier(weights_path=args.weights)
    buffer = FeatureBuffer(config=classifier.config)

    if args.weights is None:
        print("Classifier: placeholder rule (no --weights provided).")
    else:
        print(f"Classifier: ONNX from {args.weights}")
    print(
        f"Window: {buffer.window_size} frames × {buffer.num_features} features"
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
            features = compute_slouch_features(pose) if pose is not None else None
            buffer.push(features)
            window = buffer.as_window()
            prob = classifier.predict(window)

            if pose is not None:
                draw_pose(frame, pose)
            draw_probability(frame, prob, buffer.is_full())

            now = time.perf_counter()
            if now - last_print >= PRINT_INTERVAL_S:
                if pose is None:
                    print(f"prob={prob:.2f}  (no detection)")
                else:
                    print(format_summary(pose, features, prob, buffer.is_full()))
                last_print = now

            cv2.imshow("zapme — classifier demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
