"""Pure visual YOLO pose feed — for the demo audience to watch.

Opens the requested webcam, runs YOLO pose on every frame, draws
the skeleton + bounding box, and displays it in a window. Nothing
else: no slouch classifier, no debouncer, no logging clutter, no
gate. Just "what does YOLO see right now."

Designed to run on the laptop next to the actual Pi setup so the
audience can see the AI's view of the user while the Pi
independently does the EMS gating off-screen. Cameras pointed
roughly the same way will produce roughly the same skeleton.

Run:
    python scripts/yolo_view.py                # fullscreen by default
    python scripts/yolo_view.py --camera 1
    python scripts/yolo_view.py --windowed     # opt out of fullscreen

Press `q` (with the window focused) to quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Self-bootstrap so the script works regardless of whether
# `pip install -e .` succeeded. Adds the repo root (parent of
# `scripts/`) to sys.path so `from zapme...` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from zapme.src.model.vision import Pose, PoseEstimator
from zapme.src.runtime.loop import open_camera

WINDOW_NAME = "zapme — YOLO view (press q to quit)"

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
    """Overlay a detected pose onto `frame` in place.

    Args:
        frame: `(H, W, 3)` `uint8` BGR array; mutated.
        pose: Pose to render.
        kp_threshold: Per-keypoint confidence below which a keypoint is
            not drawn (and any skeleton edge that touches it is skipped).

    Preconditions:
        - `frame` is a writable `(H, W, 3)` `uint8` BGR array.
        - `pose.keypoints` is shaped `(17, 3)` (COCO format).

    Postconditions:
        - `frame` has bbox, skeleton, and keypoint markers drawn in
          place. Returns `None`.
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
            cv2.circle(frame, (int(x), int(y)), 4, KEYPOINT_COLOR, -1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the visual feed.

    Returns:
        Parsed argparse namespace.

    Preconditions:
        - `sys.argv` is set as expected for a CLI entry point.

    Postconditions:
        - Returned namespace exposes `camera`, `repo`, `filename`,
          `conf`, `windowed`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0 = laptop built-in).",
    )
    parser.add_argument(
        "--repo", type=str, default="Ultralytics/YOLO11",
        help="Hugging Face repo holding the pose checkpoint.",
    )
    parser.add_argument(
        "--filename", type=str, default="yolo11n-pose.pt",
        help="YOLO weights filename within the HF repo.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO detection confidence threshold.",
    )
    parser.add_argument(
        "--windowed", action="store_true",
        help="Open in a normal window instead of fullscreen. The "
             "default is fullscreen since this script is intended "
             "for the demo TV next to the Pi.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the visual YOLO feed until the operator presses `q`.

    Returns:
        `0` on clean exit, `1` if the camera could not be opened.

    Preconditions:
        - A webcam exists at the requested camera index.
        - Network access is available on first run for the model download.

    Postconditions:
        - The capture device is released and all OpenCV windows are
          closed before return.
    """
    args = parse_args()
    print("Loading YOLO model...", file=sys.stderr)
    estimator = PoseEstimator(
        repo_id=args.repo, filename=args.filename, confidence_threshold=args.conf
    )

    try:
        camera = open_camera(args.camera)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    print("Press q in the window to quit.", file=sys.stderr)
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue

            pose = estimator.infer(frame)
            if pose is not None:
                draw_pose(frame, pose)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
