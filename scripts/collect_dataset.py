"""Hotkey-driven labeled dataset recorder for the slouch classifier.

Run from the repo root, with the webcam pointed at the user:

    python scripts/collect_dataset.py --out data/eric_session_01.parquet

Live preview is the same as `demo_pose.py`. Labels are picked with
keystrokes; whichever label is active gets stamped onto every frame
recorded until the label is changed or recording is paused.

Hotkeys (focus the OpenCV window before pressing):

    1 — upright        (start recording labelled `upright`)
    2 — slouch         (start recording labelled `slouch`)
    3 — shrimp         (start recording labelled `shrimp`)
    0 — pause          (still preview, but do not record)
    q — quit and save  (writes the parquet file then exits)

Output is a single Parquet file with one row per recorded frame:

    ts                            float64  seconds since session start
    label                         string   one of the labels above
    det_score                     float32  YOLO detection confidence
    kp_x, kp_y, kp_conf           list[float32] length 17 (COCO order)
    shoulder_width_px             float32
    ear_drop, nose_drop,
        ear_forward               float32 (NaN if upstream missing)
    shoulder_tilt_deg,
        ear_visibility_asymmetry  float32

Frames where shoulders are too unreliable to anchor geometry (the
`features.compute_slouch_features(...) == None` case) are still
recorded — feature columns just contain `NaN`. The classifier needs
to learn that "no geometry" is a state, not a class.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from zapme.src.model.features import SlouchFeatures, compute_slouch_features
from zapme.src.model.vision import Pose, PoseEstimator

LABELS_BY_KEY: dict[int, str | None] = {
    ord("1"): "upright",
    ord("2"): "slouch",
    ord("3"): "shrimp",
    ord("0"): None,
}

LABEL_COLOR = (255, 255, 255)
PAUSED_COLOR = (60, 60, 200)


def _row_for_frame(
    elapsed_s: float,
    label: str,
    pose: Pose,
    features: SlouchFeatures | None,
) -> dict[str, object]:
    """Build a single recorded row from the live pose + feature output.

    Args:
        elapsed_s: Seconds since session start.
        label: Currently selected label string.
        pose: Pose detected on this frame.
        features: Pre-computed features for `pose`, or `None` when the
            shoulder geometry was unusable.

    Returns:
        A flat dict with the schema documented at the module level. All
        values are JSON-/Arrow-friendly Python primitives or lists.

    Preconditions:
        - `pose.keypoints` is shaped `(17, 3)` in COCO order.

    Postconditions:
        - Returned dict contains every column of the recording schema,
          using `float("nan")` for missing optional features.
    """
    kp = pose.keypoints
    if features is None:
        sw = float("nan")
        ear_drop = float("nan")
        nose_drop = float("nan")
        ear_forward = float("nan")
        tilt = float("nan")
        ear_asym = float("nan")
    else:
        sw = features.shoulder_width_px
        ear_drop = features.ear_drop if features.ear_drop is not None else float("nan")
        nose_drop = features.nose_drop if features.nose_drop is not None else float("nan")
        ear_forward = (
            features.ear_forward if features.ear_forward is not None else float("nan")
        )
        tilt = features.shoulder_tilt_deg
        ear_asym = features.ear_visibility_asymmetry

    return {
        "ts": float(elapsed_s),
        "label": label,
        "det_score": float(pose.score),
        "kp_x": [float(v) for v in kp[:, 0]],
        "kp_y": [float(v) for v in kp[:, 1]],
        "kp_conf": [float(v) for v in kp[:, 2]],
        "shoulder_width_px": sw,
        "ear_drop": ear_drop,
        "nose_drop": nose_drop,
        "ear_forward": ear_forward,
        "shoulder_tilt_deg": tilt,
        "ear_visibility_asymmetry": ear_asym,
    }


def _draw_status(
    frame: np.ndarray,
    label: str | None,
    n_recorded: int,
    n_by_label: dict[str, int],
) -> None:
    """Overlay the current label and recording counts onto `frame`.

    Args:
        frame: `(H, W, 3)` `uint8` BGR array; mutated.
        label: Active label, or `None` when paused.
        n_recorded: Total number of frames recorded so far this session.
        n_by_label: Per-label running counts.

    Preconditions:
        - `frame` is a writable `(H, W, 3)` `uint8` BGR array.

    Postconditions:
        - `frame` has the status banner drawn in place along the top.
    """
    if label is None:
        text = "PAUSED  (1 upright | 2 slouch | 3 shrimp | 0 pause | q quit)"
        color = PAUSED_COLOR
    else:
        text = f"REC: {label}  (1 upright | 2 slouch | 3 shrimp | 0 pause | q quit)"
        color = LABEL_COLOR
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    counts_line = f"total={n_recorded}  " + "  ".join(
        f"{k}={n_by_label.get(k, 0)}" for k in ("upright", "slouch", "shrimp")
    )
    cv2.putText(frame, counts_line, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def _write_parquet(rows: list[dict[str, object]], out_path: Path) -> None:
    """Persist accumulated rows to a Parquet file.

    Args:
        rows: List of per-frame dicts produced by `_row_for_frame`.
        out_path: Destination Parquet path. Parent directory is created
            if needed; existing files at `out_path` are overwritten.

    Preconditions:
        - Every dict in `rows` shares the same keys.

    Postconditions:
        - `out_path` exists and is a valid Parquet file with one row
          per element of `rows`.
        - On empty `rows`, writes nothing and prints a notice.
    """
    if not rows:
        print("No frames recorded; not writing a file.", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


def _pick_backend() -> int:
    """Pick the OpenCV capture backend that best matches the host platform.

    Returns:
        A `cv2.CAP_*` constant suitable for `cv2.VideoCapture(index, backend)`.

    Preconditions:
        - `cv2` is importable and built with the relevant backend support.

    Postconditions:
        - Returns `cv2.CAP_DSHOW` on Windows, `cv2.CAP_V4L2` on Linux,
          and `cv2.CAP_ANY` elsewhere.
        - Does not open any device.
    """
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open a webcam and request the desired capture format.

    Args:
        index: OpenCV camera index.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Requested camera frame rate. Drivers may clamp to a supported value.

    Returns:
        An opened `cv2.VideoCapture`. The caller owns it and must call
        `.release()` when done.

    Raises:
        RuntimeError: If the camera cannot be opened at `index`.

    Preconditions:
        - `index` corresponds to a webcam connected to the host.
        - `width`, `height`, and `fps` are positive.

    Postconditions:
        - Returned capture is opened (`isOpened()` is true) and has had its
          buffer size, frame size, and FPS configured on a best-effort basis.
        - Driver-accepted format may differ from requested values; the caller
          is responsible for reading back actual values via `cap.get(...)`.
    """
    cap = cv2.VideoCapture(index, _pick_backend())
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the recorder.

    Returns:
        Parsed argparse namespace.

    Preconditions:
        - `sys.argv` is populated as expected for a CLI entry point.

    Postconditions:
        - Returned namespace exposes `out`, `camera`, `repo`, `filename`,
          `conf`, `width`, `height`, `fps`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Destination Parquet file (parent directories created as needed).",
    )
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
        "--width", type=int, default=640,
        help="Requested capture width in pixels (default: 640).",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Requested capture height in pixels (default: 480).",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Requested capture frame rate (default: 30). "
             "Higher = more labelable frames per session.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the recorder loop until the user presses `q`.

    Returns:
        `0` on clean exit, `1` if the camera could not be opened.

    Preconditions:
        - A webcam is available at the requested camera index.
        - Network access is available on first run for the model download.

    Postconditions:
        - Capture device released and OpenCV windows closed before return.
        - Parquet file at `--out` exists when at least one frame was
          recorded; otherwise, no file is written.
    """
    args = parse_args()

    estimator = PoseEstimator(
        repo_id=args.repo, filename=args.filename, confidence_threshold=args.conf
    )

    try:
        cap = _open_camera(args.camera, args.width, args.height, args.fps)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {args.camera}: requested {args.width}x{args.height} @ {args.fps} FPS, "
        f"got {actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
    )

    rows: list[dict[str, object]] = []
    n_by_label: dict[str, int] = {}
    active_label: str | None = None
    session_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.", file=sys.stderr)
                break

            pose = estimator.infer(frame)
            features = compute_slouch_features(pose) if pose is not None else None

            if active_label is not None and pose is not None:
                elapsed = time.perf_counter() - session_start
                rows.append(_row_for_frame(elapsed, active_label, pose, features))
                n_by_label[active_label] = n_by_label.get(active_label, 0) + 1

            _draw_status(frame, active_label, len(rows), n_by_label)
            cv2.imshow("zapme — collect (q to save & quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in LABELS_BY_KEY:
                active_label = LABELS_BY_KEY[key]
    finally:
        cap.release()
        cv2.destroyAllWindows()
        _write_parquet(rows, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
