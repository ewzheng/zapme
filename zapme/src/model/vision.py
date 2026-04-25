"""Vision model API: image in, structured pose out.

This module is the *only* public surface for vision inference. It hides
the choice of backend (currently Ultralytics YOLO) behind a stable
contract so other models can be swapped in without touching call sites.

Model files are loaded via `_common.download_from_hf` and cached on disk
by the Hugging Face Hub client.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zapme.src.model._common import load_yolo_from_hf

DEFAULT_REPO_ID = "Ultralytics/YOLO11"
DEFAULT_FILENAME = "yolo11n-pose.pt"


@dataclass(frozen=True)
class Pose:
    """A single detected person's pose for one frame.

    Attributes:
        keypoints: `(K, 3)` float array of `(x, y, confidence)` per keypoint,
            in pixel coordinates relative to the original frame. `K = 17` for
            the COCO keypoint format used by YOLO pose models.
        bbox: `(4,)` float array `(x1, y1, x2, y2)` person bounding box in
            pixel coordinates.
        score: Detection confidence for the bounding box, in `[0, 1]`.
    """

    keypoints: np.ndarray
    bbox: np.ndarray
    score: float


class PoseEstimator:
    """Pretrained pose model wrapped behind a stable inference API.

    The estimator loads a YOLO-family pose checkpoint from a Hugging Face
    repo on construction, then accepts BGR frames (as produced by OpenCV)
    and returns the highest-confidence detected pose, or `None` if no
    person was detected with sufficient confidence.

    Instances are not thread-safe; create one per inference thread.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        revision: str | None = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        """Load and prepare the pose model.

        Args:
            repo_id: Hugging Face repo holding the model weights.
            filename: Weights filename within `repo_id`.
            revision: Optional git revision pin for reproducibility.
            confidence_threshold: Minimum bounding-box confidence below which
                detections are dropped.

        Preconditions:
            - The HF repo + filename combination resolves to a YOLO-format
              pose checkpoint compatible with the installed `ultralytics`
              version.
            - `confidence_threshold` lies in `[0, 1]`.

        Postconditions:
            - Model weights are downloaded to the HF cache (if not already
              present) and loaded into memory.
            - `self` is ready to accept `infer()` calls.
        """
        self._model = load_yolo_from_hf(
            repo_id=repo_id, filename=filename, revision=revision
        )
        self._conf = confidence_threshold

    def infer(self, frame: np.ndarray) -> Pose | None:
        """Run pose inference on a single frame.

        Args:
            frame: Input image as a `(H, W, 3)` `uint8` BGR array (the
                format OpenCV's `VideoCapture.read()` returns).

        Returns:
            The highest-confidence detected `Pose`, or `None` when no
            detection cleared `confidence_threshold`.

        Preconditions:
            - `frame` is a contiguous `(H, W, 3)` `uint8` array in BGR order.
            - The model has been initialized successfully.

        Postconditions:
            - `frame` is not mutated.
            - Returned arrays are owned by the caller (safe to mutate).
        """
        results = self._model.predict(
            source=frame, conf=self._conf, verbose=False
        )
        if not results:
            return None

        result = results[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return None

        scores = result.boxes.conf.cpu().numpy()
        best = int(np.argmax(scores))

        keypoints_xy = result.keypoints.xy[best].cpu().numpy()
        keypoints_conf = result.keypoints.conf[best].cpu().numpy()
        keypoints = np.concatenate(
            [keypoints_xy, keypoints_conf[:, None]], axis=1
        ).astype(np.float32)

        bbox = result.boxes.xyxy[best].cpu().numpy().astype(np.float32)
        score = float(scores[best])

        return Pose(keypoints=keypoints, bbox=bbox, score=score)
