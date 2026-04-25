"""Temporal slouch classifier with a swappable backend.

Consumes a sliding window of `SlouchFeatures` vectors (shape
`(NUM_FEATURES, window_size)`) and returns the probability that the
user is currently slouching.

Two backends are supported behind the same API:

- An ONNX-exported 1D CNN, trained off-Pi in PyTorch and shipped as a
  `.onnx` file. Inference uses `onnxruntime`, which is already a hard
  dependency for the YOLO pose backbone, so this adds no new install
  surface on the Pi (and crucially does **not** drag PyTorch onto the
  Pi).
- A hand-tuned placeholder rule used when no weights file is provided.
  Lets the runtime loop be exercised end-to-end before a trained
  checkpoint exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from zapme.src.model._common import create_onnx_session
from zapme.src.model.features import MLP_FEATURES, NUM_FEATURES

EAR_DROP_INDEX: int = MLP_FEATURES.index("ear_drop")

# Hand-tuned placeholder thresholds, calibrated against the early
# data we collected (upright ear_drop ≈ -0.63, slouch/shrimp ≈ -0.50).
# Replaced by the trained 1D CNN as soon as weights are available.
PLACEHOLDER_THRESHOLD: float = -0.56
PLACEHOLDER_SLOPE: float = 12.0


@dataclass(frozen=True)
class ClassifierConfig:
    """Static shape configuration for the temporal classifier.

    Attributes:
        window_size: Number of feature vectors expected per inference
            window. Must match the size the trained CNN was exported
            with; fed-in windows of any other length will raise.
        num_features: Number of features per timestep. Driven by
            `features.NUM_FEATURES`; surfaced here so a `predict()`
            shape mismatch fails loudly at construction time rather
            than producing garbage probabilities.
    """

    window_size: int = 15
    num_features: int = NUM_FEATURES


class SlouchClassifier:
    """Stateless slouch classifier over a fixed-length feature window.

    Designed so the runtime loop never has to know whether the trained
    CNN is loaded yet. Construct it with `weights_path=None` to use the
    placeholder rule during early integration; pass a real `.onnx` once
    training has produced one and `predict()` will switch over without
    any other call-site changes.
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        config: ClassifierConfig | None = None,
    ) -> None:
        """Load an ONNX checkpoint or fall back to the placeholder rule.

        Args:
            weights_path: Path to a `.onnx` file produced by the off-Pi
                training pipeline, or `None` to use the placeholder.
            config: Window-size / feature-count configuration. Defaults
                to a 50-step window matching `NUM_FEATURES`.

        Raises:
            FileNotFoundError: If `weights_path` is provided but the file
                does not exist.

        Preconditions:
            - When `weights_path` is provided, the file is a valid ONNX
              graph that takes a `(1, num_features, window_size)` float
              tensor and produces a single `(1,)` or `(1, 1)` slouch
              probability output.

        Postconditions:
            - `self` is ready to accept `predict()` calls of shape
              `(num_features, window_size)`.
        """
        self._config = config or ClassifierConfig()
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

        if weights_path is None:
            return

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Classifier weights not found at {weights_path}; pass "
                "weights_path=None to use the placeholder rule instead."
            )

        self._session = create_onnx_session(weights_path)
        self._input_name = self._session.get_inputs()[0].name

    @property
    def config(self) -> ClassifierConfig:
        """Expose the active shape configuration to callers (e.g. the buffer).

        Returns:
            The `ClassifierConfig` this instance was constructed with.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - Returned value is the same instance held internally;
              `ClassifierConfig` is frozen so this is safe.
        """
        return self._config

    def predict(self, window: np.ndarray) -> float:
        """Run the classifier on a single feature window.

        Args:
            window: Feature window shaped `(num_features, window_size)`.
                May contain `NaN` entries for timesteps where the
                upstream geometry was unavailable.

        Returns:
            Slouch probability in `[0, 1]`.

        Raises:
            ValueError: If `window.shape` does not match the classifier's
                configured shape.

        Preconditions:
            - `window` is a `np.ndarray` of `float32` (or castable to it).

        Postconditions:
            - `window` is not mutated.
            - Return value is finite and lies in `[0, 1]`.
        """
        expected = (self._config.num_features, self._config.window_size)
        if window.shape != expected:
            raise ValueError(
                f"window shape {window.shape} does not match expected {expected}"
            )

        if self._session is None:
            return _placeholder_probability(window)

        batched = window.astype(np.float32, copy=False)[None, :, :]
        outputs = self._session.run(None, {self._input_name: batched})
        prob = float(np.asarray(outputs[0]).reshape(-1)[0])
        return max(0.0, min(1.0, prob))


def _placeholder_probability(window: np.ndarray) -> float:
    """Compute slouch probability from the mean ear-drop over a window.

    Used until a trained model exists. The decision rule is a logistic
    function of `mean(ear_drop)`, hand-calibrated so an upright sample
    (mean around `-0.63`) returns a low probability and a slouch sample
    (mean around `-0.50`) returns a high one.

    Args:
        window: Feature window shaped `(num_features, window_size)`.

    Returns:
        Slouch probability in `[0, 1]`. Returns `0.0` when every
        timestep's `ear_drop` is `NaN` (no usable geometry across the
        whole window) — the gate logic should prefer the safe default.

    Preconditions:
        - `window`'s shape was validated by the caller.

    Postconditions:
        - Return value is finite and lies in `[0, 1]`.
    """
    ear_drop_series = window[EAR_DROP_INDEX, :]
    if np.all(np.isnan(ear_drop_series)):
        return 0.0
    mean_ear_drop = float(np.nanmean(ear_drop_series))
    z = PLACEHOLDER_SLOPE * (mean_ear_drop - PLACEHOLDER_THRESHOLD)
    return float(1.0 / (1.0 + np.exp(-z)))
