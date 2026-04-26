"""Rolling buffer of recent slouch-feature vectors.

The temporal classifier wants a fixed-shape `(num_features, window_size)`
tensor every time it makes a decision. Live frames arrive one at a time
and may carry `None`-valued features (when shoulder geometry was
unavailable, or individual keypoints were occluded). This module keeps
the last `window_size` vectors as a rolling window and returns them in
the shape the classifier expects, with NaN entries cleaned via the
shared policy in `model/_common.clean_feature_window` so the classifier
sees the same input distribution it was trained on.

Pure NumPy. Hardware-agnostic. Lives in `runtime/` rather than `model/`
because it owns *temporal* state (how the loop accumulates frames over
time), not model contracts.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from zapme.src.model._common import clean_feature_window
from zapme.src.model.classifier import ClassifierConfig
from zapme.src.model.features import NUM_FEATURES, SlouchFeatures


class FeatureBuffer:
    """Sliding window over the last `window_size` per-frame feature vectors.

    Insert one vector per camera frame via `push()`; ask for the current
    classifier-shaped window via `as_window()`.

    Missing-value policy: each push either stores the live feature
    vector (with `NaN` for any individual feature whose upstream
    keypoint was unreliable) or, when no shoulder geometry was usable
    at all, stores an all-`NaN` placeholder. The actual NaN handling
    happens lazily in `as_window()` via the shared
    `model/_common.clean_feature_window` cleaner — same logic the
    training pipeline uses, so the classifier never sees a different
    input distribution between train and inference.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        """Initialize an empty buffer sized to match the classifier.

        Args:
            config: Classifier shape configuration. Defaults to
                `ClassifierConfig()` (currently a 15-frame window over
                `NUM_FEATURES` features).

        Preconditions:
            - `config.num_features == NUM_FEATURES`.

        Postconditions:
            - `len(self) == 0`.
            - `self.window_size` and `self.num_features` are exposed for
              callers that need the shapes (e.g. logging, tests).
        """
        self._config = config or ClassifierConfig()
        if self._config.num_features != NUM_FEATURES:
            raise ValueError(
                f"FeatureBuffer cannot reshape: classifier expects "
                f"{self._config.num_features} features, project produces "
                f"{NUM_FEATURES}."
            )
        self._buffer: deque[np.ndarray] = deque(maxlen=self._config.window_size)

    @property
    def window_size(self) -> int:
        """Return the configured window length in frames.

        Returns:
            Window size as a positive integer.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - Returned value matches what `as_window()` will produce.
        """
        return self._config.window_size

    @property
    def num_features(self) -> int:
        """Return the per-frame feature count.

        Returns:
            Number of features per timestep.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - Returned value equals `NUM_FEATURES`.
        """
        return self._config.num_features

    def __len__(self) -> int:
        """Return the number of vectors currently stored.

        Returns:
            Count in `[0, window_size]`.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - Result reflects the current state of the rolling buffer.
        """
        return len(self._buffer)

    def is_full(self) -> bool:
        """Return whether the buffer has accumulated a full window.

        Returns:
            `True` once `len(self) == window_size`.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - Returned value is monotonically true once first reached
              (the deque is fixed-length, so it stays full).
        """
        return len(self._buffer) == self._config.window_size

    def push(self, features: SlouchFeatures | None) -> None:
        """Append the next per-frame feature vector to the rolling buffer.

        Args:
            features: Per-frame features, or `None` if shoulder geometry
                was unusable on this frame.

        Preconditions:
            - When non-`None`, `features` was produced by
              `compute_slouch_features` so `as_vector()` returns the
              expected `(NUM_FEATURES,)` shape.

        Postconditions:
            - `len(self)` increases by 1, capped at `window_size`.
            - When `features is None`, an all-`NaN` placeholder is
              appended; cleaning at `as_window()` time will fill it via
              ffill/bfill from neighboring frames.
        """
        if features is None:
            self._buffer.append(np.full(NUM_FEATURES, np.nan, dtype=np.float32))
        else:
            self._buffer.append(features.as_vector())

    def as_window(self) -> np.ndarray:
        """Return the current buffer as a cleaned `(num_features, window_size)` array.

        Pads on the left with `NaN` columns when the buffer has not yet
        accumulated a full window, then runs the same NaN cleaner the
        training pipeline used (`clean_feature_window`) so per-feature
        ffill / bfill / zero-fill is applied. The classifier therefore
        always receives a NaN-free tensor in the same input
        distribution it saw at training time.

        Returns:
            `float32` array shaped `(NUM_FEATURES, window_size)`. Newest
            vector is at column `window_size - 1`; oldest is at `0`.
            No NaN values.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - Returned array is a fresh allocation; safe for the caller
              to mutate or feed straight to `SlouchClassifier.predict`.
            - Output contains no NaN values.
        """
        window = np.full(
            (NUM_FEATURES, self._config.window_size), np.nan, dtype=np.float32
        )
        offset = self._config.window_size - len(self._buffer)
        for i, vec in enumerate(self._buffer):
            window[:, offset + i] = vec
        return clean_feature_window(window, time_axis=1)

    def reset(self) -> None:
        """Drop all stored vectors.

        Useful when the runtime re-initializes after a fault or a long
        gap (e.g. camera disconnect followed by reconnect) — old vectors
        are no longer representative of the current scene.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - `len(self) == 0`.
        """
        self._buffer.clear()
