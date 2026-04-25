"""Rolling buffer of recent slouch-feature vectors.

The temporal classifier wants a fixed-shape `(num_features, window_size)`
tensor every time it makes a decision. Live frames arrive one at a time
and may carry `None`-valued features (when shoulder geometry was
unavailable). This module manages both: it keeps the last `window_size`
feature vectors as a rolling window, applies a simple imputation
strategy for missing values, and emits the window in the shape the
classifier expects.

Pure NumPy. Hardware-agnostic. Lives in `runtime/` rather than `model/`
because it owns *temporal* state (how the loop accumulates frames over
time), not model contracts.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from zapme.src.model.classifier import ClassifierConfig
from zapme.src.model.features import NUM_FEATURES, SlouchFeatures


class FeatureBuffer:
    """Sliding window over the last `window_size` per-frame feature vectors.

    Insert one vector per camera frame via `push()`; ask for the current
    classifier-shaped window via `as_window()`.

    Missing-value policy: when a pushed `SlouchFeatures` is `None` (no
    usable shoulder geometry), the buffer carries forward the last good
    vector instead. This keeps the classifier window dense and lets the
    runtime treat "no geometry" as "no posture change," which matches
    the safe-default behavior we want at the gate. If no good vector
    has ever been observed, missing pushes are stored as all-`NaN`
    vectors so downstream cleaning still has something to work with.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        """Initialize an empty buffer sized to match the classifier.

        Args:
            config: Classifier shape configuration. Defaults to
                `ClassifierConfig()` (currently a 15-frame window over
                `NUM_FEATURES` features).

        Preconditions:
            - `config.num_features == NUM_FEATURES`. (The buffer cannot
              currently bridge a shape mismatch.)

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
        self._last_good: np.ndarray | None = None

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

        Carries forward the last good vector when `features` is `None`.
        Falls back to an all-NaN vector when no good frame has been seen
        yet (typical at startup before the user is detected).

        Args:
            features: Per-frame features, or `None` if shoulder geometry
                was unusable on this frame.

        Preconditions:
            - When non-`None`, `features` was produced by
              `compute_slouch_features` so `as_vector()` returns the
              expected `(NUM_FEATURES,)` shape.

        Postconditions:
            - `len(self)` increases by 1, capped at `window_size`.
            - `self._last_good` is updated when `features` is non-`None`.
        """
        if features is None:
            if self._last_good is not None:
                self._buffer.append(self._last_good.copy())
            else:
                self._buffer.append(np.full(NUM_FEATURES, np.nan, dtype=np.float32))
            return

        vec = features.as_vector()
        self._last_good = vec.copy()
        self._buffer.append(vec)

    def as_window(self) -> np.ndarray:
        """Return the current buffer as a `(num_features, window_size)` array.

        Pads on the left with `NaN` columns when the buffer has not yet
        accumulated a full window. The classifier's NaN-aware decoding
        handles the padding cleanly (placeholder rule uses `nanmean`;
        the trained CNN sees a normalized, NaN-filled tensor — see the
        training script's window cleaner for the matching policy).

        Returns:
            `float32` array shaped `(NUM_FEATURES, window_size)`. Newest
            vector is at column `window_size - 1`; oldest is at `0`. Any
            unfilled columns at the start are `NaN`.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - Returned array is a fresh allocation; safe for the caller
              to mutate or feed straight to `SlouchClassifier.predict`.
        """
        window = np.full(
            (NUM_FEATURES, self._config.window_size), np.nan, dtype=np.float32
        )
        offset = self._config.window_size - len(self._buffer)
        for i, vec in enumerate(self._buffer):
            window[:, offset + i] = vec
        return window

    def reset(self) -> None:
        """Drop all stored vectors and clear the carry-forward state.

        Useful when the runtime re-initializes after a fault or a long
        gap (e.g. camera disconnect followed by reconnect) — old vectors
        are no longer representative of the current scene.

        Preconditions:
            - `__init__` has completed.

        Postconditions:
            - `len(self) == 0`.
            - Subsequent `push(None)` calls store `NaN` until a good
              frame arrives, just like a fresh instance.
        """
        self._buffer.clear()
        self._last_good = None
