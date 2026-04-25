"""Train the temporal slouch classifier on collected parquet sessions.

Designed to run on a CUDA box (the HPC). Reads every Parquet under
`--data-glob`, slices each per-(session, label) contiguous run into
overlapping windows of `--window-size` frames, trains a tiny 1D CNN
to classify them as `upright` / `slouch` / `shrimp`, and exports the
result as ONNX so the Pi-side `SlouchClassifier` can pick it up
unchanged.

Held-out validation is **session-aware**: pass `--val-session` to keep
one entire session out of training, otherwise the model trivially
memorizes per-session noise and reported accuracy is meaningless.

Example:

    python scripts/train_classifier.py \\
        --data-glob 'data/*.parquet' \\
        --val-session eric_session_03 \\
        --out models/slouch_cnn.onnx
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from zapme.src.model.features import MLP_FEATURES, NUM_FEATURES, compute_slouch_features
from zapme.src.model.vision import Pose

LABEL_ORDER_MULTI: tuple[str, ...] = ("upright", "slouch", "shrimp")
LABEL_ORDER_BINARY: tuple[str, ...] = ("upright", "not_upright")
UPRIGHT_LABEL: str = "upright"


def collapse_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map the three-class labels onto the binary `upright` / `not_upright` axis.

    Args:
        df: Combined dataset whose `label` column holds values from
            `LABEL_ORDER_MULTI`.

    Returns:
        A copy of `df` with `label` rewritten so any non-`upright` value
        becomes `not_upright`. Other columns are untouched.

    Preconditions:
        - `df` contains a `label` column of strings.

    Postconditions:
        - Returned DataFrame's `label` values are members of
          `LABEL_ORDER_BINARY`.
        - Original `df` is not mutated.
    """
    out = df.copy()
    out["label"] = np.where(out["label"] == UPRIGHT_LABEL, UPRIGHT_LABEL, "not_upright")
    return out


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one training run.

    Attributes:
        window_size: Number of frames per training/inference window.
            Must match the value the runtime feeds in.
        stride: Window stride within a contiguous label run, in frames.
        batch_size: SGD batch size.
        epochs: Total training epochs.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
    """

    window_size: int = 15
    stride: int = 3
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    dropout: float = 0.4


def load_combined(data_glob: str) -> pd.DataFrame:
    """Load and concat every Parquet matching `data_glob`.

    Args:
        data_glob: Shell-style glob (e.g. `'data/*.parquet'`).

    Returns:
        A pandas DataFrame with all rows from all matching files, plus
        a synthetic `session` column derived from the file basename.

    Raises:
        FileNotFoundError: If `data_glob` matches no files.

    Preconditions:
        - Files matched by the glob were produced by `collect_dataset.py`
          and share the same column schema.

    Postconditions:
        - Returned DataFrame is the row-wise concat in glob order.
        - `session` column is present and populated.
    """
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No Parquet files matched glob: {data_glob}")
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pq.read_table(path).to_pandas()
        df["session"] = os.path.basename(path).replace(".parquet", "")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} files, {len(combined)} total rows.")
    return combined


def build_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    label_order: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice the per-frame DataFrame into fixed-length feature windows.

    A "segment" is the maximal run of consecutive rows within a single
    session whose `label` does not change. Sliding-window slicing happens
    *within* segments only, so windows never straddle a label boundary
    or a session boundary.

    Args:
        df: Combined dataset as returned by `load_combined`. Must contain
            the columns enumerated in `MLP_FEATURES`, plus `session`,
            `label`, and `ts`.
        window_size: Window length in frames.
        stride: Stride between successive window starts within a segment.

    Returns:
        Tuple `(X, y, sessions)`:
        - `X`: float32 array of shape `(N, NUM_FEATURES, window_size)`.
        - `y`: int64 array of shape `(N,)` with class indices into
          `LABEL_ORDER`.
        - `sessions`: object array of shape `(N,)` holding the source
          session of each window (used for session-aware splitting).

    Preconditions:
        - `df['ts']` is monotonic-increasing per session.
        - `df['label']` values are members of `LABEL_ORDER`.

    Postconditions:
        - Every output window contains exactly `window_size` consecutive
          frames from one (session, label) segment.
        - `NaN` entries are forward-then-backward filled within each
          window's feature row; columns that are entirely NaN within a
          window are zero-filled.
        - Features are recomputed live from the per-frame raw keypoint
          columns (`kp_x` / `kp_y` / `kp_conf`) via
          `features.compute_slouch_features`. Pre-computed feature
          columns in the parquet (if present) are ignored, so changes to
          the feature set propagate to training without re-recording.
    """
    df = df.sort_values(["session", "ts"]).reset_index(drop=True)

    segment_id = (
        (df["session"] != df["session"].shift())
        | (df["label"] != df["label"].shift())
    ).cumsum()

    X_chunks: list[np.ndarray] = []
    y_chunks: list[int] = []
    session_chunks: list[str] = []

    label_to_idx = {name: i for i, name in enumerate(label_order)}

    for _, segment in df.groupby(segment_id, sort=False):
        if len(segment) < window_size:
            continue
        label_name = segment["label"].iloc[0]
        if label_name not in label_to_idx:
            continue
        label_idx = label_to_idx[label_name]
        session_name = segment["session"].iloc[0]
        feats = _features_from_segment(segment)

        for start in range(0, len(segment) - window_size + 1, stride):
            window = feats[start : start + window_size, :]
            X_chunks.append(_clean_window(window))
            y_chunks.append(label_idx)
            session_chunks.append(session_name)

    if not X_chunks:
        raise RuntimeError(
            "Windowing produced zero examples. Check window_size against "
            "your segment lengths."
        )

    X = np.stack(X_chunks).transpose(0, 2, 1)
    y = np.asarray(y_chunks, dtype=np.int64)
    sessions = np.asarray(session_chunks, dtype=object)
    return X, y, sessions


def _features_from_segment(segment: pd.DataFrame) -> np.ndarray:
    """Recompute per-frame feature vectors from raw keypoint columns.

    Iterating on the feature set should not require re-recording any
    data. The collector saves raw `kp_x` / `kp_y` / `kp_conf` per frame;
    this helper recomputes the live feature vector from them on every
    training run, so adding or removing entries in `MLP_FEATURES`
    immediately takes effect without touching the parquets.

    Args:
        segment: Contiguous (session, label) slice with one row per
            frame. Must contain `kp_x`, `kp_y`, `kp_conf` columns
            holding length-17 lists, plus `det_score`.

    Returns:
        Float32 array shaped `(len(segment), NUM_FEATURES)` with `NaN`
        wherever the live `compute_slouch_features` returned `None`.

    Preconditions:
        - Each `kp_*` cell is a list / array of length 17 in COCO order.

    Postconditions:
        - Output ordering matches `MLP_FEATURES`.
    """
    n = len(segment)
    out = np.full((n, NUM_FEATURES), np.nan, dtype=np.float32)
    kp_x = segment["kp_x"].to_numpy()
    kp_y = segment["kp_y"].to_numpy()
    kp_conf = segment["kp_conf"].to_numpy()
    det = segment["det_score"].to_numpy()
    for i in range(n):
        keypoints = np.stack(
            [
                np.asarray(kp_x[i], dtype=np.float32),
                np.asarray(kp_y[i], dtype=np.float32),
                np.asarray(kp_conf[i], dtype=np.float32),
            ],
            axis=1,
        )
        pose = Pose(
            keypoints=keypoints,
            bbox=np.zeros(4, dtype=np.float32),
            score=float(det[i]),
        )
        feats = compute_slouch_features(pose)
        if feats is not None:
            out[i, :] = feats.as_vector()
    return out


def _clean_window(window: np.ndarray) -> np.ndarray:
    """Forward-fill, backward-fill, then zero-fill NaNs within a window.

    Args:
        window: Float array shaped `(window_size, NUM_FEATURES)`.

    Returns:
        Same shape, no NaNs. Caller is responsible for stacking.

    Preconditions:
        - `window` is a 2D float array.

    Postconditions:
        - Returned array has no NaN entries.
        - Original `window` is not mutated.
    """
    out = window.copy()
    for col in range(out.shape[1]):
        series = pd.Series(out[:, col])
        series = series.ffill().bfill()
        if series.isna().all():
            series = series.fillna(0.0)
        out[:, col] = series.to_numpy()
    return out


def session_aware_split(
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    val_session: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out one session entirely for validation.

    Args:
        X: Feature tensor `(N, C, T)`.
        y: Label vector `(N,)`.
        sessions: Per-window session names `(N,)`.
        val_session: Session to hold out for validation. When `None`, a
            random 20% slice is held out instead — fine for sanity checks
            but does not measure cross-person generalization.

    Returns:
        `(X_train, y_train, X_val, y_val)` as float32 / int64 arrays.

    Preconditions:
        - `X.shape[0] == y.shape[0] == sessions.shape[0]`.
        - When `val_session` is provided, it appears in `sessions`.

    Postconditions:
        - Train and validation sets are disjoint.
        - When `val_session` is provided, no training window comes from it.
    """
    if val_session is None:
        rng = np.random.default_rng(seed=0)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        cut = int(len(X) * 0.8)
        train_idx, val_idx = idx[:cut], idx[cut:]
    else:
        if val_session not in set(sessions):
            raise ValueError(
                f"val_session '{val_session}' not present in dataset. "
                f"Available: {sorted(set(sessions))}"
            )
        val_idx = np.where(sessions == val_session)[0]
        train_idx = np.where(sessions != val_session)[0]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class FeatureNormalize(nn.Module):
    """Bake training-set z-score normalization into the model graph.

    Stats are registered as buffers so they travel with `state_dict`
    saves and ONNX exports. The runtime can therefore pass raw
    `SlouchFeatures.as_vector()` outputs straight in without applying
    any preprocessing.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store per-feature mean and std as non-trainable buffers.

        Args:
            mean: Per-feature mean, shape `(num_features,)`.
            std: Per-feature std, shape `(num_features,)`.

        Preconditions:
            - `mean` and `std` have the same shape and length
              `NUM_FEATURES`.

        Postconditions:
            - Buffers are stored in shape `(1, num_features, 1)` for
              broadcast against `(B, C, T)` inputs.
        """
        super().__init__()
        m = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1)
        s = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization across the channel dim.

        Args:
            x: Input tensor `(B, C, T)`.

        Returns:
            `(x - mean) / (std + eps)`, same shape.

        Preconditions:
            - `x.shape[1] == self.mean.shape[1]`.

        Postconditions:
            - Output dtype matches `x.dtype`.
        """
        return (x - self.mean) / (self.std + 1e-6)


class SlouchCNN(nn.Module):
    """Two-stage temporal classifier: 1D conv mixer + MLP head.

    The conv stage exists only to mix features across time — one
    `Conv1d(NUM_FEATURES → 32, kernel=5)` with BatchNorm + ReLU gives
    each output channel a receptive field covering a third of the
    window, which is plenty for posture (which evolves over seconds, not
    sub-frames). Adaptive average pooling collapses the time axis to a
    single 32-dim summary vector. The MLP head then does the actual
    classification with dropout for regularization.

    Total parameter count is well under 10K — small enough not to
    overfit ~1300 training windows, large enough that the head can
    learn nonlinear feature interactions the placeholder rule misses.
    """

    def __init__(
        self,
        normalizer: FeatureNormalize,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Assemble the model with a baked-in normalization head.

        Args:
            normalizer: Pre-fitted `FeatureNormalize` placed before the
                conv mixer so ONNX export captures the normalization.
            num_classes: Output class count. Defaults to `2` (binary
                upright / not-upright); pass `3` for the original
                `LABEL_ORDER_MULTI` scheme.
            dropout: Dropout probability applied between the MLP head's
                hidden layers.

        Preconditions:
            - `normalizer` was fitted on the training set.

        Postconditions:
            - `self` is in train mode and ready to receive `(B, C, T)`
              tensors with `C == NUM_FEATURES`.
        """
        super().__init__()
        self.normalizer = normalizer
        self.temporal = nn.Sequential(
            nn.Conv1d(NUM_FEATURES, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        self.body = nn.Sequential(self.temporal, self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of windows.

        Args:
            x: Float tensor `(B, NUM_FEATURES, window_size)`.

        Returns:
            Logits tensor `(B, num_classes)`.

        Preconditions:
            - `x.dim() == 3` and `x.shape[1] == NUM_FEATURES`.

        Postconditions:
            - Output is on the same device as `x`.
        """
        return self.body(self.normalizer(x))


class SlouchONNXWrapper(nn.Module):
    """Inference-time wrapper that emits a single slouch probability.

    The Pi-side `SlouchClassifier.predict()` consumes a scalar in
    `[0, 1]`. This wrapper applies softmax over the trained classes and
    returns `1 - P(upright)` so the runtime contract is identical
    whether the underlying classifier was trained binary or multi-class.
    """

    def __init__(self, classifier: SlouchCNN, upright_index: int) -> None:
        """Hold a trained classifier; do not modify its parameters.

        Args:
            classifier: A trained `SlouchCNN` ready for inference.
            upright_index: Column of the `upright` class in the
                classifier's softmax output. Caller is responsible for
                passing the right value (`label_order.index("upright")`).

        Preconditions:
            - `classifier` is in eval mode (caller's responsibility).
            - `0 <= upright_index < classifier.body[-1].out_features`.

        Postconditions:
            - This wrapper carries no extra trainable parameters.
        """
        super().__init__()
        self.classifier = classifier
        self.upright_index = upright_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute slouch probability `1 - P(upright)` per batch element.

        Args:
            x: Float tensor `(B, NUM_FEATURES, window_size)`.

        Returns:
            Float tensor `(B, 1)` of slouch probabilities in `[0, 1]`.

        Preconditions:
            - The wrapped classifier was trained so that
              `softmax(logits)[:, upright_index]` is `P(upright)`.

        Postconditions:
            - Output values lie in `[0, 1]`.
        """
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        upright = probs[:, self.upright_index : self.upright_index + 1]
        return 1.0 - upright


def fit_normalizer(X_train: np.ndarray) -> FeatureNormalize:
    """Compute per-feature mean / std from training windows only.

    Args:
        X_train: Training tensor `(N, C, T)`.

    Returns:
        A `FeatureNormalize` module pre-loaded with stats.

    Preconditions:
        - `X_train` contains no NaNs (cleaned upstream).

    Postconditions:
        - Resulting normalizer's `mean` / `std` have shape `(1, C, 1)`.
    """
    flat = X_train.transpose(0, 2, 1).reshape(-1, X_train.shape[1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return FeatureNormalize(mean, std)


def train_one_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    label_order: tuple[str, ...],
    device: torch.device,
    log_prefix: str = "",
) -> tuple[SlouchCNN, dict[str, torch.Tensor], float, int, np.ndarray]:
    """Train one fold with best-checkpoint tracking.

    Builds a fresh normalizer + model from `X_train`, runs `cfg.epochs`
    epochs, tracks the best val-accuracy checkpoint, and returns it.
    The caller can either restore the best state into the returned model
    (for export) or read off the metrics for CV reporting.

    Args:
        X_train: Training tensor `(N, C, T)`.
        y_train: Training labels `(N,)`.
        X_val: Validation tensor `(M, C, T)`.
        y_val: Validation labels `(M,)`.
        cfg: Hyperparameters.
        label_order: Class-name tuple ordered by class index.
        device: `'cuda'` or `'cpu'`.
        log_prefix: Prefixed to each per-epoch print line. Useful when
            interleaving multiple folds in the same stdout.

    Returns:
        `(model, best_state, best_val_acc, best_epoch, best_confusion)`.

    Preconditions:
        - `X_train`, `X_val` shapes match `(*, NUM_FEATURES, cfg.window_size)`.
        - `cfg.epochs >= 1`.

    Postconditions:
        - Model is on `device` with parameters from the *final* epoch
          (not best). Caller is responsible for `load_state_dict`.
        - Best-checkpoint state lives on CPU.
    """
    normalizer = fit_normalizer(X_train)
    model = SlouchCNN(
        normalizer, num_classes=len(label_order), dropout=cfg.dropout
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_epoch = 0
    best_confusion = np.zeros((len(label_order), len(label_order)), dtype=np.int64)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_correct = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optim.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.item()) * X_batch.size(0)
            train_correct += int((logits.argmax(dim=-1) == y_batch).sum().item())
            train_n += X_batch.size(0)

        val_acc, val_per_class, val_confusion = evaluate(
            model, val_loader, device, label_order
        )

        train_loss = train_loss_sum / max(train_n, 1)
        train_acc = train_correct / max(train_n, 1)
        per_class_str = "  ".join(
            f"{name}={val_per_class.get(name, float('nan')):.2f}"
            for name in label_order
        )
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_confusion = val_confusion
            marker = "  *"
        print(
            f"{log_prefix}epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.3f} "
            f"| val acc={val_acc:.3f}  per-class: {per_class_str}{marker}"
        )

    return model, best_state, best_val_acc, best_epoch, best_confusion


def train_on_all(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    label_order: tuple[str, ...],
    device: torch.device,
    n_epochs: int,
) -> SlouchCNN:
    """Train a fresh model on the entire dataset, no held-out validation.

    Used for the final shipping artifact after CV has measured how well
    the same training recipe generalizes. Runs for a fixed number of
    epochs (typically the average best-epoch from CV) since there is no
    val signal to drive early stopping.

    Args:
        X: All windows `(N, C, T)`.
        y: All labels `(N,)`.
        cfg: Hyperparameters; `cfg.epochs` is overridden by `n_epochs`.
        label_order: Class-name tuple.
        device: Torch device.
        n_epochs: Number of epochs to train for.

    Returns:
        Trained `SlouchCNN` ready for ONNX export.

    Preconditions:
        - `n_epochs >= 1`.
        - `X.shape[1] == NUM_FEATURES`, `X.shape[2] == cfg.window_size`.

    Postconditions:
        - Returned model is on `device` and in eval mode.
    """
    normalizer = fit_normalizer(X)
    model = SlouchCNN(
        normalizer, num_classes=len(label_order), dropout=cfg.dropout
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_correct = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optim.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.item()) * X_batch.size(0)
            train_correct += int((logits.argmax(dim=-1) == y_batch).sum().item())
            train_n += X_batch.size(0)
        train_loss = train_loss_sum / max(train_n, 1)
        train_acc = train_correct / max(train_n, 1)
        print(
            f"[final] epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.3f}"
        )
    return model.eval()


def run_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    cfg: TrainConfig,
    label_order: tuple[str, ...],
    device: torch.device,
) -> list[dict[str, object]]:
    """Run leave-one-session-out cross-validation, one fold per session.

    Args:
        X: All windows `(N, C, T)`.
        y: All labels `(N,)`.
        sessions: Per-window session names `(N,)`.
        cfg: Hyperparameters (shared across folds).
        label_order: Class-name tuple.
        device: Torch device.

    Returns:
        List of per-fold result dicts containing `session`, `val_acc`,
        `best_epoch`, `confusion`. Order matches sorted unique sessions.

    Preconditions:
        - At least 2 unique sessions appear in `sessions`.

    Postconditions:
        - No model is retained; the function only reports metrics. Final
          model training is the caller's responsibility.
    """
    unique_sessions = sorted(set(sessions))
    results: list[dict[str, object]] = []
    for fold_idx, held_out in enumerate(unique_sessions, start=1):
        print(
            f"\n=== Fold {fold_idx}/{len(unique_sessions)}: held-out session = {held_out} ==="
        )
        val_mask = sessions == held_out
        train_mask = ~val_mask
        _, _, best_val_acc, best_epoch, best_confusion = train_one_split(
            X[train_mask], y[train_mask],
            X[val_mask], y[val_mask],
            cfg, label_order, device,
            log_prefix=f"[fold {fold_idx}] ",
        )
        results.append(
            {
                "session": held_out,
                "val_acc": float(best_val_acc),
                "best_epoch": int(best_epoch),
                "confusion": best_confusion,
            }
        )
        print(
            f"[fold {fold_idx}] best epoch={best_epoch}  best val acc={best_val_acc:.3f}"
        )
    return results


def report_cv(
    results: list[dict[str, object]],
    label_order: tuple[str, ...],
) -> tuple[float, int]:
    """Print a CV summary and return aggregate stats.

    Args:
        results: Per-fold output of `run_loso_cv`.
        label_order: Class-name tuple, used to pretty-print confusions.

    Returns:
        `(mean_val_acc, median_best_epoch)` aggregates suitable for
        configuring a final all-data training run.

    Preconditions:
        - `results` is non-empty.

    Postconditions:
        - Stdout has a per-fold + aggregate summary plus per-fold
          confusion matrices.
    """
    print("\n=== CV summary ===")
    accs = [float(r["val_acc"]) for r in results]
    epochs = [int(r["best_epoch"]) for r in results]
    for r in results:
        print(
            f"  held-out {r['session']!s:<22} "
            f"val acc={float(r['val_acc']):.3f}  best epoch={int(r['best_epoch']):>2}"
        )
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    median_epoch = int(np.median(epochs))
    print(f"  mean val acc = {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  median best epoch = {median_epoch}")

    for r in results:
        confusion = r["confusion"]
        print(f"\nConfusion (held-out {r['session']}):")
        header = "          " + "  ".join(f"{name:>10}" for name in label_order)
        print(header)
        for i, name in enumerate(label_order):
            row = "  ".join(
                f"{int(confusion[i, j]):>10d}" for j in range(len(label_order))
            )
            print(f"{name:>8}  {row}")
    return mean_acc, median_epoch


def evaluate(
    model: SlouchCNN,
    loader: DataLoader,
    device: torch.device,
    label_order: tuple[str, ...],
) -> tuple[float, dict[str, float], np.ndarray]:
    """Compute accuracy, per-class accuracy, and confusion matrix on a loader.

    Args:
        model: Trained `SlouchCNN`.
        loader: DataLoader over the eval set.
        device: Device the model lives on.

    Returns:
        Tuple `(overall_acc, per_class_acc, confusion)`:
        - `overall_acc`: float in `[0, 1]`.
        - `per_class_acc`: dict mapping label name → recall.
        - `confusion`: int matrix `(num_classes, num_classes)`,
          `[true_idx, pred_idx]`.

    Preconditions:
        - `model` is on `device`.

    Postconditions:
        - `model` is left in eval mode.
        - No optimizer state is mutated.
    """
    model.eval()
    n_total = 0
    n_correct = 0
    confusion = np.zeros((len(label_order), len(label_order)), dtype=np.int64)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch).argmax(dim=-1)
            n_correct += int((preds == y_batch).sum().item())
            n_total += int(y_batch.numel())
            for t, p in zip(y_batch.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(t), int(p)] += 1

    per_class: dict[str, float] = {}
    for i, name in enumerate(label_order):
        row_total = int(confusion[i, :].sum())
        per_class[name] = (
            float(confusion[i, i]) / row_total if row_total > 0 else float("nan")
        )

    return n_correct / max(n_total, 1), per_class, confusion


def export_onnx(
    model: SlouchCNN,
    out_path: Path,
    window_size: int,
    upright_index: int,
) -> None:
    """Export a slouch-probability ONNX graph from a trained classifier.

    Args:
        model: Trained `SlouchCNN` on any device.
        out_path: Destination `.onnx` path. Parent dir is created if
            absent. Existing files at this path are overwritten.
        window_size: Window length used at training; baked into the
            graph as a fixed input shape.
        upright_index: Column index of the `upright` class in the
            classifier's softmax output.

    Preconditions:
        - `model.normalizer` holds the training-set normalization stats.

    Postconditions:
        - `out_path` exists and is loadable by `onnxruntime`.
        - Exported graph takes `(1, NUM_FEATURES, window_size)` float32
          input and emits `(1, 1)` float32 slouch probability.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = SlouchONNXWrapper(model.cpu().eval(), upright_index=upright_index)
    dummy = torch.zeros((1, NUM_FEATURES, window_size), dtype=torch.float32)
    torch.onnx.export(
        cpu_model,
        dummy,
        str(out_path),
        input_names=["window"],
        output_names=["slouch_prob"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )
    print(f"Exported ONNX classifier to {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.

    Preconditions:
        - `sys.argv` is set as expected for a CLI entry point.

    Postconditions:
        - Returned namespace exposes every training knob.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-glob", type=str, default="data/*.parquet",
        help="Glob of session Parquets (default: data/*.parquet).",
    )
    parser.add_argument(
        "--val-session", type=str, default=None,
        help="Session to hold out for validation. None => random 20%% split.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("models/slouch_cnn.onnx"),
        help="Destination .onnx path.",
    )
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument(
        "--multiclass", action="store_true",
        help=(
            "Train on the full upright/slouch/shrimp 3-class label space. "
            "Default is binary upright vs not_upright, which matches the "
            "runtime gate's contract and is more robust on small data."
        ),
    )
    parser.add_argument(
        "--cv", action="store_true",
        help=(
            "Run leave-one-session-out cross-validation, then train a "
            "final model on ALL sessions for the median best-epoch and "
            "export that. Mutually exclusive with --val-session."
        ),
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda' / 'cpu'. Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    """Train the slouch classifier end-to-end and export ONNX.

    Returns:
        `0` on success, `1` on a recoverable failure (e.g. no data).

    Preconditions:
        - PyTorch is installed and the requested device is available.

    Postconditions:
        - On success, `--out` exists as an ONNX graph.
        - Train / val metrics are printed to stdout.
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = TrainConfig(
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    label_order = LABEL_ORDER_MULTI if args.multiclass else LABEL_ORDER_BINARY
    print(f"Label scheme: {'multiclass' if args.multiclass else 'binary'}  ({label_order})")

    df = load_combined(args.data_glob)
    if not args.multiclass:
        df = collapse_binary(df)
    X, y, sessions = build_windows(df, cfg.window_size, cfg.stride, label_order)
    print(
        f"Built {len(X)} windows of shape {X.shape[1:]}; "
        f"sessions: {sorted(set(sessions))}"
    )
    label_counts = {name: int((y == i).sum()) for i, name in enumerate(label_order)}
    print(f"Class counts: {label_counts}")

    if args.cv and args.val_session is not None:
        print("--cv and --val-session are mutually exclusive.", file=sys.stderr)
        return 1

    upright_index = label_order.index(UPRIGHT_LABEL)

    if args.cv:
        results = run_loso_cv(X, y, sessions, cfg, label_order, device)
        mean_acc, median_epoch = report_cv(results, label_order)
        n_final = max(median_epoch, 1)
        print(
            f"\n=== Final training on ALL sessions for {n_final} epochs "
            f"(median best epoch from CV) ==="
        )
        model = train_on_all(X, y, cfg, label_order, device, n_final)
        export_onnx(model, args.out, cfg.window_size, upright_index)
        print(
            f"\nCV mean val acc was {mean_acc:.3f}; final model is trained on "
            "all {} sessions and exported to {}.".format(
                len(set(sessions)), args.out
            )
        )
        return 0

    X_train, y_train, X_val, y_val = session_aware_split(
        X, y, sessions, args.val_session
    )
    print(f"Train windows: {len(X_train)}  |  Val windows: {len(X_val)}")
    if len(X_val) == 0:
        print("Validation set is empty; aborting.", file=sys.stderr)
        return 1

    model, best_state, best_val_acc, best_epoch, best_confusion = train_one_split(
        X_train, y_train, X_val, y_val, cfg, label_order, device
    )
    print(f"\nRestoring best epoch (epoch {best_epoch}, val acc {best_val_acc:.3f}).")
    model.load_state_dict(best_state)

    print("Best-epoch validation confusion matrix (rows=true, cols=pred):")
    header = "          " + "  ".join(f"{name:>10}" for name in label_order)
    print(header)
    for i, name in enumerate(label_order):
        row = "  ".join(
            f"{int(best_confusion[i, j]):>10d}" for j in range(len(label_order))
        )
        print(f"{name:>8}  {row}")

    export_onnx(model, args.out, cfg.window_size, upright_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
