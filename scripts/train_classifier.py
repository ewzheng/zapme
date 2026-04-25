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

from zapme.src.model.features import MLP_FEATURES, NUM_FEATURES

LABEL_ORDER: tuple[str, ...] = ("upright", "slouch", "shrimp")
UPRIGHT_INDEX: int = LABEL_ORDER.index("upright")


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
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


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
    """
    feature_cols = list(MLP_FEATURES)
    df = df.sort_values(["session", "ts"]).reset_index(drop=True)

    segment_id = (
        (df["session"] != df["session"].shift())
        | (df["label"] != df["label"].shift())
    ).cumsum()

    X_chunks: list[np.ndarray] = []
    y_chunks: list[int] = []
    session_chunks: list[str] = []

    label_to_idx = {name: i for i, name in enumerate(LABEL_ORDER)}

    for _, segment in df.groupby(segment_id, sort=False):
        if len(segment) < window_size:
            continue
        label_name = segment["label"].iloc[0]
        if label_name not in label_to_idx:
            continue
        label_idx = label_to_idx[label_name]
        session_name = segment["session"].iloc[0]
        feats = segment[feature_cols].to_numpy(dtype=np.float32)

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
    """Tiny 1D CNN over (B, NUM_FEATURES, window_size) feature windows.

    Three Conv1d blocks with two max-pools, then global average pooling
    and two FC layers. Total parameter count is in the low thousands —
    trains in seconds on any GPU and runs in microseconds on CPU.
    """

    def __init__(self, normalizer: FeatureNormalize, num_classes: int = 3) -> None:
        """Assemble the CNN with a baked-in normalization head.

        Args:
            normalizer: Pre-fitted `FeatureNormalize` placed before the
                first Conv1d so ONNX export captures the normalization.
            num_classes: Output class count; defaults to 3 to match
                `LABEL_ORDER`.

        Preconditions:
            - `normalizer` was fitted on the training set.

        Postconditions:
            - `self` is in train mode and ready to receive `(B, C, T)`
              tensors with `C == NUM_FEATURES`.
        """
        super().__init__()
        self.normalizer = normalizer
        self.body = nn.Sequential(
            nn.Conv1d(NUM_FEATURES, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

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
    returns `1 - P(upright)` so the runtime contract stays unchanged
    even though we trained multi-class.
    """

    def __init__(self, classifier: SlouchCNN) -> None:
        """Hold a trained classifier; do not modify its parameters.

        Args:
            classifier: A trained `SlouchCNN` ready for inference.

        Preconditions:
            - `classifier` is in eval mode (caller's responsibility).

        Postconditions:
            - This wrapper carries no extra trainable parameters.
        """
        super().__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute slouch probability `1 - P(upright)` per batch element.

        Args:
            x: Float tensor `(B, NUM_FEATURES, window_size)`.

        Returns:
            Float tensor `(B, 1)` of slouch probabilities in `[0, 1]`.

        Preconditions:
            - The wrapped classifier was trained with `LABEL_ORDER` such
              that `LABEL_ORDER[UPRIGHT_INDEX] == "upright"`.

        Postconditions:
            - Output values lie in `[0, 1]`.
        """
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        upright = probs[:, UPRIGHT_INDEX : UPRIGHT_INDEX + 1]
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


def train_loop(
    model: SlouchCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    """Run the full training loop, printing per-epoch metrics.

    Args:
        model: Initialized `SlouchCNN`.
        train_loader: DataLoader over the training set.
        val_loader: DataLoader over the held-out validation set.
        cfg: Training hyperparameters.
        device: `'cuda'` or `'cpu'` torch device.

    Preconditions:
        - `model` is on `device`.
        - Both loaders yield `(X, y)` tuples on CPU; the loop moves them.

    Postconditions:
        - `model` is in eval mode on return with weights from the final
          epoch (no early stopping; sample sizes are too small for it
          to be informative).
    """
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

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

        val_acc, val_per_class, val_confusion = evaluate(model, val_loader, device)

        train_loss = train_loss_sum / max(train_n, 1)
        train_acc = train_correct / max(train_n, 1)
        per_class_str = "  ".join(
            f"{name}={val_per_class.get(name, float('nan')):.2f}"
            for name in LABEL_ORDER
        )
        print(
            f"epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.3f} "
            f"| val acc={val_acc:.3f}  per-class: {per_class_str}"
        )

    print("Final validation confusion matrix (rows=true, cols=pred):")
    header = "          " + "  ".join(f"{name:>8}" for name in LABEL_ORDER)
    print(header)
    for i, name in enumerate(LABEL_ORDER):
        row = "  ".join(f"{val_confusion[i, j]:>8d}" for j in range(len(LABEL_ORDER)))
        print(f"{name:>8}  {row}")


def evaluate(
    model: SlouchCNN,
    loader: DataLoader,
    device: torch.device,
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
    confusion = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=np.int64)
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
    for i, name in enumerate(LABEL_ORDER):
        row_total = int(confusion[i, :].sum())
        per_class[name] = (
            float(confusion[i, i]) / row_total if row_total > 0 else float("nan")
        )

    return n_correct / max(n_total, 1), per_class, confusion


def export_onnx(
    model: SlouchCNN,
    out_path: Path,
    window_size: int,
) -> None:
    """Export a slouch-probability ONNX graph from a trained classifier.

    Args:
        model: Trained `SlouchCNN` on any device.
        out_path: Destination `.onnx` path. Parent dir is created if
            absent. Existing files at this path are overwritten.
        window_size: Window length used at training; baked into the
            graph as a fixed input shape.

    Preconditions:
        - `model.normalizer` holds the training-set normalization stats.

    Postconditions:
        - `out_path` exists and is loadable by `onnxruntime`.
        - Exported graph takes `(1, NUM_FEATURES, window_size)` float32
          input and emits `(1, 1)` float32 slouch probability.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = SlouchONNXWrapper(model.cpu().eval())
    dummy = torch.zeros((1, NUM_FEATURES, window_size), dtype=torch.float32)
    torch.onnx.export(
        cpu_model,
        dummy,
        str(out_path),
        input_names=["window"],
        output_names=["slouch_prob"],
        opset_version=17,
        dynamic_axes=None,
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    )

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    df = load_combined(args.data_glob)
    X, y, sessions = build_windows(df, cfg.window_size, cfg.stride)
    print(
        f"Built {len(X)} windows of shape {X.shape[1:]}; "
        f"sessions: {sorted(set(sessions))}"
    )
    label_counts = {name: int((y == i).sum()) for i, name in enumerate(LABEL_ORDER)}
    print(f"Class counts: {label_counts}")

    X_train, y_train, X_val, y_val = session_aware_split(
        X, y, sessions, args.val_session
    )
    print(f"Train windows: {len(X_train)}  |  Val windows: {len(X_val)}")
    if len(X_val) == 0:
        print("Validation set is empty; aborting.", file=sys.stderr)
        return 1

    normalizer = fit_normalizer(X_train)
    model = SlouchCNN(normalizer).to(device)

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

    train_loop(model, train_loader, val_loader, cfg, device)
    export_onnx(model, args.out, cfg.window_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
