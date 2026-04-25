"""Shared model-loading utilities.

This module owns *how* models are fetched and materialized in memory. The
public model APIs (e.g. `vision.py`) consume these helpers so they can
stay focused on inference contracts rather than file management or
framework instantiation.

Weights are pulled from the Hugging Face Hub via `huggingface_hub`. Files
are cached under the standard HF cache (`~/.cache/huggingface/hub` by
default), so repeated calls do not re-download.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def download_from_hf(
    repo_id: str,
    filename: str,
    revision: str | None = None,
) -> Path:
    """Download a single file from a Hugging Face Hub repo, returning its local path.

    Wraps `huggingface_hub.hf_hub_download` so callers do not need to depend
    on the HF API directly. The file is cached on disk; subsequent calls with
    the same arguments are no-ops that just return the cached path.

    Args:
        repo_id: Hugging Face repo identifier, e.g. `"Ultralytics/YOLO11"`.
        filename: Path to the target file within the repo, e.g.
            `"yolo11n-pose.pt"`.
        revision: Optional git revision (branch, tag, or commit SHA). When
            `None`, the repo's default branch is used.

    Returns:
        Absolute path to the cached file on the local filesystem.

    Raises:
        huggingface_hub.errors.HfHubHTTPError: Raised when the file or repo
            cannot be resolved (e.g. typo, gated repo, network failure).

    Preconditions:
        - `repo_id` and `filename` are non-empty strings that resolve to a
          public file on the Hugging Face Hub (or one the caller has access
          to via `huggingface-cli login`).

    Postconditions:
        - The returned path exists and points to the requested file.
        - No mutation of any caller-owned state.
    """
    cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    return Path(cached)


def load_yolo_from_hf(
    repo_id: str,
    filename: str,
    revision: str | None = None,
) -> YOLO:
    """Download a YOLO checkpoint from Hugging Face and load it into memory.

    Combines `download_from_hf` with the Ultralytics `YOLO` constructor so
    the public model API never has to touch file paths or framework
    instantiation directly.

    Args:
        repo_id: Hugging Face repo holding the YOLO checkpoint.
        filename: Weights filename within `repo_id` (e.g. `yolo11n-pose.pt`).
        revision: Optional git revision pin for reproducibility.

    Returns:
        An initialized `ultralytics.YOLO` model ready for `predict()`.

    Raises:
        huggingface_hub.errors.HfHubHTTPError: Raised when the file cannot
            be downloaded.
        Exception: Re-raises any error from the Ultralytics loader (e.g.
            corrupted weights, version mismatch).

    Preconditions:
        - `repo_id` + `filename` resolve to a YOLO-format checkpoint
          compatible with the installed `ultralytics` version.

    Postconditions:
        - Model weights are present in the HF cache.
        - Returned model is loaded and ready for inference; ownership
          transfers to the caller.
    """
    weights_path = download_from_hf(repo_id=repo_id, filename=filename, revision=revision)
    return YOLO(str(weights_path))


def create_onnx_session(model_path: Path) -> ort.InferenceSession:
    """Open an ONNX model with sane defaults for the Pi 4 deployment target.

    Pins the CPU execution provider explicitly. Other providers (CUDA,
    DirectML) are usually fine, but pinning makes behavior identical
    between the developer laptop and the Pi, which matters for the
    hand-off of the slouch classifier.

    Args:
        model_path: Filesystem path to a `.onnx` graph.

    Returns:
        An `onnxruntime.InferenceSession` ready for `run()`.

    Raises:
        FileNotFoundError: If `model_path` does not exist.
        onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile: When the
            session loader rejects the path.

    Preconditions:
        - `model_path` points to a well-formed ONNX graph.

    Postconditions:
        - Returned session is initialized with the CPU provider only.
        - No mutation of caller-owned state.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
