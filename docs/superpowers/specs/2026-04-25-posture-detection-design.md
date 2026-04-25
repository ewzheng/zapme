# zapme — Posture Detection Pipeline Design

**Date:** 2026-04-25
**Status:** Approved (pending user review of this written spec)
**Context:** Hack AZ 2026, mid-hackathon. Project background and constraints live in `.llm/llm.MD` and `.llm/documentation.MD` and are assumed by this document.

## Goal

Build the on-device vision + classification pipeline that decides when to assert the EMS gate. The hardware integration (camera, GPIO, EMS) is in scope; off-device training is in scope at the artifact level (we ship the trained MLP into the repo).

## Non-goals

- Multi-person handling. Top-confidence detection only.
- Model retraining on the Pi. The Pi runs inference only.
- On-device dataset collection during competition. Collect at home if possible.
- Web UI / phone app. The device is the device.
- Persistent metrics or cloud sync. Local logging only.

## Architecture

End-to-end flow on the Pi:

```
Webcam ──> frame capture ──> YOLO11n-pose (ONNX) ──> 17 keypoints
                                                           │
                                                feature engineering
                                                           │
                                                    MLP classifier
                                                           │
                                                  P(slouch) ──> debounce / hysteresis
                                                                       │
                                                                 GPIO gate ──> EMS
                                                                       │
                                                                  watchdog
```

Two execution environments:

- **Desktop (developer's 7900XTX box):** trains the MLP on collected keypoint datasets, exports YOLO11n-pose to ONNX, exports MLP weights. Pushes both artifacts to the repo.
- **Pi 4:** runs `onnxruntime` for YOLO inference and a tiny pure-NumPy forward pass for the MLP. **No `torch` install required on the Pi** — keeps the Pi image lean.

The `model/` ↔ `runtime/` ↔ `utils/` seam from `.llm/llm.MD` holds: pose, feature engineering, and classifier are pure model code; camera, gate, loop, and watchdog are runtime; config and logging are utils.

## Components

### `zapme/src/model/` — hardware-agnostic, unit-testable on any machine

- **`pose.py`** — `PoseEstimator` class. Loads `yolo11n-pose.onnx` via `onnxruntime`, accepts a frame (`np.ndarray`), returns the highest-confidence person's 17 COCO keypoints as a `(17, 3)` array of `(x, y, conf)`. Single responsibility: image → keypoints.
- **`features.py`** — pure functions converting keypoints → MLP feature vector. Candidate features (final list driven by data, not committed yet): neck-shoulder angle, ear-shoulder horizontal offset (head-forward distance), shoulder-hip vertical alignment, shoulder symmetry. Normalized by torso length so it is scale-invariant.
- **`classifier.py`** — `SlouchClassifier` class. Loads MLP weights from `.npz`, takes a feature vector, returns `P(slouch) ∈ [0, 1]`. Forward pass is ~10 lines of NumPy.

### `zapme/src/runtime/` — Pi-specific, default-off, watchdog-protected

- **`camera.py`** — webcam capture via OpenCV (`cv2.VideoCapture`). Yields frames as a generator. Disconnect raises; the loop's exception handler closes the gate.
- **`gate.py`** — `Gate` interface with `LgpioGate` (Pi) and `FakeGate` (Windows / dev) implementations. Default-off at construction. Context-manager lifecycle so process exit always drives the line low.
- **`loop.py`** — main loop. Pulls frames, runs pose → features → classifier, applies debounce, drives the gate, feeds the watchdog.
- **`watchdog.py`** — heartbeat-based. If the loop misses N heartbeats (hung model, dead camera), force gate-off.

### `zapme/src/utils/`

- **`config.py`** — load thresholds, debounce window, gate pin, camera index, model paths from a single TOML config file. Pydantic or dataclass-based.
- **`logging.py`** — structured logging setup, single line per gate transition.

### `scripts/`

- **`collect_dataset.py`** — runs camera + pose, lets the user label frames as slouch / not-slouch with keypress, dumps to `data/keypoints.parquet`.
- **`train_mlp.py`** — trains MLP on `data/keypoints.parquet` on the desktop, exports `models/slouch_mlp.npz`.
- **`export_yolo.py`** — exports YOLO11n-pose to ONNX, drops to `models/yolo11n-pose.onnx`.

## Data flow & key decisions

- **Camera placement:** default to **side profile** (camera on a desk arm to the user's left or right). This makes ear-shoulder-hip geometry trivially observable. Code should still work front-on with a different feature set, but the MLP will be trained and tuned for side profile first.
- **Inference cadence:** target **5–10 FPS**. YOLO11n-pose ONNX on Pi 4 CPU lands here comfortably.
- **Debounce:** require N consecutive frames over a sliding window (e.g., 4 of last 6 ≈ 0.6 s at 10 FPS) above the slouch threshold before asserting the gate. Symmetric release window to avoid chatter. Exact thresholds live in config, not constants.
- **MLP I/O:** input = ~8–12 normalized features (final list driven by data), output = single sigmoid for slouch probability. Two hidden layers of 32 units each is overkill but trains in seconds and runs in microseconds.
- **MLP serialization:** plain NumPy `.npz` of weights / biases. Pi-side classifier does the forward pass in NumPy directly.
- **Model artifacts in repo:** `models/yolo11n-pose.onnx` (~12 MB) and `models/slouch_mlp.npz` (~few KB). Both fit fine in git without LFS.

## Error handling & safety

- **Default-off everywhere.** Gate is constructed in the off state; the runtime explicitly enables it after the watchdog and loop are healthy.
- **Any exception in the loop** → outer `try` / `finally` releases the gate before propagating.
- **No detection or low-confidence pose** → treat as "no slouch" (do not assert gate); reset debounce buffer so a flaky frame does not pollute the window. Log at debug level.
- **Watchdog timeout** (no heartbeat for >1 s) → force gate off, log error, exit. Systemd or a shell wrapper restarts the process.
- **SIGINT / SIGTERM** → handler closes gate before exiting.
- **Camera disconnect** → close gate, log, exit (let the wrapper restart).

## Testing strategy

- **Model side** runs anywhere. Unit tests use checked-in sample images and canned keypoint arrays. No GPIO, no camera.
  - `tests/test_pose.py` — sanity-check that the wrapper returns sane shapes from a sample frame.
  - `tests/test_features.py` — pure-function tests with hand-crafted keypoint inputs.
  - `tests/test_classifier.py` — runs the MLP on a known input and asserts a known output (regression test for the serialization format).
- **Runtime side** uses fakes by default.
  - `FakeGate` records transitions in memory; tests assert the right transitions happen given a script of `P(slouch)` values.
  - `FakeCamera` yields canned frames.
  - `tests/test_loop.py` — drives the loop end-to-end with both fakes, checks debounce + safety behavior.
- Real-hardware tests (`@pytest.mark.hardware`) skipped by default; only run on the Pi.

## Stretch — Gemma award track

Optional, **non-blocking**, off the hot path:

- After a sustained slouch event, asynchronously kick a small Gemma model (Gemma 3n if it will run on a Pi 4 8 GB; otherwise a remote API call as a demo concession) to generate a one-line coaching message.
- Speak it via TTS or render to a small e-paper / OLED screen if time permits.
- Lives in `runtime/coach.py`, fed by an `asyncio.Queue` so it never blocks the inference loop.

This is a "do it if everything else works and we have an hour left" item. The main hardware demo does not depend on it.

## Dependencies

The current `pyproject.toml` puts everything in core deps. To honor the "no torch on the Pi" decision, the layout needs to be reorganized when implementing:

- **Core** (installs on both desktop and Pi): `huggingface_hub[cli]`, `numpy`, `pillow`, `onnxruntime`, `opencv-python`.
- **`[project.optional-dependencies.train]`** (desktop only — install with `pip install -e .[train]`): `torch`, `torchvision`, `transformers`, `datasets`, `bitsandbytes`, `ultralytics`.
- **`[project.optional-dependencies.pi]`** (marker-gated to Linux aarch64): `lgpio`, `gpiozero`.
- **`[project.optional-dependencies.dev]`**: `pytest`, `ruff`.

This split is what enables the Pi-side image to stay lean: the Pi installs `pip install -e .[pi]` and gets `onnxruntime` + GPIO bindings only, no torch / transformers / bitsandbytes. The desktop installs `pip install -e .[train,dev]` and gets the full ML stack.

`requirements.txt` should be regenerated to mirror this split with environment markers.

## Implementation approach

The plan that follows this spec will be structured for **iterative, user-paced execution**. Each step lands as an independently reviewable chunk; do not dispatch the whole plan to a subagent in one go.

Suggested ordering (subject to revision in the plan):

1. Wire up the model side end-to-end with stubs + `FakeGate` / `FakeCamera` so the runtime loop can be exercised on Windows.
2. Implement `pose.py` against a real ONNX model using a checked-in sample frame.
3. Implement `features.py` and `classifier.py` (initially with hand-set MLP weights so the loop is functional before training).
4. Implement `runtime/loop.py` + `gate.py` + `watchdog.py` against the fake backends; add the loop test.
5. Wire up real camera on Pi, real `LgpioGate`, real EMS gating; verify on hardware.
6. Collect data, train MLP, export, swap in.
7. (Stretch) Gemma coaching path.
