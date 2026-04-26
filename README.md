# zapme

A vision-driven posture corrector built for **Hack AZ 2026**. A camera watches your back, a model decides if you're slouching, and a Raspberry Pi 4 closes a relay that gates a TENS unit driving EMS pads on your shoulders. Slouch → zap → straighten up.

> ⚠️ **Not a medical device.** This drives a consumer TENS unit through an external relay. Set the TENS to a comfortable intensity *before* connecting it to the Pi, never run it unattended, and read the safety section below before plugging anything in.

## How it works

```
webcam ─► YOLO11n-pose ─► geometric features ─► temporal CNN ─► P(slouch)
                                                                    │
                                                            hysteresis debouncer
                                                                    │
                                                  warn 1 → warn 2 → fire (audio escalation)
                                                                    │
                                                      single-pulse + cooldown (Pulser)
                                                                    │
                                                       GPIO pin → opto-relay → TENS gate
                                                                    │
                                                              watchdog (default-off)
```

Two warnings are spoken before any pulse fires. After a pulse, the gate is locked low for a cooldown window so the user has time to remove the pad if anything misbehaves. A watchdog thread independently drives the line low if the loop ever stops heart‑beating.

## Hardware

| Part | Notes |
| --- | --- |
| Raspberry Pi 4 (4 GB+) | Running 64‑bit Pi OS Bookworm. |
| USB webcam *or* Pi Camera | Anything OpenCV can open. Side-profile mounting trains best. |
| Consumer TENS unit | The Pi does **not** source the EMS current — it only gates the unit's own trigger line. |
| EMS pads | Placed on the upper back / shoulder area. |
| Opto-isolated relay module | Between the Pi GPIO pin and the TENS trigger. Default polarity is **active-LOW** (line HIGH at boot keeps the EMS de-energized). |
| Speaker / earbuds | For the spoken warnings. Optional — pass `--no-audio` to silence. |

Default GPIO pin is **BCM 17**. Override with `--gate-pin`. If your relay asserts on HIGH instead of LOW, pass `--gate-active-high`.

## Install

The codebase splits dependencies three ways so the Pi stays lean:

| Extra | What it adds | When to use |
| --- | --- | --- |
| (core) | YOLO + onnxruntime + OpenCV | Always installed. |
| `[pi]` | `gpiozero` (and the apt-installed `python3-lgpio`) | On the Pi for real GPIO. |
| `[train]` | torch / transformers / pyarrow / onnx | On a desktop / GPU box for training the classifier. |
| `[dev]` | pytest + ruff | For development. |

### Desktop (Windows / macOS / Linux x86) — for dev, demos, and training

```bash
git clone https://github.com/<you>/zapme.git
cd zapme
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .[train,dev]
```

### Raspberry Pi 4 — for live deployment

The supported install path uses the apt-packaged `python3-lgpio` (pip's build from source frequently fails on Bookworm because of a missing multiarch gcc symlink). Create the venv with `--system-site-packages` so it can see the apt-installed binding:

```bash
sudo apt update
sudo apt install -y python3-lgpio mpg123     # mpg123 is for audio playback

git clone https://github.com/<you>/zapme.git
cd zapme
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .[pi]
```

Make sure your user is in the `gpio` group (`sudo usermod -aG gpio $USER`, then re-login).

## Run

### Try it without any hardware

```bash
# 1. Just the YOLO pose feed — purely visual.
python scripts/yolo_view.py

# 2. Pose + features overlay + numbers in the terminal.
python scripts/demo_pose.py

# 3. End-to-end pipeline with the (placeholder) classifier.
python scripts/demo_classifier.py

# 4. Full runtime loop with a FakeGate — every "Pulse fired" is simulated.
python scripts/dry_run.py
```

`dry_run.py` accepts every flag the real entry point does (thresholds, classifier weights, log levels, etc.) and is the recommended way to tune `--on-threshold` / `--cooldown` / `--pulse-duration` from a laptop before plugging anything in.

### On the Pi

Always start with the live-test wrapper so you get a relay sanity pulse *before* the inference loop runs:

```bash
python scripts/live_test.py
```

It will:

1. Print a loud "LIVE — REAL HARDWARE" banner.
2. Fire one short pulse (`--pulse-test 0.05` by default) so you can confirm the relay clicks.
3. Force the `lgpio` backend regardless of platform default.
4. Use shortened warning intervals so a full warn → warn → fire cycle takes ~5 s instead of 30.

Skip the relay pulse with `--no-self-test`. Pass any of the regular flags through:

```bash
python scripts/live_test.py \
  --weights models/slouch_cnn.onnx \
  --camera 0 \
  --gate-pin 17 \
  --warn-to-warn 5 --warn-to-fire 5
```

The plain entry point (no banner, no self-test) is `python -m zapme.src` — use it under systemd / a shell wrapper for unattended runs. `Ctrl-C` always drops the line and exits cleanly.

It is IMPERATIVE that you start with the TENS unit __**OFF**__ or else you will be shocked immediately. 

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--backend {fake,lgpio}` | Auto-detected (lgpio on Linux, fake elsewhere). |
| `--gate-pin 17` | BCM pin to the relay. |
| `--gate-active-high` | Flip polarity if your relay needs HIGH to assert. |
| `--weights models/slouch_cnn.onnx` | Trained classifier. Without it, a placeholder rule runs so the loop still works. |
| `--on-threshold 0.8 --off-threshold 0.4` | Hysteresis on slouch probability. |
| `--debounce-window 20 --min-on-fraction 0.6` | How aggressively the debouncer fires. |
| `--warn-to-warn 10 --warn-to-fire 10 --fire-to-warn 10` | Audio escalation timing (seconds). |
| `--cooldown 15` | Seconds the gate stays locked low after a pulse. |
| `--pulse-duration 0.3` | How long the gate is asserted on each fire. |
| `--watchdog-timeout 3` | Heartbeat budget; tripped → gate forced low and exit code 1. |
| `--no-audio` | Use a silent `FakeSpeaker`. |
| `--imgsz 320` | YOLO input size. 320 ≈ 4× the FPS of 640. |

## Collect training data

```bash
python scripts/collect_dataset.py --out data/eric_session_01.parquet
```

Live preview pops up. Hotkeys (focus the OpenCV window):

| Key | Action |
| --- | --- |
| `1` | Record as `upright` |
| `2` | Record as `slouch` |
| `3` | Record as `shrimp` |
| `0` | Pause (preview only) |
| `q` | Save Parquet + quit |

Each row is one frame: timestamp, label, raw 17-keypoint COCO array, and the live geometric features. Frames where shoulder geometry is unreliable get NaN feature values — the classifier learns to handle that as its own state.

The collector now sets capture defaults to **640×480 @ 30 FPS** with `BUFFERSIZE=1`, plus picks a sensible OpenCV backend per platform (DirectShow on Windows, V4L2 on Linux). Override with `--width / --height / --fps`.

## Train the classifier

Training runs on a desktop / GPU box (the `[train]` extra). Slices each session into windows of 15 frames, recomputes features live from the raw keypoints (so feature-set tweaks don't require re-recording), and exports an ONNX graph the Pi-side `SlouchClassifier` consumes unchanged.

```bash
# Quick run with a held-out session.
python scripts/train_classifier.py \
  --data-glob 'data/*.parquet' \
  --val-session eric_session_03 \
  --out models/slouch_cnn.onnx

# Leave-one-session-out cross-val + final all-data fit.
python scripts/train_classifier.py \
  --data-glob 'data/*.parquet' --cv \
  --out models/slouch_cnn.onnx
```

Defaults are binary `upright` vs `not_upright`. Pass `--multiclass` for the full `upright / slouch / shrimp` head. Confusion matrices are printed per fold.

## Audio

Place MP3s under `zapme/assets/`:

| File | When it plays |
| --- | --- |
| `bootup.mp3` | Once at startup, after gate + watchdog wired. |
| `firstwarn.mp3` | First detection of a sustained slouch. |
| `finalwarn.mp3` | Slouch persisted past `--warn-to-warn`. |
| `zapwarn.mp3` | Immediately before a pulse fires. |
| `zapscream.mp3` | Plays right after `zapwarn` — overlaps with the actual zap. |

Missing clips are logged and skipped — the loop never blocks on audio. Linux uses `mpg123 -q`, macOS `afplay`, Windows `ffplay -nodisp -autoexit`.

## Tests

```bash
pip install -e .[dev]
pytest
```

The model side runs anywhere. Runtime tests use `FakeGate` / `FakeSpeaker` so the full loop exercises end-to-end without GPIO. Real-hardware paths are intentionally not in the default suite.

## Repository layout

```
zapme/
├── .llm/                       # Agent context + Python doc standard (binding)
├── zapme/src/
│   ├── __main__.py             # Entry point: python -m zapme.src
│   ├── model/                  # Hardware-agnostic: vision, features, classifier
│   ├── runtime/                # Pi-specific: loop, gate, watchdog, feature buffer
│   └── utils/                  # Speaker, TENS controller, cross-cutting helpers
├── zapme/assets/               # Voice clips
├── scripts/
│   ├── yolo_view.py            # Pure visual feed (audience demo)
│   ├── demo_pose.py            # Pose + features in terminal
│   ├── demo_classifier.py      # End-to-end with a placeholder/trained classifier
│   ├── collect_dataset.py      # Hotkey-driven labeled recorder
│   ├── train_classifier.py     # Desktop training → ONNX export
│   ├── dry_run.py              # Full pipeline, FakeGate (no hardware)
│   └── live_test.py            # Real-hardware bring-up wrapper
├── tests/                      # Pytest suite (model + runtime fakes)
├── pyproject.toml
└── requirements.txt
```

## Safety notes

- **Power on the TENS unit only AFTER the Pi has booted and the runtime has claimed the GPIO line.** If the TENS unit is already powered when the Pi boots, the line floats during early boot and the unit will fire a pulse before the runtime is even up. Boot order: Pi up → runtime running (you'll hear `bootup.mp3`) → *then* switch the TENS on.
- The TENS unit's intensity is set on the unit itself, not by the Pi. **Set it to a comfortable level with the pads on your skin before letting the Pi gate it.**
- **Set the TENS output to deliver ≥3 V** to actually feel the pulse. Below that the stim is too weak to register through the EMS pads. We power the unit itself with a fresh 9 V battery so it has the headroom to hit that output level.
- The relay is **active-LOW by default** so the line idles HIGH (de-energized) at boot, before Python is even running.
- The watchdog forces the gate low on missing heartbeats, uncaught exceptions, signal handlers, and process exit. Don't disable it.
- `--cooldown 0` disables the post-pulse lockout. Don't.
- Don't run unattended. Don't use on broken/wet skin, near the heart/throat, or with implanted electronics. Standard TENS contraindications apply.

## License

Apache 2.0. See [LICENSE](LICENSE).
