# zapme — Agent Context

This document is the source of truth for agents working on `zapme`. It describes what the project is, what it runs on, how the codebase is organized, and the constraints that should shape every design decision. The Python documentation standard lives next to this file in `documentation.MD` and is binding for all code under `zapme/src/`.

## What zapme is

`zapme` is a Hack AZ 2026 project: a closed-loop posture corrector built around a Raspberry Pi 4. A camera observes the user's upper back and shoulders, a vision model classifies posture in near real time, and on a sustained slouch detection the device drives a GPIO line that gates an EMS (electrical muscle stimulation) unit attached to the user's back. The shock is the corrective signal — the user adjusts posture and the gate closes.

The Pi is both the inference host and the actuator. There is no cloud component, no companion app, and no external service in the loop. Everything between the camera and the EMS unit runs on-device.

## Target hardware

- **Compute:** Raspberry Pi 4 (assume 4 GB unless otherwise noted).
- **Camera:** Pi Camera Module or a USB UVC webcam over `/dev/video*`.
- **Actuator:** an EMS unit whose trigger input is gated by a transistor/optoisolator driven from a Pi GPIO pin. The Pi never sources the EMS current itself — the GPIO line only switches the gate.
- **GPIO access:** prefer direct register access via memory-mapped I/O on `/dev/gpiomem` (MMIO) for the hot path. Library wrappers (`RPi.GPIO`, `gpiozero`, `lgpio`) are acceptable for setup, calibration scripts, and tests, but the inference-driven trigger path should not pay their per-call overhead.

Treat the EMS output as a safety-critical actuator. Default to *off*. Any code path that can assert the trigger line must be paired with a watchdog that releases it on missing frames, model errors, exceptions, or process exit.

## Runtime topology

The end-to-end loop is intentionally simple:

```
camera frame  →  preprocess  →  model.infer  →  classify(slouch?)
                                                       │
                                              debounce / hysteresis
                                                       │
                                                  GPIO gate  →  EMS unit
```

Three hard separations matter:

1. **`model/` is hardware-agnostic.** It loads a model, accepts a numpy/torch frame, returns a posture classification. It must be unit-testable on a developer laptop without a Pi, a camera, or GPIO. Do not import anything from `runtime/` inside `model/`.
2. **`runtime/` owns everything Pi-specific.** Camera capture, GPIO setup, MMIO writes, the EMS gating policy, the main loop, and the watchdog all live here. `runtime/` may import from `model/` and `utils/`.
3. **`utils/` is the lowest layer.** Cross-cutting helpers (config loading, logging setup, timing, small data structures) with no hardware and no model dependency of their own. Both `model/` and `runtime/` may import from `utils/`; `utils/` imports from neither.

Keeping these seams clean is what lets us iterate on the model on a laptop and only deploy to the Pi for hardware-in-the-loop work.

## Repository layout

```
zapme/
├── .llm/                    # This file and the documentation standard
│   ├── llm.MD               # ← you are here (agent context)
│   └── documentation.MD     # Binding Python documentation standard
├── zapme/src/               # Main Python package
│   ├── __init__.py
│   ├── __main__.py          # Entry point: python -m zapme.src
│   ├── model/               # Vision model: load, preprocess, infer, classify
│   │   └── __init__.py
│   ├── runtime/             # Hardware I/O: camera, GPIO/EMS, main loop, watchdog
│   │   └── __init__.py
│   └── utils/               # Cross-cutting helpers (config, logging, timing, glue)
│       └── __init__.py
├── scripts/                 # One-shot utilities (calibration, model conversion, flashing)
├── tests/                   # Pytest suite (mirrors zapme/src/ layout)
├── CLAUDE.md                # Stub pointing here
├── AGENTS.md                # Stub pointing here
├── LICENSE
└── README.md
```

## Constraints that shape design

These are non-obvious and easy to violate without thinking about the deployment target.

- **Latency budget.** The full loop (capture → infer → decide → GPIO) should fit comfortably under the camera's frame interval at the chosen FPS. On a Pi 4 with a small posture classifier, plan for 5–15 FPS, not 30+. Pick the smallest model that does the job; quantize aggressively.
- **Thermal and power.** A Pi 4 under sustained ML load throttles. Avoid spinning a CPU thread to busy-wait; prefer blocking I/O and event-driven loops.
- **No floating actuator.** The EMS gate line must be driven low at startup, before any model is loaded, and held low until the runtime explicitly enables it. On any uncaught exception, signal handler, or watchdog timeout the line returns to low.
- **Debounce, don't react instantly.** A single frame of "slouch" is not a slouch. Use temporal hysteresis (e.g., N consecutive positive frames over a window) before asserting the gate, and a similar window before releasing it. Exact thresholds belong in config, not constants buried in code.
- **MMIO is unforgiving.** Direct `/dev/gpiomem` writes bypass the kernel's safety nets. Wrap them in a small module with clear ownership semantics and acquire/release lifecycle, and use it through a context manager from the main loop.
- **Hardware-agnostic model code.** Anything that imports `RPi.GPIO`, `picamera2`, `mmap`, or similar is `runtime/` code by definition. If you find yourself reaching for those inside `model/`, you are about to break the seam.

## Toolchain

- **Python 3.11.** The documentation standard assumes 3.11 generics (`list[str]`, `X | None`).
- **Build system:** not yet committed. When adding one, prefer a single `pyproject.toml` at the repo root.
- **Tests:** not yet wired up. Use `pytest` and place tests under `tests/` mirroring the `zapme/src/` layout. Tests for `model/` should run on any developer machine; tests for `runtime/` should be guarded so they only run on a Pi (or against a fake GPIO/camera backend).
- **Lint/format:** not yet committed. When adding one, `ruff` is a reasonable default — it covers the formatting and lint surface without dragging in a separate formatter.

There are intentionally no build/test/lint commands listed here yet because none have been defined. Update this section the moment they exist.

## Documentation standard (summary)

The full standard is in `documentation.MD` and is binding. The non-negotiables:

- Every function (public, private, async, helper, method, classmethod, staticmethod, `main`) has full type annotations and a Google-style docstring with explicit `Preconditions` and `Postconditions`.
- Use `X | None`, not `Optional[X]`. Use `list[str]`, not `typing.List[str]`.
- Modules and classes carry their own docstrings describing purpose, lifecycle, and invariants.
- Updating code without updating its docstring is incomplete work.

Read `documentation.MD` before writing any function in this repo.

## What to do when something is missing

This project is at the boilerplate stage. Many modules are empty stubs. When you are asked to implement a piece:

1. Re-read this file and `documentation.MD`.
2. Decide which side of the `model/` ↔ `runtime/` ↔ `utils/` seams the work belongs on, and stay on that side.
3. If the work touches the EMS trigger path, write the fail-safe (default-off, watchdog, exception handling) before the happy path.
4. If a tool, command, or convention is missing from this document, add it here as part of the same change. This file should always describe the current state of the project.
