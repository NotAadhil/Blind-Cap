# Blind Cap — Assistive Vision Prototype

> **⚠️ SAFETY DISCLAIMER**: This is an **experimental research prototype**, NOT a certified mobility aid. It cannot guarantee obstacle detection. **"No detection" does NOT mean "path is safe."** Never rely on this system as your sole means of navigation. Always use established mobility techniques, a cane, or a guide dog.

---

## What Is Blind Cap?

Blind Cap is a real-time computer-vision assistant that:

1. Captures video from a webcam.
2. Detects objects using **YOLO26n** (a fast, lightweight AI model).
3. Determines whether objects are in your walking path.
4. Estimates relative urgency using visual heuristics.
5. Prioritises the most important hazard.
6. Gives **short spoken warnings** through your speakers or headphones.
7. Prevents repetitive speech spam with smart cooldowns.
8. Maintains a smooth camera display with diagnostic overlays.

### Long-Term Vision

```
Camera on wearable cap  →  Raspberry Pi  →  Wi-Fi  →  PC (AI)  →  Speech  →  Bluetooth earphones
```

**This version** runs entirely on a Windows PC using the built-in webcam. The modular architecture means the webcam can later be swapped for a Raspberry Pi camera stream without rewriting the AI, hazard, or TTS systems.

---

## Current Limitations

- Only detects objects the pretrained COCO model knows (80 classes).
- Distance is estimated from bounding-box size — **not real depth measurement**.
- Cannot detect potholes, stairs, curbs, or transparent obstacles (glass).
- Does not work in complete darkness.
- Speech is English only (Windows SAPI5).
- **This is NOT a substitute for proper mobility training or safety equipment.**

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     BLIND CAP                            │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐   │
│  │  Camera   │──→│ Detector │──→│  Decision Engine   │   │
│  │ (threaded)│   │ OpenVINO │   │  Hazard States     │   │
│  └──────────┘   │ YOLO26n  │   │  Focus Selection   │   │
│                  └──────────┘   │  Warning Cooldowns  │   │
│                                 └─────────┬──────────┘   │
│                                           │              │
│  ┌──────────┐                  ┌──────────▼──────────┐   │
│  │ Visualizer│←────────────────│   TTS Engine        │   │
│  │   HUD     │                 │   Priority Queue    │   │
│  └──────────┘                  │   SAPI5 Worker      │   │
│                                 └────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start Guide

### 1. Install Python

Download **Python 3.11+** from [python.org](https://www.python.org/downloads/).

During installation, **check "Add Python to PATH"**.

Verify in PowerShell:

```powershell
python --version
```

### 2. Create a Virtual Environment

Open PowerShell and navigate to the project folder:

```powershell
cd "d:\Projects\Project Azure"
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> **If you get a script execution error**, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- `openvino` — Intel inference engine
- `ultralytics` — YOLO model framework
- `opencv-python` — Video capture and display
- `pyttsx3` — Text-to-speech (Windows SAPI5)
- `numpy`, `pyyaml` — Data processing
- `pywin32` — Windows COM support for TTS threading
- `pytest` — Unit testing

### 5. Check Intel GPU Visibility

```powershell
python -c "from openvino import Core; c = Core(); print(c.available_devices)"
```

Expected output (with Intel Iris Xe):
```
['CPU', 'GPU']
```

If you only see `['CPU']`, your Intel GPU drivers may need updating. Visit [Intel Driver Support](https://www.intel.com/content/www/us/en/support/detect.html).

### 6. Export the YOLO26n Model to OpenVINO

This step converts the model for Intel hardware. **Run once:**

```powershell
python export_model.py
```

This creates `models/yolo26n_openvino/` containing the optimised model files.

For better GPU performance (FP16):
```powershell
python export_model.py --half --force
```

### 7. Run Hardware Diagnostics

```powershell
python tools/hardware_check.py
```

This checks Python, OpenCV, OpenVINO, GPU, webcam, and TTS.

### 8. Test the TTS System

**Critical step** — run this before the main app:

```powershell
python tools/test_tts.py
```

You should hear 5 complete sentences. If speech stops after one word or freezes, there is a TTS issue to resolve before proceeding.

### 9. Run the Benchmark

```powershell
python tools/benchmark.py
```

Compares CPU vs GPU inference speed.

### 10. Run Blind Cap

```powershell
python main.py
```

With options:

```powershell
python main.py --device GPU --camera 0 --debug
python main.py --mute                          # start silent
python main.py --device CPU                    # force CPU inference
```

---

## Keyboard Controls

| Key       | Action                    |
|-----------|---------------------------|
| `Q` / `ESC` | Quit                   |
| `M`       | Mute / Unmute speech      |
| `S`       | Toggle Quiet mode         |
| `R`       | Repeat last warning       |
| `D`       | Toggle Debug logging      |
| `B`       | Toggle Benchmark overlay  |

**Quiet mode** suppresses INFO and most CAUTION alerts but allows WARNING and CRITICAL alerts through.

---

## Configuration

All settings are in [`config.py`](config.py). Key parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `"yolo26n.pt"` | YOLO model to use |
| `OPENVINO_DEVICE` | `"GPU"` | Inference device (`GPU`, `CPU`, `AUTO`) |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `CAMERA_WIDTH` | `640` | Capture width |
| `CAMERA_HEIGHT` | `480` | Capture height |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `PATH_LEFT_RATIO` | `0.30` | Walking corridor left boundary |
| `PATH_RIGHT_RATIO` | `0.70` | Walking corridor right boundary |
| `PERSISTENCE_FRAMES` | `3` | Frames before a detection is "stable" |
| `MIN_HAZARD_AREA_RATIO` | `0.03` | Minimum size to trigger a hazard |
| `WARNING_COOLDOWN` | `5.0` | Seconds between repeated warnings |
| `CRITICAL_COOLDOWN` | `3.0` | Seconds between critical alerts |
| `TTS_ENABLED` | `True` | Enable/disable speech |
| `TTS_RATE` | `175` | Speech speed (words per minute) |
| `TTS_MODE` | `"NORMAL"` | `NORMAL`, `MINIMAL`, or `URGENT` |

---

## How the Hazard System Works

### Walking Path Model

The camera view is divided into three vertical zones:

```
┌──────────┬────────────────┬──────────┐
│   LEFT   │  WALKING PATH  │  RIGHT   │
│  0–30%   │    30–70%      │ 70–100%  │
└──────────┴────────────────┴──────────┘
```

Only objects in the **centre corridor** are evaluated as potential hazards.

### Hazard Scoring

Each detected object gets a multi-factor score:

- **Class priority** — cars score higher than bottles
- **Confidence** — high-confidence detections score higher
- **Path overlap** — more overlap with the corridor = higher score
- **Size** — larger bounding box = visually closer (heuristic, not real depth)
- **Vertical position** — objects at the bottom of the frame are likely closer
- **Persistence** — objects seen for multiple frames score higher
- **Approach** — growing bounding-box area suggests approaching

### Hazard States

Each tracked object moves through a state machine:

```
NEW → STABLE → CLOSER → CRITICAL → CLEARED
```

- **NEW**: First detected, waiting for persistence confirmation
- **STABLE**: Confirmed detection, initial warning spoken
- **CLOSER**: Bounding box is growing (approaching heuristic)
- **CRITICAL**: Very large or high-priority — urgent warning
- **CLEARED**: Object disappeared

### Priority Levels

| Level | Examples | Response |
|-------|----------|----------|
| **CRITICAL** | Car, bus, truck directly ahead | "STOP. Car ahead." |
| **WARNING** | Person, bicycle, chair in path | "Person ahead." |
| **CAUTION** | Partial overlap, distant object | Spoken only in NORMAL mode |
| **INFO** | Side objects, background | Displayed only, not spoken |

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

Tests cover:
- **Decision engine**: regions, persistence, cooldowns, priority, approach detection, focus selection, state machine
- **TTS queue**: message ordering, deduplication, mute/quiet, shutdown
- **Detector utilities**: letterbox resizing, region classification

---

## Troubleshooting

### "No module named 'openvino'"
```powershell
pip install openvino
```

### "Camera not found"
- Check webcam connection
- Try different camera index: `python main.py --camera 1`
- Check Windows Settings → Privacy → Camera

### GPU not detected
- Install latest Intel GPU drivers
- Verify: `python -c "from openvino import Core; print(Core().available_devices)"`
- The app will automatically fall back to CPU

### TTS freezes or speaks only one word
- Test independently: `python tools/test_tts.py`
- Ensure `pywin32` is installed: `pip install pywin32`
- Try: `pip install pyttsx3==2.71` (known-stable version)

### Model not found
```powershell
python export_model.py
```

### "ModuleNotFoundError" on tools/
Tools must be run from the project root:
```powershell
cd "d:\Projects\Project Azure"
python tools/test_tts.py
```

---

## Project Structure

```
blind-cap/
│
├── main.py                  # Application entry point
├── config.py                # All configuration parameters
├── camera.py                # Webcam capture (threaded)
├── detector.py              # OpenVINO YOLO inference
├── decision_engine.py       # Hazard analysis & state machine
├── tts.py                   # Text-to-speech (priority queue + SAPI5)
├── visualization.py         # HUD rendering & bounding boxes
├── hardware.py              # System diagnostics
├── logger.py                # Logging module
├── export_model.py          # Model export tool
│
├── models/
│   └── yolo26n_openvino/    # Exported OpenVINO model
│
├── tools/
│   ├── test_tts.py          # TTS acceptance test
│   ├── benchmark.py         # CPU vs GPU benchmark
│   └── hardware_check.py    # Hardware diagnostics
│
├── tests/
│   ├── test_decision_engine.py
│   ├── test_tts_queue.py
│   └── test_detector_utils.py
│
├── logs/                    # Runtime log files
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Future: Raspberry Pi Architecture

The planned wearable system:

```
[Raspberry Pi]              [Windows PC]
Camera → JPEG stream  →  →  OpenVINO + YOLO26n
                              ↓
                           Decision Engine
                              ↓
                           Warning text/event
                              ↓
Audio ← Bluetooth  ←  ←  ←  Speech
```

The PC remains the heavy AI computer. The Pi handles camera, networking, and audio output. The current codebase is designed so that only `camera.py` needs a new `RaspberryPiStream` implementation — all other modules stay unchanged.

Future features (architecture prepared but not implemented):
- **OCR**: Read signs and text
- **Depth estimation**: Real distance measurement
- **Pothole detection**: Custom-trained model
- **Scene description**: Contextual environment summaries

---

## License

This is an experimental research project.

**Do not deploy this as a safety-critical system without proper validation, certification, and liability assessment.**
