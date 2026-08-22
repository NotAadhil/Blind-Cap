# Blind Cap: Real-Time Assistive Vision and Spatial Audio Prototype

## Safety Disclaimer
This project is an experimental research prototype, NOT a certified medical or mobility device. It cannot guarantee detection of every obstacle. Absence of a detection does not mean that the path is clear or safe. Never rely on this software as your primary or sole method of navigation. Always use standard mobility aids, canes, guide dogs, and certified orientation techniques.

---

## Overview
Blind Cap is an intelligent, real-time assistive vision system designed to help visually impaired individuals perceive their surroundings and navigate obstacles safely.

The software captures video from a camera mounted on a wearable cap or webcam, detects objects across 80 categories, tracks their physical positions and metric distances, and provides concise, event-driven voice feedback using text-to-speech.

### Key Features
- High-Accuracy Object Detection: Powered by YOLO26m (Medium FP16) accelerated with Intel OpenVINO on PC, and YOLO26n TFLite on Android.
- Dynamic Multi-Object Scene Evaluation: Does not latch onto single objects. Continuously evaluates all active obstacles, counts instances, and describes them with natural relative positions (e.g. "Two people detected: one on the left and one on the right. Chair in the center").
- Spatial Walking Corridor and Danger Zones: Divides the field of view into Left, Center (Walking Path), and Right zones. Immediately announces high-priority urgent stop warnings when obstacles enter the close walking corridor (1.8 meters and 0.9 meters).
- Metric Distance Estimation: Uses object reference heights and camera focal models to estimate real-world distances in meters.
- Event-Driven Voice Output: Announces new objects, departures, region shifts, and significant distance changes, but remains completely silent when the scene is stationary to prevent audio clutter.
- On-Demand OCR Text Reader: Reads printed signs, notices, packaging, labels, or computer screens aloud on demand.
- On-Demand Scene Summary: Generates a complete comprehensive description of the entire visual scene on demand.
- Decoupled Viewfinder Pipeline: Runs camera capture and user interface rendering asynchronously from the AI worker thread for smooth real-time performance with zero lag.
- Android Mobile App & ESP32-CAM Streaming: Includes a full standalone Android APK supporting both the phone camera and external wireless ESP32-CAM MJPEG streams over WiFi.

---

## Android Mobile Application & Releases

Download the latest pre-compiled Android APK directly from GitHub Releases:
- [**Download Latest Android APK (v1.0.18)**](https://github.com/NotAadhil/Blind-Cap/releases/download/v1.0.18/app-debug.apk)
- [**View Full Release History & Changelog (CHANGELOG.md)**](file:///D:/Projects/Project%20Azure/CHANGELOG.md)

---

## Hardware and System Requirements
- Operating System: Windows 10 or Windows 11 (64-bit).
- Python: Python 3.10, 3.11, or 3.12.
- Camera: Built-in webcam, external USB camera, or network video stream.
- Accelerator (Recommended): Intel Core processor with Intel Iris Xe Graphics, Intel Arc GPU, or modern multi-core CPU.

---

## 1-Click Quick Start (Windows)

### Option 1: Double-Click Batch File
1. Clone or download the repository:
   ```bash
   git clone https://github.com/NotAadhil/Blind-Cap.git
   cd Blind-Cap
   ```
2. Double-click the file `run.bat`.
3. The script will automatically create a Python virtual environment (.venv), install all dependencies, export the YOLO26m model for OpenVINO acceleration if needed, and open the live camera interface.

### Option 2: PowerShell
Run the following commands in PowerShell:
```powershell
git clone https://github.com/NotAadhil/Blind-Cap.git
cd Blind-Cap
.\run.ps1
```

---

## Manual Installation and Usage

### 1. Set Up Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Export Model (First Run Only)
Export the YOLO26m model to OpenVINO FP16 format for hardware acceleration:
```powershell
python export_model.py
```

### 3. Run the Application
```powershell
python main.py
```

Optional command-line arguments:
- `--model`: Select model variant (default: `yolo26m`, options: `yolo26m`, `yolo26s`, `yolo26n`).
- `--device`: Select OpenVINO device (default: `GPU`, options: `GPU`, `CPU`, `AUTO`).
- `--camera`: Specify camera index (default: `0`).
- `--mute`: Start with voice output muted.
- `--debug`: Enable verbose diagnostic logging.

---

## Interactive Hotkeys and Controls

| Key | Action | Description |
| :--- | :--- | :--- |
| `T` | Read Text (OCR) | Captures visible text or signs and reads them aloud. |
| `Space` or `C` | Full Scene Summary | Speaks a complete descriptive summary of the entire visual scene. |
| `V` or `Tab` | Switch Camera | Cycles to the next available camera input (index 0, 1, 2, etc.). |
| `M` | Mute / Unmute | Toggles all voice announcements on or off. |
| `S` | Quiet Mode | Silences routine informational announcements, keeping only urgent safety warnings. |
| `R` | Repeat Last Warning | Repeats the most recent important safety announcement. |
| `D` | Debug Overlay | Toggles detailed detection logs in the console. |
| `B` | Benchmark Overlay | Toggles real-time inference latency and FPS benchmarks on the HUD. |
| `Q` or `ESC` | Quit | Gracefully releases all hardware resources and closes the application. |

---

## System Architecture

```
Camera Input (Webcam / USB / Wireless)
            |
            v
[OpenCVCamera Thread] (Non-blocking capture buffer)
            |
            v
[AsyncVisionPipeline] (Decoupled background worker)
  |-- OpenVINODetector (YOLO26m FP16 on Intel GPU/CPU)
  |-- DepthEstimator (Monocular metric distance estimation)
  |-- MotionFilter (Ego-motion compensation)
  |-- DecisionEngine (Spatial tracking and change detection)
  |-- TTSEngine (Windows SAPI5 voice output)
  |-- OCRReader (Windows Native OCR text extraction)
            |
            v
[Visualizer] (High-speed 60 FPS HUD overlay and GUI display)
```

---

## Verification and Testing
The project includes a comprehensive automated test suite covering all modules:
```powershell
python -m pytest tests/ -v
```
All 87 unit and integration tests verify:
- Spatial tracking and region classification.
- Dynamic multi-object grouping and relative spatial formatting.
- Multi-tier danger zone escalation and hysteresis stability.
- Thread-safe priority voice queuing and non-blocking interruption.
- Hardware diagnostics and sensor fallback.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
