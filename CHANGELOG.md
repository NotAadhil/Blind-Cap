# Blind Cap Mobile App - Release History & Changelog

All notable changes, bug fixes, performance improvements, and architectural evolutions for the Blind Cap Mobile Android application are documented here.

---

## [v1.0.21] - 2026-08-23

### Summary
Comprehensive AI Pipeline Acceleration, Global Camera Motion Compensation, False-Positive Temporal Gate, and Anti-Stacking OCR Debouncer.

### Bugs Identified & Addressed
- **AI Frame Rate Drops (< 10 FPS)**:
  - Per-frame allocation in `TensorImage.load()` and `ImageProcessor.process()` created severe GC churn.
  - Redundant 300-slot iteration and class smoothing on low-scoring candidate slots wasted CPU cycles.
- **OCR Button Request Stacking**:
  - Rapid presses spawned multiple concurrent coroutines, stacking multiple `"No readable text found"` speech events in the TTS queue.
- **False-Positive Single-Frame Detections**:
  - Single-frame candidate detections at `>= 0.45` were immediately promoted to confirmed status and announced.
- **Repeated Person Announcements on Camera Movement**:
  - Camera panning caused bounding boxes to shift faster than standard IoU tracking could bridge, dropping tracks and creating new IDs upon camera stabilization.

### Fixes & Architectural Enhancements
- **Zero-Allocation AI Inference Pipeline (`TfliteYoloDetector.kt`)**:
  - Reused `TensorImage` and pre-allocated output tensors eliminate per-frame heap allocations.
  - Score gate moved before class smoothing, eliminating 90%+ redundant iterations.
  - Bounded class history cache with periodic pruning prevents memory creep.
- **Global Scene Motion Compensation (`DecisionEngine.kt`)**:
  - Multi-factor spatial association combines IoU (35%), normalized center proximity (30%), normalized area scale ratio (15%), and class matching (20%).
  - Person class center tolerance expanded to `0.55` so camera pans do not break track identity.
  - Global camera motion compensation detects uniform scene shifts and extends the coasting grace window to 30 frames (~2.2s).
- **Two-Tier Temporal Validation Gate**:
  - Requires 2+ consecutive frames of consistent detection OR high confidence (`>= 0.55`) before promoting any object to `CONFIRMED`.
  - Single-frame transient hallucinations (0.32-0.54) are held in `CANDIDATE` state and never trigger speech.
- **Anti-Stacking Concurrency & Debounce Guards (`MainActivity.kt`, `TtsManager.kt`)**:
  - `AtomicBoolean` guard prevents concurrent OCR coroutines from running simultaneously.
  - 1.5-second debounce window rejects rapid repetitive button presses.
  - Identical OCR speech requests within 2.5s are automatically filtered out.

---

## [v1.0.19] - 2026-08-22

### Summary
Fixed OCR Audio Speech Playback with Active Utterance Lifecycle Tracking and Long Text Chunking.

### Bugs Identified & Addressed
- **OCR Text Detected but Silent**: When OCR finished recognizing text, Android's `TextToSpeech` engine did not speak the recognized text aloud even though the Toast message displayed correctly.
- **Root Cause**:
  1. Triggering `tts.stop()` on the initial `"Reading text..."` prompt caused Android's TTS engine to asynchronously fire `onDone`/`onError`, resetting the active speech state and dropping the subsequent OCR text utterance.
  2. Long OCR paragraphs without chunking could fail silently in some Android TTS engines.

### Fixes & Architectural Enhancements
- **Active Utterance ID Lifecycle Matching**:
  - `UtteranceProgressListener` now tracks `activeUtteranceId` so asynchronous completion callbacks from previously stopped or aborted prompts cannot cancel or silence newly started OCR readings.
- **Automatic Text Chunking**:
  - Long OCR paragraphs over 500 characters are automatically split into natural speech chunks with sequential `QUEUE_ADD` chaining, ensuring every word is spoken clearly without truncation.
- **Status Verification & Fallback**:
  - Added return code status checks on `tts.speak()` with automatic fallback.

---

## [v1.0.18] - 2026-08-22

### Summary
OCR Task Priority Protection & Persistent Person Tracking Across Camera Movements.

### Bugs Identified & Addressed
- **OCR Interruption by Background Detections**: When reading text aloud via OCR, ordinary background detections (e.g., a person standing in the room) would immediately interrupt and kill the OCR speech.
- **Repeated Person Announcements on Camera Movement**: Slight panning or movement of the camera caused a tracked person's bounding box or region to shift slightly, triggering repetitive `"Person detected"` announcements.

### Fixes & Architectural Enhancements
- **OCR Task Priority Protection (`startOcrReading`)**:
  - Sets OCR reading task priority to `85`.
  - Background object detections (`priority <= 70`) are **completely suppressed** while OCR is actively reading.
  - Only genuine critical collision hazards (`priority >= 90`, e.g. obstacle `<= 0.9m` in walking path) can preempt OCR.
  - No background announcements accumulate behind OCR.
- **Track-Level Identity Memory (`isAnnounced`)**:
  - Each tracked physical object maintains an announcement memory flag.
  - Once a person is confirmed and announced, camera movement, panning, bounding-box shifts, distance changes, or confidence fluctuations will **never re-trigger announcements**.
  - New announcements are strictly reserved for genuinely new tracks (e.g. a second person entering) or danger corridor entries.
- **Region Hysteresis Deadband**:
  - Added a 6% frame width deadband to region boundaries in `DepthEstimator.kt`, preventing region flapping when panning.
- **Extended Coasting Grace Period**:
  - Increased track coasting tolerance to 20 frames (~1.5s) to seamlessly survive temporary detection dropouts or camera blurs.

---

## [v1.0.17] - 2026-08-22

### Summary
Multi-Object Spatial Grouping, Counting, 30% Baseline Confidence Filter, and Scene Stabilization Window.

### Bugs Identified & Addressed
- **Detector Flickering**: Borderline detections appearing and disappearing for 1 frame caused the system to think an object left and reappeared as a brand new object, repeatedly triggering TTS.
- **Single Object Monopoly**: When multiple objects were visible simultaneously (e.g., person, chair, backpack), the system would often speak only one object and ignore the rest.
- **Uncounted Repetition**: When multiple objects of the same class were in view (e.g., 3 chairs or 2 people), the system did not count them intelligently.

### Fixes & Architectural Enhancements
- **30% Baseline Confidence Filter with Dual-Threshold Hysteresis**:
  - Raw detections below 30% are filtered out.
  - New objects require `>= 30%` confidence and 2 consecutive frames to be confirmed (`CANDIDATE -> CONFIRMED`).
  - Already-tracked confirmed objects stay alive even if confidence temporarily dips to 20% during camera movement or motion blur.
- **Coasting Grace Period**:
  - Temporarily occluded or missed objects remain in a `COASTING` state for up to 16 frames (~1.1s) before expiring, preventing detector flicker from creating/destroying tracks.
- **Natural Spatial Grouping & Counting Engine**:
  - Generates natural, human-like summaries:
    - Single object: `"Person on the left."`, `"Chair in the center."`
    - Multiple instances of the same class: `"Two people detected: one on the left and one on the right."`, `"Three chairs in the center."`
    - Mixed multi-object scenes: `"Two people detected: one on the left and one on the right. Chair in the center. Backpack on the right."`
- **Scene Stabilization Window**:
  - Requires 2 stable frames before generating a speech snapshot, preventing partial or fragmented sentences.

---

## [v1.0.16] - 2026-08-22

### Summary
Complete Pipeline Overhaul: Decoupled Camera & AI Worker Threads, Zero-Repeat Scale-Invariant Tracking, Stale-Proof Preemptive TTS, and Fine-Grained FPS Telemetry.

### Bugs Identified & Addressed
- **Low Frame Rates (~10 FPS Camera, ~8 FPS AI)**: CameraX was running inference synchronously on `cameraExecutor`, blocking the camera capture thread and starving the hardware buffer pool.
- **Repeated Speech on Walking Closer**: Walking toward an object (e.g., a bed) caused its bounding box to expand, causing previous IoU tracking to fail and re-triggering `"Bed detected"` continuously.
- **Outdated / Stale TTS Queue**: Moving the camera from a bed to a chair resulted in the TTS still finishing a queue of 4 old `"Bed"` messages before mentioning the chair.

### Fixes & Architectural Enhancements
- **Fully Decoupled Camera & AI Pipeline**:
  - Camera producer extracts frames and closes `ImageProxy` in `< 1ms`, unlocking smooth **30+ FPS** camera capture.
  - Background AI worker thread polls the newest available frame via lock-free atomic pointer (`AtomicReference<Bitmap>`), eliminating frame queues.
- **Scale-Invariant Spatial Tracker**:
  - Multi-factor tracking (`IoU + Center Distance + Class Match`) correctly matches expanding bounding boxes as the same physical object when moving closer.
  - Objects progress through states (`NEW -> CONFIRMED -> PERSISTENT`), ensuring zero repeated announcements while looking at the same scene.
- **Stale-Proof Interruptible TTS Engine**:
  - Added 2.5-second Time-To-Live (TTL) on speech messages. Stale messages are dropped automatically.
  - Queue capacity restricted to 1 pending message with instant preemption for urgent obstacles.
- **Diagnostic Performance HUD**:
  - Added real-time telemetry displaying Camera FPS, AI FPS, and latency breakdown (`Preprocess ms`, `Inference ms`, `Postprocess ms`).

---

## [v1.0.15] - 2026-08-22

### Summary
External ESP32-CAM MJPEG Stream Integration and Video Input Source Switcher.

### Features Added
- **Top Bar Video Source Switcher**:
  - `[Source: Phone Cam]` / `[Source: ESP32-CAM]` toggle button in the top bar.
  - Automatically unbinds CameraX when using ESP32-CAM to save phone battery.
- **Configurable Stream URL**:
  - `[Stream IP]` settings dialog with persistent `SharedPreferences` storage (default: `http://192.168.4.1:81/stream`).
- **High-Speed Non-Blocking MJPEG Stream Reader**:
  - Added `MjpegStreamReader.kt` with automatic SOI/EOI JPEG byte parsing, auto-reconnect, and keep-only-latest frame buffering.
  - Enabled `usesCleartextTraffic="true"` and network permissions in `AndroidManifest.xml`.
- **Unified AI Pipeline**:
  - All YOLO26n detections, spatial tracking, danger zones, and OCR work identically on both Phone Camera and ESP32-CAM feeds.

---

## [v1.0.14] - 2026-08-22

### Summary
Native SIMD Preprocessing Acceleration, 2-Zone Alert Hierarchy, Standalone Bundled OCR, and Hardware Volume Key Interception.

### Fixes & Enhancements
- **Eliminated 7 FPS GC Bottleneck**:
  - Replaced JVM pixel iteration loops with **TensorFlow Lite Support `TensorImage` & `ImageProcessor`** (`ResizeOp` + `NormalizeOp`), executing in optimized native C++ SIMD (< 1.5 ms).
- **2-Zone Walking Corridor Alert Hierarchy**:
  - Danger Zone Tier 1 (`<= 1.8m` Center): `"Stop! [Object] is obstructing your path, [X] meters away."`
  - Danger Zone Tier 2 (`<= 0.9m` Center): `"Stop! [Object] is very close, [X] meters away. Please stop!"`
- **100% Offline Bundled OCR**:
  - Integrated `com.google.mlkit:text-recognition:16.0.0` with bundled native library (`libmlkit_google_ocr_pipeline.so`), removing all Play Services download dependencies.
- **Hardware Volume Key Override**:
  - Overrode `dispatchKeyEvent` in `MainActivity` to map Volume Up to OCR Text Reading and Volume Down to Full Scene Summary without triggering OS volume sliders.

---

## [v1.0.13] - 2026-08-22

### Summary
Native 320x320 Direct Buffer Pipeline and GPU Delegate Acceleration.

### Fixes & Enhancements
- Converted input graph from 640x640 to 320x320 FP16 for real-time mobile execution.
- Configured direct native byte buffer transfers and TFLite GPU Delegate with 4-thread XNNPACK CPU fallback.

---

## [v1.0.12] - 2026-08-22

### Summary
YOLO26n TFLite End-to-End Decoder Verification.

### Fixes & Enhancements
- Verified model architecture input shape `[1, 320, 320, 3]` and output tensor `[1, 300, 6]`.
- Verified official 80 COCO class labels and metric depth estimation mappings.

---

## [v1.0.11] - 2026-08-22

### Summary
Official Ultralytics YOLO26n TFLite Integration with 80 COCO Classes.

### Fixes & Enhancements
- Replaced fallback classifiers with true Ultralytics YOLO26n TFLite neural network.
- Bundled verified `labels.txt` containing all 80 standard COCO categories.

---

## [v1.0.10] - 2026-08-22

### Summary
Direct FloatBuffer Output Memory Management and Pixel Tensor G1 Tuning.

### Fixes & Enhancements
- Resolved buffer copy latency on Google Pixel 6a (Google Tensor G1 chip).
- Tuned multi-threading options for Cortex-X1 big cores.

---

## [v1.0.9] - 2026-08-22

### Summary
CameraX Rotation Matrix Fix and Native FloatBuffer Alignment.

### Fixes & Enhancements
- Fixed orientation issue where CameraX sensor rotation caused horizontal bounding box shifts.
- Corrected native FloatBuffer memory order to `ByteOrder.nativeOrder()`.

---

## [v1.0.8] - 2026-08-22

### Summary
Initial Android Mobile Application Release.

### Features
- Native Android app with CameraX viewfinder, HUD overlay, ML Kit OCR, and local TTS.
- Automated GitHub Actions build workflow for generating `app-debug.apk`.