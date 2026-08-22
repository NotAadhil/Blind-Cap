# Blind Cap Android Mobile Application

This is the standalone Android application for Blind Cap, providing real-time on-device assistive vision, spatial walking corridor tracking, Danger Zone Stop warnings, on-demand OCR, and offline speech synthesis.

## Features
- 100% On-Device Processing: Zero cloud dependencies, zero data usage.
- Hardware Acceleration: TFLite with GPU/NNAPI Delegate support.
- Spatial Walking Corridor: Left (0-35%), Center Walking Path (35-65%), Right (65-100%).
- 2-Tier Danger Zone Alerts: Urgent warnings at <= 1.8m and <= 0.9m in walking path.
- Dynamic Relative Positioning: Natural spoken conjunctions ("Person on the right and cell phone on the left").
- Spoken Baseline Diffing: Instant updates when new objects appear, stationary silence when unchanged.
- On-Demand OCR Text Reader: Triggered by pressing the physical Volume Up button.
- On-Demand Scene Summary: Triggered by pressing the physical Volume Down button.
- Touch Gestures: Single tap to repeat last warning, double tap to toggle mute.

## How to Build and Run
1. Open Android Studio.
2. Select "Open Project" and choose this `android_app` folder.
3. Ensure `yolo.tflite` or `yolo.onnx` is present in `app/src/main/assets/`.
4. Connect your Android phone via USB with USB Debugging enabled.
5. Click Run (Shift + F10) to install and launch directly on your phone.
