"""
main.py - Blind Cap Assistive Vision & Audio System (Async 60 FPS)
===================================================================
Coordinates the real-time decoupled assistive vision & audio pipeline::

    Async Pipeline -> Intel Iris Xe YOLO26n -> Depth & Motion -> 3D Spatial Audio -> HUD

Usage::

    python main.py [--device GPU|CPU|AUTO] [--camera 0] [--mode HYBRID|RADAR|SPEECH]

Accessible Hotkeys:
    T           Read Text / Signs (On-Demand OCR)
    Space / C   Speak Full Scene Summary
    A           Toggle Audio Mode (HYBRID -> RADAR -> SPEECH)
    M           Mute / Unmute
    S           Toggle Quiet Mode
    R           Repeat Last Warning
    Q / ESC     Quit Application

IMPORTANT: This is an experimental research prototype, NOT a certified
mobility aid.  "No detection" does NOT mean "path is safe."
"""

import argparse
import sys
import time
from typing import Any

import cv2

from async_pipeline import AsyncVisionPipeline
from config import Config
from detector import OpenVINODetector
from hardware import run_diagnostics
from logger import get_logger
from tts import TTSEngine
from visualization import Visualizer

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind Cap: Assistive Vision Prototype (YOLO26n + OpenVINO + 3D Audio)"
    )
    parser.add_argument(
        "--device", choices=["GPU", "CPU", "AUTO"], default="GPU",
        help="Inference device (default: GPU for Intel Iris Xe)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Webcam device index",
    )
    parser.add_argument(
        "--mode", choices=["HYBRID", "RADAR", "SPEECH"], default="HYBRID",
        help="Audio feedback mode (default: HYBRID)",
    )
    parser.add_argument(
        "--model", choices=["yolo26m", "yolo26s", "yolo26n"], default="yolo26m",
        help="YOLO model variant (default: yolo26m for highest accuracy)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--mute", action="store_true", help="Start with speech muted")
    args = parser.parse_args()

    # --- Initialise config ---
    from pathlib import Path
    config = Config()
    if args.model == "yolo26m":
        config.MODEL_NAME = "yolo26m.pt"
        config.OPENVINO_MODEL_DIR = Path("models/yolo26m_480_openvino")
        config.INFERENCE_SIZE = 480
    elif args.model == "yolo26s":
        config.MODEL_NAME = "yolo26s.pt"
        config.OPENVINO_MODEL_DIR = Path("models/yolo26s_openvino")
        config.INFERENCE_SIZE = 640
    elif args.model == "yolo26n":
        config.MODEL_NAME = "yolo26n.pt"
        config.OPENVINO_MODEL_DIR = Path("models/yolo26n_openvino")
        config.INFERENCE_SIZE = 640

    config.OPENVINO_DEVICE = args.device
    config.CAMERA_INDEX = args.camera
    config.DEBUG_MODE = args.debug
    config.AUDIO_MODE = args.mode
    if args.mute:
        config.TTS_ENABLED = False

    logger.info("Starting Blind Cap Assistive Vision System...")

    # --- 1. Hardware diagnostics ---
    run_diagnostics(verbose=True)

    # --- 2. Initialise OpenVINO Detector ---
    try:
        detector = OpenVINODetector(config=config, device=config.OPENVINO_DEVICE)
    except Exception as exc:
        logger.error("Detector init failed: %s", exc)
        print("\n[ERROR] Could not initialise OpenVINO detector. Ensure model is exported: python export_model.py\n")
        sys.exit(1)

    # --- 3. Initialise TTS & Async Vision Pipeline ---
    tts = TTSEngine(config=config)
    if args.mute:
        tts.set_muted(True)

    pipeline = AsyncVisionPipeline(
        config=config,
        detector=detector,
        tts=tts,
    )

    if not pipeline.start():
        print("\n[ERROR] Could not start video camera stream.\n")
        sys.exit(1)

    visualizer = Visualizer(config=config)
    from camera import list_available_cameras
    detected_cameras = list_available_cameras(max_tested=4, active_index=config.CAMERA_INDEX)

    # --- Startup banner ---
    print("\n" + "=" * 60)
    print("       BLIND CAP - ACCESSIBLE VISION & 3D AUDIO SYSTEM")
    print("=" * 60)
    print(f"  Model:        {config.MODEL_NAME}")
    print(f"  Device:       {detector.active_device} ({detector.device_full_name})")
    print(f"  Audio Mode:   {config.AUDIO_MODE} (3D Stereo Earcons + Voice)")
    print(f"  Camera:       Index {config.CAMERA_INDEX} (Detected cameras: {detected_cameras})")
    print("-" * 60)
    print("  Accessible Hotkeys:")
    print("    T           Read Visible Text / Signs (OCR)")
    print("    Space / C   Speak Full Scene Summary")
    print("    V / Tab     Switch Camera Input (0 / 1 / 2...)")
    print("    A           Toggle Audio Mode (HYBRID/RADAR/SPEECH)")
    print("    M           Mute / Unmute Speech")
    print("    S           Quiet Mode (Urgent alerts only)")
    print("    R           Repeat Last Important Warning")
    print("    Q / ESC     Quit")
    print("=" * 60 + "\n")

    # GUI Window setup
    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_TITLE, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    try:
        cv2.setWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    quit_requested = False

    def on_mouse_click(event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse clicks on interactive GUI buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            btn = visualizer.get_clicked_button(x, y, frame_w=config.CAMERA_WIDTH)
            if btn == "switch_camera":
                pipeline.switch_camera()
            elif btn == "scene_summary":
                pipeline.trigger_scene_summary()
            elif btn == "ocr":
                pipeline.trigger_ocr()
            elif btn == "audio_mode":
                pipeline.toggle_audio_mode()

    cv2.setMouseCallback(config.WINDOW_TITLE, on_mouse_click)

    def handle_key(k: int) -> None:
        nonlocal quit_requested
        if k in (27, ord("q"), ord("Q")):
            quit_requested = True
        elif k in (ord("t"), ord("T")):
            pipeline.trigger_ocr()
        elif k in (32, ord("c"), ord("C")):  # Space or C
            pipeline.trigger_scene_summary()
        elif k in (ord("v"), ord("V"), 9):  # V or Tab
            pipeline.switch_camera()
        elif k in (ord("a"), ord("A")):
            new_mode = pipeline.toggle_audio_mode()
            print(f"  [AUDIO MODE] -> {new_mode}")
        elif k in (ord("m"), ord("M")):
            cur_st = tts.get_status()
            tts.set_muted(not cur_st["muted"])
        elif k in (ord("s"), ord("S")):
            cur_st = tts.get_status()
            tts.set_quiet_mode(not cur_st["quiet_mode"])
        elif k in (ord("r"), ord("R")):
            tts.repeat_last()
        elif k in (ord("d"), ord("D")):
            config.DEBUG_MODE = not config.DEBUG_MODE
            logger.info("Debug mode: %s", config.DEBUG_MODE)
        elif k in (ord("b"), ord("B")):
            config.BENCHMARK_OVERLAY = not config.BENCHMARK_OVERLAY

    # --- High-Speed Main GUI Render Loop (Smooth 60 FPS) ---
    try:
        while not quit_requested:
            # Non-blocking fetch of latest frame & AI data
            data = pipeline.get_latest_data()
            frame = data.frame

            if frame is None:
                time.sleep(0.005)
                key = cv2.waitKey(1) & 0xFF
                handle_key(key)
                continue

            # Use stats directly from inference thread (includes accurate camera_fps)
            stats = dict(data.stats)

            # Render composite overlay
            rendered = visualizer.render(
                frame,
                data.detections,
                data.hazard_event,
                stats,
            )

            cv2.imshow(config.WINDOW_TITLE, rendered)

            # Fast keyboard event processing (1 ms)
            key = cv2.waitKey(1) & 0xFF
            handle_key(key)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Shutting down Blind Cap...")
        pipeline.shutdown()
        cv2.destroyAllWindows()
        logger.info("Blind Cap stopped.")


if __name__ == "__main__":
    main()
