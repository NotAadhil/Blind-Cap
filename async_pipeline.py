"""
async_pipeline.py - Lag-Free Asynchronous Decoupled Vision & Audio Pipeline
===========================================================================
Eliminates camera frame stutter and inference lag by decoupling video ingestion,
Intel OpenVINO GPU inference, audio synthesis, and display rendering into
independent high-performance worker threads.

Architecture::

    [Webcam Hardware]
           | (30-60 FPS)
    [Capture Worker Thread] ---> Latest Frame Buffer (Zero-Copy)
                                         |
                                         +---> [Inference Worker Thread] (Runs on Intel GPU)
                                         |        - YOLO26n Object Detection
                                         |        - Monocular Depth & Floor Clearance
                                         |        - Optical Flow Motion Filter
                                         |        - Scene-Aware Decision Engine
                                         |        - Voice-Only TTS Dispatch
                                         |
                                         +---> [Main GUI Render Loop] (Solid 60 FPS)
                                                  - High-speed HUD Overlay
                                                  - cv2.imshow & Keyboard Dispatch
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from audio_spatializer import AudioSpatializer
from camera import OpenCVCamera
from config import Config, DEFAULT_CONFIG, SEVERITY_CRITICAL, SEVERITY_WARNING
from decision_engine import DecisionEngine, HazardEvent
from depth_estimator import MonocularDepthEstimator
from detector import OpenVINODetector
from logger import get_logger
from motion_filter import EgoMotionFilter
from ocr_reader import OCRReader
from scene_describer import SceneDescriber
from tts import TTSEngine

logger = get_logger(__name__)


@dataclass
class PipelineFrameData:
    """Snapshot of current frame, latest AI inference results, and HUD stats."""

    frame: Optional[np.ndarray] = None
    detections: List[Dict] = field(default_factory=list)
    hazard_event: Optional[HazardEvent] = None
    depth_info: Dict = field(default_factory=dict)
    stats: Dict = field(default_factory=dict)
    timestamp: float = 0.0


class AsyncVisionPipeline:
    """
    Asynchronous vision pipeline that isolates capture, AI inference, and display.
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        detector: Optional[OpenVINODetector] = None,
        camera: Optional[OpenCVCamera] = None,
        tts: Optional[TTSEngine] = None,
    ):
        self.config = config
        self.detector = detector or OpenVINODetector(config=config)
        self.camera = camera or OpenCVCamera(
            camera_index=config.CAMERA_INDEX,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
        )
        self.tts = tts or TTSEngine(config=config)
        self.engine = DecisionEngine(config=config)
        self.spatializer = AudioSpatializer(config=config)
        self.depth_estimator = MonocularDepthEstimator(config=config)
        self.motion_filter = EgoMotionFilter(config=config)
        self.ocr_reader = OCRReader(config=config)
        self.scene_describer = SceneDescriber(config=config)

        self._running = True

        # Inter-thread synchronization slots
        self._latest_capture_frame: Optional[np.ndarray] = None
        self._capture_lock = threading.Lock()
        self._new_frame_event = threading.Event()

        self._latest_result: PipelineFrameData = PipelineFrameData()
        self._result_lock = threading.Lock()

        # Performance tracking
        self._inference_count = 0
        self._inf_fps_smooth = 0.0
        self._last_inf_time = time.perf_counter()
        self._capture_fps_smooth = 30.0
        self._last_capture_time = time.perf_counter()

        # Start inference worker thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="Inference-Worker"
        )

    def start(self) -> bool:
        """Start camera and inference threads."""
        if not self.camera.open():
            logger.error("Failed to open camera in AsyncVisionPipeline")
            return False

        # Startup speech confirmation
        if self.tts:
            self.tts.speak(
                "Blind Cap ready.",
                priority=100,
                severity=SEVERITY_WARNING,
                category="system",
                dedupe_key="startup_ready",
            )

        self._inference_thread.start()
        logger.info("AsyncVisionPipeline started successfully (Decoupled 60 FPS)")
        return True

    def _inference_loop(self) -> None:
        """
        Background worker that continuously pulls the latest frame,
        runs OpenVINO GPU inference, depth estimation, and decision analysis.
        """
        while self._running:
            # Get latest frame from camera
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Track actual camera capture FPS
            now_cap = time.perf_counter()
            cap_dt = now_cap - self._last_capture_time
            self._last_capture_time = now_cap
            cap_fps = 1.0 / cap_dt if cap_dt > 0 else 30.0
            self._capture_fps_smooth = 0.9 * self._capture_fps_smooth + 0.1 * cap_fps

            h, w = frame.shape[:2]
            now_wall = time.time()

            # 1. Optical flow ego-motion update
            if self.config.OPTICAL_FLOW_ENABLED:
                self.motion_filter.update(frame)

            # 2. OpenVINO YOLO Detection
            detections = self.detector.detect(frame)

            # 3. Monocular floor clearance & drop-off detection
            depth_info = {}
            if self.config.DEPTH_ESTIMATION_ENABLED:
                depth_info = self.depth_estimator.evaluate_corridor(frame, detections)

            # 4. Decision Engine Evaluation (Scene-aware change detection)
            hazard_event = self.engine.evaluate(
                detections,
                current_time=now_wall,
                frame_w=w,
                frame_h=h,
            )

            # 5. Audio Radar Update (no-op in silent mode)
            active_h = hazard_event.active_hazard
            self.spatializer.update_radar(active_h)

            # 6. TTS Voice Dispatch — Scene-aware descriptions only
            # If an on-demand user query (Scene Summary / OCR) is speaking,
            # protect it from interruption by routine scene alerts.
            if hazard_event.warning_text and self.tts:
                user_speaking = getattr(self.tts, "_user_query_active", False)
                if not user_speaking:
                    self.tts.speak(
                        text=hazard_event.warning_text,
                        priority=hazard_event.speak_priority,
                        severity=hazard_event.severity,
                        category=hazard_event.category,
                        dedupe_key=hazard_event.dedupe_key,
                    )

            # Performance stats calculation
            now_perf = time.perf_counter()
            dt = now_perf - self._last_inf_time
            self._last_inf_time = now_perf
            inf_fps = 1.0 / dt if dt > 0 else 0.0
            self._inf_fps_smooth = 0.9 * self._inf_fps_smooth + 0.1 * inf_fps

            tts_st = (
                self.tts.get_status()
                if self.tts
                else {"status": "OFF", "queue_size": 0, "muted": True, "quiet_mode": False}
            )

            inf_ms = self.detector.last_inference_ms
            cam_idx = getattr(self.camera, "camera_index", self.config.CAMERA_INDEX)
            stats = {
                "camera_fps": self._capture_fps_smooth,
                "camera_index": cam_idx,
                "inference_fps": self._inf_fps_smooth,
                "inference_ms": inf_ms,
                "device": self.detector.active_device,
                "model": self.config.MODEL_NAME.replace(".pt", ""),
                "detections": len(detections),
                "tts_status": tts_st["status"],
                "tts_queue": tts_st["queue_size"],
                "mode": tts_st.get("mode", "NORMAL"),
                "audio_mode": self.spatializer.audio_mode,
                "focus": hazard_event.focus_object or "",
                "severity": hazard_event.focus_severity,
                "focus_state": hazard_event.focus_state,
                "warning": hazard_event.warning_text or "",
                "clearance": depth_info.get("clearance_ratio", 1.0),
                "drop_off": depth_info.get("drop_off_detected", False),
            }

            # Update shared result snapshot
            with self._result_lock:
                self._latest_result = PipelineFrameData(
                    frame=frame,
                    detections=detections,
                    hazard_event=hazard_event,
                    depth_info=depth_info,
                    stats=stats,
                    timestamp=now_wall,
                )

            # Yield briefly to ensure video capture & rendering buffers stay smooth
            time.sleep(0.002)

    def get_latest_data(self) -> PipelineFrameData:
        """
        Instantaneous, non-blocking retrieval of latest frame and AI data
        for 60 FPS GUI rendering.
        """
        # Pull freshest camera frame
        ret, frame = self.camera.read()
        with self._result_lock:
            cur = self._latest_result
            out_frame = frame if (ret and frame is not None) else cur.frame
            return PipelineFrameData(
                frame=out_frame,
                detections=list(cur.detections),
                hazard_event=cur.hazard_event,
                depth_info=dict(cur.depth_info),
                stats=dict(cur.stats),
                timestamp=cur.timestamp,
            )

    # ------------------------------------------------------------------
    # On-Demand Accessible Commands (Hotkeys / Voice triggers)
    # ------------------------------------------------------------------
    def switch_camera(self, target_index: Optional[int] = None) -> int:
        """
        Switch or cycle to a different camera device index (e.g. 0 <-> 1).
        Announces the change via TTS and updates configuration.
        """
        from camera import list_available_cameras, OpenCVCamera

        current_idx = getattr(self.camera, "camera_index", self.config.CAMERA_INDEX)
        available = list_available_cameras(max_tested=4, active_index=current_idx)

        if target_index is not None:
            new_idx = target_index
        else:
            # Cycle to next available camera index
            if len(available) > 1 and current_idx in available:
                pos = available.index(current_idx)
                new_idx = available[(pos + 1) % len(available)]
            else:
                # If only 1 index was detected initially, try toggling 0 <-> 1 (e.g. plugged in USB cam)
                new_idx = 1 if current_idx == 0 else 0

        logger.info("Switching camera to index %d (available: %s)", new_idx, available)
        print(f"\n  [CAMERA SWITCH] Switching to Camera Index {new_idx} (Detected: {available})...\n")

        success = False
        if hasattr(self.camera, "switch_camera"):
            success = self.camera.switch_camera(new_idx)
        else:
            self.camera.release()
            self.camera = OpenCVCamera(
                camera_index=new_idx,
                width=self.config.CAMERA_WIDTH,
                height=self.config.CAMERA_HEIGHT,
            )
            success = self.camera.open()

        if success:
            self.config.CAMERA_INDEX = new_idx
            msg = f"Camera switched to input {new_idx}."
            print(f"  [CAMERA SWITCH RESULT] -> {msg}\n")
            if self.tts:
                self.tts.speak(
                    text=msg,
                    priority=95,
                    severity=SEVERITY_WARNING,
                    category="camera",
                    dedupe_key=f"cam_switch_{time.time()}",
                    force=True,
                )
            return new_idx
        else:
            msg = f"Failed to switch to Camera {new_idx}."
            print(f"  [CAMERA SWITCH ERROR] -> {msg}\n")
            if self.tts:
                self.tts.speak(
                    text=msg,
                    priority=95,
                    severity=SEVERITY_WARNING,
                    category="camera",
                    dedupe_key=f"cam_switch_err_{time.time()}",
                    force=True,
                )
            return current_idx

    def trigger_ocr(self) -> None:
        """Read visible text and signs in front of the camera on demand."""
        with self._result_lock:
            frame = self._latest_result.frame

        if frame is None or not self.tts:
            return

        logger.info("On-demand OCR triggered")
        print("\n  [ON-DEMAND OCR] Scanning frame for text...")
        res = self.ocr_reader.extract_text(frame)
        print(f"  [ON-DEMAND OCR RESULT] -> {res['text']}\n")

        # Speak out loud with forced priority (interrupts anything playing)
        self.tts.speak(
            text=res["text"],
            priority=100,
            severity=SEVERITY_CRITICAL,
            category="ocr",
            dedupe_key=f"ocr_{time.time()}",
            force=True,
        )

    def trigger_scene_summary(self) -> None:
        """Speak a comprehensive environmental scene summary on demand."""
        with self._result_lock:
            data = self._latest_result
            frame = data.frame
            dets = data.detections
            depth = data.depth_info

        if not self.tts:
            return

        logger.info("On-demand Scene Summary triggered")
        print("\n  [ON-DEMAND SCENE SUMMARY] Analyzing scene layout...")
        full_summary = self.scene_describer.describe_scene(frame, dets, depth)
        print(f"  [SCENE SUMMARY RESULT] -> {full_summary}\n")

        # Speak out loud with forced priority (interrupts anything playing)
        self.tts.speak(
            text=full_summary,
            priority=100,
            severity=SEVERITY_CRITICAL,
            category="summary",
            dedupe_key=f"summary_{time.time()}",
            force=True,
        )

    def toggle_audio_mode(self) -> str:
        """Cycle audio mode: HYBRID -> RADAR -> SPEECH -> HYBRID."""
        new_mode = self.spatializer.toggle_mode()
        if self.tts:
            self.tts.speak(
                text=f"Audio mode: {new_mode}.",
                priority=90,
                severity=SEVERITY_WARNING,
                category="mode",
                dedupe_key=f"mode_{time.time()}",
            )
        return new_mode

    def shutdown(self) -> None:
        """Gracefully stop all pipeline threads."""
        logger.info("Stopping AsyncVisionPipeline...")
        self._running = False
        self.spatializer.shutdown()
        if self.tts:
            self.tts.shutdown()
        self.camera.release()
        logger.info("AsyncVisionPipeline stopped")
