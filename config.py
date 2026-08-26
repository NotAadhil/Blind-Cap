"""
config.py - Blind Cap Configuration Module
===========================================
Central configuration for the Blind Cap assistive vision prototype.
All tuneable parameters live here so beginners can adjust behaviour
without hunting through multiple source files.

IMPORTANT: This is an experimental research prototype, NOT a certified
mobility aid.  "No detection" does NOT mean "path is safe."
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# TTS speech-mode constants
# ---------------------------------------------------------------------------
TTS_MODE_NORMAL = "NORMAL"      # Full spoken phrases:  "Person ahead."
TTS_MODE_MINIMAL = "MINIMAL"    # Short spoken words:   "Person."
TTS_MODE_URGENT = "URGENT"      # Urgent prefix:        "STOP. Person ahead."

# ---------------------------------------------------------------------------
# Audio output modes (Spatial Earcons vs Voice)
# ---------------------------------------------------------------------------
AUDIO_MODE_HYBRID = "HYBRID"    # Spatial 3D clicks for proximity + voice for critical hazard names
AUDIO_MODE_RADAR = "RADAR"      # Non-verbal 3D stereo radar beeps/clicks only (zero voice spam)
AUDIO_MODE_SPEECH = "SPEECH"    # Traditional speech-only mode

# ---------------------------------------------------------------------------
# Severity level constants (used by decision engine + TTS)
# ---------------------------------------------------------------------------
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_WARNING = "WARNING"
SEVERITY_CAUTION = "CAUTION"
SEVERITY_INFO = "INFORMATION"


@dataclass
class Config:
    """All tuneable runtime parameters for Blind Cap."""

    # -----------------------------------------------------------------------
    # 1. Model & OpenVINO Inference
    # -----------------------------------------------------------------------

    # Default model: YOLO26m (Medium variant - high accuracy, optimized 480x480 FP16)
    MODEL_NAME: str = "yolo26m.pt"

    # Fallback if the primary .pt file is unavailable upstream
    FALLBACK_MODEL_NAME: str = "yolov8m.pt"

    # Directory containing the exported OpenVINO IR files (.xml / .bin)
    OPENVINO_MODEL_DIR: Path = field(
        default_factory=lambda: Path("models/yolo26m_480_openvino")
    )

    # Preferred inference device.  "GPU" targets Intel Iris Xe iGPU.
    # Supported: "GPU", "CPU", "AUTO"
    OPENVINO_DEVICE: str = "GPU"

    # Square input dimension for YOLO inference (pixels)
    INFERENCE_SIZE: int = 480

    # Detections with confidence below this threshold are discarded.
    # Set to 0.28 for rich multi-object detection (cell phones, bottles, chairs, people)
    CONFIDENCE_THRESHOLD: float = 0.28

    # Non-Maximum Suppression (NMS) IoU threshold for deduplicating overlapping boxes.
    NMS_THRESHOLD: float = 0.45
    IOU_THRESHOLD: float = 0.45

    # -----------------------------------------------------------------------
    # 2. Camera / Video Capture
    # -----------------------------------------------------------------------

    # Webcam device index (0 = built-in / primary USB camera)
    CAMERA_INDEX: int = 0

    # Requested capture resolution (actual may differ per camera hardware)
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480

    # Target frames-per-second for the main display loop
    TARGET_FPS: int = 30

    # -----------------------------------------------------------------------
    # 3. Walking Path Corridor
    # -----------------------------------------------------------------------
    # The frame is split into three vertical zones:
    #   LEFT   [0 .. PATH_LEFT_RATIO]
    #   CENTER [PATH_LEFT_RATIO .. PATH_RIGHT_RATIO]  <- walking path
    #   RIGHT  [PATH_RIGHT_RATIO .. 1.0]
    # Values are fractions of total image width.
    # Narrowed center corridor (35% - 65%) for tight walking path focus
    PATH_LEFT_RATIO: float = 0.35
    PATH_RIGHT_RATIO: float = 0.65

    # -----------------------------------------------------------------------
    # 4. Hazard Detection & Scoring
    # -----------------------------------------------------------------------

    # Minimum bounding-box area (fraction of total frame) for visual detection tracking.
    # 0.002 allows small objects (cell phones, cups, bottles, remotes) to be tracked.
    MIN_HAZARD_AREA_RATIO: float = 0.002

    # Distance estimation thresholds (meters)
    # Critical Zone Tier 1: Initial Danger Zone entry (<= 1.8m) triggers Critical Alert 1 ("Stop! ...")
    DANGER_ZONE_DISTANCE_M: float = 1.8
    CRITICAL_ZONE_1_DISTANCE_M: float = 1.8

    # Critical Zone Tier 2: Immediate Danger zone (<= 0.9m) triggers Critical Alert 2 ("Stop! ... very close!")
    CRITICAL_DISTANCE_M: float = 0.9
    CRITICAL_ZONE_2_DISTANCE_M: float = 0.9

    # In-Frame Advisory Zone: Objects within IN_FRAME_MAX_DISTANCE_M (3.5m) trigger in-frame awareness
    IN_FRAME_MAX_DISTANCE_M: float = 3.5

    # Hysteresis buffer in meters to prevent rapid flapping across zone boundaries
    ZONE_HYSTERESIS_M: float = 0.3

    # Minimum distance delta in meters to trigger an updated distance voice announcement
    DISTANCE_UPDATE_THRESHOLD_M: float = 0.8

    CAMERA_VFOV_DEG: float = 48.0

    # Minimum bounding-box area to trigger an audio SPEECH warning in the walking corridor.
    CLOSE_HAZARD_AREA_RATIO: float = 0.08

    # Minimum area ratio for side (left/right) obstacles to trigger a speech alert.
    SIDE_CLOSE_AREA_RATIO: float = 0.045

    # Number of consecutive frames an object must be detected before it is
    # treated as a "stable" detection worthy of a warning.
    PERSISTENCE_FRAMES: int = 2

    # Number of frames an object may be missing before its tracking state
    # is cleared (grace period for flicker).
    DISAPPEAR_GRACE_FRAMES: int = 15

    # Number of recent area-ratio samples kept per tracked object so we
    # can estimate whether it is "approaching" (visually growing).
    APPROACH_HISTORY_LENGTH: int = 8

    # If the area ratio increases by more than this factor between the
    # oldest and newest history entry, the object is considered
    # "approaching".  E.g. 1.4 -> 40% growth.
    APPROACH_GROWTH_THRESHOLD: float = 1.35

    # Area ratio above which an object in the path is considered
    # "critically close" (large on screen).
    CRITICAL_AREA_RATIO: float = 0.15

    # Minimum quiet gap in seconds between consecutive spoken utterances to prevent speech overload
    TTS_MIN_GAP_SEC: float = 0.5

    # -----------------------------------------------------------------------
    # 5. Hazard Priority Map
    # -----------------------------------------------------------------------
    # Base priority score per COCO class.  Higher = more urgent.
    # Objects not listed default to DEFAULT_HAZARD_PRIORITY.
    HAZARD_PRIORITIES: Dict[str, int] = field(
        default_factory=lambda: {
            # Vehicles – immediate collision risk
            "car": 100,
            "truck": 100,
            "bus": 100,
            "motorcycle": 95,
            "bicycle": 90,
            # People & animals
            "person": 80,
            "dog": 70,
            "cat": 65,
            "horse": 60,
            "cow": 60,
            # Large indoor / outdoor obstacles
            "chair": 55,
            "couch": 55,
            "bed": 50,
            "dining table": 50,
            "bench": 50,
            "toilet": 45,
            "potted plant": 40,
            # Bags & luggage & accessories
            "suitcase": 45,
            "backpack": 40,
            "handbag": 35,
            "umbrella": 30,
            # Electronics
            "laptop": 35,
            "tv": 35,
            "keyboard": 20,
            "mouse": 20,
            # Handheld & small items
            "knife": 30,
            "scissors": 30,
            "bottle": 20,
            "cup": 20,
            "cell phone": 15,
            "remote": 15,
            "book": 10,
        }
    )
    DEFAULT_HAZARD_PRIORITY: int = 35

    # -----------------------------------------------------------------------
    # 6. Warning Cooldowns (seconds)
    # -----------------------------------------------------------------------

    # Minimum seconds between repeated CAUTION / WARNING spoken alerts
    # for the same stationary tracked object (10.0s).
    WARNING_COOLDOWN: float = 10.0

    # Minimum seconds between repeated CRITICAL spoken alerts.
    CRITICAL_COOLDOWN: float = 4.0

    # Seconds before an INFO-level speech event can repeat.
    INFO_COOLDOWN: float = 8.0

    # Smart Announcement Mode:
    # If True: An object is announced ONCE and NOT repeated while stationary
    # in the same position, until it moves (Left/Right/Approaching) or leaves view.
    # If False: Repeats periodically every WARNING_COOLDOWN seconds (e.g. 5.0s).
    SMART_ANNOUNCE_ONLY: bool = False

    # Minimum seconds between automatic scene change descriptions
    SCENE_CHANGE_COOLDOWN: float = 1.5

    # Minimum seconds between urgent path obstruction alerts
    PATH_OBSTRUCTION_COOLDOWN: float = 1.2

    # -----------------------------------------------------------------------
    # 7. Text-to-Speech (TTS) & 3D Spatial Audio
    # -----------------------------------------------------------------------

    # Master switch — set False to disable all speech at startup.
    TTS_ENABLED: bool = True

    # Speech speed in words-per-minute (pyttsx3 / SAPI5 property)
    TTS_RATE: int = 175

    # Volume from 0.0 (silent) to 1.0 (max)
    TTS_VOLUME: float = 1.0

    # Speech style: NORMAL | MINIMAL | URGENT
    TTS_MODE: str = TTS_MODE_NORMAL

    # Audio Feedback Mode: HYBRID | RADAR | SPEECH
    AUDIO_MODE: str = AUDIO_MODE_HYBRID

    # 3D Stereo spatial panning
    SPATIAL_AUDIO_ENABLED: bool = True

    # Radar click pulse intervals (seconds)
    RADAR_MIN_INTERVAL: float = 0.15
    RADAR_MAX_INTERVAL: float = 1.0

    # Soft presence beep interval for stationary obstacles (seconds)
    PRESENCE_BEEP_INTERVAL: float = 3.0
    PRESENCE_BEEP_VOLUME: float = 0.25

    # Maximum items allowed in the TTS priority queue
    TTS_MAX_QUEUE: int = 2

    # How many seconds a queued speech message remains valid.
    TTS_MESSAGE_TTL: float = 4.0

    # Maximum engine-recovery attempts before the TTS worker gives up.
    TTS_MAX_RETRIES: int = 3

    # -----------------------------------------------------------------------
    # 8. Monocular Depth & Floor Clearance
    # -----------------------------------------------------------------------
    DEPTH_ESTIMATION_ENABLED: bool = True
    MIN_CORRIDOR_CLEARANCE: float = 0.40  # 40% open ground required for "clear path"

    # -----------------------------------------------------------------------
    # 9. Optical Flow Ego-Motion Filter
    # -----------------------------------------------------------------------
    OPTICAL_FLOW_ENABLED: bool = True
    EGO_MOTION_COMPENSATION: bool = True

    # -----------------------------------------------------------------------
    # 10. On-Demand OCR Text Reader
    # -----------------------------------------------------------------------
    OCR_ENABLED: bool = True

    # -----------------------------------------------------------------------
    # 11. Quiet Mode & Debug
    # -----------------------------------------------------------------------

    # When True, suppress INFO and most CAUTION alerts.
    QUIET_MODE: bool = False

    # Show verbose detection info in the terminal
    DEBUG_MODE: bool = False

    # Show benchmark overlay (timing breakdown) on-screen
    BENCHMARK_OVERLAY: bool = False

    # OpenCV window title
    WINDOW_TITLE: str = "Oculus AI - Assistive Vision (YOLO26n + OpenVINO)"


# -------------------------------------------------------------------------
# Singleton default instance used when no explicit Config is supplied.
# -------------------------------------------------------------------------
DEFAULT_CONFIG = Config()
