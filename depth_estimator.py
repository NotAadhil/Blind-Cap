"""
depth_estimator.py - Monocular Floor Clearance & Metric Distance Estimator
==========================================================================
Software-only ground-plane clearance, metric object distance estimation,
and drop-off (stairs/curb) detection using pinhole perspective geometry.

Features:
- Metric Distance Estimation: Computes real-world distance (in meters) for all
  detected objects using pinhole camera geometry and physical height priors.
- Path Openness Ratio: Computes exact percentage of walking corridor that is clear.
- Drop-Off / Negative Obstacle Detector: Identifies stairs descending and curbs.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config, DEFAULT_CONFIG
from logger import get_logger

logger = get_logger(__name__)

# Real-world object reference heights in meters (COCO classes)
REAL_WORLD_HEIGHTS: Dict[str, float] = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.20,
    "bus": 3.20,
    "train": 3.50,
    "truck": 2.80,
    "chair": 0.85,
    "couch": 0.85,
    "bed": 0.70,
    "dining table": 0.75,
    "bench": 0.80,
    "toilet": 0.75,
    "dog": 0.60,
    "cat": 0.30,
    "horse": 1.60,
    "cow": 1.40,
    "bottle": 0.25,
    "cup": 0.12,
    "knife": 0.20,
    "scissors": 0.20,
    "cell phone": 0.15,
    "remote": 0.15,
    "backpack": 0.50,
    "suitcase": 0.70,
    "handbag": 0.35,
    "potted plant": 0.60,
    "tv": 0.60,
    "laptop": 0.25,
    "mouse": 0.05,
    "keyboard": 0.04,
    "umbrella": 0.80,
    "book": 0.25,
}
DEFAULT_REAL_HEIGHT: float = 0.90


def estimate_object_distance(
    class_name: str,
    bbox: List[int],
    frame_h: int,
    frame_w: int,
    vfov_deg: float = 48.0,
) -> float:
    """
    Monocular metric distance estimation based on pinhole geometry
    and real-world object height prior.

    Args:
        class_name: Object category (e.g. 'person', 'chair').
        bbox: [x1, y1, x2, y2] bounding box coordinates.
        frame_h: Image height in pixels.
        frame_w: Image width in pixels.
        vfov_deg: Camera vertical field of view in degrees.

    Returns:
        Estimated distance in meters (e.g. 1.8, 2.5, 4.2).
    """
    if not bbox or len(bbox) < 4 or frame_h <= 0:
        return 5.0

    x1, y1, x2, y2 = bbox
    bbox_h = max(1, y2 - y1)

    # Vertical focal length in pixels: fy = H / (2 * tan(VFOV / 2))
    vfov_rad = math.radians(vfov_deg)
    fy = frame_h / (2.0 * math.tan(vfov_rad / 2.0))

    real_h = REAL_WORLD_HEIGHTS.get(class_name.lower(), DEFAULT_REAL_HEIGHT)

    # Pinhole distance formula: D = (fy * real_h) / bbox_h
    dist_pinhole = (fy * real_h) / float(bbox_h)

    # Perspective ground-plane correction:
    # If the bottom of the object (y2) is near the bottom of the frame,
    # physical floor geometry constrains the maximum distance continuously without step jumps.
    y2_norm = max(0.0, min(1.0, y2 / float(frame_h)))
    if y2_norm > 0.65:
        floor_dist_cap = 1.2 + (1.0 - y2_norm) * 8.0
        dist_pinhole = min(dist_pinhole, floor_dist_cap)

    # Clamp distance between 0.3m and 20.0m
    dist_m = max(0.3, min(20.0, dist_pinhole))
    return round(dist_m, 1)


class MonocularDepthEstimator:
    """
    Evaluates ground-plane clearance, obstacle distances, and drop-offs.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.last_clearance_ratio: float = 1.0
        self.last_drop_off_detected: bool = False

    def evaluate_corridor(
        self,
        frame: np.ndarray,
        detections: List[Dict],
    ) -> Dict:
        """
        Analyze the walking corridor for floor clearance, obstacles, and drop-offs.
        Also calculates and attaches 'distance_m' to each detection dict.
        """
        if frame is None or frame.size == 0:
            return {
                "clearance_ratio": 1.0,
                "is_path_clear": True,
                "drop_off_detected": False,
                "obstacle_count": 0,
            }

        h, w = frame.shape[:2]
        left_x = int(w * self.config.PATH_LEFT_RATIO)
        right_x = int(w * self.config.PATH_RIGHT_RATIO)
        vfov = getattr(self.config, "CAMERA_VFOV_DEG", 48.0)

        # 1. Calculate distance for every detection and check corridor occlusion
        occupied_vertical_px = np.zeros(h, dtype=bool)
        corridor_detections = 0

        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            cname = d.get("class_name", "object")

            # Estimate and attach metric distance
            dist_m = estimate_object_distance(cname, bbox, h, w, vfov_deg=vfov)
            d["distance_m"] = dist_m

            bx1, by1, bx2, by2 = map(int, bbox)

            # Check horizontal overlap with corridor
            ov_x1 = max(bx1, left_x)
            ov_x2 = min(bx2, right_x)
            if ov_x2 > ov_x1:
                # Mark vertical range as obstructed in corridor
                y_start = max(0, min(h - 1, by1))
                y_end = max(0, min(h - 1, by2))
                occupied_vertical_px[y_start:y_end] = True
                corridor_detections += 1

        # Calculate corridor clearance (focus on bottom 70% of view - walking path)
        walking_y_start = int(h * 0.30)
        walking_h = h - walking_y_start
        obstructed_count = np.sum(occupied_vertical_px[walking_y_start:])
        clearance_ratio = float(max(0.0, min(1.0, 1.0 - (obstructed_count / max(1, walking_h)))))

        # 2. Floor drop-off / Negative obstacle analysis (Stairs, curbs)
        drop_off_detected = False
        floor_y_start = int(h * 0.65)
        floor_corridor = frame[floor_y_start:h, left_x:right_x]

        if floor_corridor.size > 0:
            gray = cv2.cvtColor(floor_corridor, cv2.COLOR_BGR2GRAY)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            abs_sobel_y = np.abs(sobel_y)

            row_energy = np.mean(abs_sobel_y, axis=1)
            high_energy_rows = np.where(row_energy > 40.0)[0]

            if len(high_energy_rows) >= 3:
                row_diffs = np.diff(high_energy_rows)
                spaced_edges = np.sum((row_diffs > 4) & (row_diffs < 25))
                if spaced_edges >= 2:
                    drop_off_detected = True

        self.last_clearance_ratio = round(clearance_ratio, 2)
        self.last_drop_off_detected = drop_off_detected

        is_clear = clearance_ratio >= self.config.MIN_CORRIDOR_CLEARANCE and not drop_off_detected

        return {
            "clearance_ratio": self.last_clearance_ratio,
            "is_path_clear": is_clear,
            "drop_off_detected": drop_off_detected,
            "obstacle_count": corridor_detections,
        }
