"""
motion_filter.py - Ego-Motion & Optical Flow Motion Filter
==========================================================
Distinguishes between user head/body movement and actual moving obstacles.

Features:
- Camera ego-motion estimation using sparse optical flow (Lucas-Kanade).
- Subtracts global camera flow from object bounding box trajectories to compute
  true relative velocity.
- Suppresses false "approaching" alarms when the user is simply walking past
  stationary obstacles.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config, DEFAULT_CONFIG
from logger import get_logger

logger = get_logger(__name__)


class EgoMotionFilter:
    """
    Computes camera ego-motion and compensates object velocity vectors.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self.last_ego_velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/frame

    def update(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Estimate camera motion (ego-velocity) from the previous to current frame.

        Returns:
            (vx_camera, vy_camera) in pixels.
        """
        if frame is None or frame.size == 0:
            return (0.0, 0.0)

        # Downscale for ultra-fast optical flow computation
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        vx_cam, vy_cam = 0.0, 0.0

        if self._prev_gray is not None and self._prev_pts is not None and len(self._prev_pts) > 8:
            # Track points using Lucas-Kanade
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                self._prev_pts,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )

            if next_pts is not None and status is not None:
                good_prev = self._prev_pts[status.flatten() == 1]
                good_next = next_pts[status.flatten() == 1]

                if len(good_prev) > 5:
                    pts_prev_2d = good_prev.reshape(-1, 2)
                    pts_next_2d = good_next.reshape(-1, 2)
                    displacements = pts_next_2d - pts_prev_2d
                    # Robust median displacement to ignore moving foreground objects
                    vx_cam = float(np.median(displacements[:, 0])) * 2.0  # scale back from 320 to 640
                    vy_cam = float(np.median(displacements[:, 1])) * 2.0

        # Refresh feature points periodically
        if self._prev_pts is None or len(self._prev_pts) < 20 or self._prev_gray is None:
            pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=50,
                qualityLevel=0.03,
                minDistance=15,
                blockSize=5,
            )
            self._prev_pts = pts
        else:
            self._prev_pts = next_pts[status.flatten() == 1].reshape(-1, 1, 2) if next_pts is not None else None

        self._prev_gray = gray
        self.last_ego_velocity = (vx_cam, vy_cam)
        return (vx_cam, vy_cam)

    def compensate_object_velocity(
        self,
        box_velocity: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Subtract camera motion from observed object bounding box motion.
        True velocity = Observed velocity - Camera velocity.
        """
        vx_obs, vy_obs = box_velocity
        vx_cam, vy_cam = self.last_ego_velocity
        return (vx_obs - vx_cam, vy_obs - vy_cam)

    def is_true_approach(
        self,
        box_area_growth: float,
        vertical_motion: float,
    ) -> bool:
        """
        Evaluate whether an obstacle is actively moving towards the user
        after accounting for forward walking ego-motion.
        """
        vy_cam = self.last_ego_velocity[1]
        # If camera is pitching down or walking forward rapidly, require higher growth threshold
        required_growth = 1.35 if vy_cam < 2.0 else 1.55
        return box_area_growth > required_growth
