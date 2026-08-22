"""
tests/test_motion_filter.py - Ego-Motion & Optical Flow Tests
==============================================================
"""

import numpy as np
import pytest
from config import Config
from motion_filter import EgoMotionFilter


@pytest.fixture
def motion_filter():
    return EgoMotionFilter(config=Config())


class TestMotionFilter:
    def test_motion_update_empty_frame(self, motion_filter):
        vx, vy = motion_filter.update(None)
        assert vx == 0.0 and vy == 0.0

    def test_motion_update_consecutive_frames(self, motion_filter):
        # Create patterned synthetic image
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.roll(frame1, shift=5, axis=1)  # simulated 5px horizontal pan

        vx1, vy1 = motion_filter.update(frame1)
        vx2, vy2 = motion_filter.update(frame2)
        # Should execute without errors and produce motion tuple
        assert isinstance(vx2, float)
        assert isinstance(vy2, float)

    def test_compensate_object_velocity(self, motion_filter):
        motion_filter.last_ego_velocity = (10.0, 2.0)
        # Observed box velocity (10.0, 2.0) moving at exact same speed as camera -> true velocity (0.0, 0.0)
        true_vx, true_vy = motion_filter.compensate_object_velocity((10.0, 2.0))
        assert abs(true_vx) < 0.01 and abs(true_vy) < 0.01

    def test_is_true_approach(self, motion_filter):
        motion_filter.last_ego_velocity = (0.0, 0.0)
        assert motion_filter.is_true_approach(box_area_growth=1.50, vertical_motion=0.0) is True
        assert motion_filter.is_true_approach(box_area_growth=1.10, vertical_motion=0.0) is False
