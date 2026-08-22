"""
tests/test_depth_estimator.py - Monocular Floor Clearance & Metric Distance Tests
==================================================================================
"""

import numpy as np
import pytest
from config import Config
from depth_estimator import MonocularDepthEstimator, estimate_object_distance


@pytest.fixture
def estimator():
    return MonocularDepthEstimator(config=Config())


class TestDepthEstimator:
    def test_empty_frame_returns_safe_defaults(self, estimator):
        res = estimator.evaluate_corridor(None, [])
        assert res["is_path_clear"] is True
        assert res["clearance_ratio"] == 1.0

    def test_clear_corridor_no_obstacles(self, estimator):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = estimator.evaluate_corridor(frame, [])
        assert res["is_path_clear"] is True
        assert res["clearance_ratio"] == 1.0
        assert res["drop_off_detected"] is False

    def test_obstructed_corridor_with_large_detection(self, estimator):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Large detection occupying walking path (center 0.35 - 0.65 -> 224 to 416)
        det = {
            "class_name": "couch",
            "bbox": [230, 150, 410, 450],
            "area_ratio": 0.25,
            "region": "center",
        }
        res = estimator.evaluate_corridor(frame, [det])
        assert res["clearance_ratio"] < 0.40
        assert res["is_path_clear"] is False
        assert res["obstacle_count"] == 1
        assert "distance_m" in det

    def test_lateral_detection_does_not_obstruct_corridor(self, estimator):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Detection strictly on the left (outside corridor 0.35 - 0.65)
        det_left = {
            "class_name": "chair",
            "bbox": [10, 200, 150, 400],
            "area_ratio": 0.08,
            "region": "left",
        }
        res = estimator.evaluate_corridor(frame, [det_left])
        assert res["clearance_ratio"] == 1.0
        assert res["is_path_clear"] is True
        assert "distance_m" in det_left

    def test_estimate_object_distance_person_close(self):
        # Large person box (height 400px out of 480px) -> should be close (~2m or less)
        dist = estimate_object_distance("person", [200, 40, 440, 440], 480, 640)
        assert 0.5 <= dist <= 2.5

    def test_estimate_object_distance_person_far(self):
        # Small person box (height 100px out of 480px) -> should be far (> 4m)
        dist = estimate_object_distance("person", [300, 100, 340, 200], 480, 640)
        assert dist >= 4.0
