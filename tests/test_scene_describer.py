"""
tests/test_scene_describer.py - AI Scene Describer Unit Tests
=============================================================
"""

import numpy as np
import pytest
from config import Config
from scene_describer import SceneDescriber


@pytest.fixture
def describer():
    return SceneDescriber(config=Config())


class TestSceneDescriber:
    def test_describe_empty_scene(self, describer):
        res = describer.describe_scene(None, [], {"clearance_ratio": 1.0, "drop_off_detected": False})
        assert "open and clear" in res.lower()

    def test_describe_drop_off(self, describer):
        res = describer.describe_scene(None, [], {"clearance_ratio": 0.5, "drop_off_detected": True})
        assert "drop-off" in res.lower() or "stairs" in res.lower()

    def test_describe_objects_with_positions(self, describer):
        dets = [
            {"class_name": "chair", "region": "left", "area_ratio": 0.08},
            {"class_name": "person", "region": "center", "area_ratio": 0.18},
        ]
        depth = {"clearance_ratio": 0.60, "drop_off_detected": False}
        res = describer.describe_scene(None, dets, depth)
        assert "person" in res.lower()
        assert "chair" in res.lower()
        assert "left" in res.lower()

    def test_describe_multiple_similar_objects(self, describer):
        dets = [
            {"class_name": "chair", "region": "left", "area_ratio": 0.08, "distance_m": 2.0},
            {"class_name": "chair", "region": "left", "area_ratio": 0.08, "distance_m": 2.2},
            {"class_name": "chair", "region": "left", "area_ratio": 0.08, "distance_m": 2.1},
        ]
        depth = {"clearance_ratio": 0.80, "drop_off_detected": False}
        res = describer.describe_scene(None, dets, depth)
        assert "3 chairs" in res.lower() or "chairs" in res.lower()
        assert "left" in res.lower()
