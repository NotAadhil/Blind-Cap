"""
tests/test_detector_utils.py - Detector Utility Function Tests
================================================================
Tests pure helper functions from detector.py.
Does NOT require a GPU or the actual YOLO model.

Run::

    python -m pytest tests/test_detector_utils.py -v
"""

import numpy as np
import pytest
from detector import letterbox, classify_region


class TestLetterbox:
    def test_square_input(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
        assert result.shape == (640, 640, 3)
        assert ratio == pytest.approx(1.0)

    def test_landscape_input(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
        assert result.shape[0] == 640
        assert result.shape[1] == 640
        assert ratio == pytest.approx(1.0)

    def test_portrait_input(self):
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
        assert result.shape[0] == 640
        assert result.shape[1] == 640

    def test_small_input_scaled_up(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result, ratio, _ = letterbox(img, new_shape=(640, 640))
        assert result.shape == (640, 640, 3)
        assert ratio == pytest.approx(6.4)

    def test_custom_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result, _, _ = letterbox(img, new_shape=(320, 320))
        assert result.shape[0] == 320
        assert result.shape[1] == 320


class TestClassifyRegion:
    def test_left_region(self):
        assert classify_region(50, 640, 0.30, 0.70) == "left"

    def test_center_region(self):
        assert classify_region(320, 640, 0.30, 0.70) == "center"

    def test_right_region(self):
        assert classify_region(600, 640, 0.30, 0.70) == "right"

    def test_boundary_left(self):
        # Exactly at left boundary → should be center
        cx = int(640 * 0.30)
        assert classify_region(cx, 640, 0.30, 0.70) == "center"

    def test_boundary_right(self):
        # Exactly at right boundary → should be right
        cx = int(640 * 0.70) + 1
        assert classify_region(cx, 640, 0.30, 0.70) == "right"

    def test_custom_ratios(self):
        assert classify_region(200, 640, 0.20, 0.80) == "center"
        assert classify_region(50, 640, 0.20, 0.80) == "left"
        assert classify_region(550, 640, 0.20, 0.80) == "right"
