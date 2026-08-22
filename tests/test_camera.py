"""
tests/test_camera.py - Camera & Multi-Camera Enumeration Tests
==============================================================
"""

import pytest
from camera import OpenCVCamera, list_available_cameras, FrameSource


class TestCamera:
    def test_list_available_cameras_returns_list(self):
        cams = list_available_cameras(max_tested=2)
        assert isinstance(cams, list)
        assert len(cams) >= 1
        assert all(isinstance(i, int) for i in cams)

    def test_opencv_camera_properties(self):
        cam = OpenCVCamera(camera_index=1, width=320, height=240)
        assert cam.camera_index == 1
        assert isinstance(cam, FrameSource)
        assert not cam.is_opened()
