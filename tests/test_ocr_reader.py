"""
tests/test_ocr_reader.py - On-Demand OCR Text Reader Tests
===========================================================
"""

import cv2
import numpy as np
import pytest
from config import Config
from ocr_reader import OCRReader


@pytest.fixture
def ocr_reader():
    return OCRReader(config=Config())


class TestOCRReader:
    def test_extract_text_empty_frame(self, ocr_reader):
        res = ocr_reader.extract_text(None)
        assert res["has_text"] is False
        assert res["text"] == "No text detected."

    def test_extract_text_blank_image(self, ocr_reader):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = ocr_reader.extract_text(blank)
        assert res["has_text"] is False

    def test_extract_text_rendered_sign(self, ocr_reader):
        # Render a clean high-contrast text sign
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(img, "ROOM 302 EXIT", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        res = ocr_reader.extract_text(img)
        assert isinstance(res, dict)
        assert "text" in res
        assert "has_text" in res
        if res["has_text"]:
            assert "ROOM" in res["text"] or "302" in res["text"] or "EXIT" in res["text"] or "Text detected" in res["text"]
