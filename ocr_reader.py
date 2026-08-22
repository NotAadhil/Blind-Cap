"""
ocr_reader.py - Real Windows Media OCR & Text Reader Engine
============================================================
Extracts and speaks text from signs, room numbers, doors, and labels using
the native offline Windows 10/11 Media OCR engine.

Triggered on-demand via hotkey 'T' or voice query.
"""

import asyncio
import re
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import Config, DEFAULT_CONFIG
from logger import get_logger

logger = get_logger(__name__)


class OCRReader:
    """
    On-demand OCR text reader powered by Windows 10/11 native Media OCR.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self._win_ocr_available = False
        self._init_ocr()

    def _init_ocr(self) -> None:
        try:
            import winrt.windows.media.ocr as ocr

            self._engine = ocr.OcrEngine.try_create_from_user_profile_languages()
            if self._engine is not None:
                self._win_ocr_available = True
                logger.info("Windows Native Media OCR engine initialised successfully")
            else:
                logger.warning("Windows OCR engine could not be created for user languages")
        except Exception as exc:
            logger.warning("Windows OCR initialization error: %s", exc)

    def extract_text(self, frame: np.ndarray) -> Dict:
        """
        Run OCR on the given frame and return clean detected text.

        Returns:
            Dictionary with:
                "text": str,           # full spoken text or "No text detected."
                "raw_items": List[str],
                "has_text": bool,
        """
        if frame is None or frame.size == 0:
            return {"text": "No text detected.", "raw_items": [], "has_text": False}

        # 1. Try native Windows Media OCR
        if self._win_ocr_available:
            try:
                text_result = self._run_windows_ocr(frame)
                if text_result and text_result.get("has_text"):
                    return text_result
            except Exception as exc:
                logger.debug("Windows OCR runtime error: %s", exc)

        # 2. Try Tesseract fallback if installed
        try:
            import pytesseract

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw = pytesseract.image_to_string(gray, config="--psm 11")
            lines = [l.strip() for l in raw.splitlines() if len(l.strip()) >= 2]
            if lines:
                return {
                    "text": "Text detected: " + ", ".join(lines[:4]),
                    "raw_items": lines,
                    "has_text": True,
                }
        except Exception:
            pass

        return {
            "text": "No text detected in view.",
            "raw_items": [],
            "has_text": False,
        }

    def _run_windows_ocr(self, frame: np.ndarray) -> Optional[Dict]:
        """Convert frame to Windows SoftwareBitmap and run asynchronous OCR."""
        import winrt.windows.graphics.imaging as imaging
        import winrt.windows.storage.streams as streams

        # Encode image to PNG buffer
        success, png_buf = cv2.imencode(".png", frame)
        if not success:
            return None

        async def _async_recognize():
            stream = streams.InMemoryRandomAccessStream()
            writer = streams.DataWriter(stream)
            writer.write_bytes(png_buf.tobytes())
            await writer.store_async()
            await writer.flush_async()
            stream.seek(0)

            decoder = await imaging.BitmapDecoder.create_async(stream)
            sb = await decoder.get_software_bitmap_async()

            result = await self._engine.recognize_async(sb)
            return result

        try:
            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        res = pool.submit(asyncio.run, _async_recognize()).result()
                else:
                    res = loop.run_until_complete(_async_recognize())
            except RuntimeError:
                res = asyncio.run(_async_recognize())

            if res and res.text:
                cleaned_text = res.text.strip()
                # Split lines
                lines = [l.text.strip() for l in res.lines if l.text.strip()]
                if cleaned_text:
                    spoken = f"Text detected: {cleaned_text}"
                    logger.info("[OCR SUCCESS] %s", spoken)
                    return {
                        "text": spoken,
                        "raw_items": lines,
                        "has_text": True,
                    }
        except Exception as exc:
            logger.debug("Windows OCR recognition error: %s", exc)

        return None
