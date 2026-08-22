"""
camera.py - Camera capture module for Blind Cap
===============================================
Threaded, non-blocking camera capture supporting multi-camera enumeration,
automatic fallback, sensor warm-up retry, and runtime dynamic camera switching.
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from logger import get_logger

logger = get_logger(__name__)


def list_available_cameras(max_tested: int = 5, active_index: Optional[int] = None) -> List[int]:
    """
    Probe camera indices from 0 up to max_tested - 1 to find connected cameras.
    If active_index is provided, it is included directly without opening a conflicting handle.
    """
    available = []
    if active_index is not None and 0 <= active_index < max_tested:
        available.append(active_index)

    for idx in range(max_tested):
        if idx == active_index:
            continue
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            if idx not in available:
                                available.append(idx)
                            break
                        time.sleep(0.03)
                    cap.release()
                    time.sleep(0.08)
                    if idx in available:
                        break
            except Exception:
                pass
    return sorted(available) if available else [0]


class FrameSource(ABC):
    """Abstract base class for all frame sources."""

    @abstractmethod
    def open(self) -> bool:
        """Open the frame source. Returns True if successful."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame. Returns (success, frame)."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release the frame source."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if the source is opened."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class OpenCVCamera(FrameSource):
    """
    OpenCV-based camera source with threaded frame capture.
    Supports dynamic camera device switching at runtime, sensor warm-up retries,
    and graceful fallback.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._cap = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = False
        self._consecutive_failures = 0
        self._is_hardware_connected = False

    @property
    def camera_index(self) -> int:
        return self._camera_index

    @property
    def is_hardware_connected(self) -> bool:
        return self._is_hardware_connected

    def _try_open_index(self, index: int) -> Optional[cv2.VideoCapture]:
        """
        Attempt to open a specific camera index with DirectShow, MSMF, and default backends.
        Provides an extended sensor warm-up retry loop to negotiate the video stream on startup.
        """
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

                    # Sensor warm-up loop (allow webcam up to 800ms to deliver first frame)
                    for attempt in range(12):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            logger.info(
                                "Camera index %d ready on attempt %d (backend=%s)",
                                index, attempt + 1, backend
                            )
                            return cap
                        time.sleep(0.06)
                    cap.release()
                    time.sleep(0.08)
            except Exception as exc:
                logger.debug("Error probing camera index %d on backend %s: %s", index, backend, exc)
        return None

    def open(self) -> bool:
        """Open the camera and start the capture thread (with automatic index fallback and warm-up)."""
        try:
            # 1. Try requested camera index first
            cap = self._try_open_index(self._camera_index)

            # 2. If requested index failed, try searching for other cameras
            if cap is None:
                logger.warning(
                    "Camera index %d not immediately accessible. Searching for working camera...",
                    self._camera_index,
                )
                for alt_idx in [0, 1, 2, 3]:
                    if alt_idx == self._camera_index:
                        continue
                    cap = self._try_open_index(alt_idx)
                    if cap is not None:
                        self._camera_index = alt_idx
                        logger.info("Auto-selected working camera at index %d", alt_idx)
                        break

            if cap is not None and cap.isOpened():
                self._cap = cap
                self._is_hardware_connected = True
                self._running = True
                self._consecutive_failures = 0
                self._thread = threading.Thread(
                    target=self._capture_loop, daemon=True, name="Camera-Capture"
                )
                self._thread.start()
                logger.info("Successfully opened camera %d and started capture thread.", self._camera_index)
                return True

            # If no physical camera could be opened, start in placeholder fallback mode
            logger.warning("No physical camera could be opened. Starting with visual fallback feed.")
            self._is_hardware_connected = False
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="Camera-Capture"
            )
            self._thread.start()
            return True

        except Exception as e:
            logger.error("Error opening camera: %s", e)
            self._is_hardware_connected = False
            self._running = True
            return True

    def switch_camera(self, new_camera_index: int) -> bool:
        """
        Switch to a different camera device index at runtime.
        Safely stops the current capture thread, releases current device,
        and starts capture on the new device.
        """
        logger.info("Switching camera from index %d to %d...", self._camera_index, new_camera_index)
        self.release()
        # Allow Windows driver 200ms to release device handle
        time.sleep(0.20)
        self._camera_index = new_camera_index
        return self.open()

    def _generate_fallback_frame(self) -> np.ndarray:
        """Generate an animated placeholder frame when no camera is connected."""
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        for y in range(self._height):
            frame[y, :] = (int(30 + 20 * (y / self._height)), 20, 20)

        cv2.rectangle(frame, (10, 10), (self._width - 10, self._height - 10), (0, 180, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "BLIND CAP - CAMERA FEED", (self._width // 2 - 160, self._height // 2 - 50),
                    font, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Camera Index {self._camera_index} Offline", (self._width // 2 - 140, self._height // 2 - 10),
                    font, 0.6, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Click [Switch Cam] or Press [V] to select another input",
                    (self._width // 2 - 230, self._height // 2 + 35),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Connect a USB / Built-in camera and switch input",
                    (self._width // 2 - 200, self._height // 2 + 65),
                    font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        return frame

    def _capture_loop(self) -> None:
        """Background thread loop that continuously reads frames with auto-recovery."""
        while self._running:
            if self._cap is not None and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self._lock:
                        self._latest_frame = frame
                        self._frame_ready = True
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures % 25 == 0:
                        logger.warning(
                            "Camera read failure (%d consecutive), attempting auto-recovery...",
                            self._consecutive_failures,
                        )
                        # Attempt live reconnect without freezing pipeline
                        recovered_cap = self._try_open_index(self._camera_index)
                        if recovered_cap is not None:
                            old_cap = self._cap
                            self._cap = recovered_cap
                            old_cap.release()
                            self._consecutive_failures = 0
                            logger.info("Camera %d auto-recovered successfully.", self._camera_index)
                    if self._consecutive_failures > 50:
                        with self._lock:
                            self._latest_frame = self._generate_fallback_frame()
                    time.sleep(0.015)
            else:
                # Try auto-connecting physical camera
                if self._consecutive_failures % 30 == 0:
                    recovered_cap = self._try_open_index(self._camera_index)
                    if recovered_cap is not None:
                        self._cap = recovered_cap
                        self._is_hardware_connected = True
                        self._consecutive_failures = 0
                        logger.info("Physical camera %d connected and resumed.", self._camera_index)
                with self._lock:
                    self._latest_frame = self._generate_fallback_frame()
                    self._frame_ready = True
                self._consecutive_failures += 1
                time.sleep(0.033)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Non-blocking read, returning the latest frame instantly."""
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def release(self) -> None:
        """Stop the background thread and release the camera."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        with self._lock:
            self._latest_frame = None
            self._frame_ready = False
        self._is_hardware_connected = False
        logger.info("Camera released.")

    def is_opened(self) -> bool:
        return self._running


class RaspberryPiStream(FrameSource):
    """
    Placeholder for future Raspberry Pi camera stream over WiFi.
    Not implemented in this version.
    """

    def __init__(self, url: str = "http://raspberrypi.local:8080/stream"):
        self.url = url

    def open(self) -> bool:
        raise NotImplementedError("RaspberryPiStream will be implemented in a future version.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return False, None

    def release(self) -> None:
        pass

    def is_opened(self) -> bool:
        return False
