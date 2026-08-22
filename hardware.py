"""
hardware.py - Hardware & Environment Diagnostics
==================================================
Checks Python, OpenCV, OpenVINO, webcam, and TTS availability.
Returns structured results for programmatic use and logs a formatted
report to the console.

Usage (standalone)::

    python hardware.py
"""

import sys
from typing import Dict, List, Tuple

from logger import get_logger

logger = get_logger(__name__)


def check_python() -> Tuple[bool, str]:
    """Verify Python version >= 3.10."""
    v = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 10)
    return ok, f"Python {v} ({'OK' if ok else 'Requires 3.10+'})"


def check_opencv() -> Tuple[bool, str]:
    try:
        import cv2
        return True, f"OpenCV {cv2.__version__}"
    except ImportError:
        return False, "Not installed - pip install opencv-python"


def check_openvino() -> Tuple[bool, str, List[Dict[str, str]]]:
    try:
        import openvino as ov
        core = ov.Core()
        devs = core.available_devices
        details = []
        for d in devs:
            try:
                name = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:
                name = d
            details.append({"id": d, "name": name})
        return True, f"OpenVINO {ov.__version__}", details
    except ImportError:
        return False, "Not installed - pip install openvino", []
    except Exception as exc:
        return False, f"Error: {exc}", []


def check_cameras(max_index: int = 4) -> List[Dict]:
    """Probe camera indices 0..*max_index* and return accessible ones."""
    results: List[Dict] = []
    try:
        import cv2
        import time
    except ImportError:
        return results
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    results.append({"index": idx, "resolution": f"{w}x{h}"})
                    break
                time.sleep(0.03)
            cap.release()
            time.sleep(0.08)
    time.sleep(0.20)  # Allow Windows DirectShow driver to fully unbind device handle before pipeline starts
    return results


def check_tts() -> Tuple[bool, str]:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.stop()
        return True, f"pyttsx3 OK ({len(voices)} voice(s))"
    except Exception as exc:
        return False, f"pyttsx3 unavailable ({exc})"


def run_diagnostics(verbose: bool = True) -> Dict:
    """
    Run full diagnostic suite.

    Returns:
        Dictionary of check results.
    """
    py_ok, py_msg = check_python()
    cv_ok, cv_msg = check_opencv()
    ov_ok, ov_msg, ov_devs = check_openvino()
    cams = check_cameras()
    tts_ok, tts_msg = check_tts()

    gpu_available = any(d["id"].startswith("GPU") for d in ov_devs)
    cpu_available = any(d["id"] == "CPU" for d in ov_devs)
    recommended = "GPU" if gpu_available else ("CPU" if cpu_available else "N/A")

    if verbose:
        print("\n" + "=" * 55)
        print("        BLIND CAP - HARDWARE DIAGNOSTICS")
        print("=" * 55)
        _row("Python", py_ok, py_msg)
        _row("OpenCV", cv_ok, cv_msg)
        _row("OpenVINO", ov_ok, ov_msg)
        _row("TTS", tts_ok, tts_msg)

        print("\n  Webcams:")
        if cams:
            for c in cams:
                print(f"    Index {c['index']}: {c['resolution']}")
        else:
            print("    None found")

        print("\n  OpenVINO Devices:")
        if ov_devs:
            for d in ov_devs:
                tag = " [TARGET]" if d["id"].startswith("GPU") else ""
                print(f"    {d['id']:<6} -> {d['name']}{tag}")
        else:
            print("    None")

        print(f"\n  Recommended device: {recommended}")
        print("=" * 55 + "\n")

    return {
        "python": py_ok,
        "opencv": cv_ok,
        "openvino": ov_ok,
        "cameras": cams,
        "tts": tts_ok,
        "ov_devices": ov_devs,
        "recommended_device": recommended,
    }


def _row(label: str, ok: bool, msg: str) -> None:
    tag = "[OK]  " if ok else "[FAIL]"
    print(f"  {tag} {label:<10} {msg}")


if __name__ == "__main__":
    run_diagnostics(verbose=True)
