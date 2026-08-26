"""
visualization.py - Real-Time HUD and Bounding Box Visualizer
==============================================================
Renders walking-path corridors, colour-coded detection bounding boxes,
distance/proximity indicators (FAR vs CLOSE vs CRITICAL), floor clearance meters,
and an accessible heads-up display (HUD).

Colour-blind friendly: every box has an explicit text label with proximity
and direction, so colour alone is never the only indicator.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    Config,
    DEFAULT_CONFIG,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    SEVERITY_CAUTION,
    SEVERITY_INFO,
)
from logger import get_logger

logger = get_logger(__name__)

# Font used throughout
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    """Renders visual overlays on camera frames."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config

        # BGR colour palette
        self.CLR_CRITICAL = (0, 0, 255)     # red
        self.CLR_WARNING = (0, 100, 255)    # orange
        self.CLR_CAUTION = (0, 215, 255)    # amber/yellow
        self.CLR_FAR = (255, 200, 0)        # cyan/light blue
        self.CLR_INFO = (0, 255, 128)       # mint/green
        self.CLR_CORRIDOR = (200, 200, 200)
        self.CLR_HUD_BG = (25, 25, 25)
        self.CLR_WHITE = (255, 255, 255)
        self.CLR_BLACK = (0, 0, 0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sev_colour(self, severity: str, proximity: str = "CLOSE") -> Tuple[int, int, int]:
        """Map severity and proximity to a BGR colour. ONLY Danger/Stop Zone turns RED."""
        if proximity == "FAR":
            return self.CLR_FAR
        if proximity == "MEDIUM":
            return self.CLR_INFO  # Normal tracking (Green/Mint), NOT Red!
        if proximity in ("CRITICAL", "CLOSE"):
            return self.CLR_CRITICAL  # RED only when within closer danger/stop range (<= 1.8m)
        return self.CLR_INFO

    def _sev_text_colour(self, severity: str, proximity: str = "CLOSE") -> Tuple[int, int, int]:
        """Text colour for a given severity background."""
        if proximity in ("CRITICAL", "CLOSE"):
            return self.CLR_WHITE
        return self.CLR_BLACK

    def _lookup_hazard_info(
        self, det: Dict, hazard_event
    ) -> Tuple[str, str, Optional[float], Optional[List[int]], str, str]:
        """Find the (severity, proximity, distance_m, smoothed_bbox, class_name, region) assigned to *det*."""
        dist_m = det.get("distance_m")
        raw_bbox = det.get("bbox", [])
        raw_cls = det.get("class_name", "?")
        raw_reg = det.get("region", "center")

        if hazard_event is None:
            return SEVERITY_INFO, "FAR", dist_m, raw_bbox, raw_cls, raw_reg

        best_match = None
        best_score = 0.0

        for h in getattr(hazard_event, "all_hazards", []):
            h_bbox = h.get("bbox", [])
            h_cls = h.get("class_name")

            iou = 0.0
            if raw_bbox and h_bbox and len(raw_bbox) == 4 and len(h_bbox) == 4:
                x1 = max(raw_bbox[0], h_bbox[0])
                y1 = max(raw_bbox[1], h_bbox[1])
                x2 = min(raw_bbox[2], h_bbox[2])
                y2 = min(raw_bbox[3], h_bbox[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                a1 = (raw_bbox[2] - raw_bbox[0]) * (raw_bbox[3] - raw_bbox[1])
                a2 = (h_bbox[2] - h_bbox[0]) * (h_bbox[3] - h_bbox[1])
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0.0

            score = iou + (0.5 if h_cls == raw_cls else 0.0)
            if score > best_score:
                best_score = score
                best_match = h

        if best_match is not None and best_score > 0.2:
            return (
                best_match.get("severity", SEVERITY_INFO),
                best_match.get("proximity", "FAR"),
                best_match.get("distance_m", dist_m),
                best_match.get("bbox", raw_bbox),
                best_match.get("class_name", raw_cls),
                best_match.get("region", raw_reg),
            )

        return SEVERITY_INFO, "FAR", dist_m, raw_bbox, raw_cls, raw_reg

    def _is_active_focus(self, det: Dict, hazard_event) -> bool:
        """Is *det* the current focus object?"""
        active = getattr(hazard_event, "active_hazard", None)
        if not active:
            return False
        det_bbox = det.get("bbox", [])
        act_bbox = active.get("bbox", [])
        if det.get("class_name") != active.get("class_name"):
            return False
        if det_bbox == act_bbox:
            return True
        if det_bbox and act_bbox and len(det_bbox) == 4 and len(act_bbox) == 4:
            x1 = max(det_bbox[0], act_bbox[0])
            y1 = max(det_bbox[1], act_bbox[1])
            x2 = min(det_bbox[2], act_bbox[2])
            y2 = min(det_bbox[3], act_bbox[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
            a2 = (act_bbox[2] - act_bbox[0]) * (act_bbox[3] - act_bbox[1])
            union = a1 + a2 - inter
            return (inter / union) >= 0.35 if union > 0 else False
        return False

    # ------------------------------------------------------------------
    # Drawing layers
    # ------------------------------------------------------------------
    def draw_corridor(self, frame: np.ndarray, stats: Optional[Dict[str, Any]] = None) -> None:
        """Draw vertical corridor boundary lines + floor clearance meter."""
        h, w = frame.shape[:2]
        xl = int(w * self.config.PATH_LEFT_RATIO)
        xr = int(w * self.config.PATH_RIGHT_RATIO)

        # Draw corridor boundary lines
        cv2.line(frame, (xl, 0), (xl, h), self.CLR_CORRIDOR, 1, cv2.LINE_AA)
        cv2.line(frame, (xr, 0), (xr, h), self.CLR_CORRIDOR, 1, cv2.LINE_AA)

        # Floor clearance indicator in center corridor
        clearance = stats.get("clearance", 1.0) if stats else 1.0
        drop_off = stats.get("drop_off", False) if stats else False

        clr_text = f"PATH CLEAR ({int(clearance*100)}%)" if clearance > 0.60 else f"OBSTRUCTED ({int(clearance*100)}%)"
        if drop_off:
            clr_text = "CAUTION: DROP-OFF / STAIRS"

        clr_color = (0, 220, 255) if clearance > 0.60 and not drop_off else (0, 100, 255)
        if drop_off:
            clr_color = self.CLR_CRITICAL

        cv2.putText(frame, "LEFT", (10, h - 12), _FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, clr_text, (xl + 8, h - 12), _FONT, 0.42, clr_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT", (xr + 10, h - 12), _FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        hazard_event: Any,
    ) -> None:
        """Draw bounding boxes with proximity, metric distance, and severity-based colours and text labels."""
        for det in detections:
            sev, prox, dist_m, smoothed_bbox, name, region = self._lookup_hazard_info(det, hazard_event)
            bbox = smoothed_bbox if smoothed_bbox and len(smoothed_bbox) == 4 else det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)
            conf = det.get("confidence", 0.0)

            is_focus = self._is_active_focus(det, hazard_event)
            colour = self._sev_colour(sev, prox)
            thick = 3 if (is_focus and prox in ("CRITICAL", "CLOSE")) else (2 if prox in ("CRITICAL", "CLOSE") else 1)

            # Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)

            # Label text showing metric distance, proximity, and region clearly
            dist_tag = f"[{dist_m:.1f}m] " if dist_m is not None else ""
            if prox == "CRITICAL":
                label = f"[EMERGENCY] {dist_tag}{name.upper()} {int(conf * 100)}% ({region})"
            elif prox == "CLOSE":
                label = f"[STOP] {dist_tag}{name.upper()} {int(conf * 100)}% ({region})"
            elif prox == "MEDIUM":
                label = f"{dist_tag}{name} {int(conf * 100)}% ({region})"
            else:
                label = f"{dist_tag}{name} {int(conf * 100)}% [Far] ({region})"

            if is_focus and prox in ("CRITICAL", "CLOSE"):
                label = f">>> {label}"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
            ly = max(0, y1 - th - 6)
            cv2.rectangle(frame, (x1, ly), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                _FONT, 0.45, self._sev_text_colour(sev, prox), 1, cv2.LINE_AA,
            )

    def draw_hud(self, frame: np.ndarray, stats: Dict[str, Any]) -> None:
        """Render semi-transparent stats panel on the left edge."""
        h, w = frame.shape[:2]
        pw = 225  # panel width

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (pw, h), self.CLR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        x = 10
        y = 26
        gap = 20  # line spacing

        def line(text: str, colour=self.CLR_WHITE, scale: float = 0.42) -> None:
            nonlocal y
            cv2.putText(frame, str(text), (x, y), _FONT, scale, colour, 1, cv2.LINE_AA)
            y += gap

        def spacer(px: int = 6) -> None:
            nonlocal y
            y += px

        # Title
        line("OCULUS AI - ASSIST", scale=0.50, colour=(0, 220, 255))
        spacer()

        # Performance
        cam_idx = stats.get("camera_index", 0)
        line(f"Cam {cam_idx} FPS:  {stats.get('camera_fps', 0):.1f}")
        line(f"Infer FPS:  {stats.get('inference_fps', 0):.1f} ({stats.get('inference_ms', 0):.0f}ms)")
        spacer()

        # Hardware
        line(f"Device: {stats.get('device', '?')}")
        line(f"Detections: {stats.get('detections', 0)}")
        spacer()

        # Audio & Feedback Modes
        aud_mode = stats.get("audio_mode", "HYBRID")
        aud_clr = (0, 255, 128) if aud_mode == "HYBRID" else ((0, 220, 255) if aud_mode == "RADAR" else self.CLR_WHITE)
        line(f"Audio: {aud_mode}", aud_clr)

        tts_status = stats.get("tts_status", "?")
        tts_clr = self.CLR_INFO if tts_status == "READY" else self.CLR_CAUTION
        if tts_status == "MUTED":
            tts_clr = (100, 100, 255)
        line(f"TTS: {tts_status}", tts_clr)
        spacer()

        # Floor Clearance
        clr_ratio = stats.get("clearance", 1.0)
        clr_text = f"Clearance: {int(clr_ratio*100)}%"
        clr_c = self.CLR_INFO if clr_ratio > 0.60 else self.CLR_WARNING
        line(clr_text, clr_c)

        drop_off = stats.get("drop_off", False)
        if drop_off:
            line("DROP-OFF DETECTED!", self.CLR_CRITICAL)
        spacer()

        # Focus Object
        line("Focus:", self.CLR_INFO)
        focus = stats.get("focus", "")
        if focus:
            line(f"  {focus}", scale=0.38)
            fs = stats.get("focus_state", "")
            if fs:
                line(f"  [{fs}]")
        else:
            line("  None")
        spacer(8)

        # Accessible Hotkey Guide
        line("Keys:", (180, 180, 180), scale=0.38)
        line("  T: Read Text (OCR)", (220, 220, 220), scale=0.35)
        line("  Space: Scene Summary", (220, 220, 220), scale=0.35)
        line("  V: Switch Camera", (220, 220, 220), scale=0.35)
        line("  A: Audio Mode | M: Mute", (180, 180, 180), scale=0.35)

    def draw_hazard_banner(self, frame: np.ndarray, hazard_event: Any) -> None:
        """Draw a prominent warning banner near the top if a CLOSE or CRITICAL hazard is active."""
        if not getattr(hazard_event, "hazard_detected", False):
            return

        active = getattr(hazard_event, "active_hazard", None)
        if not active:
            return

        prox = active.get("proximity", "FAR")
        if prox == "FAR":
            return

        sev = active.get("severity", SEVERITY_INFO)
        name = active.get("class_name", "obstacle").upper()
        region = active.get("region", "center")

        if region == "left":
            loc_str = "ON LEFT"
        elif region == "right":
            loc_str = "ON RIGHT"
        else:
            loc_str = "IN WALKING PATH" if sev == SEVERITY_WARNING else "AHEAD"

        if sev == SEVERITY_CRITICAL or prox == "CRITICAL":
            banner = f"[CRITICAL] STOP. {name} {loc_str}"
        elif sev == SEVERITY_WARNING:
            banner = f"[WARNING] {name} {loc_str}"
        elif sev == SEVERITY_CAUTION:
            banner = f"[CAUTION] {name} {loc_str}"
        else:
            banner = f"[ALERT] {name} {loc_str}"

        colour = self._sev_colour(sev, prox)
        text_clr = self._sev_text_colour(sev, prox)

        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(banner, _FONT, 0.65, 2)
        bx1 = max(0, (w - tw) // 2 - 18)
        bx2 = min(w, (w + tw) // 2 + 18)
        by1 = 40
        by2 = 40 + th + 18

        # Semi-transparent banner
        ov = frame.copy()
        cv2.rectangle(ov, (bx1, by1), (bx2, by2), colour, -1)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), colour, 2)

        cv2.putText(
            frame, banner, ((w - tw) // 2, by1 + th + 8),
            _FONT, 0.65, text_clr, 2, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Interactive GUI Buttons (Clickable on screen)
    # ------------------------------------------------------------------
    def get_button_rects(self, frame_w: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Return {button_id: (x1, y1, x2, y2)} for interactive mouse clicks."""
        bw = 155
        bh = 30
        x1 = max(10, frame_w - bw - 10)
        x2 = x1 + bw
        return {
            "switch_camera": (x1, 10, x2, 10 + bh),
            "scene_summary": (x1, 46, x2, 46 + bh),
            "ocr": (x1, 82, x2, 82 + bh),
            "audio_mode": (x1, 118, x2, 118 + bh),
        }

    def get_clicked_button(self, x: int, y: int, frame_w: int = 640) -> Optional[str]:
        """Check if mouse click (x, y) hit an interactive GUI button."""
        rects = self.get_button_rects(frame_w)
        for btn_id, (bx1, by1, bx2, by2) in rects.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return btn_id
        return None

    def draw_gui_buttons(self, frame: np.ndarray, stats: Optional[Dict[str, Any]] = None) -> None:
        """Render modern, clickable interactive GUI buttons on top-right."""
        h, w = frame.shape[:2]
        rects = self.get_button_rects(w)
        stats = stats or {}
        cam_idx = stats.get("camera_index", 0)
        aud_mode = stats.get("audio_mode", "HYBRID")

        buttons = [
            ("switch_camera", f"Cam {cam_idx} [Switch]", (0, 180, 255), (0, 0, 0)),
            ("scene_summary", "Summary (C)", (40, 200, 100), (0, 0, 0)),
            ("ocr", "Read Text (T)", (200, 160, 40), (0, 0, 0)),
            ("audio_mode", f"Audio: {aud_mode}", (160, 80, 220), (255, 255, 255)),
        ]

        overlay = frame.copy()
        for btn_id, label, bg_col, text_col in buttons:
            x1, y1, x2, y2 = rects[btn_id]
            # Button background
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

            # Button text
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.38, 1)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(frame, label, (tx, ty), _FONT, 0.38, text_col, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ------------------------------------------------------------------
    # Public composite
    # ------------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        hazard_event: Any,
        stats: Dict[str, Any],
    ) -> np.ndarray:
        """Composite all visual layers and return the rendered frame."""
        if frame is None:
            return frame
        out = frame.copy()
        self.draw_corridor(out, stats)
        self.draw_detections(out, detections, hazard_event)
        self.draw_hud(out, stats)
        self.draw_hazard_banner(out, hazard_event)
        self.draw_gui_buttons(out, stats)
        return out
