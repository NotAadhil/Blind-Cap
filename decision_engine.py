"""
decision_engine.py - Scene-Aware Hazard Decision & Spatial Tracking Engine
==========================================================================
Evaluates detections, tracks spatial objects across frames using IoU & center
distance, estimates real-world metric distance (in meters), stabilizes object
classifications via confidence-weighted voting, determines proximity (FAR vs
MEDIUM vs CLOSE vs CRITICAL), and generates spoken warnings with 2-zone distance
awareness (50% reduced danger threshold):

1. Danger / Stop Zone (<= 1.8m in Walking Path):
   - Box turns RED.
   - Urgent Alert: "Stop! Person detected, 0.8 meters away." / "Stop! Chair detected, 1.2 meters away."
2. In-Frame Advisory Zone (1.8m to 3.5m):
   - Box remains normal colour (Green / Cyan / Amber - NOT Red).
   - In-frame announcement: "Person detected, 3 meters away." / "Chair detected on your left, 2 meters away."
3. Dynamic Distance Updates:
   - As an object moves closer or farther, distance updates dynamically.
4. Path Cleared:
   - When all obstacles leave view: "Path is clear."
"""

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from config import (
    Config,
    DEFAULT_CONFIG,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    SEVERITY_CAUTION,
    SEVERITY_INFO,
    TTS_MODE_NORMAL,
    TTS_MODE_MINIMAL,
    TTS_MODE_URGENT,
)
from depth_estimator import estimate_object_distance
from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hazard state constants
# ---------------------------------------------------------------------------
STATE_FAR = "FAR"
STATE_NEW = "NEW"
STATE_APPROACHING = "APPROACHING"
STATE_STABLE = "STABLE"
STATE_CLOSER = "CLOSER"
STATE_CRITICAL_1 = "CRITICAL_1"
STATE_CRITICAL_2 = "CRITICAL_2"
STATE_CRITICAL = "CRITICAL"  # alias for backward compatibility
STATE_RECOVERY = "RECOVERY"
STATE_CLEARED = "CLEARED"


# ---------------------------------------------------------------------------
# Natural language formatting helpers
# ---------------------------------------------------------------------------
def pluralize(word: str, count: int = 2) -> str:
    """Pluralize an object name naturally."""
    if count == 1:
        return word
    w = word.lower()
    if w == "person":
        return "people"
    if w in ("couch", "bench"):
        return f"{word}es"
    if w.endswith(("s", "sh", "ch", "x", "z")):
        return f"{word}es"
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    if w == "scissors":
        return "pairs of scissors"
    return f"{word}s"


def format_item_list(items: List[str]) -> str:
    """Format ['A', 'B', 'C'] -> 'A, B, and C', ['A', 'B'] -> 'A and B', ['A'] -> 'A'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def format_distance_str(distance_m: Optional[float]) -> str:
    """Format metric distance naturally for voice alerts (e.g. '0.8 meters away', '3 meters away')."""
    if distance_m is None or distance_m <= 0:
        return ""
    if distance_m < 1.0:
        return f"{distance_m:.1f} meters away"
    rounded = round(distance_m, 1)
    if abs(rounded - round(rounded)) < 0.15:
        int_val = int(round(rounded))
        unit = "meter" if int_val == 1 else "meters"
        return f"{int_val} {unit} away"
    return f"{rounded:.1f} meters away"


# ---------------------------------------------------------------------------
# Spatial helper functions
# ---------------------------------------------------------------------------
def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_center_distance(c1: List[int], c2: List[int], frame_w: int, frame_h: int) -> float:
    """Compute normalized Euclidean distance between two center points."""
    if not c1 or not c2 or len(c1) < 2 or len(c2) < 2:
        return 1.0
    diag = (frame_w**2 + frame_h**2) ** 0.5
    if diag <= 0:
        return 1.0
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return ((dx**2 + dy**2) ** 0.5) / diag


def classify_region_with_hysteresis(
    cx: int,
    frame_w: int,
    current_region: str,
    left_ratio: float = 0.35,
    right_ratio: float = 0.65,
    hysteresis_ratio: float = 0.05,
) -> str:
    """
    Classify region (left, center, right) with spatial hysteresis
    to prevent border oscillation when standing still.
    """
    if frame_w <= 0:
        return "center"
    norm_x = cx / float(frame_w)

    if current_region == "center":
        if norm_x < (left_ratio - hysteresis_ratio):
            return "left"
        elif norm_x > (right_ratio + hysteresis_ratio):
            return "right"
        return "center"
    elif current_region == "left":
        if norm_x > (left_ratio + hysteresis_ratio):
            return "center"
        return "left"
    elif current_region == "right":
        if norm_x < (right_ratio - hysteresis_ratio):
            return "center"
        return "right"
    else:
        if norm_x < left_ratio:
            return "left"
        elif norm_x > right_ratio:
            return "right"
        return "center"


def smooth_bbox(
    old_bbox: List[int],
    new_bbox: List[int],
    deadband_px: int = 4,
) -> List[int]:
    """
    Smooth bounding box with deadband to eliminate visual jitter and trembling
    when standing still, while preserving responsiveness during movement.
    """
    if not old_bbox or len(old_bbox) < 4:
        return new_bbox
    if not new_bbox or len(new_bbox) < 4:
        return old_bbox

    max_delta = max(abs(old_bbox[i] - new_bbox[i]) for i in range(4))
    if max_delta <= deadband_px:
        return old_bbox

    alpha = 0.60 if max_delta > 20 else 0.30
    return [
        int(round((1.0 - alpha) * old_bbox[i] + alpha * new_bbox[i]))
        for i in range(4)
    ]


def smooth_distance(
    old_dist: float,
    new_dist: float,
    deadband_m: float = 0.15,
) -> float:
    """
    Smooth metric distance with deadband to ignore tiny noise fluctuations.
    """
    if old_dist <= 0:
        return round(new_dist, 1)

    delta = abs(new_dist - old_dist)
    if delta <= deadband_m:
        return old_dist

    alpha = 0.60 if delta > 0.8 else 0.25
    return round((1.0 - alpha) * old_dist + alpha * new_dist, 1)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrackedObject:
    """Persistent state for a single tracked physical object across frames."""

    track_id: int
    class_name: str
    track_key: str = ""  # string identifier for backward compatibility

    state: str = STATE_NEW
    proximity: str = "FAR"  # "FAR", "MEDIUM", "CLOSE", "CRITICAL"
    estimated_distance_m: float = 5.0  # Metric distance in meters
    frames_seen: int = 0
    frames_missing: int = 0

    last_confidence: float = 0.0
    last_area_ratio: float = 0.0
    last_bbox: List[int] = field(default_factory=list)
    last_center: List[int] = field(default_factory=list)
    last_region: str = "center"

    # History of (class_name, confidence) for majority voting & temporal filtering
    class_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # Rolling window of area_ratio values for approach detection
    area_history: deque = field(default_factory=lambda: deque(maxlen=8))

    last_warning_time: float = 0.0
    last_state_change_time: float = 0.0
    priority_score: int = 0

    # Multi-tier critical state machine
    # 0 = normal, 1 = critical tier 1 announced (<=1.8m), 2 = critical tier 2 announced (<=0.9m)
    critical_tier: int = 0
    critical_announced_time: float = 0.0
    has_announced_obstruction: bool = False
    obstruction_announced_time: float = 0.0
    frames_in_center: int = 0
    frames_outside_center: int = 0

    # Presence & Distance Announcement State
    has_announced_presence: bool = False
    last_announced_tier: int = 0
    last_announced_distance_m: Optional[float] = None
    last_announced_region: str = ""
    last_distance_announced_time: float = 0.0
    has_announced_in_frame: bool = False


@dataclass
class HazardEvent:
    """Result returned from :meth:`DecisionEngine.evaluate` each frame."""

    # Speech fields (None means "don't speak this frame")
    warning_text: Optional[str] = None
    speak_priority: int = 0
    severity: str = SEVERITY_INFO
    category: str = ""
    dedupe_key: str = ""

    # Detection summary
    hazard_detected: bool = False
    active_hazard: Optional[Dict] = None
    all_hazards: List[Dict] = field(default_factory=list)

    # Focus display (for HUD)
    focus_object: Optional[str] = None
    focus_severity: str = ""
    focus_state: str = ""
    focus_proximity: str = "FAR"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    Per-frame evaluator that uses spatial object tracking and 2-zone distance
    awareness to generate natural spoken descriptions and urgent path alerts.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self._tracked: Dict[int, TrackedObject] = {}
        self._next_track_id: int = 1
        self._focus_key: Optional[int] = None
        self._last_spoken_warning: Optional[str] = None
        self._frame_count: int = 0

        # Scene change tracking
        self._last_spoken_scene_signature: Optional[Tuple[Tuple[Tuple[str, str], int], ...]] = None
        self._last_spoken_scene_text: Optional[str] = None
        self._last_scene_speak_time: float = 0.0
        self._last_had_objects: bool = False

    # ------------------------------------------------------------------
    # Class Resolution (Confidence-Weighted Majority Voting)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_class_name(tracked: TrackedObject) -> str:
        """
        Confidence-weighted majority voting over the recent class history.
        Filters out transient single-frame detector hallucinations (e.g. knife vs cell phone).
        """
        if not tracked.class_history:
            return tracked.class_name

        scores: Dict[str, float] = {}
        for cls_name, conf in tracked.class_history:
            scores[cls_name] = scores.get(cls_name, 0.0) + conf

        best_cls = max(scores.keys(), key=lambda c: scores[c])
        return best_cls

    # ------------------------------------------------------------------
    # Spatial Tracking Matcher
    # ------------------------------------------------------------------
    def _match_detections_to_tracks(
        self,
        detections: List[Dict],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Robust spatial object matching using IoU, Euclidean center distance,
        class consistency, and region alignment.
        """
        if not self._tracked:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(self._tracked.keys()), []

        track_ids = list(self._tracked.keys())
        candidates: List[Tuple[float, int, int]] = []

        # Pass 1: Strict same-class matching
        for tid in track_ids:
            t = self._tracked[tid]
            for d_idx, det in enumerate(detections):
                det_cls = det.get("class_name", "")
                if det_cls != t.class_name:
                    continue  # Keep distinct classes completely isolated!

                iou = compute_iou(t.last_bbox, det.get("bbox", []))
                cdist = compute_center_distance(t.last_center, det.get("center", []), frame_w, frame_h)

                match_score = 0.0
                if iou >= 0.10:
                    match_score = 1.0 + iou
                elif cdist <= 0.30:
                    match_score = 0.5 + (0.30 - cdist)

                if match_score > 0 and det.get("region") == t.last_region:
                    match_score += 0.15

                if match_score > 0.20:
                    candidates.append((match_score, tid, d_idx))

        # Greedy bipartite matching
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()
        pairs: List[Tuple[int, int]] = []

        for score, tid, d_idx in candidates:
            if tid not in matched_tracks and d_idx not in matched_dets:
                matched_tracks.add(tid)
                matched_dets.add(d_idx)
                pairs.append((tid, d_idx))

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_dets = [d_idx for d_idx in range(len(detections)) if d_idx not in matched_dets]

        # Pass 2: High-overlap matching (IoU >= 0.50) for transient single-object label switches
        # (e.g. knife -> cell phone on the exact same physical box)
        for tid in list(unmatched_tracks):
            t = self._tracked[tid]
            for d_idx in list(unmatched_dets):
                if d_idx in matched_dets or tid in matched_tracks:
                    continue
                iou = compute_iou(t.last_bbox, detections[d_idx].get("bbox", []))
                if iou >= 0.50:
                    matched_tracks.add(tid)
                    matched_dets.add(d_idx)
                    pairs.append((tid, d_idx))
                    break

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_dets = [d_idx for d_idx in range(len(detections)) if d_idx not in matched_dets]

        return pairs, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _compute_hazard_score(
        self, tracked: TrackedObject, frame_w: int, frame_h: int
    ) -> Tuple[int, str]:
        """
        Multi-factor hazard score with spatial awareness and distance weighting.
        Returns ``(score, severity_string)``.
        """
        base_score = self.config.HAZARD_PRIORITIES.get(
            tracked.class_name, self.config.DEFAULT_HAZARD_PRIORITY
        )
        score = base_score

        # Distance bonus: closer distance = higher hazard priority
        if tracked.estimated_distance_m <= 0.9:
            score += 25
        elif tracked.estimated_distance_m <= 1.8:
            score += 15
        elif tracked.estimated_distance_m <= 3.5:
            score += 5

        # Confidence bonus
        if tracked.last_confidence > 0.9:
            score += 10
        elif tracked.last_confidence > 0.8:
            score += 5

        # Path overlap & Spatial region
        region = tracked.last_region
        if region == "center":
            if tracked.last_bbox and len(tracked.last_bbox) == 4:
                x1, y1, x2, y2 = tracked.last_bbox
                corridor_left = int(frame_w * self.config.PATH_LEFT_RATIO)
                corridor_right = int(frame_w * self.config.PATH_RIGHT_RATIO)
                bbox_w = max(1, x2 - x1)

                overlap_start = max(x1, corridor_left)
                overlap_end = min(x2, corridor_right)
                overlap = max(0, overlap_end - overlap_start)
                overlap_ratio = overlap / bbox_w
                score += int(20 * overlap_ratio)

                # Vertical weight (lower bbox bottom -> closer to camera)
                if frame_h > 0:
                    y2_ratio = y2 / frame_h
                    if y2_ratio > 0.75:
                        score += 15
                    elif y2_ratio > 0.55:
                        score += 10
        else:
            # Lateral awareness
            score = int(score * 0.70)
            if tracked.last_bbox and len(tracked.last_bbox) == 4:
                x1, y1, x2, y2 = tracked.last_bbox
                if frame_h > 0:
                    y2_ratio = y2 / frame_h
                    if y2_ratio > 0.75:
                        score += 10
                    elif y2_ratio > 0.55:
                        score += 5

        # Size weight (larger area = closer)
        score += int(min(tracked.last_area_ratio * 120, 25))

        # Persistence
        if tracked.frames_seen > self.config.PERSISTENCE_FRAMES:
            score += 5

        # Approach
        if self._is_approaching(tracked):
            score += 10

        # Map score -> severity
        if score >= 80:
            severity = SEVERITY_CRITICAL
        elif score >= 50:
            severity = SEVERITY_WARNING
        elif score >= 25:
            severity = SEVERITY_CAUTION
        else:
            severity = SEVERITY_INFO

        return score, severity

    # ------------------------------------------------------------------
    # State & Proximity Determination (Multi-Tier with Hysteresis)
    # ------------------------------------------------------------------
    def _determine_state_and_proximity(
        self, tracked: TrackedObject, score: int, severity: str
    ) -> Tuple[str, str, bool]:
        """
        Transition hazard state and proximity using multi-tier critical zones and hysteresis:
        - CRITICAL_2 (Immediate Danger): <= 0.9m in center corridor
        - CRITICAL_1 (Danger Zone Entry): <= 1.8m in center corridor
        - APPROACHING (In-Frame Advisory): 1.8m - 3.5m
        - FAR: > 3.5m
        """
        danger_dist = getattr(self.config, "DANGER_ZONE_DISTANCE_M", 1.8)
        crit_dist = getattr(self.config, "CRITICAL_DISTANCE_M", 0.9)
        in_frame_dist = getattr(self.config, "IN_FRAME_MAX_DISTANCE_M", 3.5)
        hysteresis = getattr(self.config, "ZONE_HYSTERESIS_M", 0.3)

        critical_area = self.config.CRITICAL_AREA_RATIO
        close_area = (
            self.config.CLOSE_HAZARD_AREA_RATIO
            if tracked.last_region == "center"
            else getattr(self.config, "SIDE_CLOSE_AREA_RATIO", 0.045)
        )

        d = tracked.estimated_distance_m
        in_center = (tracked.last_region == "center")
        old_state = tracked.state

        # Proximity classification
        if d <= crit_dist or tracked.last_area_ratio >= critical_area:
            proximity = "CRITICAL"
        elif d <= danger_dist or tracked.last_area_ratio >= close_area:
            proximity = "CLOSE"
        elif d <= in_frame_dist:
            proximity = "MEDIUM"
        else:
            proximity = "FAR"

        tracked.proximity = proximity

        if proximity == "FAR":
            new_state = STATE_FAR
            tracked.critical_tier = 0
            return new_state, proximity, (old_state != STATE_FAR)

        # Persistence check
        if tracked.frames_seen < self.config.PERSISTENCE_FRAMES:
            return STATE_NEW, proximity, (old_state != STATE_NEW)

        # State transition logic with hysteresis:
        if in_center:
            if d <= crit_dist or tracked.last_area_ratio >= critical_area:
                new_state = STATE_CRITICAL_2
            elif d <= danger_dist or tracked.last_area_ratio >= close_area:
                # If currently in CRITICAL_2, apply hysteresis before stepping down to CRITICAL_1
                if old_state == STATE_CRITICAL_2 and d < (crit_dist + 0.2):
                    new_state = STATE_CRITICAL_2
                else:
                    new_state = STATE_CRITICAL_1
            elif d <= in_frame_dist:
                # If previously in a critical state, apply hysteresis before leaving critical zone
                if old_state in (STATE_CRITICAL_1, STATE_CRITICAL_2, STATE_CRITICAL) and d < (danger_dist + hysteresis):
                    new_state = old_state
                else:
                    new_state = STATE_APPROACHING if self._is_approaching(tracked) else STATE_STABLE
                    # Reset critical tier upon clean recovery
                    if old_state in (STATE_CRITICAL_1, STATE_CRITICAL_2, STATE_CRITICAL):
                        tracked.critical_tier = 0
            else:
                new_state = STATE_FAR
                tracked.critical_tier = 0
        else:
            # Side / lateral obstacle
            if d <= in_frame_dist:
                new_state = STATE_APPROACHING if self._is_approaching(tracked) else STATE_STABLE
            else:
                new_state = STATE_FAR
            tracked.critical_tier = 0

        state_changed = (new_state != old_state)
        tracked.state = new_state
        return new_state, proximity, state_changed

    # ------------------------------------------------------------------
    # Approach heuristic
    # ------------------------------------------------------------------
    def _is_approaching(self, tracked: TrackedObject) -> bool:
        """Returns True if object bounding box is expanding over time."""
        hist = tracked.area_history
        if len(hist) < 3:
            return False
        items = list(hist)
        early_avg = sum(items[:2]) / 2.0
        recent_avg = sum(items[-2:]) / 2.0
        if early_avg <= 0:
            return False
        return (recent_avg / early_avg) > self.config.APPROACH_GROWTH_THRESHOLD

    # ------------------------------------------------------------------
    # Scene signature & Multi-Object Natural Description with Relative Positions
    # ------------------------------------------------------------------
    def _build_scene_signature(self) -> Tuple[Tuple[Tuple[str, str], int], ...]:
        """
        Build a canonical hashable signature of all active objects in view:
        ((class_name, region), count).
        """
        counts: Dict[Tuple[str, str], int] = Counter()
        for t in self._tracked.values():
            if t.proximity in ("CLOSE", "CRITICAL", "MEDIUM") and t.state not in (STATE_NEW, STATE_FAR):
                counts[(t.class_name, t.last_region)] += 1
        return tuple(sorted(counts.items()))

    def _format_scene_description(self, sig: Tuple[Tuple[Tuple[str, str], int], ...]) -> str:
        """
        Generate a natural, spoken scene description dynamically from all currently detected objects.
        Examples:
        - "Person detected, 2 meters away."
        - "Person and cell phone detected, 1.8 meters away."
        - "Person on the right and cell phone on the left."
        - "Person on the left and cell phone on the right."
        - "Person ahead and chair on the left."
        - "Path is clear."
        """
        if not sig:
            return "Path is clear."

        left_items: Dict[str, Tuple[int, List[float]]] = {}
        center_items: Dict[str, Tuple[int, List[float]]] = {}
        right_items: Dict[str, Tuple[int, List[float]]] = {}

        for t in self._tracked.values():
            if t.proximity in ("CLOSE", "CRITICAL", "MEDIUM") and t.state not in (STATE_NEW, STATE_FAR):
                reg = t.last_region
                cls = t.class_name
                d = t.estimated_distance_m
                target_dict = left_items if reg == "left" else (right_items if reg == "right" else center_items)
                if cls not in target_dict:
                    target_dict[cls] = [1, [d]]
                else:
                    target_dict[cls][0] += 1
                    target_dict[cls][1].append(d)

        def _format_items_phrase(items: Dict[str, Tuple[int, List[float]]]) -> str:
            phrases = []
            sorted_items = sorted(
                items.items(),
                key=lambda x: (0 if x[0].lower() == "person" else 1, x[0]),
            )
            for cls_name, (count, dists) in sorted_items:
                if count > 1:
                    phrases.append(f"{count} {pluralize(cls_name)}")
                else:
                    phrases.append(cls_name)
            return format_item_list(phrases)

        active_regions = [r for r, d in [("center", center_items), ("left", left_items), ("right", right_items)] if d]
        if not active_regions:
            return "Path is clear."

        # Case 1: All objects in a single region
        if len(active_regions) == 1:
            r = active_regions[0]
            items_dict = center_items if r == "center" else (left_items if r == "left" else right_items)
            items_str = _format_items_phrase(items_dict)
            all_dists = [d for v in items_dict.values() for d in v[1]]
            avg_dist = sum(all_dists) / len(all_dists) if all_dists else None
            dist_str = format_distance_str(avg_dist)
            dist_suffix = f", {dist_str}" if dist_str else ""

            if r == "center":
                res = f"{items_str} detected{dist_suffix}"
            elif r == "left":
                res = f"{items_str} on the left{dist_suffix}"
            else:
                res = f"{items_str} on the right{dist_suffix}"

        # Case 2: Objects distributed across multiple regions (e.g. Person on right and cell phone on left)
        else:
            region_clauses = []
            # Order: person first if present, or center -> right -> left
            region_order = ["center", "right", "left"]
            if "person" in left_items and "person" not in right_items and "person" not in center_items:
                region_order = ["left", "right", "center"]

            for r in region_order:
                items_dict = center_items if r == "center" else (left_items if r == "left" else right_items)
                if not items_dict:
                    continue
                items_str = _format_items_phrase(items_dict)
                all_dists = [d for v in items_dict.values() for d in v[1]]
                avg_dist = sum(all_dists) / len(all_dists) if all_dists else None
                dist_str = format_distance_str(avg_dist)
                dist_suffix = f", {dist_str}" if dist_str else ""

                if r == "center":
                    region_clauses.append(f"{items_str} ahead{dist_suffix}")
                elif r == "left":
                    region_clauses.append(f"{items_str} on the left{dist_suffix}")
                else:
                    region_clauses.append(f"{items_str} on the right{dist_suffix}")

            if len(region_clauses) == 2:
                res = f"{region_clauses[0]} and {region_clauses[1]}"
            else:
                res = f"{region_clauses[0]}, {region_clauses[1]}, and {region_clauses[2]}"

        res = res.strip()
        if not res[0].isupper():
            res = res[0].upper() + res[1:]
        if not res.endswith("."):
            res += "."
        return res

    def _format_path_obstruction(self, tracks: List[TrackedObject], tier: int = 1) -> str:
        """
        Generate an urgent path-obstruction warning for one or multiple objects
        in the Danger Zone (tier 1: <= 1.8m, tier 2: <= 0.9m) in the center walking corridor.
        Examples:
        - Tier 1: "Stop! Person and cell phone detected, 1.4 meters away."
        - Tier 2: "Stop! Person is very close, 0.7 meters away. Please stop!"
        """
        if not tracks:
            return "Stop! Obstacle detected nearby."

        counts = Counter(t.class_name for t in tracks)
        total = len(tracks)
        min_dist = min(t.estimated_distance_m for t in tracks)
        dist_str = format_distance_str(min_dist)
        dist_suffix = f", {dist_str}" if dist_str else ""

        if total >= 5:
            if tier >= 2:
                return f"Stop! Multiple obstacles are very close{dist_suffix}. Please stop!"
            return f"Stop! Multiple obstacles detected{dist_suffix}."

        phrases = []
        sorted_counts = sorted(
            counts.items(),
            key=lambda x: (0 if x[0].lower() == "person" else 1, x[0]),
        )
        for cls_name, count in sorted_counts:
            if count > 1:
                phrases.append(f"{count} {pluralize(cls_name)}")
            else:
                phrases.append(cls_name)

        if phrases:
            phrases[0] = phrases[0].capitalize()
            for idx in range(1, len(phrases)):
                if not any(phrases[idx].startswith(f"{n} ") for n in range(2, 20)) and not phrases[idx].startswith("multiple "):
                    phrases[idx] = phrases[idx].lower()

        subject = format_item_list(phrases)
        if not subject[0].isupper():
            subject = subject[0].upper() + subject[1:]

        if tier >= 2:
            return f"Stop! {subject} is very close{dist_suffix}. Please stop!"
        return f"Stop! {subject} detected{dist_suffix}."

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        detections: List[Dict],
        current_time: Optional[float] = None,
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> HazardEvent:
        """
        Process one frame's detections and return a :class:`HazardEvent`.
        """
        if current_time is None:
            current_time = time.time()
        self._frame_count += 1
        vfov = getattr(self.config, "CAMERA_VFOV_DEG", 48.0)

        # --- 1. Spatial Matching & Track Update -------------------------
        pairs, unmatched_tracks, unmatched_dets = self._match_detections_to_tracks(
            detections, frame_w, frame_h
        )

        # Update matched tracks
        for tid, d_idx in pairs:
            det = detections[d_idx]
            t = self._tracked[tid]
            t.frames_seen += 1
            t.frames_missing = 0
            t.last_confidence = det.get("confidence", 0.0)
            t.last_area_ratio = det.get("area_ratio", 0.0)

            # Smooth bounding box with deadband to eliminate stationary jitter
            new_bbox = det.get("bbox", [])
            t.last_bbox = smooth_bbox(t.last_bbox, new_bbox, deadband_px=4)

            # Smooth center coordinates
            if t.last_bbox and len(t.last_bbox) == 4:
                t.last_center = [
                    t.last_bbox[0] + (t.last_bbox[2] - t.last_bbox[0]) // 2,
                    t.last_bbox[1] + (t.last_bbox[3] - t.last_bbox[1]) // 2,
                ]

            # Region classification with spatial hysteresis
            t.last_region = classify_region_with_hysteresis(
                t.last_center[0] if t.last_center else 320,
                frame_w,
                t.last_region,
                left_ratio=self.config.PATH_LEFT_RATIO,
                right_ratio=self.config.PATH_RIGHT_RATIO,
            )

            # Temporal class smoothing (majority voting)
            raw_cls = det.get("class_name", "object")
            t.class_history.append((raw_cls, t.last_confidence))
            t.class_name = self._resolve_class_name(t)
            t.track_key = f"{t.class_name}_{t.track_id}"
            t.area_history.append(t.last_area_ratio)

            # Metric distance estimation with deadband smoothing
            dist = det.get("distance_m")
            if dist is None:
                dist = estimate_object_distance(t.class_name, t.last_bbox, frame_h, frame_w, vfov_deg=vfov)
            t.estimated_distance_m = smooth_distance(t.estimated_distance_m, dist, deadband_m=0.15)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            tid = self._next_track_id
            self._next_track_id += 1
            raw_cls = det.get("class_name", "object")
            conf = det.get("confidence", 0.0)
            area = det.get("area_ratio", 0.0)
            bbox = det.get("bbox", [])

            dist = det.get("distance_m")
            if dist is None:
                dist = estimate_object_distance(raw_cls, bbox, frame_h, frame_w, vfov_deg=vfov)

            t = TrackedObject(
                track_id=tid,
                class_name=raw_cls,
                track_key=f"{raw_cls}_{tid}",
                estimated_distance_m=dist,
                frames_seen=1,
                last_confidence=conf,
                last_area_ratio=area,
                last_bbox=bbox,
                last_center=det.get("center", [0, 0]),
                last_region=det.get("region", "center"),
            )
            t.class_history.append((raw_cls, conf))
            t.area_history.append(area)
            self._tracked[tid] = t

        # Update unmatched missing tracks
        remove_ids: List[int] = []
        disappear_grace = getattr(self.config, "DISAPPEAR_GRACE_FRAMES", 15)
        for tid in unmatched_tracks:
            t = self._tracked[tid]
            t.frames_missing += 1
            if t.frames_missing > disappear_grace:
                t.state = STATE_CLEARED
                remove_ids.append(tid)

        for tid in remove_ids:
            del self._tracked[tid]
            if self._focus_key == tid:
                self._focus_key = None

        # --- 2. Score & Determine Proximity/State for All Tracks --------
        hazards: List[Dict] = []
        best_score = -1
        best_tid: Optional[int] = None

        for tid, t in self._tracked.items():
            if t.last_area_ratio < self.config.MIN_HAZARD_AREA_RATIO:
                continue

            score, severity = self._compute_hazard_score(t, frame_w, frame_h)
            t.priority_score = score

            new_state, proximity, changed = self._determine_state_and_proximity(t, score, severity)
            t.proximity = proximity
            if changed:
                t.state = new_state
                t.last_state_change_time = current_time

            # Center walking corridor tracking
            if t.last_region == "center" and t.proximity in ("CLOSE", "CRITICAL"):
                t.frames_in_center += 1
                t.frames_outside_center = 0
            else:
                t.frames_outside_center += 1
                if t.frames_outside_center > 15:
                    t.has_announced_obstruction = False

            hazards.append(
                {
                    "track_key": t.track_key,
                    "class_name": t.class_name,
                    "region": t.last_region,
                    "proximity": t.proximity,
                    "distance_m": t.estimated_distance_m,
                    "score": score,
                    "severity": severity,
                    "state": t.state,
                    "bbox": t.last_bbox,
                    "area_ratio": t.last_area_ratio,
                }
            )

            # Focus scoring: prefer Danger Zone objects
            effective_score = score + (30 if proximity in ("CLOSE", "CRITICAL") else (10 if proximity == "MEDIUM" else 0))
            if effective_score > best_score:
                best_score = effective_score
                best_tid = tid

        # --- 3. Sticky Focus Selection ---------------------------------
        if best_tid is not None:
            if self._focus_key is not None and self._focus_key in self._tracked:
                cur_obj = self._tracked[self._focus_key]
                cur_effective = cur_obj.priority_score + (30 if cur_obj.proximity in ("CLOSE", "CRITICAL") else (10 if cur_obj.proximity == "MEDIUM" else 0))
                if best_score > cur_effective + 15:
                    self._focus_key = best_tid
            else:
                self._focus_key = best_tid
        elif not hazards:
            self._focus_key = None

        # --- 4. Build HazardEvent --------------------------------------
        event = HazardEvent(all_hazards=hazards)

        if self._focus_key is not None and self._focus_key in self._tracked:
            fobj = self._tracked[self._focus_key]
            score, severity = self._compute_hazard_score(fobj, frame_w, frame_h)

            region_str = f"({fobj.last_region})" if fobj.last_region != "center" else "ahead"
            dist_str = f"[{fobj.estimated_distance_m:.1f}m]"
            event.hazard_detected = True
            event.focus_object = f"{fobj.class_name} {dist_str} [{fobj.proximity}] {region_str}"
            event.focus_severity = severity
            event.focus_state = fobj.state
            event.focus_proximity = fobj.proximity
            event.category = fobj.class_name
            event.dedupe_key = fobj.track_key
            event.active_hazard = {
                "class_name": fobj.class_name,
                "region": fobj.last_region,
                "proximity": fobj.proximity,
                "distance_m": fobj.estimated_distance_m,
                "bbox": fobj.last_bbox,
                "area_ratio": fobj.last_area_ratio,
                "severity": severity,
                "score": score,
                "state": fobj.state,
            }

        # --- 5. Event-Driven Multi-Tier Critical & Scene Speech Dispatch ---
        current_sig = self._build_scene_signature()
        elapsed_since_scene = current_time - self._last_scene_speak_time

        crit1_thresh = getattr(self.config, "DANGER_ZONE_DISTANCE_M", 1.8)
        crit2_thresh = getattr(self.config, "CRITICAL_DISTANCE_M", 0.9)
        obstruction_cooldown = getattr(self.config, "PATH_OBSTRUCTION_COOLDOWN", 2.0)
        scene_cooldown = getattr(self.config, "SCENE_CHANGE_COOLDOWN", 3.0)

        # Check for unannounced Critical Tier 2 (<= 0.9m) tracks
        new_crit2_tracks: List[TrackedObject] = []
        # Check for unannounced Critical Tier 1 (<= 1.8m) tracks
        new_crit1_tracks: List[TrackedObject] = []

        for t in self._tracked.values():
            if t.last_region == "center" and t.state not in (STATE_NEW, STATE_FAR):
                if (t.estimated_distance_m <= crit2_thresh or t.state == STATE_CRITICAL_2) and t.critical_tier < 2:
                    new_crit2_tracks.append(t)
                elif (t.estimated_distance_m <= crit1_thresh or t.state == STATE_CRITICAL_1) and t.critical_tier < 1:
                    new_crit1_tracks.append(t)

        # Check for genuine new unannounced tracks in scene
        has_new_unannounced = any(
            t.state in (STATE_STABLE, STATE_APPROACHING) and not t.has_announced_presence
            for t in self._tracked.values()
        )

        # Check for genuine significant distance shift (>= 0.8m) on existing tracks
        has_distance_shift = any(
            t.has_announced_presence
            and t.last_announced_distance_m is not None
            and abs(t.estimated_distance_m - t.last_announced_distance_m) >= 0.8
            and (current_time - t.last_distance_announced_time) >= 3.0
            for t in self._tracked.values()
        )

        # Check for genuine region shift
        has_region_shift = any(
            t.has_announced_presence
            and t.last_announced_region != ""
            and t.last_region != t.last_announced_region
            for t in self._tracked.values()
        )

        # Scene change detection compared to what was ACTUALLY LAST SPOKEN
        sig_changed = (current_sig != self._last_spoken_scene_signature) and bool(current_sig)
        cleared_changed = (not current_sig) and (self._last_spoken_scene_signature is not None)

        # Check for genuine significant distance shift (>= 0.8m) on existing tracks
        has_distance_shift = any(
            t.has_announced_presence
            and t.last_announced_distance_m is not None
            and abs(t.estimated_distance_m - t.last_announced_distance_m) >= 0.8
            and (current_time - t.last_distance_announced_time) >= 2.5
            for t in self._tracked.values()
        )

        # Priority 1: Critical Tier 2 (Immediate Danger <= 0.9m)
        if new_crit2_tracks and elapsed_since_scene >= 1.0:
            all_crit2 = [
                t for t in self._tracked.values()
                if t.last_region == "center" and t.state not in (STATE_NEW, STATE_FAR)
                and (t.estimated_distance_m <= crit2_thresh or t.state == STATE_CRITICAL_2)
            ]
            tracks_to_format = all_crit2 if all_crit2 else new_crit2_tracks
            text = self._format_path_obstruction(tracks_to_format, tier=2)
            event.warning_text = text
            event.severity = SEVERITY_CRITICAL
            event.speak_priority = 100
            event.dedupe_key = f"crit2_{time.time()}"
            event.category = "obstruction"
            for t in tracks_to_format:
                t.critical_tier = 2
                t.last_announced_tier = 2
                t.has_announced_obstruction = True
                t.has_announced_presence = True
                t.critical_announced_time = current_time
                t.last_announced_distance_m = t.estimated_distance_m
                t.last_distance_announced_time = current_time
                t.last_announced_region = t.last_region
            self._last_scene_speak_time = current_time
            self._last_spoken_warning = text
            self._last_spoken_scene_signature = current_sig
            self._last_spoken_scene_text = text
            self._last_had_objects = True
            logger.info("[CRITICAL TIER 2 - IMMEDIATE DANGER] %s", text)

        # Priority 2: Critical Tier 1 (Danger Zone Entry <= 1.8m)
        elif new_crit1_tracks and elapsed_since_scene >= obstruction_cooldown:
            all_crit1 = [
                t for t in self._tracked.values()
                if t.last_region == "center" and t.state not in (STATE_NEW, STATE_FAR)
                and (t.estimated_distance_m <= crit1_thresh or t.state in (STATE_CRITICAL_1, STATE_CRITICAL_2))
            ]
            tracks_to_format = all_crit1 if all_crit1 else new_crit1_tracks
            text = self._format_path_obstruction(tracks_to_format, tier=1)
            event.warning_text = text
            event.severity = SEVERITY_CRITICAL
            event.speak_priority = 90
            event.dedupe_key = f"crit1_{time.time()}"
            event.category = "obstruction"
            for t in tracks_to_format:
                t.critical_tier = 1
                t.last_announced_tier = 1
                t.has_announced_obstruction = True
                t.has_announced_presence = True
                t.critical_announced_time = current_time
                t.last_announced_distance_m = t.estimated_distance_m
                t.last_distance_announced_time = current_time
                t.last_announced_region = t.last_region
            self._last_scene_speak_time = current_time
            self._last_spoken_warning = text
            self._last_spoken_scene_signature = current_sig
            self._last_spoken_scene_text = text
            self._last_had_objects = True
            logger.info("[CRITICAL TIER 1 - DANGER ZONE ENTRY] %s", text)

        # Priority 3: Dynamic Scene Description (Speaks whenever scene content changes or distance shifts)
        elif (sig_changed or has_distance_shift) and elapsed_since_scene >= scene_cooldown:
            if current_sig:
                text = self._format_scene_description(current_sig)
                event.warning_text = text
                event.severity = SEVERITY_WARNING
                event.speak_priority = 60
                event.dedupe_key = f"scene_{time.time()}"
                event.category = "scene"
                self._last_spoken_warning = text
                self._last_spoken_scene_signature = current_sig
                self._last_spoken_scene_text = text
                self._last_scene_speak_time = current_time
                self._last_had_objects = True
                for t in self._tracked.values():
                    if t.proximity in ("MEDIUM", "CLOSE", "CRITICAL") and t.state not in (STATE_NEW, STATE_FAR):
                        t.has_announced_presence = True
                        t.last_announced_distance_m = t.estimated_distance_m
                        t.last_distance_announced_time = current_time
                        t.last_announced_region = t.last_region
                logger.info("[SCENE ANNOUNCEMENT] %s", text)

        # Priority 4: Scene Cleared (when all objects leave view)
        elif cleared_changed and elapsed_since_scene >= scene_cooldown:
            text = "Path is clear."
            event.warning_text = text
            event.severity = SEVERITY_INFO
            event.speak_priority = 40
            event.dedupe_key = f"clear_{time.time()}"
            event.category = "scene"
            self._last_spoken_warning = text
            self._last_spoken_scene_signature = None
            self._last_spoken_scene_text = text
            self._last_had_objects = False
            self._last_scene_speak_time = current_time
            logger.info("[SCENE CLEARED] %s", text)

        return event

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_focus_info(self) -> Dict:
        """Return current focus object info for the HUD."""
        if self._focus_key is None or self._focus_key not in self._tracked:
            return {}
        f = self._tracked[self._focus_key]
        return {
            "track_key": f.track_key,
            "class_name": f.class_name,
            "region": f.last_region,
            "proximity": f.proximity,
            "distance_m": f.estimated_distance_m,
            "state": f.state,
            "score": f.priority_score,
            "bbox": f.last_bbox,
        }

    @property
    def last_spoken_warning(self) -> Optional[str]:
        return self._last_spoken_warning

    def reset(self) -> None:
        """Clear all tracking state."""
        self._tracked.clear()
        self._next_track_id = 1
        self._focus_key = None
        self._last_spoken_warning = None
        self._frame_count = 0
        self._last_spoken_scene_signature = None
        self._last_spoken_scene_text = None
        self._last_scene_speak_time = 0.0
        self._last_had_objects = False
        logger.info("Decision engine reset")
