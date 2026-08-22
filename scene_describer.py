"""
scene_describer.py - AI Scene Describer & Spatial Environment Understander
===========================================================================
Generates natural, human-like spatial scene descriptions tailored specifically
for blind navigation and situational awareness.

Combines real-time object spatial locations, depth clearance, metric distance,
and multi-object natural language aggregation into coherent contextual descriptions.
"""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from config import Config, DEFAULT_CONFIG
from logger import get_logger

logger = get_logger(__name__)


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


class SceneDescriber:
    """
    Synthesizes rich, structured situational awareness descriptions of the scene.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config

    def describe_scene(
        self,
        frame: Optional[np.ndarray],
        detections: List[Dict],
        depth_info: Dict,
    ) -> str:
        """
        Generate a comprehensive, natural language scene summary.

        Args:
            frame: Current camera BGR frame.
            detections: List of detections from OpenVINODetector.
            depth_info: Dictionary from MonocularDepthEstimator.

        Returns:
            Fluent, concise descriptive sentence for TTS delivery.
        """
        sentences = []

        # 1. Floor & Navigation Path Status
        clearance = depth_info.get("clearance_ratio", 1.0)
        drop_off = depth_info.get("drop_off_detected", False)

        if drop_off:
            sentences.append("Caution: Step drop-off or stairs detected directly in your walking path.")
        elif clearance > 0.75:
            sentences.append("Your walking path is open and clear ahead.")
        elif clearance > 0.40:
            sentences.append("Your walking path is partially clear with obstacles nearby.")
        else:
            sentences.append("Your walking path is obstructed ahead.")

        # 2. Object Spatial Relationships (Grouped by Left, Center, Right)
        if detections:
            regions_data: Dict[str, List[Dict]] = {"center": [], "left": [], "right": []}
            for d in detections:
                r = d.get("region", "center")
                if r in regions_data:
                    regions_data[r].append(d)

            obj_descriptions = []

            for region_key, label_prefix in [
                ("center", "Straight ahead"),
                ("left", "On your left"),
                ("right", "On your right"),
            ]:
                dets = regions_data[region_key]
                if not dets:
                    continue

                # Count classes and average distances
                counts = Counter(d.get("class_name", "object") for d in dets)
                distances: Dict[str, List[float]] = {}
                for d in dets:
                    cname = d.get("class_name", "object")
                    dm = d.get("distance_m")
                    if dm is not None:
                        distances.setdefault(cname, []).append(dm)

                region_phrases = []
                for cname, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                    avg_dist = np.mean(distances[cname]) if cname in distances and distances[cname] else None

                    if avg_dist is not None:
                        if avg_dist <= 1.2:
                            dist_desc = "very close"
                        elif avg_dist <= 3.0:
                            dist_desc = f"about {avg_dist:.1f} meters away"
                        else:
                            dist_desc = f"about {int(round(avg_dist))} meters away"
                    else:
                        dist_desc = ""

                    dist_suffix = f" ({dist_desc})" if dist_desc else ""

                    if count >= 4:
                        region_phrases.append(f"a lot of {pluralize(cname)}{dist_suffix}")
                    elif count >= 2:
                        region_phrases.append(f"{count} {pluralize(cname)}{dist_suffix}")
                    else:
                        region_phrases.append(f"{cname}{dist_suffix}")

                if region_phrases:
                    combined = format_item_list(region_phrases[:3])
                    obj_descriptions.append(f"{label_prefix}: {combined}")

            if obj_descriptions:
                sentences.append(". ".join(obj_descriptions) + ".")
        else:
            sentences.append("No specific objects identified in immediate view.")

        full_description = " ".join(sentences)
        logger.info("[SCENE SUMMARY] %s", full_description)
        return full_description
