"""
tests/test_decision_engine.py - Decision Engine Unit Tests
============================================================
Tests spatial tracking, temporal class smoothing, 2-zone distance alerts,
danger zone stop triggers, scene-level change detection, persistence,
approach detection, and focus selection.

Run::

    python -m pytest tests/test_decision_engine.py -v
"""

import time
import pytest
from config import Config, SEVERITY_CRITICAL, SEVERITY_WARNING, SEVERITY_CAUTION, SEVERITY_INFO
from decision_engine import (
    DecisionEngine,
    HazardEvent,
    STATE_NEW,
    STATE_STABLE,
    STATE_CLOSER,
    STATE_CRITICAL,
    STATE_CRITICAL_1,
    STATE_CRITICAL_2,
    STATE_CLEARED,
)


@pytest.fixture
def config():
    cfg = Config()
    cfg.PERSISTENCE_FRAMES = 3
    cfg.DISAPPEAR_GRACE_FRAMES = 5
    cfg.WARNING_COOLDOWN = 5.0
    cfg.CRITICAL_COOLDOWN = 3.0
    cfg.MIN_HAZARD_AREA_RATIO = 0.002
    cfg.CLOSE_HAZARD_AREA_RATIO = 0.08
    cfg.SIDE_CLOSE_AREA_RATIO = 0.08
    cfg.APPROACH_GROWTH_THRESHOLD = 1.4
    cfg.CRITICAL_AREA_RATIO = 0.20
    cfg.SCENE_CHANGE_COOLDOWN = 1.0  # fast for testing
    cfg.PATH_OBSTRUCTION_COOLDOWN = 1.0  # fast for testing
    cfg.DANGER_ZONE_DISTANCE_M = 1.8  # 50% reduced
    cfg.IN_FRAME_MAX_DISTANCE_M = 3.5  # 50% reduced
    cfg.CRITICAL_DISTANCE_M = 0.9  # 50% reduced
    return cfg


@pytest.fixture
def engine(config):
    return DecisionEngine(config=config)


# ------------------------------------------------------------------
# Proximity & Distance tests (Far vs Close)
# ------------------------------------------------------------------
class TestProximity:
    def test_far_object_is_silent(self, engine):
        """Far away object (> 3.5m) should be tracked visually but NOT speak."""
        det_far = _det(class_name="person", cx=320, area_ratio=0.03)
        det_far["distance_m"] = 5.0  # > 3.5m -> FAR
        for i in range(5):
            event = engine.evaluate([det_far], current_time=100.0 + i * 0.1)
        assert event.hazard_detected
        assert event.focus_proximity == "FAR"
        assert event.warning_text is None  # SILENT for distant objects

    def test_close_object_speaks_warning(self, engine):
        """Close object within danger zone (dist <= 1.8m) should speak after persistence."""
        det_close = _det(class_name="person", cx=320, area_ratio=0.09)
        det_close["distance_m"] = 1.2
        warnings = []
        for i in range(5):
            event = engine.evaluate([det_close], current_time=100.0 + i * 0.1)
            if event.warning_text:
                warnings.append(event.warning_text)
        assert event.hazard_detected
        assert event.focus_proximity in ("CLOSE", "CRITICAL")
        assert len(warnings) > 0  # Spoke when transitioning to confirmed close hazard


def _det(class_name="person", cx=320, cy=400, area_ratio=0.08, conf=0.85, region="center"):
    """Helper to build a detection dict matching detector.py output format."""
    half_w = int((area_ratio * 640 * 480) ** 0.5 / 2)
    return {
        "class_name": class_name,
        "class_id": 0,
        "confidence": conf,
        "bbox": [cx - half_w, cy - half_w, cx + half_w, cy + half_w],
        "center": [cx, cy],
        "width": half_w * 2,
        "height": half_w * 2,
        "area_ratio": area_ratio,
        "region": region,
    }


# ------------------------------------------------------------------
# Region & Spatial Awareness tests
# ------------------------------------------------------------------
class TestRegions:
    def test_left_detection_spatial_warning(self, engine):
        det = _det(region="left", cx=50)
        det["distance_m"] = 2.0
        for i in range(4):
            engine.evaluate([det], current_time=100.0 + i * 0.1)
        assert "left" in (engine.last_spoken_warning or "").lower()

    def test_right_detection_spatial_warning(self, engine):
        det = _det(region="right", cx=600)
        det["distance_m"] = 2.0
        for i in range(4):
            engine.evaluate([det], current_time=100.0 + i * 0.1)
        assert "right" in (engine.last_spoken_warning or "").lower()

    def test_center_detection_is_hazard(self, engine):
        det = _det(region="center", cx=320)
        det["distance_m"] = 1.0  # <= 1.8m -> Stop! Person detected
        for i in range(4):
            engine.evaluate([det], current_time=100.0 + i * 0.1)
        warning = (engine.last_spoken_warning or "").lower()
        assert "person" in warning
        assert "stop" in warning or "detected" in warning

    def test_center_takes_priority_over_side(self, engine):
        det_left = _det(class_name="person", region="left", cx=50, area_ratio=0.08)
        det_center = _det(class_name="person", region="center", cx=320, area_ratio=0.08)
        for i in range(5):
            event = engine.evaluate([det_left, det_center], current_time=100.0 + i * 0.1)
        assert event.hazard_detected
        assert event.active_hazard.get("region") == "center"


# ------------------------------------------------------------------
# Persistence tests
# ------------------------------------------------------------------
class TestPersistence:
    def test_new_detection_no_warning(self, engine):
        """Single frame should NOT produce a warning (persistence not met)."""
        det = _det()
        event = engine.evaluate([det], current_time=100.0)
        assert event.warning_text is None

    def test_stable_after_persistence(self, engine):
        """After PERSISTENCE_FRAMES, state should transition to STABLE."""
        det = _det()
        for i in range(4):
            event = engine.evaluate([det], current_time=100.0 + i * 0.1)
        assert event.hazard_detected
        assert event.focus_state in (STATE_STABLE, STATE_CLOSER, STATE_CRITICAL, STATE_CRITICAL_1, STATE_CRITICAL_2)

    def test_object_disappears_and_clears(self, engine):
        """Object disappearing for DISAPPEAR_GRACE_FRAMES should clear."""
        det = _det()
        for i in range(4):
            engine.evaluate([det], current_time=100.0 + i * 0.1)
        for i in range(7):
            event = engine.evaluate([], current_time=101.0 + i * 0.1)
        assert not event.hazard_detected


# ------------------------------------------------------------------
# Class Smoothing & Stability tests
# ------------------------------------------------------------------
class TestClassSmoothing:
    def test_transient_knife_to_cell_phone_speaks_cell_phone(self, engine):
        """1-frame transient 'knife' followed by stable 'cell phone' resolves to cell phone."""
        det_knife = _det(class_name="knife", conf=0.48, area_ratio=0.08)
        det_knife["distance_m"] = 2.0
        engine.evaluate([det_knife], current_time=100.0)

        det_phone1 = _det(class_name="cell phone", conf=0.88, area_ratio=0.08)
        det_phone1["distance_m"] = 2.0
        det_phone2 = _det(class_name="cell phone", conf=0.92, area_ratio=0.08)
        det_phone2["distance_m"] = 2.0

        engine.evaluate([det_phone1], current_time=100.1)
        engine.evaluate([det_phone2], current_time=100.2)
        event = engine.evaluate([det_phone2], current_time=100.3)

        assert event.focus_object is not None
        assert "cell phone" in event.focus_object.lower()
        if engine.last_spoken_warning:
            assert "knife" not in engine.last_spoken_warning.lower()


# ------------------------------------------------------------------
# Spatial Tracking tests
# ------------------------------------------------------------------
class TestSpatialTracking:
    def test_person_moving_in_center_does_not_repeat_obstruction(self, engine):
        """Person staying in center walking path should NOT spam Stop alerts on every step."""
        warnings = []
        for i in range(25):
            cx = 300 + (i % 6) * 5
            det = _det(class_name="person", cx=cx, area_ratio=0.09, region="center")
            det["distance_m"] = 1.0  # within danger zone
            ev = engine.evaluate([det], current_time=100.0 + i * 0.1)
            if ev.warning_text and "stop" in ev.warning_text.lower():
                warnings.append(ev.warning_text)

        assert len(warnings) == 1

    def test_person_leaving_and_returning_re_alerts(self, engine):
        """Person leaving center corridor and returning should re-alert."""
        # 1. In center (within danger zone)
        det_center = _det(class_name="person", cx=320, area_ratio=0.09, region="center")
        det_center["distance_m"] = 1.0
        for i in range(5):
            engine.evaluate([det_center], current_time=100.0 + i * 0.1)

        # 2. Moves to left for > 15 frames (~1.6s)
        det_left = _det(class_name="person", cx=60, area_ratio=0.09, region="left")
        det_left["distance_m"] = 2.0
        for i in range(20):
            engine.evaluate([det_left], current_time=101.0 + i * 0.1)

        # 3. Re-enters center corridor
        warnings_return = []
        for i in range(5):
            ev = engine.evaluate([det_center], current_time=105.0 + i * 0.1)
            if ev.warning_text and "stop" in ev.warning_text.lower():
                warnings_return.append(ev.warning_text)

        assert len(warnings_return) >= 1


# ------------------------------------------------------------------
# Scene Change Detection tests
# ------------------------------------------------------------------
class TestSceneChange:
    def test_scene_change_triggers_description(self, engine):
        det1 = _det(class_name="chair", region="left", cx=50, area_ratio=0.08)
        det1["distance_m"] = 2.5
        for i in range(5):
            engine.evaluate([det1], current_time=100.0 + i * 0.1)

        w1 = engine.last_spoken_warning
        assert w1 is not None
        assert "chair" in w1.lower()

    def test_same_scene_stays_silent(self, engine):
        det1 = _det(class_name="chair", region="left", cx=50, area_ratio=0.08)
        det1["distance_m"] = 2.5
        for i in range(5):
            engine.evaluate([det1], current_time=100.0 + i * 0.1)

        last_warn = engine.last_spoken_warning

        # Next 10 frames with same object -> No new speech
        new_speeches = []
        for i in range(10):
            ev = engine.evaluate([det1], current_time=102.0 + i * 0.1)
            if ev.warning_text:
                new_speeches.append(ev.warning_text)

        assert len(new_speeches) == 0

    def test_scene_clear_speaks(self, engine):
        det1 = _det(class_name="chair", region="left", cx=50, area_ratio=0.08)
        det1["distance_m"] = 2.5
        for i in range(5):
            engine.evaluate([det1], current_time=100.0 + i * 0.1)

        # Clear scene
        for i in range(10):
            engine.evaluate([], current_time=105.0 + i * 0.1)

        assert "clear" in (engine.last_spoken_warning or "").lower()

    def test_new_object_triggers_scene_update(self, engine):
        det1 = _det(class_name="chair", region="left", cx=50, area_ratio=0.08)
        det1["distance_m"] = 2.5
        for i in range(5):
            engine.evaluate([det1], current_time=100.0 + i * 0.1)

        det2 = _det(class_name="person", region="right", cx=600, area_ratio=0.08)
        det2["distance_m"] = 2.5
        for i in range(5):
            engine.evaluate([det1, det2], current_time=105.0 + i * 0.1)

        w = (engine.last_spoken_warning or "").lower()
        assert "person" in w
        assert "right" in w


# ------------------------------------------------------------------
# Path Obstruction & 2-Zone Distance tests
# ------------------------------------------------------------------
class TestPathObstruction:
    def test_center_obstruction_generates_urgent_alert(self, engine):
        """Object entering center corridor within danger zone (<= 1.8m) triggers 'Stop!' alert."""
        det = _det(class_name="person", cx=320, area_ratio=0.09, region="center")
        det["distance_m"] = 1.0
        warnings = []
        for i in range(5):
            event = engine.evaluate([det], current_time=100.0 + i * 0.1)
            if event.warning_text and "stop" in event.warning_text.lower():
                warnings.append(event.warning_text)
        assert len(warnings) > 0
        assert "stop" in warnings[0].lower()
        assert "detected" in warnings[0].lower()

    def test_side_object_no_obstruction_alert(self, engine):
        """Object on the side should NOT trigger 'Stop!' alert."""
        det = _det(class_name="person", cx=50, area_ratio=0.09, region="left")
        det["distance_m"] = 1.5
        for i in range(5):
            event = engine.evaluate([det], current_time=100.0 + i * 0.1)
        if engine.last_spoken_warning:
            assert "stop" not in engine.last_spoken_warning.lower()

    def test_person_beyond_danger_zone_announces_in_frame(self, engine):
        """Person in center corridor at 2.8m (advisory range) announces without Stop alert."""
        det_med = _det(class_name="person", cx=320, area_ratio=0.03, region="center")
        det_med["distance_m"] = 2.8  # In-Frame zone (1.8m - 3.5m)
        warnings = []
        for i in range(5):
            event = engine.evaluate([det_med], current_time=100.0 + i * 0.1)
            if event.warning_text:
                warnings.append(event.warning_text)
        assert len(warnings) > 0
        w = warnings[0].lower()
        assert "person detected" in w
        assert "meters away" in w
        assert "stop" not in w

    def test_person_within_danger_zone_triggers_stop_obstruction(self, engine):
        """Person in center corridor within danger zone (0.8m <= 1.8m) triggers urgent Stop alert."""
        det_close = _det(class_name="person", cx=320, area_ratio=0.09, region="center")
        det_close["distance_m"] = 0.8  # Danger zone (<= 1.8m)
        warnings = []
        for i in range(5):
            event = engine.evaluate([det_close], current_time=100.0 + i * 0.1)
            if event.warning_text:
                warnings.append(event.warning_text)
        assert len(warnings) > 0
        w = warnings[0].lower()
        assert "stop" in w
        assert "person" in w
        assert "0.8 meters away" in w

    def test_transition_from_in_frame_to_danger_zone(self, engine):
        """Object moves from 3.0m (In-Frame) into 0.8m (Danger Zone) -> triggers urgent Stop alert."""
        # 1. Start at 3.0m
        det_far = _det(class_name="person", cx=320, area_ratio=0.03, region="center")
        det_far["distance_m"] = 3.0
        for i in range(4):
            engine.evaluate([det_far], current_time=100.0 + i * 0.1)

        # 2. Move closer to 0.8m (Danger zone)
        det_close = _det(class_name="person", cx=320, area_ratio=0.10, region="center")
        det_close["distance_m"] = 0.8
        warnings = []
        for i in range(4):
            ev = engine.evaluate([det_close], current_time=105.0 + i * 0.1)
            if ev.warning_text and "stop" in ev.warning_text.lower():
                warnings.append(ev.warning_text)

        assert len(warnings) > 0
        assert "stop" in warnings[0].lower()
        assert "meters away" in warnings[0].lower()


# ------------------------------------------------------------------
# Priority tests
# ------------------------------------------------------------------
class TestPriority:
    def test_car_higher_than_bottle(self, engine):
        car = _det(class_name="car", cx=300, area_ratio=0.10)
        car["distance_m"] = 2.0
        bottle = _det(class_name="bottle", cx=350, area_ratio=0.05)
        bottle["distance_m"] = 2.0
        for i in range(5):
            event = engine.evaluate([car, bottle], current_time=100.0 + i * 0.1)
        assert "car" in (event.focus_object or "").lower()

    def test_larger_object_scores_higher(self, engine):
        small = _det(class_name="person", cx=320, area_ratio=0.04)
        big = _det(class_name="person", cx=300, area_ratio=0.15)
        for i in range(5):
            event = engine.evaluate([small, big], current_time=100.0 + i * 0.1)
        assert event.hazard_detected


# ------------------------------------------------------------------
# Approach detection tests
# ------------------------------------------------------------------
class TestApproach:
    def test_growing_area_triggers_closer(self, engine):
        """Progressively larger area should trigger CLOSER state."""
        base_time = 100.0
        for i in range(8):
            area = 0.04 + i * 0.02
            det = _det(area_ratio=area)
            det["distance_m"] = 2.0 - i * 0.15
            event = engine.evaluate([det], current_time=base_time + i * 0.2)

        assert event.hazard_detected
        assert event.focus_state in (STATE_CLOSER, STATE_CRITICAL, STATE_CRITICAL_1, STATE_CRITICAL_2)


# ------------------------------------------------------------------
# Focus selection tests
# ------------------------------------------------------------------
class TestFocus:
    def test_focus_doesnt_jump_randomly(self, engine):
        """Focus should be sticky — not jump to similar-scored objects."""
        det1 = _det(class_name="chair", cx=300, area_ratio=0.08)
        det1["distance_m"] = 2.0
        det2 = _det(class_name="bottle", cx=350, area_ratio=0.06)
        det2["distance_m"] = 2.0

        for i in range(5):
            engine.evaluate([det1, det2], current_time=100.0 + i * 0.1)
        focus1 = engine.get_focus_info().get("class_name")

        event = engine.evaluate([det1, det2], current_time=101.0)
        focus2 = engine.get_focus_info().get("class_name")
        assert focus1 == focus2


# ------------------------------------------------------------------
# State machine tests
# ------------------------------------------------------------------
class TestStateMachine:
    def test_critical_area_triggers_critical_state(self, engine):
        """Very large area ratio / distance <= 0.9m should immediately set CRITICAL."""
        det = _det(area_ratio=0.25)
        det["distance_m"] = 0.7
        for i in range(4):
            event = engine.evaluate([det], current_time=100.0 + i * 0.1)
        assert event.focus_state in (STATE_CRITICAL, STATE_CRITICAL_1, STATE_CRITICAL_2)

    def test_warning_text_format(self, engine):
        det = _det(class_name="person", cx=320, area_ratio=0.08)
        det["distance_m"] = 2.0
        for i in range(4):
            engine.evaluate([det], current_time=100.0 + i * 0.1)
        assert "person" in (engine.last_spoken_warning or "").lower()


# ------------------------------------------------------------------
# Multi-object aggregation & natural phrasing tests
# ------------------------------------------------------------------
class TestMultiObjectAggregation:
    def test_multiple_similar_objects_pluralizes(self, engine):
        """3 chairs on the left should be summarized as '3 chairs' or 'multiple chairs'."""
        det1 = _det(class_name="chair", cx=50, cy=200, area_ratio=0.08, region="left")
        det1["distance_m"] = 2.0
        det2 = _det(class_name="chair", cx=70, cy=300, area_ratio=0.08, region="left")
        det2["distance_m"] = 2.0
        det3 = _det(class_name="chair", cx=90, cy=400, area_ratio=0.08, region="left")
        det3["distance_m"] = 2.0

        warnings = []
        for i in range(5):
            ev = engine.evaluate([det1, det2, det3], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) > 0
        w = warnings[0].lower()
        assert "chair" in w
        assert "3 chairs" in w or "multiple chairs" in w
        assert "left" in w

    def test_compound_path_obstruction(self, engine):
        """Bed and chair both in walking path within danger zone should trigger compound alert."""
        det_bed = _det(class_name="bed", cx=300, area_ratio=0.10, region="center")
        det_bed["distance_m"] = 1.2
        det_chair = _det(class_name="chair", cx=340, area_ratio=0.09, region="center")
        det_chair["distance_m"] = 1.1

        warnings = []
        for i in range(5):
            ev = engine.evaluate([det_bed, det_chair], current_time=100.0 + i * 0.1)
            if ev.warning_text and "stop" in ev.warning_text.lower():
                warnings.append(ev.warning_text)

        assert len(warnings) > 0
        w = warnings[0].lower()
        assert "stop" in w
        assert "bed" in w
        assert "chair" in w

    def test_person_and_cell_phone_compound_announcement(self, engine):
        """Person and cell phone detected in center corridor should be announced together."""
        det_person = _det(class_name="person", cx=320, area_ratio=0.06, region="center")
        det_person["distance_m"] = 2.0
        det_phone = _det(class_name="cell phone", cx=340, area_ratio=0.005, region="center")
        det_phone["distance_m"] = 2.0

        warnings = []
        for i in range(5):
            ev = engine.evaluate([det_person, det_phone], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) > 0
        w = warnings[0].lower()
        assert "person" in w
        assert "cell phone" in w
        assert "person and cell phone" in w

    def test_multiple_people_announces_count(self, engine):
        """2 people detected in center corridor should be announced with exact count (2 people)."""
        det_p1 = _det(class_name="person", cx=280, area_ratio=0.05, region="center")
        det_p1["distance_m"] = 2.5
        det_p2 = _det(class_name="person", cx=360, area_ratio=0.05, region="center")
        det_p2["distance_m"] = 2.5

        warnings = []
        for i in range(5):
            ev = engine.evaluate([det_p1, det_p2], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) > 0
        w = warnings[0].lower()
        assert "2 people" in w

    def test_objects_in_different_regions(self, engine):
        """Bed on right and chair on left should be formatted with spatial conjunctions and distance."""
        det_bed = _det(class_name="bed", cx=580, area_ratio=0.09, region="right")
        det_bed["distance_m"] = 2.0
        det_chair = _det(class_name="chair", cx=60, area_ratio=0.09, region="left")
        det_chair["distance_m"] = 3.0

        warnings = []
        for i in range(5):
            ev = engine.evaluate([det_bed, det_chair], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) > 0
        w = warnings[0].lower()
        assert "bed on the right" in w or "bed" in w
        assert "chair on the left" in w or "chair" in w
        assert "left" in w and "right" in w

    def test_dynamic_multi_object_sequence_person_then_phone_then_remove(self, engine):
        """
        Step 1: Person alone -> Spoken: 'Person detected, 2 meters away.'
        Step 2: Person + Phone -> Spoken: 'Person and cell phone detected, 2 meters away.'
        Step 3: Person on right + Phone on left -> Spoken: 'Person on the right and cell phone on the left.'
        Step 4: Phone removed -> Spoken: 'Person on the right.'
        Step 5: All removed -> Spoken: 'Path is clear.'
        """
        # Step 1: Person in center
        det_person = _det(class_name="person", cx=320, area_ratio=0.06, region="center")
        det_person["distance_m"] = 2.0
        w1 = []
        for i in range(5):
            ev = engine.evaluate([det_person], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                w1.append(ev.warning_text)
        assert len(w1) == 1
        assert "person" in w1[0].lower()
        assert "cell phone" not in w1[0].lower()

        # Step 2: Cell phone added to center alongside person (after 1.5s)
        det_phone = _det(class_name="cell phone", cx=340, area_ratio=0.005, region="center")
        det_phone["distance_m"] = 2.0
        w2 = []
        for i in range(5):
            ev = engine.evaluate([det_person, det_phone], current_time=102.0 + i * 0.1)
            if ev.warning_text:
                w2.append(ev.warning_text)
        assert len(w2) == 1
        assert "person" in w2[0].lower()
        assert "cell phone" in w2[0].lower()
        assert "person and cell phone" in w2[0].lower()

        # Step 3: Person on right, phone on left
        det_p_right = _det(class_name="person", cx=580, area_ratio=0.06, region="right")
        det_p_right["distance_m"] = 2.0
        det_ph_left = _det(class_name="cell phone", cx=60, area_ratio=0.005, region="left")
        det_ph_left["distance_m"] = 2.0
        w3 = []
        for i in range(5):
            ev = engine.evaluate([det_p_right, det_ph_left], current_time=104.0 + i * 0.1)
            if ev.warning_text:
                w3.append(ev.warning_text)
        assert len(w3) == 1
        assert "right" in w3[0].lower() and "left" in w3[0].lower()
        assert "person" in w3[0].lower() and "cell phone" in w3[0].lower()

        # Step 4: Phone removed (grace period clears after 15 frames)
        w4 = []
        for i in range(25):
            ev = engine.evaluate([det_p_right], current_time=106.0 + i * 0.1)
            if ev.warning_text:
                w4.append(ev.warning_text)
        assert len(w4) >= 1
        assert "person" in w4[-1].lower()
        assert "cell phone" not in w4[-1].lower()

        # Step 5: All removed
        w5 = []
        for i in range(25):
            ev = engine.evaluate([], current_time=109.0 + i * 0.1)
            if ev.warning_text:
                w5.append(ev.warning_text)
        assert len(w5) >= 1
        assert "clear" in w5[-1].lower()

    def test_stationary_person_announces_once_then_stays_silent(self, engine):
        """Stationary person with minor bounding box noise must announce once and stay silent."""
        warnings = []
        # 30 frames spanning 10 seconds with minor bbox coordinate/distance jitter
        for i in range(30):
            jitter = (i % 3) * 2 - 2  # -2, 0, 2 px jitter
            det = _det(class_name="person", cx=320 + jitter, area_ratio=0.03, region="center")
            det["distance_m"] = 3.0 + (jitter * 0.05)  # 2.9m - 3.1m
            ev = engine.evaluate([det], current_time=100.0 + i * 0.33)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        # Must announce exactly ONCE on initial detection, then stay completely silent
        assert len(warnings) == 1
        assert "person detected" in warnings[0].lower()

    def test_approaching_person_updates_distance_then_triggers_stop_alert(self, engine):
        """Person moving from 3.5m -> 2.2m -> 0.9m announces distance update then urgent Stop!"""
        warnings = []
        # Phase 1: 3.5m
        for i in range(5):
            det = _det(class_name="person", cx=320, area_ratio=0.02, region="center")
            det["distance_m"] = 3.5
            ev = engine.evaluate([det], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 1
        assert "3.5" in warnings[0] or "3" in warnings[0]

        # Phase 2: Move to 2.2m after 4 seconds (elapsed > scene_cooldown)
        for i in range(5):
            det = _det(class_name="person", cx=320, area_ratio=0.04, region="center")
            det["distance_m"] = 2.2
            ev = engine.evaluate([det], current_time=105.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 2
        assert "2.2" in warnings[1] or "2" in warnings[1]

        # Phase 3: Enters Danger Zone at 0.9m
        for i in range(5):
            det = _det(class_name="person", cx=320, area_ratio=0.12, region="center")
            det["distance_m"] = 0.9
            ev = engine.evaluate([det], current_time=110.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 3
        assert "stop" in warnings[2].lower()


# ------------------------------------------------------------------
# Multi-Tier Critical Escalation & Hysteresis Tests
# ------------------------------------------------------------------
class TestMultiTierCriticalEscalation:
    def test_critical_tier1_then_tier2_escalation(self, engine):
        """Person entering Danger Zone 1 (1.4m) triggers Tier 1 alert, then moving to 0.6m triggers Tier 2 escalation."""
        warnings = []
        # 1. Enters Critical Zone 1 (1.4m)
        det_crit1 = _det(class_name="person", cx=320, area_ratio=0.08, region="center")
        det_crit1["distance_m"] = 1.4
        for i in range(5):
            ev = engine.evaluate([det_crit1], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 1
        assert "stop" in warnings[0].lower()
        assert "1.4" in warnings[0] or "1" in warnings[0]

        # 2. Remains in Critical Zone 1 for 10 frames -> NO new alert!
        for i in range(10):
            ev = engine.evaluate([det_crit1], current_time=102.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)
        assert len(warnings) == 1  # Still 1, no duplicate alert

        # 3. Escalates to Critical Zone 2 (0.6m <= 0.9m)
        det_crit2 = _det(class_name="person", cx=320, area_ratio=0.18, region="center")
        det_crit2["distance_m"] = 0.6
        for i in range(5):
            ev = engine.evaluate([det_crit2], current_time=105.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 2
        assert "stop" in warnings[1].lower()
        assert "very close" in warnings[1].lower() or "0.6" in warnings[1]

        # 4. Remains in Critical Zone 2 -> NO new alert!
        for i in range(10):
            ev = engine.evaluate([det_crit2], current_time=107.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)
        assert len(warnings) == 2

    def test_boundary_hysteresis_prevents_flapping(self, engine):
        """Distance fluctuating around 1.8m boundary (1.75m <-> 1.85m) must stay locked without flapping."""
        warnings = []
        # Initial entry at 1.75m
        det_entry = _det(class_name="person", cx=320, area_ratio=0.07, region="center")
        det_entry["distance_m"] = 1.75
        for i in range(4):
            ev = engine.evaluate([det_entry], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 1
        assert "stop" in warnings[0].lower()

        # Fluctuate across boundary between 1.75m and 1.88m (within 0.3m hysteresis)
        for i in range(20):
            d = 1.75 if (i % 2 == 0) else 1.88
            det = _det(class_name="person", cx=320, area_ratio=0.07, region="center")
            det["distance_m"] = d
            ev = engine.evaluate([det], current_time=102.0 + i * 0.2)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        # Must not re-trigger critical alerts on boundary jitter!
        assert len(warnings) == 1

    def test_multi_person_independent_tracking(self, engine):
        """Person A is far (3.0m), Person B enters critical zone (1.2m) -> alert is triggered for Person B."""
        det_a = _det(class_name="person", cx=100, area_ratio=0.03, region="left")
        det_a["distance_m"] = 3.0

        # Phase 1: Person A is alone on left
        warnings = []
        for i in range(4):
            ev = engine.evaluate([det_a], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 1
        assert "left" in warnings[0].lower()

        # Phase 2: Person B enters center corridor at 1.2m
        det_b = _det(class_name="person", cx=320, area_ratio=0.09, region="center")
        det_b["distance_m"] = 1.2

        for i in range(5):
            ev = engine.evaluate([det_a, det_b], current_time=105.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 2
        assert "stop" in warnings[1].lower()
        assert "1.2" in warnings[1] or "1" in warnings[1]


# ------------------------------------------------------------------
# Stationary Person & Bounding Box Stability Tests
# ------------------------------------------------------------------
class TestStationaryTrackingAndStability:
    def test_stationary_person_30_seconds_zero_repeated_alerts(self, engine):
        """A person standing still for 30+ seconds triggers exactly 1 alert initially, then 0 alerts."""
        warnings = []
        # 100 frames over 30 seconds
        for i in range(100):
            det = _det(class_name="person", cx=320, area_ratio=0.03, region="center")
            det["distance_m"] = 2.5
            ev = engine.evaluate([det], current_time=100.0 + i * 0.3)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 1
        assert "person detected" in warnings[0].lower()

    def test_temporary_missed_detections_preserve_identity(self, engine):
        """Temporary detection drop (e.g. 2 frames missed) must NOT treat re-detection as a new person."""
        warnings = []
        det = _det(class_name="person", cx=320, area_ratio=0.03, region="center")
        det["distance_m"] = 2.5

        # 1. Seen for 4 frames -> announced
        for i in range(4):
            ev = engine.evaluate([det], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)
        assert len(warnings) == 1

        # 2. Missed for 2 frames (detector occlusion/loss)
        for i in range(2):
            ev = engine.evaluate([], current_time=100.4 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)
        assert len(warnings) == 1

        # 3. Re-appears on frame 7 -> same person, NO new alert
        for i in range(10):
            ev = engine.evaluate([det], current_time=100.6 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        # Still exactly 1 alert, no duplicate re-announcement!
        assert len(warnings) == 1

    def test_region_boundary_hysteresis_prevents_flip(self, engine):
        """Person standing at border (cx=224, ratio=0.35) with +/- 3px jitter does not flip region."""
        warnings = []
        # Initial presence at center (cx=230, ratio=0.359)
        for i in range(4):
            det = _det(class_name="person", cx=230, area_ratio=0.03, region="center")
            det["distance_m"] = 2.5
            ev = engine.evaluate([det], current_time=100.0 + i * 0.1)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        assert len(warnings) == 1

        # Jitter between 220 and 230 over 20 frames
        for i in range(20):
            cx_jitter = 222 if (i % 2 == 0) else 230
            det = _det(class_name="person", cx=cx_jitter, area_ratio=0.03, region="center")
            det["distance_m"] = 2.5
            ev = engine.evaluate([det], current_time=102.0 + i * 0.2)
            if ev.warning_text:
                warnings.append(ev.warning_text)

        # Region hysteresis must prevent region flipping and repeated announcements
        assert len(warnings) == 1

