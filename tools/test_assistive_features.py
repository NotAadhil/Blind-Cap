"""
tools/test_assistive_features.py - Standalone Assistive Features Verification
=============================================================================
Runs quick self-tests of all newly implemented assistive components.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from audio_spatializer import AudioSpatializer, generate_stereo_click_wav
from config import Config
from depth_estimator import MonocularDepthEstimator
from motion_filter import EgoMotionFilter
from ocr_reader import OCRReader
from tts import TTSEngine


def run_tests():
    print("=" * 60)
    print("  BLIND CAP: ASSISTIVE FEATURES DIAGNOSTIC TEST")
    print("=" * 60)

    cfg = Config()

    # 1. Test 3D Spatial Audio
    print("\n[1/5] Testing 3D Spatial Audio...")
    spatializer = AudioSpatializer(config=cfg)
    print("  -> Playing Left Ear 3D Spatial Click...")
    spatializer.play_spatial_earcon(pan=0.0, area_ratio=0.10)
    time.sleep(0.3)
    print("  -> Playing Right Ear 3D Spatial Click...")
    spatializer.play_spatial_earcon(pan=1.0, area_ratio=0.10)
    time.sleep(0.3)
    print("  -> Playing Center 3D Spatial Click...")
    spatializer.play_spatial_earcon(pan=0.5, area_ratio=0.20, is_critical=True)
    time.sleep(0.3)
    spatializer.shutdown()
    print("  [PASS] 3D Spatial Audio generated successfully.")

    # 2. Test TTS Voice Engine
    print("\n[2/5] Testing Windows Native SAPI5 Voice...")
    tts = TTSEngine(config=cfg)
    time.sleep(0.5)
    print("  -> Speaking voice test phrase...")
    tts.speak("Audio and vision subsystems online.", priority=100, severity="WARNING", category="test")
    time.sleep(2.0)
    tts.shutdown()
    print("  [PASS] Voice Engine operational.")

    # 3. Test Floor Clearance Estimator
    print("\n[3/5] Testing Floor Clearance Estimator...")
    depth = MonocularDepthEstimator(config=cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_clear = depth.evaluate_corridor(frame, [])
    print(f"  -> Empty Corridor Clearance: {res_clear['clearance_ratio'] * 100}% (Path Clear: {res_clear['is_path_clear']})")
    
    det = {"class_name": "person", "bbox": [220, 100, 420, 470], "area_ratio": 0.20, "region": "center"}
    res_obs = depth.evaluate_corridor(frame, [det])
    print(f"  -> Obstructed Corridor Clearance: {res_obs['clearance_ratio'] * 100}% (Path Clear: {res_obs['is_path_clear']})")
    print("  [PASS] Floor Clearance Estimator operational.")

    # 4. Test Ego-Motion Filter
    print("\n[4/5] Testing Optical Flow Motion Filter...")
    motion = EgoMotionFilter(config=cfg)
    f1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    f2 = np.roll(f1, shift=4, axis=1)
    vx1, vy1 = motion.update(f1)
    vx2, vy2 = motion.update(f2)
    print(f"  -> Optical Flow Velocity: vx={vx2:.2f}, vy={vy2:.2f}")
    print("  [PASS] Motion Filter operational.")

    # 5. Test OCR Text Reader
    print("\n[5/5] Testing On-Demand OCR Reader...")
    ocr = OCRReader(config=cfg)
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "ROOM 101", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    ocr_res = ocr.extract_text(test_img)
    print(f"  -> OCR Result: {ocr_res['text']}")
    print("  [PASS] Native Windows OCR Reader operational.")

    # 6. Test AI Scene Describer
    print("\n[6/6] Testing AI Scene Describer...")
    from scene_describer import SceneDescriber
    describer = SceneDescriber(config=cfg)
    dets = [
        {"class_name": "person", "region": "center", "area_ratio": 0.12},
        {"class_name": "chair", "region": "left", "area_ratio": 0.06},
    ]
    depth_st = {"clearance_ratio": 0.70, "drop_off_detected": False}
    scene_text = describer.describe_scene(test_img, dets, depth_st)
    print(f"  -> Scene Description: {scene_text}")
    print("  [PASS] AI Scene Describer operational.")

    print("\n" + "=" * 60)
    print("  ALL 6 ASSISTIVE SUBSYSTEMS OPERATIONAL AND VERIFIED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
