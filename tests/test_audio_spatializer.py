"""
tests/test_audio_spatializer.py - Audio Spatializer Unit Tests (Silent Mode)
=============================================================================
Tests that the silent-mode audio spatializer API works without crashing.
"""

import time
import pytest
from audio_spatializer import AudioSpatializer, generate_stereo_click_wav
from config import Config, AUDIO_MODE_HYBRID, AUDIO_MODE_RADAR, AUDIO_MODE_SPEECH


@pytest.fixture
def config():
    cfg = Config()
    cfg.TTS_ENABLED = False  # Keep test silent
    cfg.AUDIO_MODE = AUDIO_MODE_HYBRID
    return cfg


class TestAudioSpatializer:
    def test_generate_stereo_click_wav_returns_bytes(self):
        """In silent mode, synthesize returns empty bytes (no-op)."""
        wav = generate_stereo_click_wav(frequency_hz=880.0, duration_ms=20.0, pan=0.0)
        assert isinstance(wav, bytes)

    def test_audio_mode_toggling(self, config):
        """Test cycling through HYBRID -> RADAR -> SPEECH -> HYBRID."""
        spatializer = AudioSpatializer(config=config)
        assert spatializer.audio_mode == AUDIO_MODE_HYBRID

        mode2 = spatializer.toggle_mode()
        assert mode2 == AUDIO_MODE_RADAR

        mode3 = spatializer.toggle_mode()
        assert mode3 == AUDIO_MODE_SPEECH

        mode4 = spatializer.toggle_mode()
        assert mode4 == AUDIO_MODE_HYBRID

        spatializer.shutdown()

    def test_play_spatial_earcon_does_not_crash(self, config):
        """Test calling play_spatial_earcon (no-op) with various pans."""
        spatializer = AudioSpatializer(config=config)
        # Left, center, right — all no-ops
        spatializer.play_spatial_earcon(pan=0.0, area_ratio=0.05)
        spatializer.play_spatial_earcon(pan=0.5, area_ratio=0.10)
        spatializer.play_spatial_earcon(pan=1.0, area_ratio=0.25, is_critical=True)
        spatializer.shutdown()

    def test_update_radar_hazard(self, config):
        """Test sending hazard updates (no-op radar)."""
        spatializer = AudioSpatializer(config=config)
        hazard = {
            "class_name": "person",
            "region": "left",
            "proximity": "CLOSE",
            "area_ratio": 0.12,
            "bbox": [50, 100, 150, 400],
        }
        spatializer.update_radar(hazard)
        time.sleep(0.1)
        spatializer.update_radar(None)
        spatializer.shutdown()
