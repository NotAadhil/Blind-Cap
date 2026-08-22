"""
audio_spatializer.py - 3D Spatial Audio Module (Silent Mode)
===========================================================
All audio earcon, beep, click, and radar sounds have been removed.
The system is now voice-only via TTS.

This module retains its public API as no-ops so that callers in
async_pipeline.py and main.py do not need to be modified.
"""

from typing import Dict, Optional

from config import (
    Config,
    DEFAULT_CONFIG,
    AUDIO_MODE_HYBRID,
    AUDIO_MODE_RADAR,
    AUDIO_MODE_SPEECH,
)
from logger import get_logger

logger = get_logger(__name__)


def synthesize_spatial_stereo_buffer(
    frequency_hz: float = 880.0,
    duration_ms: float = 40.0,
    pan: float = 0.5,
    volume: float = 0.5,
    sample_rate: int = 44100,
) -> bytes:
    """No-op. Retained for backward compatibility with tests."""
    return b""


# Alias for backward compatibility
generate_stereo_click_wav = synthesize_spatial_stereo_buffer


class AudioSpatializer:
    """Silent audio spatializer. All sound methods are no-ops."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.audio_mode = config.AUDIO_MODE
        self._running = True
        self._active_hazard_info: Optional[Dict] = None
        logger.info("Audio Spatializer initialised (Silent Mode — voice-only via TTS)")

    def play_spatial_earcon(
        self,
        pan: float = 0.5,
        area_ratio: float = 0.08,
        is_critical: bool = False,
    ) -> None:
        """No-op. All earcon sounds removed."""
        pass

    def play_directional_chime(self, direction: str = "center") -> None:
        """No-op. All chime sounds removed."""
        pass

    def play_presence_beep(self, direction: str = "center") -> None:
        """No-op. All presence beep sounds removed."""
        pass

    def play_stereo_startup_demo(self) -> None:
        """No-op. Startup demo sounds removed."""
        pass

    def update_radar(self, active_hazard: Optional[Dict]) -> None:
        """Store current hazard info (no audio action)."""
        self._active_hazard_info = active_hazard

    def set_audio_mode(self, mode: str) -> None:
        if mode in (AUDIO_MODE_HYBRID, AUDIO_MODE_RADAR, AUDIO_MODE_SPEECH):
            self.audio_mode = mode
            logger.info("Audio Mode changed to: %s", self.audio_mode)

    def toggle_mode(self) -> str:
        if self.audio_mode == AUDIO_MODE_HYBRID:
            self.set_audio_mode(AUDIO_MODE_RADAR)
        elif self.audio_mode == AUDIO_MODE_RADAR:
            self.set_audio_mode(AUDIO_MODE_SPEECH)
        else:
            self.set_audio_mode(AUDIO_MODE_HYBRID)
        return self.audio_mode

    def shutdown(self) -> None:
        self._running = False
        logger.info("Audio Spatializer shut down cleanly")
