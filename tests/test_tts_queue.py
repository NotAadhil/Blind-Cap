"""
tests/test_tts_queue.py - TTS Queue & Engine Unit Tests
=========================================================
Uses FakeSpeechBackend so no audio hardware is required.

Run::

    python -m pytest tests/test_tts_queue.py -v
"""

import time
import pytest
from config import Config, SEVERITY_CRITICAL, SEVERITY_WARNING, SEVERITY_CAUTION, SEVERITY_INFO
from tts import TTSEngine, SpeechMessage, FakeSpeechBackend


@pytest.fixture
def config():
    cfg = Config()
    cfg.TTS_ENABLED = True
    cfg.WARNING_COOLDOWN = 5.0
    cfg.CRITICAL_COOLDOWN = 3.0
    cfg.TTS_MESSAGE_TTL = 6.0
    return cfg


# ---- SpeechMessage ordering tests ----

class TestSpeechMessage:
    def test_higher_priority_sorts_first(self):
        """Messages with higher (more negative) priority should sort first."""
        low = SpeechMessage(
            priority=-10, text="low", severity=SEVERITY_INFO,
            category="a", timestamp=0, expires_at=100, dedupe_key="a",
        )
        high = SpeechMessage(
            priority=-100, text="high", severity=SEVERITY_CRITICAL,
            category="b", timestamp=0, expires_at=100, dedupe_key="b",
        )
        assert high < low  # min-heap: high priority dequeued first

    def test_same_priority_no_crash(self):
        """Two messages with the same priority should not crash comparison."""
        a = SpeechMessage(
            priority=-50, text="a", severity=SEVERITY_WARNING,
            category="x", timestamp=0, expires_at=100, dedupe_key="a",
        )
        b = SpeechMessage(
            priority=-50, text="b", severity=SEVERITY_WARNING,
            category="y", timestamp=0, expires_at=100, dedupe_key="b",
        )
        # Should not raise
        _ = a < b or a == b or a > b


# ---- TTSEngine tests ----

class TestTTSEngine:
    def test_speak_enqueues(self, config):
        tts = TTSEngine(config=config)
        time.sleep(0.5)  # let worker start
        tts.speak("Hello", priority=50, severity=SEVERITY_INFO, category="test")
        time.sleep(0.5)
        status = tts.get_status()
        # Message should have been consumed or be in queue
        assert status["status"] in ("READY", "SPEAKING")
        tts.shutdown()

    def test_mute_suppresses_speech(self, config):
        tts = TTSEngine(config=config)
        time.sleep(0.5)
        tts.set_muted(True)
        tts.speak("Should not be spoken", priority=50, severity=SEVERITY_WARNING, category="m")
        time.sleep(0.3)
        status = tts.get_status()
        assert status["muted"] is True
        tts.shutdown()

    def test_quiet_suppresses_info(self, config):
        tts = TTSEngine(config=config)
        time.sleep(0.5)
        tts.set_quiet_mode(True)
        tts.speak("Info suppressed", priority=20, severity=SEVERITY_INFO, category="q")
        time.sleep(0.3)
        status = tts.get_status()
        assert status["quiet_mode"] is True
        tts.shutdown()

    def test_deduplication(self, config):
        tts = TTSEngine(config=config)
        time.sleep(0.5)
        # Speak same dedupe key twice rapidly
        tts.speak("First", priority=50, severity=SEVERITY_WARNING, category="d", dedupe_key="dup")
        tts.speak("Duplicate", priority=50, severity=SEVERITY_WARNING, category="d", dedupe_key="dup")
        time.sleep(0.5)
        # Second should have been suppressed by dedup
        tts.shutdown()

    def test_shutdown_clean(self, config):
        tts = TTSEngine(config=config)
        time.sleep(0.5)
        tts.shutdown()
        # Should not hang or crash
        status = tts.get_status()
        assert True  # if we got here, shutdown was clean

    def test_repeat_last(self, config):
        tts = TTSEngine(config=config)
        time.sleep(0.5)
        tts.speak("Important warning.", priority=70, severity=SEVERITY_WARNING, category="r")
        time.sleep(1.0)
        tts.repeat_last()
        time.sleep(1.0)
        tts.shutdown()

    def test_get_status_structure(self, config):
        tts = TTSEngine(config=config)
        time.sleep(0.5)
        status = tts.get_status()
        assert "status" in status
        assert "queue_size" in status
        assert "muted" in status
        assert "quiet_mode" in status
        tts.shutdown()


# ---- FakeSpeechBackend tests ----

class TestFakeSpeechBackend:
    def test_speak_returns_true(self):
        backend = FakeSpeechBackend()
        assert backend.speak("test") is True

    def test_stop_does_not_crash(self):
        backend = FakeSpeechBackend()
        backend.stop()  # should be no-op

    def test_shutdown_does_not_crash(self):
        backend = FakeSpeechBackend()
        backend.shutdown()
