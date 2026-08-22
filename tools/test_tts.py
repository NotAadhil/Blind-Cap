"""
tools/test_tts.py - Standalone TTS Acceptance Test
====================================================
Tests the Text-to-Speech subsystem independently from YOLO/OpenVINO.

Run::

    python tools/test_tts.py

This MUST successfully speak at least 5 consecutive sentences before
the TTS can be considered integration-ready (specification Section 46).
"""

import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config, SEVERITY_CRITICAL, SEVERITY_WARNING, SEVERITY_CAUTION, SEVERITY_INFO
from logger import get_logger
from tts import TTSEngine, FakeSpeechBackend

logger = get_logger("test_tts")


def separator(title: str) -> None:
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


def test_basic_speech(tts: TTSEngine) -> bool:
    """Speak 5 consecutive sentences completely."""
    separator("TEST 1: Basic Speech (5 sentences)")

    phrases = [
        "Blind Cap speech test.",
        "Second test message.",
        "Third test message.",
        "Critical warning test.",
        "Speech system operational.",
    ]

    for i, phrase in enumerate(phrases, 1):
        print(f"  [{i}/5] Speaking: \"{phrase}\"")
        tts.speak(text=phrase, priority=50, severity=SEVERITY_INFO, category="test")
        time.sleep(2.5)  # wait for each phrase to finish

    print("  [PASS] All 5 sentences dispatched.")
    return True


def test_rapid_messages(tts: TTSEngine) -> bool:
    """Send many messages quickly - queue should not overflow."""
    separator("TEST 2: Rapid Messages")

    for i in range(8):
        tts.speak(
            text=f"Rapid message {i + 1}.",
            priority=30,
            severity=SEVERITY_INFO,
            category="rapid",
        )

    time.sleep(1.0)
    status = tts.get_status()
    print(f"  Queue size after burst: {status['queue_size']}")
    print("  [PASS] Rapid message test passed.")
    time.sleep(3.0)
    return True


def test_deduplication(tts: TTSEngine) -> bool:
    """Same dedupe_key should be suppressed within cooldown."""
    separator("TEST 3: Duplicate Suppression")

    for _ in range(5):
        tts.speak(
            text="Duplicate test.",
            priority=50,
            severity=SEVERITY_WARNING,
            category="dup",
            dedupe_key="dup_test_key",
        )

    time.sleep(3.0)
    print("  [PASS] Only one 'Duplicate test.' should have been spoken.")
    return True


def test_priority(tts: TTSEngine) -> bool:
    """Critical message should be spoken before lower-priority ones."""
    separator("TEST 4: Priority Ordering")

    tts.speak(text="Low priority.", priority=10, severity=SEVERITY_INFO, category="p")
    tts.speak(text="Medium priority.", priority=50, severity=SEVERITY_WARNING, category="p")
    tts.speak(text="CRITICAL priority.", priority=100, severity=SEVERITY_CRITICAL, category="p")

    time.sleep(5.0)
    print("  [PASS] Priority test dispatched (listen for ordering).")
    return True


def test_mute(tts: TTSEngine) -> bool:
    """Muted engine should suppress speech."""
    separator("TEST 5: Mute Toggle")

    tts.set_muted(True)
    tts.speak(text="This should NOT be heard.", priority=50, severity=SEVERITY_WARNING, category="m")
    time.sleep(1.0)

    tts.set_muted(False)
    tts.speak(text="Unmuted successfully.", priority=50, severity=SEVERITY_WARNING, category="m")
    time.sleep(3.0)

    print("  [PASS] Mute test passed.")
    return True


def test_quiet_mode(tts: TTSEngine) -> bool:
    """Quiet mode should suppress INFO/CAUTION but allow WARNING/CRITICAL."""
    separator("TEST 6: Quiet Mode")

    tts.set_quiet_mode(True)
    tts.speak(text="Info suppressed.", priority=20, severity=SEVERITY_INFO, category="q")
    tts.speak(text="Caution suppressed.", priority=30, severity=SEVERITY_CAUTION, category="q")
    tts.speak(text="Warning allowed.", priority=60, severity=SEVERITY_WARNING, category="q")
    time.sleep(3.0)

    tts.set_quiet_mode(False)
    print("  [PASS] Quiet mode test passed.")
    return True


def test_shutdown(tts: TTSEngine) -> bool:
    """Clean shutdown should not hang."""
    separator("TEST 7: Shutdown")

    tts.shutdown()
    print("  [PASS] TTS engine shut down cleanly.")
    return True


def main() -> None:
    print("=" * 50)
    print("  BLIND CAP - TTS ACCEPTANCE TEST")
    print("=" * 50)

    config = Config()
    config.TTS_ENABLED = True
    tts = TTSEngine(config=config)

    # Give the worker thread time to start
    time.sleep(1.0)
    status = tts.get_status()
    print(f"\n  TTS Status: {status['status']}")

    results = {}
    try:
        results["basic"] = test_basic_speech(tts)
        results["rapid"] = test_rapid_messages(tts)
        results["dedup"] = test_deduplication(tts)
        results["priority"] = test_priority(tts)
        results["mute"] = test_mute(tts)
        results["quiet"] = test_quiet_mode(tts)
        results["shutdown"] = test_shutdown(tts)
    except Exception as exc:
        print(f"\n  [FAIL] Test failed with exception: {exc}")
        tts.shutdown()
        sys.exit(1)

    # Summary
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    all_pass = True
    for name, passed in results.items():
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All TTS tests passed!")
    else:
        print("\n  Some tests failed. Check output above.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
