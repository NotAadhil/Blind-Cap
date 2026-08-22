"""
tts.py - Asynchronous Text-to-Speech Engine
=============================================
Non-blocking, priority-aware speech synthesis for Blind Cap warnings.

Features:
- Native Windows SAPI.SpVoice with instant asynchronous override (<50ms).
- Instant interrupt for on-demand OCR and Scene Description requests.
- Hard queue anti-overload protection.
- Global minimum speech gap to prevent voice spamming.
- pyttsx3 fallback support.
"""

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

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


# ---------------------------------------------------------------------------
# Speech backend abstraction
# ---------------------------------------------------------------------------
class SpeechBackend(ABC):
    """Interface for a speech synthesis backend."""

    @abstractmethod
    def speak(self, text: str, interrupt_event: Optional[threading.Event] = None) -> bool:
        """Speak *text* inside the worker thread, aborting if interrupt_event is set."""

    @abstractmethod
    def stop(self) -> None:
        """Interrupt any speech currently in progress."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release all engine resources."""


class SapiSpeechBackend(SpeechBackend):
    """
    Windows SAPI5 backend.
    Uses native win32com SAPI.SpVoice directly with async interruptible playback.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0):
        self._rate = rate
        self._volume = volume
        self._speaker = None
        self._pyttsx3_engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        # Try native win32com SAPI.SpVoice first (rock-solid on Windows)
        try:
            import win32com.client

            self._speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self._speaker.Volume = int(max(0.0, min(1.0, self._volume)) * 100)
            # Map wpm (~150-220) to SAPI Rate (-10 to 10): 175 wpm -> 1
            sapi_rate = int(max(-10, min(10, (self._rate - 150) / 25)))
            self._speaker.Rate = sapi_rate
            logger.info("Native Windows SAPI5 voice engine initialised (Volume=%d, Rate=%d)", self._speaker.Volume, sapi_rate)
            return
        except Exception as exc:
            logger.warning("Could not initialise native win32com SAPI.SpVoice (%s), trying pyttsx3 fallback...", exc)

        # Fallback to pyttsx3
        try:
            import pyttsx3

            self._pyttsx3_engine = pyttsx3.init(driverName="sapi5")
            self._pyttsx3_engine.setProperty("rate", self._rate)
            self._pyttsx3_engine.setProperty("volume", self._volume)
            logger.info("pyttsx3 speech engine initialised (rate=%d)", self._rate)
        except Exception as exc:
            logger.error("Failed to initialise TTS engine: %s", exc)

    def speak(self, text: str, interrupt_event: Optional[threading.Event] = None) -> bool:
        if self._speaker is not None:
            try:
                # Flag 1 = SVSFlagsAsync (start speech asynchronously)
                self._speaker.Speak(text, 1)
                # Slice wait to allow sub-50ms instant interruption
                while True:
                    done = self._speaker.WaitUntilDone(50)
                    if done:
                        return True
                    if interrupt_event is not None and interrupt_event.is_set():
                        # Flag 2 = SVSFPurgeBeforeSpeak (instantly cancels speech)
                        self._speaker.Speak("", 2)
                        return False
            except Exception as exc:
                logger.error("Native SAPI speak error: %s", exc)
                return False

        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                return True
            except Exception as exc:
                logger.error("pyttsx3 speak error: %s", exc)
                return False

        return False

    def stop(self) -> None:
        if self._speaker is not None:
            try:
                # Flag 2 = SVSFPurgeBeforeSpeak (immediately cancels any ongoing speech)
                self._speaker.Speak("", 2)
            except Exception:
                pass
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass

    def shutdown(self) -> None:
        self.stop()
        self._speaker = None
        self._pyttsx3_engine = None


class FakeSpeechBackend(SpeechBackend):
    """
    Testing backend - prints spoken text to the logger instead of producing
    audio. Requires no speakers or sound hardware.
    """

    def speak(self, text: str, interrupt_event: Optional[threading.Event] = None) -> bool:
        logger.info("[SPEAK] %s", text)
        return True

    def stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Structured speech message
# ---------------------------------------------------------------------------
@dataclass(order=True)
class SpeechMessage:
    """
    A single speech event placed into the TTS priority queue.
    """

    priority: int  # negated when enqueued
    text: str = field(compare=False)
    severity: str = field(compare=False)
    category: str = field(compare=False)
    timestamp: float = field(compare=False)
    expires_at: float = field(compare=False)
    dedupe_key: str = field(compare=False)


# ---------------------------------------------------------------------------
# TTS Engine (main public API)
# ---------------------------------------------------------------------------
class TTSEngine:
    """
    Thread-safe, anti-overload TTS manager with instant override capability.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG, backend: Optional[SpeechBackend] = None):
        self._config = config
        self._custom_backend = backend
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._muted: bool = not config.TTS_ENABLED
        self._quiet_mode: bool = config.QUIET_MODE
        self._running: bool = True
        self._interrupt_event: threading.Event = threading.Event()

        # State visible to the main thread for HUD display
        self._status: str = "READY"
        self._current_severity: str = SEVERITY_INFO
        self._current_priority: int = 0
        self._last_speech_finish_time: float = 0.0
        self._user_query_active: bool = False

        # Deduplication: dedupe_key -> last-spoken timestamp
        self._spoken_keys: Dict[str, float] = {}
        self._last_important: Optional[SpeechMessage] = None

        # The backend reference (set inside the worker thread)
        self._backend: Optional[SpeechBackend] = None

        self._worker = threading.Thread(
            target=self._worker_run, daemon=True, name="TTS-Worker"
        )
        self._worker.start()

    # ---- public API (called from main thread) -------------------------

    def speak(
        self,
        text: str,
        priority: int = 50,
        severity: str = SEVERITY_INFO,
        category: str = "",
        dedupe_key: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Thread-safe enqueue. Discards message if muted, or if quiet mode is active
        and severity is INFO/CAUTION. If a command of higher importance/priority arrives,
        ongoing lower-priority speech is immediately interrupted.
        """
        if self._muted:
            return

        if self._quiet_mode and severity in (SEVERITY_INFO, SEVERITY_CAUTION):
            return

        now = time.time()

        # Deduplication & Cooldown check
        if not force:
            if dedupe_key:
                last = self._spoken_keys.get(dedupe_key, 0.0)
                cooldown = self._cooldown_for(severity)
                if (now - last) < cooldown:
                    return

            # Check if new message has higher priority than currently playing message
            is_higher_priority = (priority > self._current_priority) and (self._status == "SPEAKING")

            # If TTS is currently speaking:
            if self._status == "SPEAKING":
                if self._current_severity == SEVERITY_CRITICAL and severity == SEVERITY_CRITICAL and not is_higher_priority:
                    return  # Let current critical warning finish!
                elif not is_higher_priority and severity != SEVERITY_CRITICAL:
                    return  # Drop lower-priority while speaking

            # Minimum quiet gap between utterances
            min_gap = getattr(self._config, "TTS_MIN_GAP_SEC", 0.5)
            if (now - self._last_speech_finish_time) < min_gap and severity != SEVERITY_CRITICAL and not is_higher_priority:
                return

        msg = SpeechMessage(
            priority=-priority,  # negate for min-heap -> highest first
            text=text,
            severity=severity,
            category=category,
            timestamp=now,
            expires_at=now + (25.0 if force else self._config.TTS_MESSAGE_TTL),
            dedupe_key=dedupe_key or f"msg_{now}",
        )

        # Check if we should interrupt ongoing lower priority speech
        is_higher_priority = (priority > self._current_priority) and (self._status == "SPEAKING")
        should_interrupt = force or is_higher_priority or (
            severity == SEVERITY_CRITICAL
            and not self._user_query_active
            and self._current_severity != SEVERITY_CRITICAL
        )

        if should_interrupt:
            if force:
                self._user_query_active = True
            self._interrupt_event.set()
            if self._backend is not None:
                self._backend.stop()
            # Drain queue of lower-priority pending messages
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        # Max pending queue size is 1 to prevent any stale message backlog
        if not force and not self._queue.empty() and severity != SEVERITY_CRITICAL and not is_higher_priority:
            return

        if dedupe_key:
            self._spoken_keys[dedupe_key] = now

        logger.info("[VOICE ALERT] %s (%s)", text, severity)
        self._queue.put((msg.priority, msg))

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self._interrupt_event.set()
            if self._backend:
                self._backend.stop()
        logger.info("TTS %s", "MUTED" if muted else "UNMUTED")

    def set_quiet_mode(self, quiet: bool) -> None:
        self._quiet_mode = quiet
        logger.info("Quiet mode %s", "ON" if quiet else "OFF")

    def repeat_last(self) -> None:
        """Re-queue the last important (WARNING/CRITICAL) warning."""
        if self._last_important:
            self.speak(
                text=self._last_important.text,
                priority=100,
                severity=self._last_important.severity,
                category=self._last_important.category,
                dedupe_key=f"repeat_{time.time()}",
                force=True,
            )

    def get_status(self) -> Dict:
        """Return TTS state for the HUD overlay."""
        return {
            "status": "MUTED" if self._muted else self._status,
            "queue_size": self._queue.qsize(),
            "muted": self._muted,
            "quiet_mode": self._quiet_mode,
        }

    @property
    def enabled(self) -> bool:
        return not self._muted

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._running = False
        self._interrupt_event.set()
        # Push a sentinel so the worker wakes up
        self._queue.put((-99999, None))
        if self._backend:
            self._backend.stop()
        self._worker.join(timeout=3.0)
        logger.info("TTS engine shut down")

    # ---- internal -----------------------------------------------------

    def _cooldown_for(self, severity: str) -> float:
        if severity == SEVERITY_CRITICAL:
            return self._config.CRITICAL_COOLDOWN
        if severity in (SEVERITY_WARNING, SEVERITY_CAUTION):
            return self._config.WARNING_COOLDOWN
        return self._config.INFO_COOLDOWN

    def _worker_run(self) -> None:
        """
        Dedicated worker thread. Owns the SAPI5 engine lifecycle.
        COM is initialised and cleaned up here.
        """
        com_init = False
        try:
            try:
                import pythoncom  # type: ignore[import-untyped]

                pythoncom.CoInitialize()
                com_init = True
            except ImportError:
                pass  # pythoncom not available - SAPI5 may still work

            if self._custom_backend is not None:
                backend = self._custom_backend
            else:
                backend = SapiSpeechBackend(
                    rate=self._config.TTS_RATE,
                    volume=self._config.TTS_VOLUME,
                )
            self._backend = backend
            retries = 0

            while self._running:
                # Pull next message
                try:
                    _neg_pri, msg = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if msg is None:  # shutdown sentinel
                    break

                # Discard expired messages
                if time.time() > msg.expires_at:
                    continue

                # Respect mute / quiet at dequeue time as well
                if self._muted:
                    continue
                if self._quiet_mode and msg.severity in (
                    SEVERITY_INFO,
                    SEVERITY_CAUTION,
                ):
                    continue

                # Reset interrupt event for this utterance
                self._interrupt_event.clear()

                # Speak
                self._status = "SPEAKING"
                self._current_severity = msg.severity
                self._current_priority = -msg.priority
                try:
                    ok = backend.speak(msg.text, self._interrupt_event)
                finally:
                    self._user_query_active = False
                    self._current_priority = 0

                self._last_speech_finish_time = time.time()

                if ok:
                    retries = 0
                    if msg.severity in (SEVERITY_WARNING, SEVERITY_CRITICAL):
                        self._last_important = msg
                else:
                    if not self._interrupt_event.is_set():
                        retries += 1
                        if retries <= self._config.TTS_MAX_RETRIES:
                            logger.warning(
                                "TTS playback failed - recovering (attempt %d/%d)",
                                retries,
                                self._config.TTS_MAX_RETRIES,
                            )
                            backend.shutdown()
                            backend = SapiSpeechBackend(
                                rate=self._config.TTS_RATE,
                                volume=self._config.TTS_VOLUME,
                            )
                            self._backend = backend
                        else:
                            logger.error(
                                "TTS max retries (%d) exceeded - worker stopping",
                                self._config.TTS_MAX_RETRIES,
                            )
                            self._status = "ERROR"
                            break

                self._status = "READY"
                self._current_severity = SEVERITY_INFO

            # Clean up backend
            backend.shutdown()
            self._backend = None

        finally:
            if com_init:
                try:
                    import pythoncom  # type: ignore[import-untyped]

                    pythoncom.CoUninitialize()
                except Exception:
                    pass
