package com.blindcap.app.speech

import android.content.Context
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

enum class TtsMode {
    OBJECT_DETECTION,
    OCR,
    CURRENCY,
    COLOR,
    BARCODE,
    FACE,
    VOICE_ASSISTANT,
    SYSTEM // Startup, Mode switch announcements, SOS, Settings, System alerts
}

data class SpeechMessage(
    val priority: Int, // Higher number = higher urgency (1..100)
    val text: String,
    val severity: String, // "CRITICAL", "WARNING", "CAUTION", "INFO"
    val mode: TtsMode = TtsMode.SYSTEM,
    val generationId: Long = 0L,
    val timestampMs: Long = SystemClock.elapsedRealtime(),
    val maxAgeMs: Long = 3000L
)

class TtsManager(
    private val context: Context,
    private val onReadyCallback: (() -> Unit)? = null
) : TextToSpeech.OnInitListener {

    private val tag = "TtsManager"
    private var tts: TextToSpeech? = null
    private var isInitialized = false

    // Single source of truth for speech session generation ID
    private val currentGeneration = AtomicLong(1L)

    @Volatile
    var activeMode: TtsMode = TtsMode.SYSTEM
        private set

    // Strict single-slot bounded pending buffer (newest message always replaces stale pending)
    private var pendingMessage: SpeechMessage? = null

    private var currentlySpeakingText: String? = null
    private var currentlySpeakingMode: TtsMode = TtsMode.SYSTEM
    private var currentlySpeakingGen: Long = 0L
    private var currentPriority: Int = 0
    private var lastSpokenTime: Long = 0L
    private var lastSpokenText: String? = null
    private var lastOcrSpeakTime: Long = 0L

    private val utteranceIdCounter = AtomicLong(1L)
    @Volatile
    private var activeUtteranceId: String? = null

    var isSpeaking: Boolean = false
        private set

    @Volatile
    var isOcrActive: Boolean = false
        private set

    var isMuted: Boolean = false
    var isQuietMode: Boolean = false
    var lastImportantWarning: String? = null

    init {
        tts = TextToSpeech(context.applicationContext, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(tag, "Language US not supported on this device TTS engine")
            } else {
                isInitialized = true
                tts?.setSpeechRate(1.08f)
                setupUtteranceListener()
                Log.i(tag, "TextToSpeech initialized successfully")
                onReadyCallback?.invoke()
            }
        } else {
            Log.e(tag, "TextToSpeech initialization failed (status: $status)")
        }
    }

    private fun setupUtteranceListener() {
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                if (utteranceId == activeUtteranceId) {
                    isSpeaking = true
                }
            }

            override fun onDone(utteranceId: String?) {
                handleUtteranceFinished(utteranceId)
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                handleUtteranceFinished(utteranceId)
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                Log.e(tag, "TTS utterance error ($errorCode) for ID: $utteranceId")
                handleUtteranceFinished(utteranceId)
            }
        })
    }

    @Synchronized
    private fun handleUtteranceFinished(utteranceId: String?) {
        // Only handle completion for the CURRENT active utterance; ignore callbacks from previously cancelled ones!
        if (utteranceId != activeUtteranceId && activeUtteranceId != null) {
            Log.d(tag, "Ignored completion of cancelled utterance: $utteranceId (active: $activeUtteranceId)")
            return
        }

        isSpeaking = false
        isOcrActive = false
        activeUtteranceId = null
        currentlySpeakingText = null
        currentPriority = 0

        // Check if there is a pending fresh message (non-stale and matches current generation + mode)
        val next = pendingMessage
        pendingMessage = null

        if (next != null) {
            val now = SystemClock.elapsedRealtime()
            val age = now - next.timestampMs

            // Validate generation ID, active mode, and age
            val isGenValid = next.generationId == currentGeneration.get()
            val isModeValid = next.mode == TtsMode.SYSTEM || next.mode == TtsMode.VOICE_ASSISTANT || next.mode == activeMode || next.priority >= 80
            val isFresh = age <= next.maxAgeMs

            if (isGenValid && isModeValid && isFresh) {
                executeSpeech(next)
            } else {
                Log.i(tag, "Discarded invalid pending TTS message (genValid=$isGenValid, modeValid=$isModeValid, age=${age}ms): ${next.text}")
            }
        }
    }

    /**
     * Atomically switches active mode, invalidates in-flight background requests,
     * immediately cuts off current speech audio, and clears the pending queue.
     * Returns the newly generated session generation ID.
     */
    @Synchronized
    fun switchMode(newMode: TtsMode): Long {
        val newGen = currentGeneration.incrementAndGet()
        activeMode = newMode

        Log.i(tag, "Switching mode to $newMode (New Generation Token: $newGen)")

        // Stop speech engine immediately
        try {
            tts?.stop()
        } catch (_: Exception) {}

        // Reset all speech states
        isSpeaking = false
        isOcrActive = false
        pendingMessage = null
        activeUtteranceId = null
        currentlySpeakingText = null
        currentPriority = 0

        return newGen
    }

    /**
     * Cancels any speech originating from a specific mode.
     */
    @Synchronized
    fun cancelModeSpeech(mode: TtsMode) {
        if (currentlySpeakingMode == mode || isSpeaking) {
            stop()
        }
        if (pendingMessage?.mode == mode) {
            pendingMessage = null
        }
    }

    /**
     * Returns the current generation ID for tagging background worker tasks.
     */
    fun getGenerationId(): Long = currentGeneration.get()

    /**
     * Dedicated method for user-requested OCR reading with priority locking and debounce.
     */
    @Synchronized
    fun startOcrReading(ocrText: String, generationId: Long = currentGeneration.get()) {
        if (isMuted || !isInitialized) return
        val trimmed = ocrText.trim()
        if (trimmed.isEmpty()) return

        // Validate generation token
        if (generationId != currentGeneration.get()) {
            Log.d(tag, "Discarded OCR reading with stale generation token ($generationId != ${currentGeneration.get()})")
            return
        }

        val now = SystemClock.elapsedRealtime()
        // Debounce: if the same OCR text was spoken within 2.5 seconds, ignore
        if (trimmed == lastSpokenText && (now - lastOcrSpeakTime < 2500L)) {
            Log.d(tag, "Debounced duplicate OCR reading request")
            return
        }

        Log.i(tag, "Starting active OCR reading task: $trimmed")
        isOcrActive = true
        lastOcrSpeakTime = now
        pendingMessage = null

        // Stop any background speech immediately and start OCR
        tts?.stop()
        isSpeaking = false
        currentPriority = 85

        val msg = SpeechMessage(
            priority = 85,
            text = trimmed,
            severity = "INFO",
            mode = TtsMode.OCR,
            generationId = generationId,
            maxAgeMs = 15000L
        )
        executeSpeech(msg)
    }

    /**
     * Primary speech dispatcher with generation validation, mode verification,
     * deduplication, instant preemption, and bounded queue anti-spam protection.
     */
    @Synchronized
    fun speak(
        text: String,
        priority: Int = 50,
        severity: String = "INFO",
        mode: TtsMode = TtsMode.SYSTEM,
        generationId: Long = currentGeneration.get()
    ) {
        if (isMuted || !isInitialized) return
        if (isQuietMode && (severity == "INFO" || severity == "CAUTION")) return

        val trimmed = text.trim()
        if (trimmed.isEmpty()) return

        // 1. Generation Token Check (Ensures requests created under old modes/sessions are discarded)
        if (generationId != currentGeneration.get()) {
            Log.d(tag, "Discarded speech from stale generation ($generationId != ${currentGeneration.get()}): '$trimmed'")
            return
        }

        // 2. Active Mode Check: SYSTEM, VOICE_ASSISTANT, and on-demand user actions (priority >= 80) are ALWAYS allowed!
        val isAllowed = (mode == TtsMode.SYSTEM || mode == TtsMode.VOICE_ASSISTANT || mode == activeMode || priority >= 80)
        if (!isAllowed) {
            Log.d(tag, "Discarded speech from inactive mode ($mode != $activeMode, priority=$priority): '$trimmed'")
            return
        }

        // 3. OCR Protection: If OCR is actively being read, suppress routine background detections
        if (isOcrActive) {
            if (priority < 90) {
                return
            } else {
                Log.w(tag, "Critical hazard preemption during OCR: $trimmed")
                isOcrActive = false
            }
        }

        // 4. Deduplication & Rate Limiting: Suppress identical phrase spam within 2.5s window
        val now = SystemClock.elapsedRealtime()
        if (trimmed == lastSpokenText && (now - lastSpokenTime < 2500L)) {
            Log.d(tag, "Debounced duplicate phrase: '$trimmed'")
            return
        }

        val maxAge = if (priority >= 65) 5000L else 3000L
        val msg = SpeechMessage(
            priority = priority,
            text = trimmed,
            severity = severity,
            mode = mode,
            generationId = generationId,
            maxAgeMs = maxAge
        )

        if (severity == "CRITICAL" || severity == "WARNING") {
            lastImportantWarning = trimmed
        }

        // 5. Instant Preemption: If higher priority, stop current speech and play immediately
        if (isSpeaking && priority > currentPriority) {
            Log.i(tag, "Preempting lower priority speech (${currentPriority} -> ${priority}) for: $trimmed")
            pendingMessage = null
            tts?.stop()
            isSpeaking = false
            currentPriority = priority
            executeSpeech(msg)
            return
        }

        // 6. If idle, speak immediately
        if (!isSpeaking) {
            executeSpeech(msg)
        } else {
            // 7. Bounded Queue: Replace pending slot with ONLY the latest relevant message (no queue backlog)
            pendingMessage = msg
        }
    }

    /**
     * Speaks the latest message immediately, replacing any pending queue and cutting off current speech if higher or equal priority.
     */
    @Synchronized
    fun speakLatest(
        text: String,
        priority: Int = 60,
        severity: String = "INFO",
        mode: TtsMode = TtsMode.SYSTEM
    ) {
        val gen = currentGeneration.get()
        if (isSpeaking) {
            tts?.stop()
            isSpeaking = false
        }
        pendingMessage = null
        speak(text, priority, severity, mode, gen)
    }

    @Synchronized
    private fun executeSpeech(msg: SpeechMessage) {
        if (!isInitialized || isMuted) return

        isSpeaking = true
        currentlySpeakingText = msg.text
        currentlySpeakingMode = msg.mode
        currentlySpeakingGen = msg.generationId
        currentPriority = msg.priority
        lastSpokenTime = SystemClock.elapsedRealtime()
        lastSpokenText = msg.text

        val id = "utt_${utteranceIdCounter.getAndIncrement()}_${SystemClock.elapsedRealtime()}"
        activeUtteranceId = id

        // Split very long OCR texts into chunks if over 500 characters
        val text = msg.text
        if (text.length > 500) {
            val chunks = text.chunked(400)
            for (i in chunks.indices) {
                val chunk = chunks[i]
                val chunkId = "${id}_$i"
                if (i == chunks.lastIndex) {
                    activeUtteranceId = chunkId
                }
                val queueMode = if (i == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
                val result = tts?.speak(chunk, queueMode, null, chunkId)
                if (result != TextToSpeech.SUCCESS) {
                    Log.e(tag, "TTS speak failed for chunk $i (code: $result)")
                }
            }
        } else {
            val result = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
            if (result != TextToSpeech.SUCCESS) {
                Log.e(tag, "TTS speak failed (code: $result)")
            }
        }
    }

    fun repeatLast() {
        val last = lastImportantWarning ?: "No recent warning."
        speak(last, priority = 70, severity = "WARNING", mode = TtsMode.SYSTEM)
    }

    fun clearQueue() {
        pendingMessage = null
    }

    fun cancelCurrent() {
        stop()
    }

    @Synchronized
    fun stop() {
        currentGeneration.incrementAndGet()
        pendingMessage = null
        isOcrActive = false
        activeUtteranceId = null
        try {
            tts?.stop()
        } catch (_: Exception) {}
        isSpeaking = false
        currentlySpeakingText = null
        currentPriority = 0
    }

    fun shutdown() {
        stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
