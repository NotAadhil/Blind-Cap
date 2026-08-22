package com.blindcap.app.speech

import android.content.Context
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

data class SpeechMessage(
    val priority: Int, // Higher number = higher urgency
    val text: String,
    val severity: String, // "CRITICAL", "WARNING", "CAUTION", "INFO"
    val timestampMs: Long = SystemClock.elapsedRealtime(),
    val maxAgeMs: Long = 2500L // Messages older than 2.5s are considered stale and dropped
)

class TtsManager(private val context: Context, private val onReadyCallback: (() -> Unit)? = null) :
    TextToSpeech.OnInitListener {

    private val tag = "TtsManager"
    private var tts: TextToSpeech? = null
    private var isInitialized = false

    private var pendingMessage: SpeechMessage? = null
    private var currentlySpeakingText: String? = null
    private var currentPriority: Int = 0
    private var lastSpokenTime: Long = 0L

    var isSpeaking: Boolean = false
        private set

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
                tts?.setSpeechRate(1.10f) // Optimized natural speech rate for low latency
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
                isSpeaking = true
            }

            override fun onDone(utteranceId: String?) {
                handleSpeechFinished()
            }

            override fun onError(utteranceId: String?) {
                handleSpeechFinished()
            }
        })
    }

    @Synchronized
    private fun handleSpeechFinished() {
        isSpeaking = false
        isOcrActive = false
        currentlySpeakingText = null
        currentPriority = 0

        // Check if there is a pending fresh message (non-stale)
        val next = pendingMessage
        pendingMessage = null

        if (next != null) {
            val age = SystemClock.elapsedRealtime() - next.timestampMs
            if (age <= next.maxAgeMs) {
                executeSpeech(next)
            } else {
                Log.i(tag, "Discarded stale TTS message (${age}ms old): ${next.text}")
            }
        }
    }

    /**
     * Dedicated method for user-requested OCR reading that locks priority and protects against background detection interruptions.
     */
    @Synchronized
    fun startOcrReading(ocrText: String) {
        if (isMuted || !isInitialized) return
        val trimmed = ocrText.trim()
        if (trimmed.isEmpty()) return

        Log.i(tag, "Starting active OCR reading task: $trimmed")
        isOcrActive = true
        pendingMessage = null

        // Stop any background speech immediately and start OCR
        tts?.stop()
        isSpeaking = false
        currentPriority = 85 // OCR priority above routine detections (50-70)

        val msg = SpeechMessage(priority = 85, text = trimmed, severity = "INFO", maxAgeMs = 10000L)
        executeSpeech(msg)
    }

    @Synchronized
    fun speak(text: String, priority: Int, severity: String) {
        if (isMuted || !isInitialized) return
        if (isQuietMode && (severity == "INFO" || severity == "CAUTION")) return

        val trimmed = text.trim()
        if (trimmed.isEmpty()) return

        // 1. OCR Protection: If OCR is actively being read, do NOT let routine background detections interrupt or queue!
        if (isOcrActive) {
            if (priority < 90) {
                // Background detections (priority 40..70) are silently suppressed during OCR reading
                return
            } else {
                // Only critical collision danger (priority 90+) can interrupt OCR
                Log.w(tag, "Critical hazard preemption during OCR: $trimmed")
                isOcrActive = false
            }
        }

        // Deduplication: Do not repeat identical speech if currently playing or played within last 3 seconds
        val now = SystemClock.elapsedRealtime()
        if (trimmed == currentlySpeakingText && (now - lastSpokenTime < 3000L)) {
            return
        }

        val msg = SpeechMessage(priority, trimmed, severity)

        if (severity == "CRITICAL" || severity == "WARNING") {
            lastImportantWarning = trimmed
        }

        // 2. Instant Preemption: If higher priority, cut off current speech immediately
        if (isSpeaking && priority > currentPriority) {
            Log.i(tag, "Preempting lower priority speech (${currentPriority} -> ${priority}) for: $trimmed")
            pendingMessage = null
            tts?.stop()
            isSpeaking = false
            currentPriority = priority
            executeSpeech(msg)
            return
        }

        // 3. If idle, speak immediately
        if (!isSpeaking) {
            executeSpeech(msg)
        } else {
            // 4. Stale queue prevention: Replace pending slot with only the freshest message
            pendingMessage = msg
        }
    }

    @Synchronized
    private fun executeSpeech(msg: SpeechMessage) {
        if (!isInitialized || isMuted) return
        isSpeaking = true
        currentlySpeakingText = msg.text
        currentPriority = msg.priority
        lastSpokenTime = SystemClock.elapsedRealtime()

        val utteranceId = "utterance_${SystemClock.elapsedRealtime()}"
        tts?.speak(msg.text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    fun repeatLast() {
        val last = lastImportantWarning ?: "No recent warning."
        speak(last, priority = 70, severity = "WARNING")
    }

    fun stop() {
        pendingMessage = null
        isOcrActive = false
        tts?.stop()
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