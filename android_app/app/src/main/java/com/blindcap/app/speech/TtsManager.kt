package com.blindcap.app.speech

import android.content.Context
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

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
        currentlySpeakingText = null
        currentPriority = 0

        // Check if there is a pending fresh message
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

    @Synchronized
    fun speak(text: String, priority: Int, severity: String) {
        if (isMuted || !isInitialized) return
        if (isQuietMode && (severity == "INFO" || severity == "CAUTION")) return

        val trimmed = text.trim()
        if (trimmed.isEmpty()) return

        // Deduplication: Do not repeat identical speech if currently playing or played within last 2.5 seconds
        val now = SystemClock.elapsedRealtime()
        if (trimmed == currentlySpeakingText && (now - lastSpokenTime < 2500L)) {
            return
        }

        val msg = SpeechMessage(priority, trimmed, severity)

        if (severity == "CRITICAL" || severity == "WARNING") {
            lastImportantWarning = trimmed
        }

        // 1. Instant Preemption: If higher priority (or urgent danger), cut off current speech immediately!
        if (isSpeaking && priority > currentPriority) {
            Log.i(tag, "Preempting lower priority speech for: $trimmed")
            pendingMessage = null
            tts?.stop()
            isSpeaking = false
            currentPriority = priority
            executeSpeech(msg)
            return
        }

        // 2. If idle, speak immediately
        if (!isSpeaking) {
            executeSpeech(msg)
        } else {
            // 3. Stale queue prevention: Replace any pending message with only the newest message
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