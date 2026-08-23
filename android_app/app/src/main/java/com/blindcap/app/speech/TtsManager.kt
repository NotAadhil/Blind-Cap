package com.blindcap.app.speech

import android.content.Context
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

data class SpeechMessage(
    val priority: Int, // Higher number = higher urgency
    val text: String,
    val severity: String, // "CRITICAL", "WARNING", "CAUTION", "INFO"
    val timestampMs: Long = SystemClock.elapsedRealtime(),
    val maxAgeMs: Long = 3000L
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
        // Only handle completion for the CURRENT active utterance, ignore callbacks from previously cancelled ones!
        if (utteranceId != activeUtteranceId && activeUtteranceId != null) {
            Log.d(tag, "Ignored completion of old/cancelled utterance: $utteranceId (active: $activeUtteranceId)")
            return
        }

        isSpeaking = false
        isOcrActive = false
        activeUtteranceId = null
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
     * Includes debouncing: rejects repeated identical or rapid successive OCR calls.
     */
    @Synchronized
    fun startOcrReading(ocrText: String) {
        if (isMuted || !isInitialized) return
        val trimmed = ocrText.trim()
        if (trimmed.isEmpty()) return

        val now = SystemClock.elapsedRealtime()
        // Debounce: if the same OCR text was spoken within 2.5 seconds, ignore
        if (trimmed == currentlySpeakingText && (now - lastOcrSpeakTime < 2500L)) {
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
        currentPriority = 85 // OCR priority above routine detections (50-70)

        val msg = SpeechMessage(priority = 85, text = trimmed, severity = "INFO", maxAgeMs = 15000L)
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

        val id = "utt_${utteranceIdCounter.getAndIncrement()}_${SystemClock.elapsedRealtime()}"
        activeUtteranceId = id

        // Split very long OCR texts into chunks if over 500 characters to prevent TTS engine dropouts
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
        speak(last, priority = 70, severity = "WARNING")
    }

    fun stop() {
        pendingMessage = null
        isOcrActive = false
        activeUtteranceId = null
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