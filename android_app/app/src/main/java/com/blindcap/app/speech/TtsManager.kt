package com.blindcap.app.speech

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.PriorityQueue
import java.util.concurrent.ConcurrentHashMap

data class SpeechMessage(
    val priority: Int, // Higher number = higher urgency
    val text: String,
    val severity: String, // "CRITICAL", "WARNING", "CAUTION", "INFO"
    val timestamp: Long = System.currentTimeMillis()
) : Comparable<SpeechMessage> {
    override fun compareTo(other: SpeechMessage): Int {
        // Max-heap: higher priority comes first
        return other.priority.compareTo(this.priority)
    }
}

class TtsManager(private val context: Context, private val onReadyCallback: (() -> Unit)? = null) :
    TextToSpeech.OnInitListener {

    private val tag = "TtsManager"
    private var tts: TextToSpeech? = null
    private var isInitialized = false

    private val speechQueue = PriorityQueue<SpeechMessage>()
    private val spokenKeys = ConcurrentHashMap<String, Long>()
    private var isSpeaking = false
    private var currentPriority = 0

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
                tts?.setSpeechRate(1.08f) // Slightly faster for efficient listening
                setupUtteranceListener()
                Log.info(tag, "TextToSpeech initialized successfully")
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
                isSpeaking = false
                currentPriority = 0
                processNextInQueue()
            }

            override fun onError(utteranceId: String?) {
                isSpeaking = false
                currentPriority = 0
                processNextInQueue()
            }
        })
    }

    @Synchronized
    fun speak(text: String, priority: Int, severity: String) {
        if (isMuted) return
        if (isQuietMode && (severity == "INFO" || severity == "CAUTION")) return

        val msg = SpeechMessage(priority, text, severity)

        if (severity == "CRITICAL" || severity == "WARNING") {
            lastImportantWarning = text
        }

        // Sub-50ms Instant Preemption: if message is higher priority than what is currently playing, cut off immediately!
        if (isSpeaking && priority > currentPriority) {
            tts?.stop()
            isSpeaking = false
            currentPriority = priority
            executeSpeech(msg)
            return
        }

        if (!isSpeaking) {
            executeSpeech(msg)
        } else {
            speechQueue.offer(msg)
        }
    }

    @Synchronized
    private fun executeSpeech(msg: SpeechMessage) {
        if (!isInitialized || isMuted) return
        isSpeaking = true
        currentPriority = msg.priority
        val utteranceId = "utterance_${System.currentTimeMillis()}"
        tts?.speak(msg.text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    @Synchronized
    private fun processNextInQueue() {
        if (!isSpeaking && speechQueue.isNotEmpty()) {
            val next = speechQueue.poll()
            if (next != null) {
                executeSpeech(next)
            }
        }
    }

    fun repeatLast() {
        val last = lastImportantWarning ?: "No recent warning."
        speak(last, priority = 70, severity = "WARNING")
    }

    fun stop() {
        speechQueue.clear()
        tts?.stop()
        isSpeaking = false
        currentPriority = 0
    }

    fun shutdown() {
        stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
