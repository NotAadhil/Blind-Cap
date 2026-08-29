package com.blindcap.app.speech

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import java.util.Locale

enum class VoiceCommandType {
    COLOR_QUERY,
    CURRENCY_QUERY,
    SCENE_QUERY,
    OCR_QUERY,
    BARCODE_QUERY,
    FACE_QUERY,
    SOS_QUERY,
    STOP_SPEECH,
    UNKNOWN
}

data class VoiceCommand(
    val type: VoiceCommandType,
    val rawText: String
)

class VoiceCommandManager(
    private val context: Context,
    private val onCommandReceived: (VoiceCommand) -> Unit,
    private val onStatusChanged: (Boolean, String) -> Unit
) {

    private val tag = "VoiceCommandMgr"
    private val mainHandler = Handler(Looper.getMainLooper())
    private var speechRecognizer: SpeechRecognizer? = null

    @Volatile
    var isListening = false
        private set

    private var timeoutRunnable: Runnable? = null

    init {
        mainHandler.post {
            ensureRecognizerInitialized()
        }
    }

    private fun ensureRecognizerInitialized() {
        if (speechRecognizer != null) return

        try {
            val appContext = context.applicationContext
            if (SpeechRecognizer.isRecognitionAvailable(appContext)) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(appContext).apply {
                    setRecognitionListener(createListener())
                }
                Log.i(tag, "SpeechRecognizer initialized successfully")
            } else {
                Log.w(tag, "Speech recognition is NOT available on this device")
            }
        } catch (e: Exception) {
            Log.e(tag, "Failed to initialize SpeechRecognizer: ${e.message}", e)
        }
    }

    fun startListening() {
        mainHandler.post {
            try {
                // Cancel any previous session
                try {
                    speechRecognizer?.cancel()
                    speechRecognizer?.destroy()
                } catch (_: Exception) {}
                speechRecognizer = null

                ensureRecognizerInitialized()

                if (speechRecognizer == null) {
                    isListening = false
                    onStatusChanged(false, "Speech recognizer unavailable")
                    return@post
                }

                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
                }

                isListening = true
                onStatusChanged(true, "Listening... Speak now")
                Log.i(tag, "SpeechRecognizer started listening")
                speechRecognizer?.startListening(intent)

                // 7-second safety timeout
                timeoutRunnable?.let { mainHandler.removeCallbacks(it) }
                timeoutRunnable = Runnable {
                    if (isListening) {
                        Log.i(tag, "Speech recognition safety timeout reached")
                        stopListening()
                    }
                }
                mainHandler.postDelayed(timeoutRunnable!!, 7000L)

            } catch (e: Exception) {
                Log.e(tag, "Error in startListening: ${e.message}", e)
                isListening = false
                onStatusChanged(false, "Voice assistant error")
            }
        }
    }

    fun stopListening() {
        mainHandler.post {
            timeoutRunnable?.let { mainHandler.removeCallbacks(it) }
            if (!isListening) return@post

            try {
                speechRecognizer?.stopListening()
            } catch (e: Exception) {
                Log.e(tag, "Error stopping SpeechRecognizer: ${e.message}", e)
            } finally {
                isListening = false
                onStatusChanged(false, "Processing query...")
            }
        }
    }

    fun cancel() {
        mainHandler.post {
            timeoutRunnable?.let { mainHandler.removeCallbacks(it) }
            try {
                speechRecognizer?.cancel()
            } catch (_: Exception) {}
            isListening = false
            onStatusChanged(false, "")
        }
    }

    private fun createListener(): RecognitionListener {
        return object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Log.i(tag, "RecognitionListener: onReadyForSpeech")
                onStatusChanged(true, "Listening... Speak now")
            }

            override fun onBeginningOfSpeech() {
                Log.i(tag, "RecognitionListener: onBeginningOfSpeech")
                onStatusChanged(true, "Hearing you...")
            }

            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                Log.i(tag, "RecognitionListener: onEndOfSpeech")
                isListening = false
                onStatusChanged(false, "Analyzing query...")
            }

            override fun onError(error: Int) {
                isListening = false
                timeoutRunnable?.let { mainHandler.removeCallbacks(it) }

                val errorMsg = when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH -> "No speech recognized"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Listening timed out"
                    SpeechRecognizer.ERROR_AUDIO -> "Audio recording issue"
                    SpeechRecognizer.ERROR_CLIENT -> "Voice assistant ready"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission required"
                    else -> "Voice assistant ready"
                }
                Log.w(tag, "SpeechRecognizer error: $error ($errorMsg)")
                onStatusChanged(false, errorMsg)
            }

            override fun onResults(results: Bundle?) {
                isListening = false
                timeoutRunnable?.let { mainHandler.removeCallbacks(it) }

                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val raw = matches[0].trim()
                    Log.i(tag, "Speech recognized text: '$raw'")
                    val command = parseCommand(raw)
                    onCommandReceived(command)
                } else {
                    onStatusChanged(false, "No speech heard")
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val partial = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!partial.isNullOrEmpty()) {
                    onStatusChanged(true, partial[0])
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
    }

    private fun parseCommand(text: String): VoiceCommand {
        val lower = text.lowercase(Locale.getDefault())

        val type = when {
            // Color queries
            lower.contains("color") || lower.contains("colour") || lower.contains("shade") -> VoiceCommandType.COLOR_QUERY

            // Currency queries
            lower.contains("currency") || lower.contains("money") || lower.contains("rupee") ||
            lower.contains("banknote") || lower.contains("note") || lower.contains("cash") ||
            lower.contains("how much") -> VoiceCommandType.CURRENCY_QUERY

            // Scene description queries
            lower.contains("scene") || lower.contains("what is in front") || lower.contains("what's in front") ||
            lower.contains("what is ahead") || lower.contains("what's ahead") || lower.contains("surroundings") ||
            lower.contains("obstacles") || lower.contains("what do you see") || lower.contains("describe") ||
            lower.contains("where am i") -> VoiceCommandType.SCENE_QUERY

            // OCR text reading queries
            lower.contains("read") || lower.contains("text") || lower.contains("written") ||
            lower.contains("read this") || lower.contains("document") || lower.contains("book") -> VoiceCommandType.OCR_QUERY

            // Barcode queries
            lower.contains("barcode") || lower.contains("qr") || lower.contains("scan product") ||
            lower.contains("product") || lower.contains("scan") -> VoiceCommandType.BARCODE_QUERY

            // Face recognition queries
            lower.contains("who is this") || lower.contains("who's this") || lower.contains("who is in front") ||
            lower.contains("recognize") || lower.contains("face") -> VoiceCommandType.FACE_QUERY

            // SOS queries
            lower.contains("help") || lower.contains("emergency") || lower.contains("sos") ||
            lower.contains("danger") -> VoiceCommandType.SOS_QUERY

            // Stop speech
            lower.contains("stop") || lower.contains("be quiet") || lower.contains("silence") ||
            lower.contains("shut up") || lower.contains("cancel") -> VoiceCommandType.STOP_SPEECH

            else -> VoiceCommandType.UNKNOWN
        }

        return VoiceCommand(type, text)
    }

    fun destroy() {
        mainHandler.post {
            try {
                speechRecognizer?.cancel()
                speechRecognizer?.destroy()
                speechRecognizer = null
            } catch (_: Exception) {}
        }
    }
}
