package com.blindcap.app.speech

import android.content.Context
import android.content.Intent
import android.os.Bundle
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

    private val tag = "VoiceCommandManager"
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false

    init {
        try {
            if (SpeechRecognizer.isRecognitionAvailable(context)) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                    setRecognitionListener(createListener())
                }
            } else {
                Log.w(tag, "Speech recognition service not available on device")
            }
        } catch (e: Exception) {
            Log.e(tag, "Error initializing SpeechRecognizer: ${e.message}", e)
        }
    }

    fun startListening() {
        if (isListening) return
        try {
            if (speechRecognizer == null) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                    setRecognitionListener(createListener())
                }
            }

            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            }

            speechRecognizer?.startListening(intent)
            isListening = true
            onStatusChanged(true, "Listening...")
        } catch (e: Exception) {
            Log.e(tag, "Failed to start speech recognition: ${e.message}", e)
            isListening = false
            onStatusChanged(false, "Voice input unavailable")
        }
    }

    fun stopListening() {
        if (!isListening) return
        try {
            speechRecognizer?.stopListening()
        } catch (e: Exception) {
            Log.e(tag, "Error stopping SpeechRecognizer: ${e.message}", e)
        } finally {
            isListening = false
            onStatusChanged(false, "Processing...")
        }
    }

    fun cancel() {
        try {
            speechRecognizer?.cancel()
        } catch (_: Exception) {}
        isListening = false
        onStatusChanged(false, "")
    }

    private fun createListener(): RecognitionListener {
        return object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                onStatusChanged(true, "Listening...")
            }

            override fun onBeginningOfSpeech() {
                onStatusChanged(true, "Hearing speech...")
            }

            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                isListening = false
                onStatusChanged(false, "Analyzing...")
            }

            override fun onError(error: Int) {
                isListening = false
                val errorMsg = when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH -> "No speech recognized"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Speech timeout"
                    SpeechRecognizer.ERROR_AUDIO -> "Audio error"
                    else -> "Voice recognition error"
                }
                onStatusChanged(false, errorMsg)
            }

            override fun onResults(results: Bundle?) {
                isListening = false
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val raw = matches[0].trim()
                    val command = parseCommand(raw)
                    onCommandReceived(command)
                } else {
                    onStatusChanged(false, "No speech heard")
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
    }

    private fun parseCommand(text: String): VoiceCommand {
        val lower = text.lowercase(Locale.getDefault())

        val type = when {
            // Color query
            lower.contains("color") || lower.contains("colour") || lower.contains("what shade") -> VoiceCommandType.COLOR_QUERY

            // Currency query
            lower.contains("currency") || lower.contains("money") || lower.contains("rupee") ||
            lower.contains("banknote") || lower.contains("note") || lower.contains("cash") ||
            lower.contains("how much") -> VoiceCommandType.CURRENCY_QUERY

            // Scene description query
            lower.contains("scene") || lower.contains("what is in front") || lower.contains("what's ahead") ||
            lower.contains("surroundings") || lower.contains("obstacles") || lower.contains("what do you see") ||
            lower.contains("describe") -> VoiceCommandType.SCENE_QUERY

            // OCR text reading query
            lower.contains("read") || lower.contains("text") || lower.contains("written") ||
            lower.contains("read this") || lower.contains("document") -> VoiceCommandType.OCR_QUERY

            // Barcode query
            lower.contains("barcode") || lower.contains("qr") || lower.contains("scan product") ||
            lower.contains("what product") -> VoiceCommandType.BARCODE_QUERY

            // SOS query
            lower.contains("help") || lower.contains("emergency") || lower.contains("sos") -> VoiceCommandType.SOS_QUERY

            // Stop speech
            lower.contains("stop") || lower.contains("be quiet") || lower.contains("silence") -> VoiceCommandType.STOP_SPEECH

            else -> VoiceCommandType.UNKNOWN
        }

        return VoiceCommand(type, text)
    }

    fun destroy() {
        try {
            speechRecognizer?.destroy()
            speechRecognizer = null
        } catch (_: Exception) {}
    }
}
