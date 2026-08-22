package com.blindcap.app.ocr

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

class OcrManager {

    private val tag = "OcrManager"
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    suspend fun extractText(bitmap: Bitmap): String = withContext(Dispatchers.Default) {
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        suspendCancellableCoroutine<String> { continuation ->
            recognizer.process(inputImage)
                .addOnSuccessListener { visionText ->
                    val cleanText = visionText.text.trim()
                    if (cleanText.isEmpty()) {
                        continuation.resume("No readable text found in view.")
                    } else {
                        // Clean up excess newlines for natural speech
                        val spokenText = cleanText.replace("\n", " ").replace("  ", " ")
                        continuation.resume("Text detected: $spokenText")
                    }
                }
                .addOnFailureListener { exc ->
                    Log.e(tag, "ML Kit OCR failed: ${exc.message}")
                    continuation.resume("Text recognition failed.")
                }
        }
    }

    fun close() {
        recognizer.close()
    }
}
