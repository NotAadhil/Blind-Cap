package com.blindcap.app.ai

import android.graphics.Bitmap
import android.graphics.Color
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.regex.Pattern
import kotlin.coroutines.resume

data class CurrencyResult(
    val denomination: String,
    val confidence: Float,
    val isDefinitive: Boolean,
    val spokenText: String
)

class CurrencyDetector {

    private val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    /**
     * Multi-modal Indian Rupee (INR) Banknote Identifier:
     * 1. Direct ML Kit OCR text reading (extracts "500", "200", "100", "50", "20", "10", "₹500", "RESERVE BANK OF INDIA")
     * 2. Calibrated chromatic spectrum analysis (Mahatma Gandhi New Series banknotes)
     */
    suspend fun detectCurrency(bitmap: Bitmap): CurrencyResult {
        // Step 1: Run ML Kit OCR on the banknote frame
        val ocrText = extractTextFromBitmap(bitmap)
        val ocrResult = evaluateOcrText(ocrText)

        // If OCR found a definitive denomination, return immediately!
        if (ocrResult != null) {
            return ocrResult
        }

        // Step 2: Fallback to robust Chromatic Distribution Analysis
        return analyzeBanknoteColor(bitmap)
    }

    private suspend fun extractTextFromBitmap(bitmap: Bitmap): String = suspendCancellableCoroutine { cont ->
        try {
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            textRecognizer.process(inputImage)
                .addOnSuccessListener { visionText ->
                    cont.resume(visionText.text)
                }
                .addOnFailureListener {
                    cont.resume("")
                }
        } catch (e: Exception) {
            cont.resume("")
        }
    }

    private fun evaluateOcrText(rawText: String): CurrencyResult? {
        if (rawText.isBlank()) return null
        val upper = rawText.uppercase()

        // Match 500 Rupees
        if (upper.contains("500") || upper.contains("₹500") || upper.contains("FIVE HUNDRED") || upper.contains("५००")) {
            return CurrencyResult("500 Rupees", 0.98f, true, "500 Rupees note.")
        }
        // Match 200 Rupees
        if (upper.contains("200") || upper.contains("₹200") || upper.contains("TWO HUNDRED") || upper.contains("२००")) {
            return CurrencyResult("200 Rupees", 0.98f, true, "200 Rupees note.")
        }
        // Match 100 Rupees
        if (upper.contains("100") || upper.contains("₹100") || upper.contains("ONE HUNDRED") || upper.contains("१००")) {
            return CurrencyResult("100 Rupees", 0.98f, true, "100 Rupees note.")
        }
        // Match 50 Rupees
        if (Pattern.compile("\b50\b").matcher(upper).find() || upper.contains("₹50") || upper.contains("FIFTY RUPEES") || upper.contains("५०")) {
            return CurrencyResult("50 Rupees", 0.98f, true, "50 Rupees note.")
        }
        // Match 20 Rupees
        if (Pattern.compile("\b20\b").matcher(upper).find() || upper.contains("₹20") || upper.contains("TWENTY RUPEES") || upper.contains("२०")) {
            return CurrencyResult("20 Rupees", 0.98f, true, "20 Rupees note.")
        }
        // Match 10 Rupees
        if (Pattern.compile("\b10\b").matcher(upper).find() || upper.contains("₹10") || upper.contains("TEN RUPEES") || upper.contains("१०")) {
            return CurrencyResult("10 Rupees", 0.98f, true, "10 Rupees note.")
        }

        return null
    }

    private fun analyzeBanknoteColor(bitmap: Bitmap): CurrencyResult {
        val width = bitmap.width
        val height = bitmap.height

        val cropX = (width * 0.10).toInt()
        val cropY = (height * 0.10).toInt()
        val cropW = (width * 0.80).toInt()
        val cropH = (height * 0.80).toInt()

        val sampleStep = 6
        var totalPixels = 0

        var count500 = 0 // Stone Grey / Olive (Hue: 60-140, Saturation: 0.05-0.35)
        var count200 = 0 // Bright Saffron / Orange (Hue: 20-55, Saturation >= 0.35)
        var count100 = 0 // Lavender / Soft Violet (Hue: 220-305, Saturation: 0.12-0.65)
        var count50 = 0  // Fluorescent Cyan / Light Blue (Hue: 160-220, Saturation >= 0.28)
        var count20 = 0  // Mustard / Greenish Yellow (Hue: 52-85, Saturation >= 0.30)
        var count10 = 0  // Chocolate Brown (Hue: 5-30, Saturation: 0.22-0.70, Value: 0.18-0.58)

        val hsv = FloatArray(3)

        for (y in cropY until (cropY + cropH) step sampleStep) {
            for (x in cropX until (cropX + cropW) step sampleStep) {
                val pixel = bitmap.getPixel(x, y)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                totalPixels++

                Color.RGBToHSV(r, g, b, hsv)
                val hue = hsv[0]
                val sat = hsv[1]
                val value = hsv[2]

                if (value < 0.15f || value > 0.98f) continue

                // ₹500: Stone Grey / Olive
                if (hue in 55f..145f && sat in 0.05f..0.38f && value in 0.25f..0.80f) {
                    count500++
                }
                // ₹200: Bright Saffron / Orange
                else if (hue in 18f..55f && sat >= 0.32f && value >= 0.40f) {
                    count200++
                }
                // ₹100: Lavender / Violet-Purple
                else if (hue in 215f..310f && sat in 0.10f..0.65f && value in 0.30f..0.90f) {
                    count100++
                }
                // ₹50: Fluorescent Cyan / Light Blue
                else if (hue in 160f..220f && sat >= 0.25f && value >= 0.35f) {
                    count50++
                }
                // ₹20: Greenish Yellow / Mustard
                else if (hue in 52f..85f && sat >= 0.28f && value >= 0.38f) {
                    count20++
                }
                // ₹10: Chocolate Brown
                else if (hue in 5f..30f && sat in 0.20f..0.75f && value in 0.18f..0.60f) {
                    count10++
                }
            }
        }

        if (totalPixels == 0) {
            return CurrencyResult("Unknown", 0f, false, "Banknote not detected. Please bring note closer to camera.")
        }

        val scores = listOf(
            Triple("500 Rupees", count500.toFloat() / totalPixels, "₹500 note"),
            Triple("200 Rupees", count200.toFloat() / totalPixels, "₹200 note"),
            Triple("100 Rupees", count100.toFloat() / totalPixels, "₹100 note"),
            Triple("50 Rupees", count50.toFloat() / totalPixels, "₹50 note"),
            Triple("20 Rupees", count20.toFloat() / totalPixels, "₹20 note"),
            Triple("10 Rupees", count10.toFloat() / totalPixels, "₹10 note")
        ).sortedByDescending { it.second }

        val top = scores[0]
        val second = scores[1]

        val confidence = (top.second * 4.0f).coerceIn(0f, 1f)

        return if (confidence >= 0.45f && top.second > 0.12f) {
            CurrencyResult(
                denomination = top.first,
                confidence = confidence,
                isDefinitive = true,
                spokenText = "${top.first} note."
            )
        } else if (confidence >= 0.25f && top.second > 0.08f) {
            CurrencyResult(
                denomination = top.first,
                confidence = confidence,
                isDefinitive = false,
                spokenText = "Likely ${top.first}. Hold steady for confirmation."
            )
        } else {
            CurrencyResult(
                denomination = "Unknown",
                confidence = confidence,
                isDefinitive = false,
                spokenText = "Banknote not recognized. Please align note in center view."
            )
        }
    }

    fun close() {
        textRecognizer.close()
    }
}
