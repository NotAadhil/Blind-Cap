package com.blindcap.app.ai

import android.graphics.Bitmap
import android.graphics.Color

data class ColorResult(
    val colorName: String,
    val brightnessLevel: String,
    val hexCode: String,
    val spokenDescription: String
)

class ColorDetector {

    /**
     * Extracts dominant color from the center focus reticle (60x60 px)
     * and evaluates scene ambient brightness (Low / Medium / High).
     */
    fun detectColor(bitmap: Bitmap): ColorResult {
        val centerX = bitmap.width / 2
        val centerY = bitmap.height / 2
        val patchRadius = 30

        val startX = (centerX - patchRadius).coerceAtLeast(0)
        val endX = (centerX + patchRadius).coerceAtMost(bitmap.width - 1)
        val startY = (centerY - patchRadius).coerceAtLeast(0)
        val endY = (centerY + patchRadius).coerceAtMost(bitmap.height - 1)

        var totalR = 0L
        var totalG = 0L
        var totalB = 0L
        var count = 0

        for (y in startY..endY step 2) {
            for (x in startX..endX step 2) {
                val pixel = bitmap.getPixel(x, y)
                totalR += Color.red(pixel)
                totalG += Color.green(pixel)
                totalB += Color.blue(pixel)
                count++
            }
        }

        if (count == 0) {
            return ColorResult("Unknown", "Medium", "#888888", "Could not sample color.")
        }

        val avgR = (totalR / count).toInt()
        val avgG = (totalG / count).toInt()
        val avgB = (totalB / count).toInt()

        val hex = String.format("#%02X%02X%02X", avgR, avgG, avgB)

        val hsv = FloatArray(3)
        Color.RGBToHSV(avgR, avgG, avgB, hsv)
        val hue = hsv[0]
        val sat = hsv[1]
        val value = hsv[2]

        val brightness = when {
            value < 0.22f -> "Low Light"
            value > 0.78f -> "Bright Light"
            else -> "Normal Light"
        }

        val colorName = when {
            // Greyscale
            value < 0.15f -> "Jet Black"
            value < 0.28f && sat < 0.20f -> "Charcoal Grey"
            sat < 0.12f && value > 0.85f -> "White"
            sat < 0.15f && value in 0.50f..0.85f -> "Grey"
            sat < 0.18f && value in 0.28f..0.50f -> "Dark Grey"
            sat < 0.22f && value > 0.85f -> "Off-White"

            // Browns & Beiges
            hue in 15f..45f && sat in 0.20f..0.55f && value in 0.20f..0.55f -> "Brown"
            hue in 25f..50f && sat in 0.18f..0.45f && value in 0.60f..0.88f -> "Beige"

            // Primary & Secondary Colors
            hue in 0f..12f || hue >= 345f -> if (value < 0.45f) "Dark Red" else "Red"
            hue in 12f..24f -> if (value < 0.50f) "Maroon" else "Crimson"
            hue in 24f..42f -> if (sat > 0.70f) "Bright Orange" else "Orange"
            hue in 42f..68f -> if (value > 0.70f) "Bright Yellow" else "Mustard Yellow"
            hue in 68f..90f -> "Lime Green"
            hue in 90f..150f -> if (value < 0.45f) "Forest Green" else "Green"
            hue in 150f..175f -> "Emerald Green"
            hue in 175f..200f -> "Cyan"
            hue in 200f..225f -> if (value > 0.70f) "Sky Blue" else "Light Blue"
            hue in 225f..255f -> if (value < 0.45f) "Navy Blue" else "Blue"
            hue in 255f..285f -> if (value < 0.45f) "Indigo" else "Purple"
            hue in 285f..320f -> "Violet"
            hue in 320f..345f -> if (value > 0.65f) "Pink" else "Magenta"

            else -> "Multi-tone"
        }

        val spoken = "$colorName. Scene brightness: $brightness."
        return ColorResult(colorName, brightness, hex, spoken)
    }
}
