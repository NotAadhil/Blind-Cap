package com.blindcap.app.engine

import kotlin.math.max

class DepthEstimator {

    companion object {
        // Calibrated reference heights in meters (COCO 80 classes)
        val REAL_WORLD_HEIGHTS = mapOf(
            "person" to 1.70f,
            "bicycle" to 1.10f,
            "car" to 1.50f,
            "motorcycle" to 1.20f,
            "bus" to 3.20f,
            "train" to 3.50f,
            "truck" to 2.80f,
            "chair" to 0.85f,
            "couch" to 0.85f,
            "bed" to 0.70f,
            "dining table" to 0.75f,
            "bench" to 0.80f,
            "toilet" to 0.75f,
            "dog" to 0.60f,
            "cat" to 0.30f,
            "horse" to 1.60f,
            "cow" to 1.40f,
            "bottle" to 0.25f,
            "cup" to 0.12f,
            "knife" to 0.20f,
            "scissors" to 0.20f,
            "cell phone" to 0.15f,
            "remote" to 0.15f,
            "backpack" to 0.50f,
            "suitcase" to 0.70f,
            "handbag" to 0.35f,
            "potted plant" to 0.60f,
            "tv" to 0.60f,
            "laptop" to 0.25f,
            "mouse" to 0.05f,
            "keyboard" to 0.04f,
            "umbrella" to 0.80f,
            "book" to 0.25f
        )
        const val DEFAULT_REAL_HEIGHT = 0.90f
        const val FOCAL_RATIO = 1.05f // Normalized vertical focal length estimate for phone cameras
    }

    /**
     * Estimate physical distance in meters using pinhole optical camera model.
     */
    fun estimateDistance(className: String, bboxHeightNorm: Float): Float {
        val refHeight = REAL_WORLD_HEIGHTS[className.lowercase()] ?: DEFAULT_REAL_HEIGHT
        val hClamped = max(0.015f, bboxHeightNorm)
        val distance = (refHeight * FOCAL_RATIO) / hClamped
        return max(0.3f, minOf(distance, 15.0f))
    }

    /**
     * Classify horizontal region with deadband hysteresis to prevent boundary flickering.
     */
    fun classifyRegion(cxNorm: Float, currentRegion: String? = null): String {
        if (currentRegion == null) {
            return when {
                cxNorm < 0.35f -> "left"
                cxNorm > 0.65f -> "right"
                else -> "center"
            }
        }
        // Hysteresis deadband: prevents jitter when panning or holding camera near boundaries
        return when (currentRegion) {
            "left" -> if (cxNorm > 0.42f) "center" else "left"
            "right" -> if (cxNorm < 0.58f) "center" else "right"
            "center" -> when {
                cxNorm < 0.28f -> "left"
                cxNorm > 0.72f -> "right"
                else -> "center"
            }
            else -> "center"
        }
    }
}