package com.blindcap.app.ai

import android.graphics.RectF

data class Detection(
    var className: String,
    val classId: Int,
    val confidence: Float,
    val bbox: RectF, // Normalized coordinates [0.0, 1.0] (left, top, right, bottom)
    val center: Pair<Float, Float>, // Normalized center point (cx, cy)
    val areaRatio: Float,
    val region: String = "center", // "left", "center", "right"
    var estimatedDistanceM: Float = 2.5f,
    val frameId: Long = 0L
)
