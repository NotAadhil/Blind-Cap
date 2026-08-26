package com.blindcap.app.ai

import android.graphics.RectF
import android.os.SystemClock

enum class FaceObservationSource {
    MLKIT_DETECTION,    // Exact face detector result
    TRACK_ANCHOR        // Real-time track-following head box derived from live YOLO person track
}

/**
 * Transient, timestamped visual face observation representing where a face is RIGHT NOW.
 * Decoupled from historical recognition events and TTS triggers.
 */
data class FaceObservation(
    val trackId: Int,
    val bbox: RectF,                        // Normalized [0..1] face rectangle
    val identity: String? = null,           // e.g. "Aadhil" or null if unknown
    val isKnown: Boolean = false,           // true if enrolled contact
    val similarity: Float = 0f,             // Cosine similarity
    val isFacingUser: Boolean = false,      // Gaze direction
    val timestamp: Long = SystemClock.elapsedRealtime(),
    val source: FaceObservationSource = FaceObservationSource.MLKIT_DETECTION
) {
    fun getAgeMs(currentTimeMs: Long = SystemClock.elapsedRealtime()): Long {
        return (currentTimeMs - timestamp).coerceAtLeast(0L)
    }

    fun isStale(currentTimeMs: Long = SystemClock.elapsedRealtime(), maxAgeMs: Long = 350L): Boolean {
        return getAgeMs(currentTimeMs) > maxAgeMs
    }
}
