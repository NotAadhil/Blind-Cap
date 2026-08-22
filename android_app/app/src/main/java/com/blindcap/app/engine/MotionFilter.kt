package com.blindcap.app.engine

import java.util.ArrayDeque
import kotlin.math.abs

class MotionFilter {

    private val areaHistory = ArrayDeque<Float>(8)
    private var lastAnnouncedDistance: Float? = null

    companion object {
        const val APPROACH_GROWTH_THRESHOLD = 1.25f
        const val SIGNIFICANT_DISTANCE_DELTA = 0.80f // meters
    }

    fun updateArea(areaRatio: Float) {
        if (areaHistory.size >= 8) {
            areaHistory.removeFirst()
        }
        areaHistory.addLast(areaRatio)
    }

    /**
     * Returns true if object bounding box is expanding significantly over time (approaching).
     */
    fun isApproaching(): Boolean {
        if (areaHistory.size < 3) return false
        val list = areaHistory.toList()
        val earlyAvg = (list[0] + list[1]) / 2.0f
        val recentAvg = (list[list.size - 1] + list[list.size - 2]) / 2.0f
        if (earlyAvg <= 0.0001f) return false
        return (recentAvg / earlyAvg) > APPROACH_GROWTH_THRESHOLD
    }

    fun hasSignificantDistanceShift(currentDistance: Float): Boolean {
        val last = lastAnnouncedDistance ?: return true
        return abs(currentDistance - last) >= SIGNIFICANT_DISTANCE_DELTA
    }

    fun recordAnnouncedDistance(distance: Float) {
        lastAnnouncedDistance = distance
    }

    fun reset() {
        areaHistory.clear()
        lastAnnouncedDistance = null
    }
}
