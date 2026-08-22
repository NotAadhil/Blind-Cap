package com.blindcap.app.engine

import android.graphics.RectF
import android.os.SystemClock
import com.blindcap.app.ai.Detection
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class HazardEvent(
    val warningText: String? = null,
    val speakPriority: Int = 0,
    val severity: String = "INFO",
    val category: String = "",
    val dedupeKey: String = "",
    val hazardDetected: Boolean = false,
    val activeHazard: TrackedItem? = null,
    val allHazards: List<TrackedItem> = emptyList()
)

data class TrackedItem(
    val id: Int,
    var className: String,
    var confidence: Float,
    var bbox: RectF,
    var center: Pair<Float, Float>,
    var region: String,
    var distanceM: Float,
    var framesSeen: Int = 1,
    var framesMissing: Int = 0,
    var lastWarningTime: Long = 0L,
    var criticalTier: Int = 0,
    var lastAnnouncedDistanceM: Float? = null,
    var hasAnnouncedInFrame: Boolean = false
)

class DecisionEngine {

    private var nextTrackId = 1
    val trackedObjects = mutableMapOf<Int, TrackedItem>()

    private var lastSpokenSceneSignature: String? = null
    private var lastSceneSpeakTime: Long = 0L
    private var lastObstructionSpeakTime: Long = 0L
    private var lastHadObjects: Boolean = false

    private val obstructionCooldownMs = 3500L
    private val sceneChangeCooldownMs = 4000L

    fun evaluate(detections: List<Detection>): HazardEvent {
        val now = SystemClock.elapsedRealtime()

        // 1. Spatial Tracking Match (IoU & Center Distance)
        val matchedTrackIds = mutableSetOf<Int>()
        val currentItems = mutableListOf<TrackedItem>()

        for (det in detections) {
            var bestTrackId: Int? = null
            var bestScore = 0.0f

            for ((id, track) in trackedObjects) {
                if (id in matchedTrackIds) continue
                val iou = computeIoU(det.bbox, track.bbox)
                val cDist = computeCenterDist(det.center, track.center)

                if (iou > 0.25f || cDist < 0.20f) {
                    val score = iou * 0.7f + (1.0f - cDist) * 0.3f
                    if (score > bestScore) {
                        bestScore = score
                        bestTrackId = id
                    }
                }
            }

            if (bestTrackId != null) {
                matchedTrackIds.add(bestTrackId)
                val track = trackedObjects[bestTrackId]!!
                track.framesSeen++
                track.framesMissing = 0
                track.className = det.className
                track.confidence = det.confidence
                track.bbox = det.bbox
                track.center = det.center
                track.region = det.region
                track.distanceM = det.estimatedDistanceM
                currentItems.add(track)
            } else {
                val newId = nextTrackId++
                val newTrack = TrackedItem(
                    id = newId,
                    className = det.className,
                    confidence = det.confidence,
                    bbox = det.bbox,
                    center = det.center,
                    region = det.region,
                    distanceM = det.estimatedDistanceM
                )
                trackedObjects[newId] = newTrack
                matchedTrackIds.add(newId)
                currentItems.add(newTrack)
            }
        }

        // 2. Remove Stale Tracks (missing for >= 3 frames)
        val iterator = trackedObjects.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key !in matchedTrackIds) {
                entry.value.framesMissing++
                if (entry.value.framesMissing >= 3) {
                    iterator.remove()
                }
            }
        }

        val activeTracks = trackedObjects.values.filter { it.framesSeen >= 2 && it.framesMissing == 0 }

        // 3. Path Cleared Transition
        if (activeTracks.isEmpty()) {
            if (lastHadObjects) {
                lastHadObjects = false
                lastSpokenSceneSignature = null
                return HazardEvent(
                    warningText = "Path is clear.",
                    speakPriority = 40,
                    severity = "INFO",
                    category = "path_clear",
                    hazardDetected = false
                )
            }
            return HazardEvent(hazardDetected = false)
        }

        lastHadObjects = true

        // 4. Alert Hierarchy: Danger Zone Tier 1 (<=1.8m) and Tier 2 (<=0.9m) in Center Corridor
        val centerObstacles = activeTracks.filter { it.region == "center" && it.distanceM <= 1.8f }
        if (centerObstacles.isNotEmpty()) {
            val closest = centerObstacles.minByOrNull { it.distanceM }!!
            val tier = if (closest.distanceM <= 0.9f) 2 else 1

            val shouldSpeakObstruction = (closest.criticalTier != tier) ||
                    (now - lastObstructionSpeakTime > obstructionCooldownMs)

            if (shouldSpeakObstruction) {
                closest.criticalTier = tier
                lastObstructionSpeakTime = now
                val warning = formatPathObstruction(centerObstacles, tier)
                return HazardEvent(
                    warningText = warning,
                    speakPriority = if (tier >= 2) 90 else 80,
                    severity = if (tier >= 2) "CRITICAL" else "WARNING",
                    category = closest.className,
                    hazardDetected = true,
                    activeHazard = closest,
                    allHazards = activeTracks
                )
            }
        }

        // 5. In-Frame Scene Announcement & Dynamic Distance Updates
        val sceneSig = buildSceneSignature(activeTracks)
        val sceneChanged = (sceneSig != lastSpokenSceneSignature)
        val cooldownPassed = (now - lastSceneSpeakTime > sceneChangeCooldownMs)

        if (sceneChanged && cooldownPassed) {
            lastSpokenSceneSignature = sceneSig
            lastSceneSpeakTime = now
            val sceneDesc = formatSceneDescription(activeTracks)
            return HazardEvent(
                warningText = sceneDesc,
                speakPriority = 50,
                severity = "CAUTION",
                category = "scene",
                hazardDetected = true,
                allHazards = activeTracks
            )
        }

        return HazardEvent(
            hazardDetected = true,
            allHazards = activeTracks
        )
    }

    private fun buildSceneSignature(tracks: List<TrackedItem>): String {
        return tracks.sortedBy { it.id }.joinToString("|") {
            "${it.className}_${it.region}_${(it.distanceM * 2).roundToInt() / 2f}"
        }
    }

    fun getFullSceneSummary(detections: List<Detection>): String {
        val items = if (detections.isNotEmpty()) {
            detections.map {
                TrackedItem(
                    id = 0,
                    className = it.className,
                    confidence = it.confidence,
                    bbox = it.bbox,
                    center = it.center,
                    region = it.region,
                    distanceM = it.estimatedDistanceM
                )
            }
        } else {
            trackedObjects.values.toList()
        }

        if (items.isEmpty()) return "The path ahead is completely clear with no detected obstacles."
        val desc = formatSceneDescription(items)
        return "Scene summary: $desc"
    }

    private fun formatSceneDescription(tracks: List<TrackedItem>): String {
        if (tracks.isEmpty()) return "Path is clear."

        val leftItems = tracks.filter { it.region == "left" }
        val centerItems = tracks.filter { it.region == "center" }
        val rightItems = tracks.filter { it.region == "right" }

        val activeRegions = listOf(
            Triple("center", centerItems, "ahead"),
            Triple("left", leftItems, "on the left"),
            Triple("right", rightItems, "on the right")
        ).filter { it.second.isNotEmpty() }

        if (activeRegions.isEmpty()) return "Path is clear."

        if (activeRegions.size == 1) {
            val (reg, items, _) = activeRegions[0]
            val itemsStr = formatItemsList(items)
            val avgDist = items.map { it.distanceM }.average().toFloat()
            val distSuffix = formatDistanceStr(avgDist)

            return when (reg) {
                "center" -> "$itemsStr detected$distSuffix."
                "left" -> "$itemsStr on your left$distSuffix."
                else -> "$itemsStr on your right$distSuffix."
            }
        }

        val regionClauses = mutableListOf<String>()
        for ((reg, items, posLabel) in activeRegions) {
            val itemsStr = formatItemsList(items)
            val avgDist = items.map { it.distanceM }.average().toFloat()
            val distSuffix = formatDistanceStr(avgDist)
            regionClauses.add("$itemsStr $posLabel$distSuffix")
        }

        return when (regionClauses.size) {
            2 -> "${regionClauses[0]} and ${regionClauses[1]}."
            else -> "${regionClauses[0]}, ${regionClauses[1]}, and ${regionClauses[2]}."
        }
    }

    private fun formatPathObstruction(items: List<TrackedItem>, tier: Int): String {
        val itemsStr = formatItemsList(items)
        val minDist = items.minOfOrNull { it.distanceM } ?: 1.0f
        val distSuffix = formatDistanceStr(minDist)
        return if (tier >= 2) {
            "Stop! $itemsStr is very close$distSuffix. Please stop!"
        } else {
            "Stop! $itemsStr is obstructing your path$distSuffix."
        }
    }

    private fun formatItemsList(items: List<TrackedItem>): String {
        val counts = items.groupingBy { it.className }.eachCount()
        val phrases = counts.map { (name, count) ->
            if (count > 1) "$count ${pluralize(name)}" else name
        }
        return when (phrases.size) {
            0 -> "Obstacle"
            1 -> phrases[0]
            2 -> "${phrases[0]} and ${phrases[1]}"
            else -> phrases.dropLast(1).joinToString(", ") + ", and " + phrases.last()
        }.replaceFirstChar { it.uppercase() }
    }

    private fun pluralize(name: String): String {
        return when (name.lowercase()) {
            "person" -> "people"
            "knife" -> "knives"
            "bench" -> "benches"
            "couch" -> "couches"
            "glass", "wine glass" -> "wine glasses"
            else -> if (name.endsWith("s") || name.endsWith("sh") || name.endsWith("ch")) "${name}es" else "${name}s"
        }
    }

    private fun formatDistanceStr(distM: Float): String {
        val rounded = (distM * 10.0f).roundToInt() / 10.0f
        return if (rounded in 0.4f..6.0f) {
            ", $rounded meters away"
        } else {
            ""
        }
    }

    private fun computeIoU(b1: RectF, b2: RectF): Float {
        val left = max(b1.left, b2.left)
        val top = max(b1.top, b2.top)
        val right = min(b1.right, b2.right)
        val bottom = min(b1.bottom, b2.bottom)

        val interW = max(0f, right - left)
        val interH = max(0f, bottom - top)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val area1 = b1.width() * b1.height()
        val area2 = b2.width() * b2.height()
        val unionArea = area1 + area2 - interArea
        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    private fun computeCenterDist(c1: Pair<Float, Float>, c2: Pair<Float, Float>): Float {
        val dx = c1.first - c2.first
        val dy = c1.second - c2.second
        return sqrt(dx * dx + dy * dy)
    }

    fun reset() {
        trackedObjects.clear()
        lastSpokenSceneSignature = null
        lastSceneSpeakTime = 0L
        lastHadObjects = false
    }
}