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

enum class TrackState {
    NEW,
    CONFIRMED,
    ANNOUNCED,
    PERSISTENT,
    DEPARTED
}

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
    var state: TrackState = TrackState.NEW,
    var isObstructionAnnounced: Boolean = false,
    var criticalTier: Int = 0,
    val classHistory: ArrayDeque<String> = ArrayDeque(6)
) {
    init {
        classHistory.add(className)
    }

    fun updateClass(newClass: String) {
        if (classHistory.size >= 6) classHistory.removeFirst()
        classHistory.addLast(newClass)
        // Majority voting to eliminate classification flicker
        className = classHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: newClass
    }

    fun smoothBbox(newBbox: RectF) {
        val alpha = 0.5f
        bbox = RectF(
            bbox.left * (1 - alpha) + newBbox.left * alpha,
            bbox.top * (1 - alpha) + newBbox.top * alpha,
            bbox.right * (1 - alpha) + newBbox.right * alpha,
            bbox.bottom * (1 - alpha) + newBbox.bottom * alpha
        )
        center = Pair((bbox.left + bbox.right) * 0.5f, (bbox.top + bbox.bottom) * 0.5f)
    }

    fun smoothDistance(newDist: Float) {
        val delta = abs(newDist - distanceM)
        if (delta > 0.15f) {
            val alpha = if (delta > 0.6f) 0.5f else 0.25f
            distanceM = (distanceM * (1 - alpha) + newDist * alpha)
        }
    }
}

class DecisionEngine {

    private var nextTrackId = 1
    val trackedObjects = mutableMapOf<Int, TrackedItem>()

    private var lastAnnouncedSceneFingerprint: Set<String> = emptySet()
    private var lastSpokenEventText: String? = null
    private var lastSpokenEventTime: Long = 0L
    private var lastHadActiveObjects: Boolean = false

    private val minFramesToConfirm = 2
    private val maxFramesMissing = 15 // Persist tracks for ~1 second through camera blur / panning

    fun evaluate(detections: List<Detection>): HazardEvent {
        val now = SystemClock.elapsedRealtime()

        // 1. Robust Spatial Match (IoU + Center Distance + Class Consistency)
        val matchedTrackIds = mutableSetOf<Int>()

        for (det in detections) {
            var bestTrackId: Int? = null
            var bestScore = 0.0f

            for ((id, track) in trackedObjects) {
                if (id in matchedTrackIds) continue

                val iou = computeIoU(det.bbox, track.bbox)
                val cDist = computeCenterDist(det.center, track.center)
                val classMatches = (det.className.equals(track.className, ignoreCase = true))

                // Scale invariant score: handles walking closer (expanding box) and lateral movement
                val score = iou * 0.35f + (1.0f - cDist).coerceIn(0f, 1f) * 0.45f + (if (classMatches) 0.20f else 0f)

                if (iou > 0.12f || (cDist < 0.35f && classMatches) || (cDist < 0.22f)) {
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
                track.confidence = det.confidence
                track.updateClass(det.className)
                track.smoothBbox(det.bbox)
                track.smoothDistance(det.estimatedDistanceM)
                track.region = det.region

                if (track.framesSeen >= minFramesToConfirm && track.state == TrackState.NEW) {
                    track.state = TrackState.CONFIRMED
                }
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
            }
        }

        // 2. Age out missing tracks
        val iterator = trackedObjects.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key !in matchedTrackIds) {
                entry.value.framesMissing++
                if (entry.value.framesMissing >= maxFramesMissing) {
                    entry.value.state = TrackState.DEPARTED
                    iterator.remove()
                }
            }
        }

        // Active confirmed objects (visible now or temporarily occluded for < 4 frames)
        val activeTracks = trackedObjects.values.filter {
            it.state != TrackState.NEW && it.state != TrackState.DEPARTED && it.framesMissing < 4
        }

        // 3. Path Cleared Event (All obstacles left the scene)
        if (activeTracks.isEmpty()) {
            if (lastHadActiveObjects) {
                lastHadActiveObjects = false
                lastAnnouncedSceneFingerprint = emptySet()
                lastSpokenEventText = "Path is clear."
                lastSpokenEventTime = now
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

        lastHadActiveObjects = true

        // 4. Alert Hierarchy: Danger Zone Obstruction (<= 1.8m in Center Corridor)
        val centerObstacles = activeTracks.filter { it.region == "center" && it.distanceM <= 1.8f }
        if (centerObstacles.isNotEmpty()) {
            val closest = centerObstacles.minByOrNull { it.distanceM }!!
            val newTier = if (closest.distanceM <= 0.9f) 2 else 1

            // Trigger obstruction alert ONLY on first discovery or if transitioning from Tier 1 -> Tier 2
            val isNewObstruction = !closest.isObstructionAnnounced || (closest.criticalTier < newTier)
            if (isNewObstruction) {
                closest.isObstructionAnnounced = true
                closest.criticalTier = newTier
                closest.state = TrackState.PERSISTENT

                val warning = formatPathObstruction(centerObstacles, newTier)
                lastSpokenEventText = warning
                lastSpokenEventTime = now
                return HazardEvent(
                    warningText = warning,
                    speakPriority = if (newTier >= 2) 90 else 80,
                    severity = if (newTier >= 2) "CRITICAL" else "WARNING",
                    category = closest.className,
                    hazardDetected = true,
                    activeHazard = closest,
                    allHazards = activeTracks
                )
            }
        }

        // 5. Scene Level Announcement (Triggered ONLY when the set of visible objects/regions changes)
        // Distance fluctuations do NOT change the fingerprint. Walking closer does NOT re-trigger speech.
        val currentFingerprint = activeTracks.map { "${it.className.lowercase()}_${it.region}" }.toSet()

        val newlyAppearedObjects = activeTracks.filter { it.state == TrackState.CONFIRMED }
        val sceneCompositionChanged = (currentFingerprint != lastAnnouncedSceneFingerprint) &&
                (lastAnnouncedSceneFingerprint.isEmpty() || !lastAnnouncedSceneFingerprint.containsAll(currentFingerprint))

        if (newlyAppearedObjects.isNotEmpty() && sceneCompositionChanged) {
            // Mark all newly confirmed objects as ANNOUNCED / PERSISTENT
            for (t in newlyAppearedObjects) {
                t.state = TrackState.PERSISTENT
            }
            lastAnnouncedSceneFingerprint = currentFingerprint

            val sceneDesc = formatSceneDescription(activeTracks)
            if (sceneDesc != lastSpokenEventText || (now - lastSpokenEventTime > 8000L)) {
                lastSpokenEventText = sceneDesc
                lastSpokenEventTime = now
                return HazardEvent(
                    warningText = sceneDesc,
                    speakPriority = 50,
                    severity = "CAUTION",
                    category = "scene",
                    hazardDetected = true,
                    allHazards = activeTracks
                )
            }
        }

        // If scene is unchanged and no new hazards appeared, RETURN SILENCE
        return HazardEvent(
            hazardDetected = true,
            allHazards = activeTracks
        )
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
            Triple("left", leftItems, "on your left"),
            Triple("right", rightItems, "on your right")
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
        for ((_, items, posLabel) in activeRegions) {
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
        return if (rounded in 0.4f..5.5f) {
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
        lastAnnouncedSceneFingerprint = emptySet()
        lastSpokenEventText = null
        lastSpokenEventTime = 0L
        lastHadActiveObjects = false
    }
}