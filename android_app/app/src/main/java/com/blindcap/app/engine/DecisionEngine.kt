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
    CANDIDATE,  // Accumulating temporal confirmation frames
    CONFIRMED,  // Stable, confirmed active physical object
    COASTING,   // Temporarily missing (occlusion/blur grace period)
    DEPARTED    // Expired and removed
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
    var state: TrackState = TrackState.CANDIDATE,
    var isAnnounced: Boolean = false,
    var isObstructionAnnounced: Boolean = false,
    var criticalTier: Int = 0,
    val classHistory: ArrayDeque<String> = ArrayDeque(8)
) {
    init {
        classHistory.add(className)
    }

    fun updateClass(newClass: String) {
        if (classHistory.size >= 8) classHistory.removeFirst()
        classHistory.addLast(newClass)
        // Majority voting to eliminate classification flicker
        className = classHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: newClass
    }

    fun smoothBbox(newBbox: RectF) {
        val alpha = 0.45f
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
            val alpha = if (delta > 0.6f) 0.40f else 0.20f
            distanceM = (distanceM * (1 - alpha) + newDist * alpha)
        }
    }
}

class DecisionEngine {

    private var nextTrackId = 1
    val trackedObjects = mutableMapOf<Int, TrackedItem>()
    private val depthEstimator = DepthEstimator()

    private var lastSpokenEventText: String? = null
    private var lastSpokenEventTime: Long = 0L
    private var lastHadActiveObjects: Boolean = false

    private val minFramesToConfirm = 2
    private val maxFramesMissing = 20 // 20 frames (~1.5 seconds) grace period for motion blur & temporary loss

    fun evaluate(detections: List<Detection>): HazardEvent {
        val now = SystemClock.elapsedRealtime()

        // 1. Robust Multi-Object Spatial & Motion-Tolerant Matching
        val matchedTrackIds = mutableSetOf<Int>()

        for (det in detections) {
            var bestTrackId: Int? = null
            var bestScore = 0.0f

            for ((id, track) in trackedObjects) {
                if (id in matchedTrackIds) continue

                val iou = computeIoU(det.bbox, track.bbox)
                val cDist = computeCenterDist(det.center, track.center)
                val classMatches = det.className.equals(track.className, ignoreCase = true)

                // Scale & motion tolerant matching: handles camera movement, walking closer, and lateral motion
                val score = iou * 0.40f + (1.0f - cDist).coerceIn(0f, 1f) * 0.35f + (if (classMatches) 0.25f else 0f)

                if (iou > 0.12f || (cDist < 0.38f && classMatches) || (cDist < 0.20f)) {
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
                // Region update with deadband hysteresis
                track.region = depthEstimator.classifyRegion(track.center.first, track.region)

                // Promote candidate to confirmed if seen for >= 2 frames with adequate confidence
                if (track.framesSeen >= minFramesToConfirm && track.state == TrackState.CANDIDATE) {
                    track.state = TrackState.CONFIRMED
                } else if (track.state == TrackState.COASTING) {
                    track.state = TrackState.CONFIRMED
                }
            } else {
                // New detection candidate
                val newId = nextTrackId++
                val initialRegion = depthEstimator.classifyRegion(det.center.first)
                val newTrack = TrackedItem(
                    id = newId,
                    className = det.className,
                    confidence = det.confidence,
                    bbox = det.bbox,
                    center = det.center,
                    region = initialRegion,
                    distanceM = det.estimatedDistanceM,
                    state = if (det.confidence >= 0.45f) TrackState.CONFIRMED else TrackState.CANDIDATE
                )
                trackedObjects[newId] = newTrack
                matchedTrackIds.add(newId)
            }
        }

        // 2. Age out missing tracks with coasting grace period
        val iterator = trackedObjects.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key !in matchedTrackIds) {
                entry.value.framesMissing++
                if (entry.value.framesMissing >= maxFramesMissing) {
                    entry.value.state = TrackState.DEPARTED
                    iterator.remove()
                } else if (entry.value.framesMissing >= 2 && entry.value.state == TrackState.CONFIRMED) {
                    entry.value.state = TrackState.COASTING
                }
            }
        }

        // Active confirmed objects (visible now or coasting for < 6 frames)
        val activeTracks = trackedObjects.values.filter {
            it.state == TrackState.CONFIRMED && it.framesMissing < 6 && it.confidence >= 0.25f
        }

        // 3. Path Cleared Event (All obstacles left the scene)
        if (activeTracks.isEmpty()) {
            if (lastHadActiveObjects) {
                lastHadActiveObjects = false
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

            val isNewObstruction = !closest.isObstructionAnnounced || (closest.criticalTier < newTier)
            if (isNewObstruction) {
                closest.isObstructionAnnounced = true
                closest.criticalTier = newTier

                val warning = formatPathObstruction(centerObstacles, newTier)
                lastSpokenEventText = warning
                lastSpokenEventTime = now
                return HazardEvent(
                    warningText = warning,
                    speakPriority = if (newTier >= 2) 90 else 70, // 90 for Tier 2 imminent danger, 70 for Tier 1
                    severity = if (newTier >= 2) "CRITICAL" else "WARNING",
                    category = closest.className,
                    hazardDetected = true,
                    activeHazard = closest,
                    allHazards = activeTracks
                )
            }
        }

        // 5. Intelligent Multi-Object Scene Understanding (Triggered ONLY when NEW unannounced tracks appear)
        val unannouncedConfirmedObjects = activeTracks.filter { !it.isAnnounced }

        if (unannouncedConfirmedObjects.isNotEmpty()) {
            // Mark all active confirmed tracks as announced so camera movement will NEVER re-trigger speech
            for (t in activeTracks) {
                t.isAnnounced = true
            }

            val sceneDesc = formatSceneDescription(activeTracks)
            if (sceneDesc.isNotEmpty() && (sceneDesc != lastSpokenEventText || (now - lastSpokenEventTime > 8000L))) {
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

        // Unchanged scene remains completely silent
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
                    distanceM = it.estimatedDistanceM,
                    state = TrackState.CONFIRMED
                )
            }
        } else {
            trackedObjects.values.filter { it.state == TrackState.CONFIRMED }.toList()
        }

        if (items.isEmpty()) return "The path ahead is completely clear with no detected obstacles."
        val desc = formatSceneDescription(items)
        return "Scene summary: $desc"
    }

    private fun formatSceneDescription(tracks: List<TrackedItem>): String {
        if (tracks.isEmpty()) return "Path is clear."

        // Group by class and sort by proximity (closest first, with person prioritized)
        val classGroups = tracks.groupBy { it.className.lowercase() }
            .entries.sortedBy { (_, items) ->
                val isPerson = items.first().className.equals("person", ignoreCase = true)
                val minDist = items.minOf { it.distanceM }
                if (isPerson) minDist - 2.0f else minDist
            }.take(4) // Max 4 most prominent categories

        val sentences = mutableListOf<String>()

        for ((_, items) in classGroups) {
            val count = items.size
            val rawName = items.first().className
            val singleName = rawName.lowercase()
            val pluralName = pluralize(singleName)

            val leftCount = items.count { it.region == "left" }
            val centerCount = items.count { it.region == "center" }
            val rightCount = items.count { it.region == "right" }

            if (count == 1) {
                // Single object: "Person on the left." / "Chair in the center."
                val reg = items.first().region
                val posStr = when (reg) {
                    "center" -> "in the center"
                    "left" -> "on the left"
                    else -> "on the right"
                }
                sentences.add("${singleName.replaceFirstChar { it.uppercase() }} $posStr")
            } else {
                // Multiple objects of same class: "Two people detected: one on the left and one on the right."
                val countWord = numberToWord(count)
                val posBreakdown = mutableListOf<String>()

                if (leftCount > 0) {
                    val w = if (leftCount == count) "on the left" else "${numberToWord(leftCount)} on the left"
                    posBreakdown.add(w)
                }
                if (centerCount > 0) {
                    val w = if (centerCount == count) "in the center" else "${numberToWord(centerCount)} in the center"
                    posBreakdown.add(w)
                }
                if (rightCount > 0) {
                    val w = if (rightCount == count) "on the right" else "${numberToWord(rightCount)} on the right"
                    posBreakdown.add(w)
                }

                if (posBreakdown.size == 1) {
                    sentences.add("${countWord.replaceFirstChar { it.uppercase() }} $pluralName ${posBreakdown[0]}")
                } else if (posBreakdown.size == 2) {
                    sentences.add("${countWord.replaceFirstChar { it.uppercase() }} $pluralName detected: ${posBreakdown[0]} and ${posBreakdown[1]}")
                } else {
                    sentences.add("${countWord.replaceFirstChar { it.uppercase() }} $pluralName detected: ${posBreakdown[0]}, ${posBreakdown[1]}, and ${posBreakdown[2]}")
                }
            }
        }

        return sentences.joinToString(". ") + "."
    }

    private fun formatPathObstruction(items: List<TrackedItem>, tier: Int): String {
        val count = items.size
        val firstItem = items.first()
        val name = if (count > 1) "${numberToWord(count)} ${pluralize(firstItem.className)}" else firstItem.className.lowercase()
        val minDist = items.minOfOrNull { it.distanceM } ?: 1.0f
        val distSuffix = formatDistanceStr(minDist)

        return if (tier >= 2) {
            "Stop! $name is very close$distSuffix. Please stop!"
        } else {
            "Stop! $name is obstructing your path$distSuffix."
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

    private fun numberToWord(num: Int): String {
        return when (num) {
            1 -> "one"
            2 -> "two"
            3 -> "three"
            4 -> "four"
            5 -> "five"
            6 -> "six"
            7 -> "seven"
            8 -> "eight"
            else -> num.toString()
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
        lastSpokenEventText = null
        lastSpokenEventTime = 0L
        lastHadActiveObjects = false
    }
}