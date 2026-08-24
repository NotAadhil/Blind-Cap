package com.blindcap.app.engine

import android.graphics.RectF
import android.os.SystemClock
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.RecognizedFace
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

enum class TrackState {
    CANDIDATE,  // Accumulating temporal confirmation frames
    CONFIRMED,  // Stable, confirmed active physical object
    COASTING,   // Temporarily missing (occlusion / blur / pan grace period)
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
    var lastAnnouncedTimeMs: Long = 0L,
    var faceName: String? = null,
    var isFacingUser: Boolean = false,
    val classHistory: ArrayDeque<String> = ArrayDeque(8)
) {
    init {
        classHistory.add(className)
    }

    fun updateClass(newClass: String) {
        // If track is already assigned a specific face name, lock the face name permanently
        if (faceName != null) {
            className = faceName!!
            return
        }

        if (classHistory.size >= 8) classHistory.removeFirst()
        classHistory.addLast(newClass)
        className = classHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: newClass
    }

    fun attachFace(name: String?, facingUser: Boolean) {
        if (name != null && name.isNotBlank()) {
            faceName = name
            className = name
        }
        if (facingUser) {
            isFacingUser = true
        }
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
    private val highConfidenceInstantThreshold = 0.52f
    private val minConfidenceThreshold = 0.30f
    private val maxFramesMissing = 24

    fun evaluate(
        detections: List<Detection>,
        recognizedFaces: List<RecognizedFace> = emptyList()
    ): HazardEvent {
        val now = SystemClock.elapsedRealtime()

        // 1. Filter raw detections
        val validDetections = detections.filter { it.confidence >= 0.25f }

        // 2. Associate detected faces with person detections (attach recognized contact name or eye contact)
        for (det in validDetections) {
            val isPersonLike = det.className.equals("person", ignoreCase = true) || det.className.equals("face", ignoreCase = true)
            if (isPersonLike) {
                for (face in recognizedFaces) {
                    val fcx = face.bbox.centerX()
                    val fcy = face.bbox.centerY()
                    // Check if face center falls inside or near the person detection bounding box
                    val marginX = det.bbox.width() * 0.15f
                    val marginY = det.bbox.height() * 0.15f
                    if (fcx >= det.bbox.left - marginX && fcx <= det.bbox.right + marginX &&
                        fcy >= det.bbox.top - marginY && fcy <= det.bbox.bottom + marginY) {
                        if (face.isKnown && face.name != null) {
                            det.className = face.name
                        }
                        break
                    }
                }
            }
        }

        var meanDx = 0f
        var meanDy = 0f
        var motionSampleCount = 0

        // 3. Spatial & Motion-Tolerant Object Tracking
        val matchedTrackIds = mutableSetOf<Int>()

        for (det in validDetections) {
            var bestTrackId: Int? = null
            var bestScore = 0.0f

            for ((id, track) in trackedObjects) {
                if (id in matchedTrackIds) continue

                val iou = computeIoU(det.bbox, track.bbox)
                val cDist = computeCenterDist(det.center, track.center)
                val classMatches = det.className.equals(track.className, ignoreCase = true) ||
                        (track.faceName != null && det.className.equals("person", ignoreCase = true))
                val isPerson = classMatches && (det.className.equals("person", ignoreCase = true) || track.faceName != null)

                val area1 = max(0.001f, det.areaRatio)
                val area2 = max(0.001f, track.bbox.width() * track.bbox.height())
                val scaleRatio = min(area1, area2) / max(area1, area2)

                val centerScore = (1.0f - cDist).coerceIn(0f, 1f)
                val score = iou * 0.35f + centerScore * 0.30f + scaleRatio * 0.15f + (if (classMatches) 0.20f else 0f)

                val maxAllowedDist = if (isPerson) 0.55f else 0.42f
                val minAllowedIou = if (isPerson) 0.08f else 0.12f

                if (iou > minAllowedIou || (cDist < maxAllowedDist && classMatches) || (cDist < 0.22f)) {
                    if (score > bestScore) {
                        bestScore = score
                        bestTrackId = id
                    }
                }
            }

            if (bestTrackId != null) {
                matchedTrackIds.add(bestTrackId)
                val track = trackedObjects[bestTrackId]!!

                val dx = det.center.first - track.center.first
                val dy = det.center.second - track.center.second
                meanDx += dx
                meanDy += dy
                motionSampleCount++

                track.framesSeen++
                track.framesMissing = 0
                track.confidence = det.confidence
                track.updateClass(det.className)
                track.smoothBbox(det.bbox)
                track.smoothDistance(det.estimatedDistanceM)
                track.region = depthEstimator.classifyRegion(track.center.first, track.region)

                // Attach face identification if matched
                for (face in recognizedFaces) {
                    val fcx = face.bbox.centerX()
                    val fcy = face.bbox.centerY()
                    if (fcx >= track.bbox.left && fcx <= track.bbox.right && fcy >= track.bbox.top && fcy <= track.bbox.bottom) {
                        track.attachFace(if (face.isKnown) face.name else null, face.isFacingUser)
                        break
                    }
                }

                if (track.framesSeen >= minFramesToConfirm || track.confidence >= highConfidenceInstantThreshold) {
                    track.state = TrackState.CONFIRMED
                } else if (track.state == TrackState.COASTING) {
                    track.state = TrackState.CONFIRMED
                }
            } else {
                if (det.confidence >= 0.30f) {
                    val newId = nextTrackId++
                    val initialRegion = depthEstimator.classifyRegion(det.center.first)
                    val isInstantConfirm = det.confidence >= highConfidenceInstantThreshold
                    val newTrack = TrackedItem(
                        id = newId,
                        className = det.className,
                        confidence = det.confidence,
                        bbox = det.bbox,
                        center = det.center,
                        region = initialRegion,
                        distanceM = det.estimatedDistanceM,
                        state = if (isInstantConfirm) TrackState.CONFIRMED else TrackState.CANDIDATE
                    )

                    // Attach face info immediately on spawn if available
                    for (face in recognizedFaces) {
                        val fcx = face.bbox.centerX()
                        val fcy = face.bbox.centerY()
                        if (fcx >= det.bbox.left && fcx <= det.bbox.right && fcy >= det.bbox.top && fcy <= det.bbox.bottom) {
                            newTrack.attachFace(if (face.isKnown) face.name else null, face.isFacingUser)
                            break
                        }
                    }

                    trackedObjects[newId] = newTrack
                    matchedTrackIds.add(newId)
                }
            }
        }

        val isGlobalCameraPan = motionSampleCount >= 2 && (abs(meanDx / motionSampleCount) > 0.05f || abs(meanDy / motionSampleCount) > 0.05f)

        // 4. Age out missing tracks
        val iterator = trackedObjects.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key !in matchedTrackIds) {
                entry.value.framesMissing++

                val effectiveMaxMissing = if (isGlobalCameraPan) maxFramesMissing + 6 else maxFramesMissing

                if (entry.value.framesMissing >= effectiveMaxMissing) {
                    entry.value.state = TrackState.DEPARTED
                    iterator.remove()
                } else if (entry.value.framesMissing >= 2 && entry.value.state == TrackState.CONFIRMED) {
                    entry.value.state = TrackState.COASTING
                }
            }
        }

        val activeTracks = trackedObjects.values.filter {
            it.state == TrackState.CONFIRMED && it.framesMissing < 12 && (it.confidence >= minConfidenceThreshold || it.framesMissing > 0)
        }

        // 5. Path Cleared Event
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

        // 6. Danger Zone Obstruction (<= 1.8m in Center Corridor)
        val centerObstacles = activeTracks.filter { it.region == "center" && it.distanceM <= 1.8f }
        if (centerObstacles.isNotEmpty()) {
            val closest = centerObstacles.minByOrNull { it.distanceM }!!
            val newTier = if (closest.distanceM <= 0.9f) 2 else 1

            val isNewObstruction = !closest.isObstructionAnnounced || (closest.criticalTier < newTier)
            if (isNewObstruction) {
                closest.isObstructionAnnounced = true
                closest.criticalTier = newTier

                val warning = formatPathObstruction(closest, newTier)
                lastSpokenEventText = warning
                lastSpokenEventTime = now
                return HazardEvent(
                    warningText = warning,
                    speakPriority = if (newTier >= 2) 90 else 70,
                    severity = if (newTier >= 2) "CRITICAL" else "WARNING",
                    category = closest.className,
                    hazardDetected = true,
                    activeHazard = closest,
                    allHazards = activeTracks
                )
            }
        }

        // 7. Multi-Object Scene Understanding
        val unannouncedConfirmedObjects = activeTracks.filter { !it.isAnnounced }

        if (unannouncedConfirmedObjects.isNotEmpty()) {
            for (t in activeTracks) {
                t.isAnnounced = true
                t.lastAnnouncedTimeMs = now
            }

            val sceneDesc = formatSceneDescription(activeTracks)
            if (sceneDesc.isNotEmpty() && sceneDesc != lastSpokenEventText) {
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

        val sentences = mutableListOf<String>()

        // Process named contacts & people with eye contact first
        val specialPeople = tracks.filter { it.faceName != null || it.isFacingUser }
        val otherObjects = tracks.filter { it !in specialPeople }

        for (person in specialPeople) {
            val distStr = formatDistanceStr(person.distanceM)
            val posStr = when (person.region) {
                "center" -> "ahead"
                "left" -> "on your left"
                else -> "on your right"
            }

            val sentence = when {
                person.faceName != null && person.isFacingUser ->
                    "${person.faceName} is looking at you $posStr$distStr"
                person.faceName != null ->
                    "${person.faceName} $posStr$distStr"
                person.isFacingUser ->
                    "Person looking towards you $posStr$distStr"
                else ->
                    "Person $posStr$distStr"
            }
            sentences.add(sentence)
        }

        // Group remaining objects by category
        val classGroups = otherObjects.groupBy { it.className.lowercase() }
            .entries.sortedBy { (_, items) -> items.minOf { it.distanceM } }
            .take(3)

        for ((_, items) in classGroups) {
            val count = items.size
            val rawName = items.first().className
            val singleName = rawName.lowercase()
            val pluralName = pluralize(singleName)

            val leftCount = items.count { it.region == "left" }
            val centerCount = items.count { it.region == "center" }
            val rightCount = items.count { it.region == "right" }

            if (count == 1) {
                val reg = items.first().region
                val posStr = when (reg) {
                    "center" -> "in the center"
                    "left" -> "on the left"
                    else -> "on the right"
                }
                sentences.add("${singleName.replaceFirstChar { it.uppercase() }} $posStr")
            } else {
                val countWord = numberToWord(count)
                val posBreakdown = mutableListOf<String>()

                if (leftCount > 0) posBreakdown.add(if (leftCount == count) "on the left" else "${numberToWord(leftCount)} on the left")
                if (centerCount > 0) posBreakdown.add(if (centerCount == count) "in the center" else "${numberToWord(centerCount)} in the center")
                if (rightCount > 0) posBreakdown.add(if (rightCount == count) "on the right" else "${numberToWord(rightCount)} on the right")

                if (posBreakdown.size == 1) {
                    sentences.add("${countWord.replaceFirstChar { it.uppercase() }} $pluralName ${posBreakdown[0]}")
                } else {
                    sentences.add("${countWord.replaceFirstChar { it.uppercase() }} $pluralName detected: ${posBreakdown.joinToString(" and ")}")
                }
            }
        }

        return sentences.joinToString(". ") + "."
    }

    private fun formatPathObstruction(item: TrackedItem, tier: Int): String {
        val name = if (item.faceName != null) item.faceName!! else item.className.lowercase()
        val distSuffix = formatDistanceStr(item.distanceM)

        return if (tier >= 2) {
            if (item.faceName != null) "Stop! $name is right ahead of you$distSuffix. Please stop!"
            else "Stop! $name is very close$distSuffix. Please stop!"
        } else {
            if (item.faceName != null) "Attention! $name is in your walking path$distSuffix."
            else "Stop! $name is obstructing your path$distSuffix."
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