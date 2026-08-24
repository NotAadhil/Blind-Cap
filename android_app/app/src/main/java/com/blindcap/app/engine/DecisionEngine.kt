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

enum class RecognitionState {
    UNKNOWN,      // Person detected, face not yet identified
    CONFIRMING,   // Face match candidate observed
    RECOGNIZED,   // Confidently identified, pending speech announcement
    ANNOUNCED,    // Name announced via TTS
    TRACKING      // Actively tracking identified person
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
    var recognitionState: RecognitionState = RecognitionState.UNKNOWN,
    var faceName: String? = null,
    var faceConfidence: Float = 0f,
    var faceConfirmHits: Int = 0,
    var isFacingUser: Boolean = false,
    val classHistory: ArrayDeque<String> = ArrayDeque(8)
) {
    init {
        classHistory.add(className)
    }

    fun updateClass(newClass: String) {
        if (faceName != null) {
            className = faceName!!
            return
        }

        if (classHistory.size >= 8) classHistory.removeFirst()
        classHistory.addLast(newClass)
        className = classHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: newClass
    }

    fun attachFace(name: String?, confidence: Float, facingUser: Boolean) {
        if (name != null && name.isNotBlank() && confidence >= 0.44f) {
            faceConfirmHits++
            faceConfidence = confidence

            if (faceName != name) {
                faceName = name
                className = name
                // Reset announcement state so the newly recognized identity is guaranteed to be spoken
                recognitionState = RecognitionState.RECOGNIZED
                isAnnounced = false
            }
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
    private val maxFramesMissing = 18 // ~1.2s grace period: cleanly re-arms when camera returns

    fun evaluate(
        detections: List<Detection>,
        recognizedFaces: List<RecognizedFace> = emptyList()
    ): HazardEvent {
        val now = SystemClock.elapsedRealtime()

        // 1. Filter raw detections by confidence gate
        val filteredDetections = detections.filter { it.confidence >= 0.25f }

        // 2. Spatial Overlap & Containment Deduplication (Person NMS)
        // Merges multi-zone body boxes, torso vs full body, and overlapping person candidates
        val deduplicatedDetections = deduplicateDetections(filteredDetections)

        // 3. Associate detected faces with person detections
        for (det in deduplicatedDetections) {
            val isPersonLike = det.className.equals("person", ignoreCase = true) || det.className.equals("face", ignoreCase = true)
            if (isPersonLike) {
                for (face in recognizedFaces) {
                    val fcx = face.bbox.centerX()
                    val fcy = face.bbox.centerY()
                    val marginX = det.bbox.width() * 0.20f
                    val marginY = det.bbox.height() * 0.20f
                    if (fcx >= det.bbox.left - marginX && fcx <= det.bbox.right + marginX &&
                        fcy >= det.bbox.top - marginY && fcy <= det.bbox.bottom + marginY) {
                        if (face.isKnown && face.name != null && face.confidence >= 0.44f) {
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

        // 4. Robust Spatial & Motion-Tolerant Tracking
        val matchedTrackIds = mutableSetOf<Int>()

        for (det in deduplicatedDetections) {
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
                track.region = classifyTrackRegion(track.bbox, track.center.first, track.region)

                // Match face recognition state
                for (face in recognizedFaces) {
                    val fcx = face.bbox.centerX()
                    val fcy = face.bbox.centerY()
                    if (fcx >= track.bbox.left && fcx <= track.bbox.right && fcy >= track.bbox.top && fcy <= track.bbox.bottom) {
                        track.attachFace(if (face.isKnown) face.name else null, face.confidence, face.isFacingUser)
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
                    val initialRegion = classifyTrackRegion(det.bbox, det.center.first)
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

                    for (face in recognizedFaces) {
                        val fcx = face.bbox.centerX()
                        val fcy = face.bbox.centerY()
                        if (fcx >= det.bbox.left && fcx <= det.bbox.right && fcy >= det.bbox.top && fcy <= det.bbox.bottom) {
                            newTrack.attachFace(if (face.isKnown) face.name else null, face.confidence, face.isFacingUser)
                            break
                        }
                    }

                    trackedObjects[newId] = newTrack
                    matchedTrackIds.add(newId)
                }
            }
        }

        val isGlobalCameraPan = motionSampleCount >= 2 && (abs(meanDx / motionSampleCount) > 0.05f || abs(meanDy / motionSampleCount) > 0.05f)

        // 5. Age out missing tracks (18 frame grace period allows instant re-arming upon re-entry)
        val iterator = trackedObjects.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key !in matchedTrackIds) {
                entry.value.framesMissing++

                val effectiveMaxMissing = if (isGlobalCameraPan) maxFramesMissing + 4 else maxFramesMissing

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

        // 6. Path Cleared Event
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

        // 7. ALERT TIER 1: Imminent Obstruction Danger (<= 1.8m in Walking Path) - Priority 70-90
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

        // 8. ALERT TIER 2: Newly Recognized Face Event - Priority 65
        // Fires immediately when a person transitions from UNKNOWN to RECOGNIZED
        val newlyRecognizedContacts = activeTracks.filter {
            it.faceName != null && it.recognitionState == RecognitionState.RECOGNIZED && !it.isAnnounced
        }

        if (newlyRecognizedContacts.isNotEmpty()) {
            val contact = newlyRecognizedContacts.first()
            contact.isAnnounced = true
            contact.recognitionState = RecognitionState.ANNOUNCED
            contact.lastAnnouncedTimeMs = now

            val announcement = formatContactAnnouncement(contact)
            lastSpokenEventText = announcement
            lastSpokenEventTime = now
            return HazardEvent(
                warningText = announcement,
                speakPriority = 65, // High priority: preempts general scene, never dropped
                severity = "INFO",
                category = "face_recognition",
                hazardDetected = true,
                activeHazard = contact,
                allHazards = activeTracks
            )
        }

        // 9. ALERT TIER 3: Multi-Object Scene Understanding - Priority 50
        val unannouncedConfirmedObjects = activeTracks.filter { !it.isAnnounced }

        if (unannouncedConfirmedObjects.isNotEmpty()) {
            for (t in activeTracks) {
                t.isAnnounced = true
                t.lastAnnouncedTimeMs = now
                if (t.faceName != null && t.recognitionState == RecognitionState.UNKNOWN) {
                    t.recognitionState = RecognitionState.ANNOUNCED
                }
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

    /**
     * Deduplicates raw bounding boxes using IoU, Containment, and Zone-spanning proximity.
     * Prevents 1 physical person spanning multiple regions from being treated as 2 people.
     */
    private fun deduplicateDetections(rawList: List<Detection>): List<Detection> {
        if (rawList.size <= 1) return rawList

        val persons = rawList.filter { it.className.equals("person", ignoreCase = true) || it.className.equals("face", ignoreCase = true) }
            .sortedByDescending { it.confidence }
        val nonPersons = rawList.filter { !it.className.equals("person", ignoreCase = true) && !it.className.equals("face", ignoreCase = true) }

        val mergedPersons = mutableListOf<Detection>()

        for (p in persons) {
            var merged = false
            for (i in mergedPersons.indices) {
                val existing = mergedPersons[i]
                if (shouldMergePersons(p.bbox, existing.bbox, p.center, existing.center)) {
                    val left = min(p.bbox.left, existing.bbox.left)
                    val top = min(p.bbox.top, existing.bbox.top)
                    val right = max(p.bbox.right, existing.bbox.right)
                    val bottom = max(p.bbox.bottom, existing.bbox.bottom)
                    val mergedBbox = RectF(left, top, right, bottom)
                    val cx = (left + right) * 0.5f
                    val cy = (top + bottom) * 0.5f

                    mergedPersons[i] = existing.copy(
                        confidence = max(p.confidence, existing.confidence),
                        bbox = mergedBbox,
                        center = Pair(cx, cy),
                        areaRatio = mergedBbox.width() * mergedBbox.height(),
                        estimatedDistanceM = min(p.estimatedDistanceM, existing.estimatedDistanceM),
                        region = classifyTrackRegion(mergedBbox, cx)
                    )
                    merged = true
                    break
                }
            }
            if (!merged) {
                mergedPersons.add(p)
            }
        }

        return mergedPersons + nonPersons
    }

    private fun shouldMergePersons(
        b1: RectF,
        b2: RectF,
        c1: Pair<Float, Float>,
        c2: Pair<Float, Float>
    ): Boolean {
        val iou = computeIoU(b1, b2)
        val cont = computeContainment(b1, b2)
        val cdist = computeCenterDist(c1, c2)
        val dx = abs(c1.first - c2.first)
        val dy = abs(c1.second - c2.second)

        val hOverlap = max(0f, min(b1.right, b2.right) - max(b1.left, b2.left))
        val minW = min(b1.width(), b2.width())
        val hRatio = if (minW > 0f) hOverlap / minW else 0f

        val vOverlap = max(0f, min(b1.bottom, b2.bottom) - max(b1.top, b2.top))
        val minH = min(b1.height(), b2.height())
        val vRatio = if (minH > 0f) vOverlap / minH else 0f

        // Condition 1: Significant 2D IoU or Containment
        if (iou > 0.25f || cont > 0.45f) return true

        // Condition 2: Same physical body spanning adjacent zones (high vertical overlap > 60% and horizontal overlap/proximity)
        if (vRatio > 0.60f && (hRatio > 0.15f || dx < 0.28f) && dy < 0.20f) return true

        // Condition 3: Proximity close center distance
        if (cdist < 0.22f) return true

        return false
    }

    /**
     * Classifies primary region for a bounding box, ensuring multi-zone spanning objects are centered/ahead.
     */
    private fun classifyTrackRegion(bbox: RectF, cx: Float, currentRegion: String? = null): String {
        if (bbox.left < 0.35f && bbox.right > 0.65f) {
            return "center"
        }
        return depthEstimator.classifyRegion(cx, currentRegion)
    }

    fun getFullSceneSummary(detections: List<Detection>): String {
        val items = if (detections.isNotEmpty()) {
            val deduped = deduplicateDetections(detections)
            deduped.map {
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

    private fun formatContactAnnouncement(contact: TrackedItem): String {
        val name = contact.faceName ?: contact.className
        val distStr = formatDistanceStr(contact.distanceM)
        val posStr = when (contact.region) {
            "center" -> "ahead"
            "left" -> "on your left"
            else -> "on your right"
        }

        return if (contact.isFacingUser) {
            "$name is looking at you $posStr$distStr."
        } else {
            "$name $posStr$distStr."
        }
    }

    private fun formatSceneDescription(tracks: List<TrackedItem>): String {
        if (tracks.isEmpty()) return "Path is clear."

        val sentences = mutableListOf<String>()

        // 1. Separate person tracks vs other object tracks
        val personTracks = tracks.filter { it.className.equals("person", ignoreCase = true) || it.faceName != null }
        val otherObjects = tracks.filter { it !in personTracks }

        // Format Person Detections based strictly on UNIQUE tracked individuals
        if (personTracks.size == 1) {
            val person = personTracks.first()
            val name = person.faceName ?: "Person"
            val distStr = formatDistanceStr(person.distanceM)
            val posStr = when (person.region) {
                "center" -> "ahead"
                "left" -> "on your left"
                else -> "on your right"
            }

            val sentence = when {
                person.faceName != null && person.isFacingUser ->
                    "$name is looking at you $posStr$distStr"
                person.faceName != null ->
                    "$name $posStr$distStr"
                person.isFacingUser ->
                    "Person looking towards you $posStr$distStr"
                else ->
                    "Person $posStr$distStr"
            }
            sentences.add(sentence)
        } else if (personTracks.size >= 2) {
            val countWord = numberToWord(personTracks.size)
            val personDetails = personTracks.map { p ->
                val name = p.faceName ?: "one"
                val posStr = when (p.region) {
                    "center" -> "ahead"
                    "left" -> "on your left"
                    else -> "on your right"
                }
                "$name $posStr"
            }
            sentences.add("${countWord.replaceFirstChar { it.uppercase() }} people detected: ${personDetails.joinToString(" and ")}")
        }

        // Format remaining object categories
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

    private fun computeContainment(b1: RectF, b2: RectF): Float {
        val left = max(b1.left, b2.left)
        val top = max(b1.top, b2.top)
        val right = min(b1.right, b2.right)
        val bottom = min(b1.bottom, b2.bottom)

        val interW = max(0f, right - left)
        val interH = max(0f, bottom - top)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val minArea = min(b1.width() * b1.height(), b2.width() * b2.height())
        return if (minArea > 0f) interArea / minArea else 0f
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