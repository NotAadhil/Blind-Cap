package com.blindcap.app.engine

import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.FaceObservation
import com.blindcap.app.ai.FaceObservationSource
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
    val allHazards: List<TrackedItem> = emptyList(),
    val faceObservations: List<FaceObservation> = emptyList()
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
    var lastFaceSeenTimeMs: Long = 0L,
    var relativeFaceOffset: RectF? = null,
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

    fun attachFace(name: String?, confidence: Float, facingUser: Boolean, faceBbox: RectF? = null) {
        lastFaceSeenTimeMs = SystemClock.elapsedRealtime()
        if (facingUser) {
            isFacingUser = true
        }

        if (faceBbox != null && bbox.width() > 0 && bbox.height() > 0) {
            // Compute normalized relative offset inside person body
            relativeFaceOffset = RectF(
                ((faceBbox.left - bbox.left) / bbox.width()).coerceIn(0f, 1f),
                ((faceBbox.top - bbox.top) / bbox.height()).coerceIn(0f, 1f),
                ((faceBbox.right - bbox.left) / bbox.width()).coerceIn(0f, 1f),
                ((faceBbox.bottom - bbox.top) / bbox.height()).coerceIn(0f, 1f)
            )
        }

        if (name != null && name.isNotBlank() && confidence >= 0.70f) {
            faceConfirmHits++
            faceConfidence = confidence

            if (faceName != name) {
                faceName = name
                className = name
                // Transition state: RECOGNIZED -> ready for announcement
                recognitionState = RecognitionState.RECOGNIZED
                isAnnounced = false
            }
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

    /**
     * Derive current real-time face bounding box at 30 FPS based on live person tracking.
     */
    fun computeCurrentFaceBbox(): RectF {
        val rel = relativeFaceOffset
        return if (rel != null && rel.width() > 0.05f && rel.height() > 0.05f) {
            val left = bbox.left + rel.left * bbox.width()
            val top = bbox.top + rel.top * bbox.height()
            val right = bbox.left + rel.right * bbox.width()
            val bottom = bbox.top + rel.bottom * bbox.height()
            RectF(left, top, right, bottom)
        } else {
            // Anatomical head position (top 30% of body, centered)
            val w = bbox.width()
            val h = bbox.height()
            RectF(
                bbox.left + w * 0.20f,
                bbox.top,
                bbox.right - w * 0.20f,
                bbox.top + h * 0.32f
            )
        }
    }
}

class DecisionEngine {

    private val tag = "DecisionEngine"
    private var nextTrackId = 1
    val trackedObjects = mutableMapOf<Int, TrackedItem>()
    private val depthEstimator = DepthEstimator()

    private var lastSpokenEventText: String? = null
    private var lastSpokenEventTime: Long = 0L
    private var lastHadActiveObjects: Boolean = false

    private val minFramesToConfirm = 2
    private val highConfidenceInstantThreshold = 0.52f
    private val minConfidenceThreshold = 0.30f
    private val maxFramesMissing = 18 // ~1.2s grace period: cleanly re-arms upon departure/re-entry

    fun evaluate(
        detections: List<Detection>,
        recognizedFaces: List<RecognizedFace> = emptyList()
    ): HazardEvent {
        val now = SystemClock.elapsedRealtime()

        // 1. Filter raw detections by confidence gate
        val filteredDetections = detections.filter { it.confidence >= 0.25f }

        // 2. Spatial Overlap & Containment Deduplication (Person NMS)
        val deduplicatedDetections = deduplicateDetections(filteredDetections)

        // 3. Associate detected faces with person detections (Strictly person bounding boxes only)
        for (det in deduplicatedDetections) {
            val isPerson = det.className.equals("person", ignoreCase = true)
            if (isPerson) {
                for (face in recognizedFaces) {
                    val fcx = face.bbox.centerX()
                    val fcy = face.bbox.centerY()
                    if (fcx >= det.bbox.left && fcx <= det.bbox.right &&
                        fcy >= det.bbox.top && fcy <= (det.bbox.top + det.bbox.height() * 0.45f)) {
                        if (face.isKnown && face.name != null && face.confidence >= 0.70f) {
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

                // Match face recognition state (Strictly for person tracks)
                if (track.className.equals("person", ignoreCase = true) || track.faceName != null) {
                    for (face in recognizedFaces) {
                        val fcx = face.bbox.centerX()
                        val fcy = face.bbox.centerY()
                        if (fcx >= track.bbox.left && fcx <= track.bbox.right &&
                            fcy >= track.bbox.top && fcy <= (track.bbox.top + track.bbox.height() * 0.45f)) {
                            track.attachFace(
                                name = if (face.isKnown) face.name else null,
                                confidence = face.confidence,
                                facingUser = face.isFacingUser,
                                faceBbox = face.bbox
                            )
                            break
                        }
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

                    if (det.className.equals("person", ignoreCase = true)) {
                        for (face in recognizedFaces) {
                            val fcx = face.bbox.centerX()
                            val fcy = face.bbox.centerY()
                            if (fcx >= det.bbox.left && fcx <= det.bbox.right &&
                                fcy >= det.bbox.top && fcy <= (det.bbox.top + det.bbox.height() * 0.45f)) {
                                newTrack.attachFace(
                                    name = if (face.isKnown) face.name else null,
                                    confidence = face.confidence,
                                    facingUser = face.isFacingUser,
                                    faceBbox = face.bbox
                                )
                                break
                            }
                        }
                    }

                    trackedObjects[newId] = newTrack
                    matchedTrackIds.add(newId)
                }
            }
        }

        val isGlobalCameraPan = motionSampleCount >= 2 && (abs(meanDx / motionSampleCount) > 0.05f || abs(meanDy / motionSampleCount) > 0.05f)

        // 5. Age out missing tracks
        val iterator = trackedObjects.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key !in matchedTrackIds) {
                entry.value.framesMissing++

                val effectiveMaxMissing = if (isGlobalCameraPan) maxFramesMissing + 4 else maxFramesMissing

                if (entry.value.framesMissing >= effectiveMaxMissing) {
                    Log.d(tag, "[FACE_TRACK_DEPARTED] track=${entry.value.id}, identity=${entry.value.faceName ?: "person"}")
                    Log.d(tag, "[FACE_BOX_CLEARED] reason=track_departed, trackId=${entry.value.id}")
                    entry.value.state = TrackState.DEPARTED
                    iterator.remove()
                } else if (entry.value.framesMissing >= 2 && entry.value.state == TrackState.CONFIRMED) {
                    Log.d(tag, "[FACE_MISSING] track=${entry.value.id}, framesMissing=${entry.value.framesMissing}")
                    entry.value.state = TrackState.COASTING
                }
            }
        }

        val activeTracks = trackedObjects.values.filter {
            it.state == TrackState.CONFIRMED && it.framesMissing < 12 && (it.confidence >= minConfidenceThreshold || it.framesMissing > 0)
        }

        // 6. Generate Live Dynamic Face Observations for all active person tracks (30 FPS)
        val faceObservations = ArrayList<FaceObservation>(4)
        for (track in activeTracks) {
            val isPerson = track.className.equals("person", ignoreCase = true) || track.faceName != null
            if (isPerson && track.framesMissing == 0) {
                val liveFaceBox = track.computeCurrentFaceBbox()
                val isDirectMlKit = (now - track.lastFaceSeenTimeMs) <= 350L

                val obs = FaceObservation(
                    trackId = track.id,
                    bbox = liveFaceBox,
                    identity = track.faceName,
                    isKnown = track.faceName != null,
                    similarity = track.faceConfidence,
                    isFacingUser = track.isFacingUser,
                    timestamp = now,
                    source = if (isDirectMlKit) FaceObservationSource.MLKIT_DETECTION else FaceObservationSource.TRACK_ANCHOR
                )
                faceObservations.add(obs)
                Log.d(tag, "[FACE_BOX_UPDATED] track=${track.id}, identity=${track.faceName ?: "Unknown"}, box=(${(liveFaceBox.left*100).toInt()}%, ${(liveFaceBox.top*100).toInt()}%), age=0ms")
            }
        }

        // 7. Path Cleared Event
        if (activeTracks.isEmpty()) {
            if (lastHadActiveObjects) {
                lastHadActiveObjects = false
                lastSpokenEventText = "Path is clear."
                lastSpokenEventTime = now
                Log.d(tag, "[FACE_BOX_CLEARED] reason=all_tracks_cleared")
                return HazardEvent(
                    warningText = "Path is clear.",
                    speakPriority = 40,
                    severity = "INFO",
                    category = "path_clear",
                    hazardDetected = false,
                    faceObservations = emptyList()
                )
            }
            return HazardEvent(hazardDetected = false, faceObservations = emptyList())
        }

        lastHadActiveObjects = true

        // 8. ALERT TIER 1: Imminent Obstruction Danger (<= 1.8m in Walking Path) - Priority 70-90
        val centerObstacles = activeTracks.filter { it.region == "center" && it.distanceM <= 1.8f }
        if (centerObstacles.isNotEmpty()) {
            val closest = centerObstacles.minByOrNull { it.distanceM }!!
            val newTier = if (closest.distanceM <= 0.9f) 2 else 1

            if (closest.criticalTier != newTier || !closest.isObstructionAnnounced) {
                closest.criticalTier = newTier
                closest.isObstructionAnnounced = true
                val warning = formatObstructionWarning(closest, newTier)
                lastSpokenEventText = warning
                lastSpokenEventTime = now
                return HazardEvent(
                    warningText = warning,
                    speakPriority = if (newTier >= 2) 90 else 70,
                    severity = if (newTier >= 2) "CRITICAL" else "WARNING",
                    category = closest.className,
                    hazardDetected = true,
                    activeHazard = closest,
                    allHazards = activeTracks,
                    faceObservations = faceObservations
                )
            }
        }

        // 9. ALERT TIER 2: Newly Recognized Face Event - Priority 65
        // Fires ONCE when a person transitions to RECOGNIZED
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
            Log.d(tag, "[FACE_RECOGNIZED] spoken_announcement=\"$announcement\"")
            return HazardEvent(
                warningText = announcement,
                speakPriority = 65,
                severity = "INFO",
                category = "face_recognition",
                hazardDetected = true,
                activeHazard = contact,
                allHazards = activeTracks,
                faceObservations = faceObservations
            )
        }

        // 10. ALERT TIER 3: Multi-Object Scene Understanding - Priority 50
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
                    allHazards = activeTracks,
                    faceObservations = faceObservations
                )
            }
        }

        return HazardEvent(
            hazardDetected = true,
            allHazards = activeTracks,
            faceObservations = faceObservations
        )
    }

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
                    val mergedCenter = Pair((left + right) * 0.5f, (top + bottom) * 0.5f)
                    val maxConf = max(p.confidence, existing.confidence)
                    val dist = depthEstimator.estimateDistance("person", mergedBbox.height())
                    val region = depthEstimator.classifyRegion(mergedCenter.first)

                    mergedPersons[i] = Detection(
                        className = "person",
                        classId = 0,
                        confidence = maxConf,
                        bbox = mergedBbox,
                        center = mergedCenter,
                        areaRatio = mergedBbox.width() * mergedBbox.height(),
                        region = region,
                        estimatedDistanceM = dist
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

    private fun shouldMergePersons(b1: RectF, b2: RectF, c1: Pair<Float, Float>, c2: Pair<Float, Float>): Boolean {
        val iou = computeIoU(b1, b2)
        if (iou >= 0.28f) return true

        val a1 = b1.width() * b1.height()
        val a2 = b2.width() * b2.height()
        val interArea = computeIntersectionArea(b1, b2)
        val containment = if (min(a1, a2) > 0) interArea / min(a1, a2) else 0f
        if (containment >= 0.65f) return true

        val dx = abs(c1.first - c2.first)
        val dy = abs(c1.second - c2.second)
        val avgW = (b1.width() + b2.width()) * 0.5f
        if (dx < avgW * 0.55f && dy < 0.28f) return true

        return false
    }

    private fun classifyTrackRegion(bbox: RectF, centerX: Float, prevRegion: String? = null): String {
        val left = bbox.left
        val right = bbox.right

        if (left < 0.65f && right > 0.35f) {
            val centerOverlap = min(right, 0.65f) - max(left, 0.35f)
            val bboxWidth = max(0.01f, right - left)
            if (centerOverlap / bboxWidth >= 0.25f) {
                return "center"
            }
        }

        if (prevRegion != null) {
            val deadband = 0.04f
            return when (prevRegion) {
                "left" -> if (centerX > 0.35f + deadband) "center" else "left"
                "right" -> if (centerX < 0.65f - deadband) "center" else "right"
                "center" -> when {
                    centerX < 0.35f - deadband -> "left"
                    centerX > 0.65f + deadband -> "right"
                    else -> "center"
                }
                else -> depthEstimator.classifyRegion(centerX)
            }
        }

        return depthEstimator.classifyRegion(centerX)
    }

    private fun formatObstructionWarning(track: TrackedItem, tier: Int): String {
        val distStr = "%.1f meters".format(track.distanceM)
        val name = if (track.faceName != null) track.faceName!! else track.className
        return if (tier >= 2) {
            "Stop! $name directly in your path, $distStr away."
        } else {
            "Caution, $name ahead in walking path, $distStr away."
        }
    }

    private fun formatContactAnnouncement(track: TrackedItem): String {
        val name = track.faceName ?: "Person"
        val dist = track.distanceM
        val distStr = if (dist < 1.2f) "close to you" else "%.1f meters ahead".format(dist)
        val gazeStr = if (track.isFacingUser) " looking at you" else ""
        return "$name is $distStr$gazeStr."
    }

    private fun formatSceneDescription(tracks: List<TrackedItem>): String {
        if (tracks.isEmpty()) return ""

        val leftItems = tracks.filter { it.region == "left" }
        val centerItems = tracks.filter { it.region == "center" }
        val rightItems = tracks.filter { it.region == "right" }

        val parts = mutableListOf<String>()

        if (centerItems.isNotEmpty()) {
            val names = centerItems.joinToString(", ") { if (it.faceName != null) it.faceName!! else it.className }
            parts.add("$names ahead")
        }
        if (leftItems.isNotEmpty()) {
            val names = leftItems.joinToString(", ") { if (it.faceName != null) it.faceName!! else it.className }
            parts.add("$names on your left")
        }
        if (rightItems.isNotEmpty()) {
            val names = rightItems.joinToString(", ") { if (it.faceName != null) it.faceName!! else it.className }
            parts.add("$names on your right")
        }

        return parts.joinToString(". ") + "."
    }

    private fun computeIoU(b1: RectF, b2: RectF): Float {
        val inter = computeIntersectionArea(b1, b2)
        if (inter <= 0f) return 0f
        val union = (b1.width() * b1.height()) + (b2.width() * b2.height()) - inter
        return if (union > 0f) inter / union else 0f
    }

    private fun computeIntersectionArea(b1: RectF, b2: RectF): Float {
        val left = max(b1.left, b2.left)
        val top = max(b1.top, b2.top)
        val right = min(b1.right, b2.right)
        val bottom = min(b1.bottom, b2.bottom)
        val w = max(0f, right - left)
        val h = max(0f, bottom - top)
        return w * h
    }

    private fun computeCenterDist(c1: Pair<Float, Float>, c2: Pair<Float, Float>): Float {
        val dx = c1.first - c2.first
        val dy = c1.second - c2.second
        return sqrt(dx * dx + dy * dy)
    }

    fun clearAllTracks() {
        trackedObjects.clear()
        lastHadActiveObjects = false
        lastSpokenEventText = null
    }

    fun reset() {
        clearAllTracks()
    }

    fun getFullSceneSummary(rawDetections: List<Detection> = emptyList()): String {
        val active = trackedObjects.values.filter { it.state == TrackState.CONFIRMED && it.framesMissing == 0 }
        if (active.isEmpty()) {
            return if (rawDetections.isEmpty()) "Path is clear. No obstacles detected."
            else {
                val names = rawDetections.joinToString(", ") { it.className }
                "Detected $names ahead."
            }
        }
        return formatSceneDescription(active.toList())
    }
}
