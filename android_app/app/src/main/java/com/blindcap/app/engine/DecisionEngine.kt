package com.blindcap.app.engine

import com.blindcap.app.ai.Detection
import kotlin.math.abs
import kotlin.math.roundToInt

data class HazardEvent(
    val warningText: String? = null,
    val speakPriority: Int = 0,
    val severity: String = "INFO", // "CRITICAL", "WARNING", "CAUTION", "INFO"
    val activeObstacles: List<Detection> = emptyList(),
    val focusObject: String? = null,
    val focusSeverity: String = "INFO"
)

data class TrackedItem(
    val className: String,
    var region: String,
    var distanceM: Float,
    var framesSeen: Int = 1,
    var framesMissing: Int = 0,
    var criticalTier: Int = 0,
    var hasAnnouncedPresence: Boolean = false,
    var lastAnnouncedDistanceM: Float? = null,
    var lastDistanceAnnouncedTime: Long = 0L,
    val motionFilter: MotionFilter = MotionFilter()
)

class DecisionEngine {

    companion object {
        const val DANGER_ZONE_1_DISTANCE_M = 1.80f
        const val DANGER_ZONE_2_DISTANCE_M = 0.90f
        const val SCENE_CHANGE_COOLDOWN_MS = 1200L
        const val OBSTRUCTION_COOLDOWN_MS = 1200L
    }

    private val trackedObjects = mutableMapOf<String, TrackedItem>()
    private var nextTrackId = 1

    private var lastSpokenSceneSignature: String? = null
    private var lastSceneSpeakTime: Long = 0L
    private var lastHadObjects: Boolean = false

    fun evaluate(detections: List<Detection>, currentTimeMs: Long = System.currentTimeMillis()): HazardEvent {
        // --- 1. Track objects across frames with spatial matching ---
        val matchedKeys = mutableSetOf<String>()

        for (det in detections) {
            val binIdx = (det.center.first * 5.0f).toInt().coerceIn(0, 4)
            val trackKey = "${det.className}_$binIdx"
            matchedKeys.add(trackKey)

            val existing = trackedObjects[trackKey]
            if (existing != null) {
                existing.framesSeen++
                existing.framesMissing = 0
                existing.region = det.region
                existing.distanceM = det.estimatedDistanceM
                existing.motionFilter.updateArea(det.areaRatio)
            } else {
                val item = TrackedItem(
                    className = det.className,
                    region = det.region,
                    distanceM = det.estimatedDistanceM
                )
                item.motionFilter.updateArea(det.areaRatio)
                trackedObjects[trackKey] = item
            }
        }

        // Clean up missing tracks
        val toRemove = mutableListOf<String>()
        for ((key, item) in trackedObjects) {
            if (!matchedKeys.contains(key)) {
                item.framesMissing++
                if (item.framesMissing > 12) {
                    toRemove.add(key)
                }
            }
        }
        toRemove.forEach { trackedObjects.remove(it) }

        // --- 2. Filter stable active tracks (seen >= 2 frames) ---
        val activeTracks = trackedObjects.values.filter { it.framesSeen >= 2 && it.framesMissing == 0 }

        val elapsedSinceScene = currentTimeMs - lastSceneSpeakTime

        // Check for urgent Danger Zone Obstructions in Walking Corridor (Center)
        val centerObstacles = activeTracks.filter { it.region == "center" }
        val newCrit2 = centerObstacles.filter { it.distanceM <= DANGER_ZONE_2_DISTANCE_M && it.criticalTier < 2 }
        val newCrit1 = centerObstacles.filter { it.distanceM <= DANGER_ZONE_1_DISTANCE_M && it.criticalTier < 1 }

        // Priority 1: Critical Tier 2 Immediate Danger (<= 0.9m)
        if (newCrit2.isNotEmpty() && elapsedSinceScene >= 1000L) {
            val text = formatPathObstruction(newCrit2, tier = 2)
            for (item in newCrit2) {
                item.criticalTier = 2
                item.hasAnnouncedPresence = true
                item.lastAnnouncedDistanceM = item.distanceM
                item.lastDistanceAnnouncedTime = currentTimeMs
            }
            lastSceneSpeakTime = currentTimeMs
            lastSpokenSceneSignature = buildSceneSignature(activeTracks)
            lastHadObjects = true
            return HazardEvent(
                warningText = text,
                speakPriority = 100,
                severity = "CRITICAL",
                activeObstacles = detections,
                focusObject = newCrit2[0].className,
                focusSeverity = "CRITICAL"
            )
        }

        // Priority 2: Critical Tier 1 Danger Zone Entry (<= 1.8m)
        if (newCrit1.isNotEmpty() && elapsedSinceScene >= OBSTRUCTION_COOLDOWN_MS) {
            val text = formatPathObstruction(newCrit1, tier = 1)
            for (item in newCrit1) {
                item.criticalTier = 1
                item.hasAnnouncedPresence = true
                item.lastAnnouncedDistanceM = item.distanceM
                item.lastDistanceAnnouncedTime = currentTimeMs
            }
            lastSceneSpeakTime = currentTimeMs
            lastSpokenSceneSignature = buildSceneSignature(activeTracks)
            lastHadObjects = true
            return HazardEvent(
                warningText = text,
                speakPriority = 90,
                severity = "CRITICAL",
                activeObstacles = detections,
                focusObject = newCrit1[0].className,
                focusSeverity = "CRITICAL"
            )
        }

        // --- 3. Dynamic Multi-Object Scene Evaluation with Spoken Baseline Diffing ---
        val currentSig = buildSceneSignature(activeTracks)
        val sigChanged = (currentSig != lastSpokenSceneSignature) && activeTracks.isNotEmpty()
        val clearedChanged = activeTracks.isEmpty() && (lastSpokenSceneSignature != null)

        val hasDistanceShift = activeTracks.any {
            it.hasAnnouncedPresence &&
            it.lastAnnouncedDistanceM != null &&
            abs(it.distanceM - it.lastAnnouncedDistanceM!!) >= 0.80f &&
            (currentTimeMs - it.lastDistanceAnnouncedTime) >= 2500L
        }

        if ((sigChanged || hasDistanceShift) && elapsedSinceScene >= SCENE_CHANGE_COOLDOWN_MS) {
            val text = formatSceneDescription(activeTracks)
            lastSpokenSceneSignature = currentSig
            lastSceneSpeakTime = currentTimeMs
            lastHadObjects = true
            for (t in activeTracks) {
                t.hasAnnouncedPresence = true
                t.lastAnnouncedDistanceM = t.distanceM
                t.lastDistanceAnnouncedTime = currentTimeMs
            }
            return HazardEvent(
                warningText = text,
                speakPriority = 60,
                severity = "WARNING",
                activeObstacles = detections,
                focusObject = activeTracks.firstOrNull()?.className,
                focusSeverity = "WARNING"
            )
        }

        if (clearedChanged && elapsedSinceScene >= SCENE_CHANGE_COOLDOWN_MS) {
            lastSpokenSceneSignature = null
            lastSceneSpeakTime = currentTimeMs
            lastHadObjects = false
            return HazardEvent(
                warningText = "Path is clear.",
                speakPriority = 40,
                severity = "INFO",
                activeObstacles = emptyList(),
                focusObject = null,
                focusSeverity = "INFO"
            )
        }

        return HazardEvent(
            warningText = null,
            activeObstacles = detections,
            focusObject = activeTracks.firstOrNull()?.className,
            focusSeverity = if (activeTracks.any { it.criticalTier > 0 }) "CRITICAL" else "INFO"
        )
    }

    private fun buildSceneSignature(tracks: List<TrackedItem>): String {
        return tracks.groupBy { it.className to it.region }
            .map { (key, list) -> "${key.first}_${key.second}_${list.size}" }
            .sorted()
            .joinToString("|")
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
                "left" -> "$itemsStr on the left$distSuffix."
                else -> "$itemsStr on the right$distSuffix."
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
        return if (rounded in 0.5f..5.0f) {
            ", $rounded meters away"
        } else {
            ""
        }
    }

    fun getFullSceneSummary(detections: List<Detection>): String {
        if (detections.isEmpty()) {
            return if (trackedObjects.isEmpty()) {
                "The path ahead is completely clear. No objects detected."
            } else {
                val desc = formatSceneDescription(trackedObjects.values.toList())
                "Scene summary: $desc"
            }
        }
        // Describe live detections directly so button always reflects what camera sees
        val grouped = detections.groupBy { it.region }
        val parts = mutableListOf<String>()
        grouped["center"]?.let { items ->
            val names = items.groupingBy { it.className }.eachCount()
                .map { (n, c) -> if (c > 1) "$c ${pluralize(n)}" else n }
                .joinToString(" and ")
            val dist = items.minOfOrNull { it.estimatedDistanceM } ?: 0f
            parts.add("$names ahead${if (dist in 0.3f..5f) ", ${(dist * 10).roundToInt() / 10.0f} meters away" else ""}")
        }
        grouped["left"]?.let { items ->
            val names = items.groupingBy { it.className }.eachCount()
                .map { (n, c) -> if (c > 1) "$c ${pluralize(n)}" else n }
                .joinToString(" and ")
            parts.add("$names on your left")
        }
        grouped["right"]?.let { items ->
            val names = items.groupingBy { it.className }.eachCount()
                .map { (n, c) -> if (c > 1) "$c ${pluralize(n)}" else n }
                .joinToString(" and ")
            parts.add("$names on your right")
        }
        return if (parts.isEmpty()) "Path is clear." else parts.joinToString(". ") + "."
    }

    fun reset() {
        trackedObjects.clear()
        lastSpokenSceneSignature = null
        lastSceneSpeakTime = 0L
        lastHadObjects = false
    }
}
