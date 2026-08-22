package com.blindcap.app.ai

import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import com.blindcap.app.engine.DepthEstimator
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class MlKitObjectDetector(
    private val depthEstimator: DepthEstimator
) {
    private val tag = "MlKitObjectDetector"

    var lastInferenceMs: Float = 0f
    var lastError: String? = null
    var activeDevice: String = "ML Kit (On-Device)"

    // ML Kit category index -> readable label map
    private val categoryMap = mapOf(
        1 to "fashion accessory",
        2 to "food",
        3 to "furniture",
        4 to "plant",
        5 to "vehicle",
        0 to "unknown object"
    )

    // ML Kit category -> common obstacle label for TTS
    private val obstacleLabel = mapOf(
        "fashion accessory" to "object",
        "food" to "object",
        "furniture" to "furniture",
        "plant" to "plant",
        "vehicle" to "vehicle",
        "unknown object" to "object"
    )

    private val detector: ObjectDetector = ObjectDetection.getClient(
        ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build()
    )

    fun detect(bitmap: Bitmap): List<Detection> {
        val startTime = SystemClock.elapsedRealtime()
        val results = mutableListOf<Detection>()
        val latch = CountDownLatch(1)

        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            detector.process(image)
                .addOnSuccessListener { objects ->
                    for (obj in objects) {
                        val box = obj.boundingBox
                        val bw = bitmap.width.toFloat()
                        val bh = bitmap.height.toFloat()

                        val left = (box.left.toFloat() / bw).coerceIn(0f, 1f)
                        val top = (box.top.toFloat() / bh).coerceIn(0f, 1f)
                        val right = (box.right.toFloat() / bw).coerceIn(0f, 1f)
                        val bottom = (box.bottom.toFloat() / bh).coerceIn(0f, 1f)

                        val bbox = RectF(left, top, right, bottom)
                        val cx = (left + right) / 2f
                        val cy = (top + bottom) / 2f
                        val areaRatio = bbox.width() * bbox.height()

                        // Get label from tracking label or category
                        val trackingLabel = obj.labels.firstOrNull()?.text
                        val categoryIdx = obj.labels.firstOrNull()?.index ?: 0
                        val rawLabel = trackingLabel
                            ?: categoryMap[categoryIdx]
                            ?: "object"

                        val confidence = obj.labels.firstOrNull()?.confidence ?: 0.7f
                        val distanceM = depthEstimator.estimateDistance(rawLabel, bbox.height())
                        val region = depthEstimator.classifyRegion(cx)

                        results.add(
                            Detection(
                                className = rawLabel,
                                classId = categoryIdx,
                                confidence = confidence,
                                bbox = bbox,
                                center = Pair(cx, cy),
                                areaRatio = areaRatio,
                                region = region,
                                estimatedDistanceM = distanceM
                            )
                        )
                    }
                    latch.countDown()
                }
                .addOnFailureListener { e ->
                    lastError = "ML Kit: ${e.message}"
                    Log.e(tag, "ML Kit detection failed: ${e.message}", e)
                    latch.countDown()
                }

            latch.await(500, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {
            lastError = "Detector crash: ${e.message}"
            Log.e(tag, "Detection crash: ${e.message}", e)
        }

        lastInferenceMs = (SystemClock.elapsedRealtime() - startTime).toFloat()
        if (lastError == null && results.isNotEmpty()) {
            Log.d(tag, "Detected ${results.size} objects in ${lastInferenceMs}ms")
        }
        return results
    }

    fun close() {
        detector.close()
    }
}