package com.blindcap.app.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.blindcap.app.engine.DepthEstimator
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class DetectionTimings(
    val preprocessMs: Float = 0f,
    val inferenceMs: Float = 0f,
    val postprocessMs: Float = 0f,
    val totalMs: Float = 0f,
    val p95Ms: Float = 0f
)

class TfliteYoloDetector(
    private val context: Context,
    private val modelFileName: String = "yolo26n_320_float32.tflite",
    private val labelsFileName: String = "coco_labels.txt",
    private val confThreshold: Float = 0.35f,
    private val iouThreshold: Float = 0.45f
) {

    private val tag = "TfliteYoloDetector"
    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    val inputSize = 320
    private val depthEstimator = DepthEstimator()

    var activeDevice: String = "Initializing..."
        private set

    var lastTimings: DetectionTimings = DetectionTimings()
        private set

    var lastError: String? = null
        private set

    // Direct ByteBuffer for Float32 tensor [1, 320, 320, 3] = 1 * 320 * 320 * 3 * 4 = 1,228,800 bytes
    private val inputByteBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    // Pre-allocated scaled bitmap and canvas for hardware downsampling
    private val scaledBitmap: Bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val scaledCanvas: Canvas = Canvas(scaledBitmap)
    private val scalePaint: Paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val srcRect: Rect = Rect()
    private val dstRect: Rect = Rect(0, 0, inputSize, inputSize)
    private val intPixels: IntArray = IntArray(inputSize * inputSize)

    // Fixed output buffer - reused across every call, never reallocated
    // Shape: [1, 300, 6] -> [x1, y1, x2, y2, score, class_id]
    private val outputBuffer: Array<Array<FloatArray>> = Array(1) { Array(300) { FloatArray(6) } }

    // Rolling latency history for P50 / P95 calculation (last 60 frames)
    private val latencyHistory = FloatArray(60)
    private var latencyIndex = 0
    private var latencyCount = 0

    init {
        loadLabels()
        initInterpreter()
    }

    private fun loadLabels() {
        try {
            val reader = BufferedReader(InputStreamReader(context.assets.open(labelsFileName)))
            reader.forEachLine { line ->
                val trimmed = line.trim()
                if (trimmed.isNotEmpty()) labels.add(trimmed)
            }
            reader.close()
            Log.i(tag, "Loaded ${labels.size} COCO class labels")
        } catch (e: Exception) {
            lastError = "Label load: ${e.message}"
            Log.e(tag, "Failed to load labels: ${e.message}", e)
        }
    }

    private fun initInterpreter() {
        try {
            val modelBytes = context.assets.open(modelFileName).readBytes()
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(modelBytes)
                rewind()
            }

            val cpuOptions = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
            interpreter = Interpreter(modelBuffer, cpuOptions)
            activeDevice = "YOLO26n 320 (XNNPACK 4T)"
            Log.i(tag, "Initialized TFLite with XNNPACK 4T CPU successfully (~18ms inference)")
        } catch (e: Exception) {
            lastError = "Init TFLite: ${e.message}"
            Log.e(tag, "Failed to initialize TFLite interpreter: ${e.message}", e)
        }
    }

    /**
     * Run detection on the provided bitmap.
     * Versioned with frameId to prevent asynchronous out-of-order stale application.
     */
    @Synchronized
    fun detect(bitmap: Bitmap, frameId: Long = 0L): List<Detection> {
        val interp = interpreter
        if (interp == null) {
            lastError = "Interpreter not initialized"
            return emptyList()
        }

        val t0 = SystemClock.elapsedRealtimeNanos()
        val rawDetections = ArrayList<Detection>(16)

        try {
            // 1. Direct hardware-scaled pixel downsampling into pre-allocated buffer
            srcRect.set(0, 0, bitmap.width, bitmap.height)
            scaledCanvas.drawBitmap(bitmap, srcRect, dstRect, scalePaint)
            scaledBitmap.getPixels(intPixels, 0, inputSize, 0, 0, inputSize, inputSize)

            // Direct zero-copy write to native Float32 ByteBuffer
            inputByteBuffer.rewind()
            val normInv = 1.0f / 255.0f
            val totalPixels = inputSize * inputSize

            for (i in 0 until totalPixels) {
                val pixel = intPixels[i]
                val r = ((pixel shr 16) and 0xFF) * normInv
                val g = ((pixel shr 8) and 0xFF) * normInv
                val b = (pixel and 0xFF) * normInv
                inputByteBuffer.putFloat(r)
                inputByteBuffer.putFloat(g)
                inputByteBuffer.putFloat(b)
            }
            val t1 = SystemClock.elapsedRealtimeNanos()

            // CRITICAL REGRESSION FIX: Explicitly zero-out output buffer before inference
            // Prevents previous frame detections from surviving when current frame has fewer objects!
            val detections300 = outputBuffer[0]
            for (i in 0 until 300) {
                val row = detections300[i]
                row[0] = 0f
                row[1] = 0f
                row[2] = 0f
                row[3] = 0f
                row[4] = 0f
                row[5] = -1f
            }

            // 2. Hardware inference into fixed pre-allocated outputBuffer
            interp.run(inputByteBuffer, outputBuffer)
            val t2 = SystemClock.elapsedRealtimeNanos()

            // 3. Postprocessing: filter and decode
            val minRawScore = (confThreshold * 0.70f).coerceAtLeast(0.20f)
            val invSize = 1.0f / inputSize.toFloat()

            for (i in 0 until 300) {
                val row = detections300[i]
                val score = row[4]

                // Score gate FIRST - skip all remaining calculations for low-confidence or zeroed slots
                if (score < minRawScore || score.isNaN() || score <= 0f) continue

                val classId = row[5].roundToInt()
                if (classId !in labels.indices) continue

                val x1 = row[0] * invSize
                val y1 = row[1] * invSize
                val x2 = row[2] * invSize
                val y2 = row[3] * invSize

                val left  = max(0f, min(1f, x1))
                val top   = max(0f, min(1f, y1))
                val right = max(0f, min(1f, x2))
                val bottom= max(0f, min(1f, y2))

                // Skip degenerate or invalid bboxes
                if (right <= left || bottom <= top || (right - left) < 0.01f || (bottom - top) < 0.01f) continue

                val bbox = RectF(left, top, right, bottom)
                val cx = (left + right) * 0.5f
                val cy = (top + bottom) * 0.5f
                val areaRatio = bbox.width() * bbox.height()

                val rawClassName = labels[classId]
                val distanceM = depthEstimator.estimateDistance(rawClassName, bbox.height())
                val region = depthEstimator.classifyRegion(cx)

                rawDetections.add(
                    Detection(
                        className = rawClassName,
                        classId = classId,
                        confidence = score,
                        bbox = bbox,
                        center = Pair(cx, cy),
                        areaRatio = areaRatio,
                        region = region,
                        estimatedDistanceM = distanceM,
                        frameId = frameId
                    )
                )
            }

            val t3 = SystemClock.elapsedRealtimeNanos()
            val totalFrameMs = (t3 - t0) / 1_000_000f

            // Record latency history for rolling P95
            latencyHistory[latencyIndex] = totalFrameMs
            latencyIndex = (latencyIndex + 1) % latencyHistory.size
            if (latencyCount < latencyHistory.size) latencyCount++

            val p95 = computeP95Latency()

            lastTimings = DetectionTimings(
                preprocessMs  = (t1 - t0) / 1_000_000f,
                inferenceMs   = (t2 - t1) / 1_000_000f,
                postprocessMs = (t3 - t2) / 1_000_000f,
                totalMs       = totalFrameMs,
                p95Ms         = p95
            )
            lastError = null

            return applyNms(rawDetections)

        } catch (e: Exception) {
            lastError = "TFLite inference: ${e.message}"
            Log.e(tag, "Inference error: ${e.message}", e)
        }

        return rawDetections
    }

    private fun computeP95Latency(): Float {
        if (latencyCount == 0) return 0f
        val copy = FloatArray(latencyCount)
        System.arraycopy(latencyHistory, 0, copy, 0, latencyCount)
        copy.sort()
        val p95Idx = ((latencyCount * 0.95f).toInt()).coerceIn(0, latencyCount - 1)
        return copy[p95Idx]
    }

    private fun applyNms(detections: List<Detection>): List<Detection> {
        if (detections.size <= 1) return detections

        val sorted = detections.sortedByDescending { it.confidence }
        val selected = ArrayList<Detection>(detections.size)

        for (det in sorted) {
            var shouldKeep = true
            for (kept in selected) {
                if (det.classId == kept.classId) {
                    val iou = computeIoU(det.bbox, kept.bbox)
                    if (iou > iouThreshold) {
                        shouldKeep = false
                        break
                    }
                }
            }
            if (shouldKeep) {
                selected.add(det)
            }
        }
        return selected
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

    fun close() {
        interpreter?.close()
        scaledBitmap.recycle()
    }
}
