package com.blindcap.app.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.blindcap.app.engine.DepthEstimator
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
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
    val totalMs: Float = 0f
)

class TfliteYoloDetector(
    private val context: Context,
    private val depthEstimator: DepthEstimator,
    private val modelFileName: String = "yolo.tflite",
    private val labelsFileName: String = "labels.txt"
) {

    private val tag = "Yolo26nTflite"
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private val labels = mutableListOf<String>()

    val inputSize = 320
    var confThreshold: Float = 0.30f // Configurable baseline confidence threshold

    var activeDevice: String = "Initializing..."
    var lastTimings: DetectionTimings = DetectionTimings()
    var lastError: String? = null

    // -----------------------------------------------------------------------
    // Pre-allocated reusable resources - never allocate inside detect()
    // -----------------------------------------------------------------------

    // Pre-allocated scaled bitmap for SIMD preprocessing (avoids per-frame heap allocation)
    private val scaledBitmap: Bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val scaledCanvas: Canvas = Canvas(scaledBitmap)

    // SIMD-accelerated TFLite Support ImageProcessor
    private val imageProcessor: ImageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0f, 255f))
        .build()

    // Reused TensorImage (load() reuses the backing buffer)
    private val tensorImage = TensorImage(DataType.FLOAT32)

    // Fixed output buffer - reused across every call, never reallocated
    // Shape: [1, 300, 6] -> [x1, y1, x2, y2, score, class_id]
    private val outputBuffer: Array<Array<FloatArray>> = Array(1) { Array(300) { FloatArray(6) } }

    // Class label smoothing history: slot -> (className, confidenceSum)
    // Capped at 4 most recent frames per slot (sparse - only populated on positive detections)
    private val classLabelHistory = HashMap<Int, ArrayDeque<Pair<String, Float>>>(64)
    private var classHistoryPruneCounter = 0

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
            Log.e(tag, "Error loading labels: ${e.message}", e)
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

            var initializedWithGpu = false
            try {
                gpuDelegate = GpuDelegate()
                val gpuOptions = Interpreter.Options().apply {
                    addDelegate(gpuDelegate)
                }
                interpreter = Interpreter(modelBuffer, gpuOptions)
                activeDevice = "YOLO26n 320 (GPU)"
                initializedWithGpu = true
                Log.i(tag, "Initialized TFLite with GPU Delegate successfully")
            } catch (e: Exception) {
                Log.w(tag, "GPU Delegate unavailable, falling back to XNNPACK 4T: ${e.message}")
                gpuDelegate?.close()
                gpuDelegate = null
            }

            if (!initializedWithGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseXNNPACK(true)
                }
                interpreter = Interpreter(modelBuffer, cpuOptions)
                activeDevice = "YOLO26n 320 (XNNPACK 4T)"
                Log.i(tag, "Initialized TFLite with XNNPACK 4T CPU successfully")
            }
        } catch (e: Exception) {
            lastError = "Init TFLite: ${e.message}"
            Log.e(tag, "Failed to initialize TFLite interpreter: ${e.message}", e)
        }
    }

    /**
     * Run detection on the provided bitmap.
     * Reuses all pre-allocated buffers to minimize GC pressure.
     * Returns detections with confidence >= (confThreshold * 0.70) for tracking state machine.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val interp = interpreter
        if (interp == null) {
            lastError = "Interpreter not initialized"
            return emptyList()
        }

        val t0 = SystemClock.elapsedRealtimeNanos()
        val rawDetections = mutableListOf<Detection>()

        try {
            // 1. Preprocessing: load into the REUSED TensorImage (no new allocation)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            val t1 = SystemClock.elapsedRealtimeNanos()

            // 2. Hardware inference into fixed pre-allocated outputBuffer
            interp.run(processedImage.buffer, outputBuffer)
            val t2 = SystemClock.elapsedRealtimeNanos()

            // 3. Postprocessing: filter and decode
            // Use a slightly lower raw gate so the tracking state machine can see borderline tracks,
            // but never lower than 0.18 to avoid excessive garbage detections
            val minRawScore = (confThreshold * 0.70f).coerceAtLeast(0.18f)
            val detections300 = outputBuffer[0]
            val invSize = 1.0f / inputSize.toFloat()

            for (i in 0 until 300) {
                val row = detections300[i]
                val score = row[4]

                // Score gate FIRST - skip all remaining work for low-confidence slots
                if (score < minRawScore) continue

                val classId = row[5].roundToInt()
                if (classId !in labels.indices) continue

                val x1 = row[0] * invSize
                val y1 = row[1] * invSize
                val x2 = row[2] * invSize
                val y2 = row[3] * invSize

                val rawClassName = labels[classId]
                val left  = max(0f, min(1f, x1))
                val top   = max(0f, min(1f, y1))
                val right = max(0f, min(1f, x2))
                val bottom= max(0f, min(1f, y2))

                // Skip degenerate bboxes
                if (right <= left || bottom <= top) continue

                val bbox = RectF(left, top, right, bottom)
                val cx = (left + right) * 0.5f
                val cy = (top + bottom) * 0.5f
                val areaRatio = bbox.width() * bbox.height()

                // Class smoothing only for slots that actually have a detection this frame
                val smoothedClass = smoothClassLabel(i, rawClassName, score)
                val distanceM = depthEstimator.estimateDistance(smoothedClass, bbox.height())
                // Region classification happens in DecisionEngine (it has hysteresis context)
                // Provide a raw initial region here for the data class
                val region = depthEstimator.classifyRegion(cx)

                rawDetections.add(
                    Detection(
                        className = smoothedClass,
                        classId = classId,
                        confidence = score,
                        bbox = bbox,
                        center = Pair(cx, cy),
                        areaRatio = areaRatio,
                        region = region,
                        estimatedDistanceM = distanceM
                    )
                )
            }

            val t3 = SystemClock.elapsedRealtimeNanos()
            lastTimings = DetectionTimings(
                preprocessMs  = (t1 - t0) / 1_000_000f,
                inferenceMs   = (t2 - t1) / 1_000_000f,
                postprocessMs = (t3 - t2) / 1_000_000f,
                totalMs       = (t3 - t0) / 1_000_000f
            )
            lastError = null

            // Periodically prune unused class history slots to prevent unbounded memory growth
            classHistoryPruneCounter++
            if (classHistoryPruneCounter >= 500) {
                classHistoryPruneCounter = 0
                classLabelHistory.clear()
            }

            return applyNms(rawDetections)

        } catch (e: Exception) {
            lastError = "TFLite inference: ${e.message}"
            Log.e(tag, "Inference error: ${e.message}", e)
        }

        return rawDetections
    }

    /**
     * Intra-class Non-Maximum Suppression to remove redundant candidate boxes.
     */
    private fun applyNms(detections: List<Detection>, iouThreshold: Float = 0.45f): List<Detection> {
        if (detections.size <= 1) return detections

        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()

        for (det in sorted) {
            var shouldKeep = true
            for (kept in selected) {
                if (det.classId == kept.classId || (det.className.equals("person", true) && kept.className.equals("person", true))) {
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

    /**
     * Weighted majority-vote class smoothing to suppress label flicker.
     * Only called for slots that passed the score gate (eliminates 90%+ calls).
     */
    private fun smoothClassLabel(slot: Int, newClass: String, confidence: Float): String {
        val history = classLabelHistory.getOrPut(slot) { ArrayDeque(4) }
        if (history.size >= 4) history.removeFirst()
        history.addLast(Pair(newClass, confidence))

        val scoreMap = HashMap<String, Float>(4)
        for ((cls, conf) in history) {
            scoreMap[cls] = (scoreMap[cls] ?: 0f) + conf
        }
        return scoreMap.maxByOrNull { it.value }?.key ?: newClass
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        scaledBitmap.recycle()
    }
}