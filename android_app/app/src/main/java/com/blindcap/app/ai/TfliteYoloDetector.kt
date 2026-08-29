package com.blindcap.app.ai

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.blindcap.app.engine.DepthEstimator
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

enum class DetectorState {
    CREATING,
    LOADING_MODEL,
    CREATING_RUNTIME,
    READY,
    FAILED
}

data class DetectionTimings(
    val preprocessMs: Float = 0f,
    val inferenceMs: Float = 0f,
    val postprocessMs: Float = 0f,
    val totalMs: Float = 0f,
    val p95Ms: Float = 0f
)

class TfliteYoloDetector(
    private val context: Context,
    private val depthEstimator: DepthEstimator = DepthEstimator(),
    private val modelFileName: String = "yolo.tflite",
    private val labelsFileName: String = "labels.txt",
    var confThreshold: Float = 0.22f,
    private val iouThreshold: Float = 0.45f
) {

    private val tag = "TfliteYoloDetector"
    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    val inputSize = 320

    var detectorState: DetectorState = DetectorState.CREATING
        private set

    var activeDevice: String = "Initializing..."
        private set

    var lastTimings: DetectionTimings = DetectionTimings()
        private set

    var lastError: String? = null
        private set

    // Reusable TensorImage and ImageProcessor (Float32 normalized [0.0, 1.0])
    private val tensorImage = TensorImage(DataType.FLOAT32)
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0f, 255f))
        .build()

    // Fixed output buffer - shape: [1, 300, 6] -> [x1, y1, x2, y2, score, class_id]
    private val outputBuffer: Array<Array<FloatArray>> = Array(1) { Array(300) { FloatArray(6) } }

    // Rolling latency history for P50 / P95 calculation (last 60 frames)
    private val latencyHistory = FloatArray(60)
    private var latencyIndex = 0
    private var latencyCount = 0

    init {
        Log.i(tag, "=== STARTING OBJECT DETECTOR INITIALIZATION ===")
        detectorState = DetectorState.CREATING
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
            Log.i(tag, "SUCCESS: Loaded ${labels.size} class labels from $labelsFileName")
        } catch (e: Exception) {
            lastError = "Label load (${labelsFileName}): ${e.javaClass.simpleName}: ${e.message}"
            Log.e(tag, "ERROR loading labels ($labelsFileName)", e)
        }
    }

    private fun initInterpreter() {
        detectorState = DetectorState.LOADING_MODEL
        try {
            val modelBuffer: ByteBuffer = try {
                val afd: AssetFileDescriptor = context.assets.openFd(modelFileName)
                val inputStream = FileInputStream(afd.fileDescriptor)
                val fileChannel = inputStream.channel
                val startOffset = afd.startOffset
                val declaredLength = afd.declaredLength
                val mapped = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
                Log.i(tag, "SUCCESS: Memory-mapped $modelFileName (size: $declaredLength bytes)")
                mapped
            } catch (afdEx: Exception) {
                Log.w(tag, "Asset openFd fallback (${afdEx.message}), loading via byte stream...")
                val modelBytes = context.assets.open(modelFileName).readBytes()
                val direct = ByteBuffer.allocateDirect(modelBytes.size).apply {
                    order(ByteOrder.nativeOrder())
                    put(modelBytes)
                    rewind()
                }
                Log.i(tag, "SUCCESS: Loaded direct byte buffer for $modelFileName (${modelBytes.size} bytes)")
                direct
            }

            detectorState = DetectorState.CREATING_RUNTIME
            val cpuOptions = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
            interpreter = Interpreter(modelBuffer, cpuOptions)

            // Validate Input and Output Tensors
            val inputTensor = interpreter?.getInputTensor(0)
            val outputTensor = interpreter?.getOutputTensor(0)
            val inputShapeStr = inputTensor?.shape()?.contentToString() ?: "unknown"
            val outputShapeStr = outputTensor?.shape()?.contentToString() ?: "unknown"
            Log.i(tag, "TFLite Model Verified: Input Shape=$inputShapeStr, Output Shape=$outputShapeStr")

            activeDevice = "YOLO26n 320 (XNNPACK 4T)"
            detectorState = DetectorState.READY
            lastError = null
            Log.i(tag, "=== OBJECT DETECTOR READY ===")
        } catch (e: Exception) {
            detectorState = DetectorState.FAILED
            lastError = "Detector Init Failed: ${e.javaClass.simpleName}: ${e.message}"
            activeDevice = "FAILED: ${e.javaClass.simpleName}"
            Log.e(tag, "CRITICAL INITIALIZATION FAILURE: Could not initialize TFLite interpreter from $modelFileName", e)
        }
    }

    @Synchronized
    fun detect(bitmap: Bitmap): List<Detection> {
        if (bitmap.isRecycled) {
            return emptyList()
        }
        val interp = interpreter
        if (interp == null || detectorState != DetectorState.READY) {
            lastError = if (lastError != null) lastError else "Interpreter not initialized (state=$detectorState)"
            Log.e(tag, "detect called but interpreter is NOT ready! state=$detectorState, error=$lastError")
            return emptyList()
        }

        val t0 = SystemClock.elapsedRealtimeNanos()
        val rawDetections = ArrayList<Detection>(16)

        try {
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            val t1 = SystemClock.elapsedRealtimeNanos()

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

            interp.run(processedImage.buffer, outputBuffer)
            val t2 = SystemClock.elapsedRealtimeNanos()

            val minRawScore = 0.15f
            val invSize = 1.0f / inputSize.toFloat()

            for (i in 0 until 300) {
                val row = detections300[i]
                val score = row[4]

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

                if (right <= left || bottom <= top) continue

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
                        estimatedDistanceM = distanceM
                    )
                )
            }

            val t3 = SystemClock.elapsedRealtimeNanos()
            val totalFrameMs = (t3 - t0) / 1_000_000f

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
            lastError = "TFLite inference: ${e.javaClass.simpleName}: ${e.message}"
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

    fun close() {
        detectorState = DetectorState.CREATING
        interpreter?.close()
        interpreter = null
    }
}
