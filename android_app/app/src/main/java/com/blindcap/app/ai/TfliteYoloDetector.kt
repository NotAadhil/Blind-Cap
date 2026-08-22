package com.blindcap.app.ai

import android.content.Context
import android.graphics.Bitmap
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
import java.util.ArrayDeque
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

    private val inputSize = 320
    private val confThreshold = 0.25f

    var activeDevice: String = "Initializing..."
    var lastTimings: DetectionTimings = DetectionTimings()
    var lastError: String? = null

    // High performance TFLite Support ImageProcessor (native C++ SIMD)
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0f, 255f))
        .build()

    private var tensorImage = TensorImage(DataType.FLOAT32)

    // Output shape for YOLO26n End-to-End: [1, 300, 6] -> [x1, y1, x2, y2, score, class_id]
    private val outputBuffer: Array<Array<FloatArray>> = Array(1) { Array(300) { FloatArray(6) } }
    private val classHistoryMap = mutableMapOf<Int, ArrayDeque<Pair<String, Float>>>()

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
            Log.i(tag, "Loaded ${labels.size} official COCO class labels from assets")
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
                activeDevice = "YOLO26n 320 (GPU Accel)"
                initializedWithGpu = true
                Log.i(tag, "Initialized TFLite with GPU Delegate acceleration successfully")
            } catch (e: Exception) {
                Log.w(tag, "GPU Delegate unavailable, falling back to 4T CPU: ${e.message}")
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

    fun detect(bitmap: Bitmap): List<Detection> {
        val interp = interpreter
        if (interp == null) {
            lastError = "Interpreter not initialized"
            return emptyList()
        }

        val t0 = SystemClock.elapsedRealtimeNanos()
        val rawDetections = mutableListOf<Detection>()

        try {
            // 1. Preprocessing (Native C++ SIMD)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            val t1 = SystemClock.elapsedRealtimeNanos()

            // 2. Hardware Inference
            interp.run(processedImage.buffer, outputBuffer)
            val t2 = SystemClock.elapsedRealtimeNanos()

            // 3. Postprocessing & Filtering
            val detections300 = outputBuffer[0]
            val invSize = 1.0f / inputSize.toFloat()

            for (i in 0 until 300) {
                val row = detections300[i]
                val score = row[4]
                if (score < confThreshold) continue

                val classId = row[5].roundToInt()
                if (classId !in labels.indices) continue

                val x1 = row[0] * invSize
                val y1 = row[1] * invSize
                val x2 = row[2] * invSize
                val y2 = row[3] * invSize

                val rawClassName = labels[classId]
                val left = max(0f, min(1f, x1))
                val top = max(0f, min(1f, y1))
                val right = max(0f, min(1f, x2))
                val bottom = max(0f, min(1f, y2))

                val bbox = RectF(left, top, right, bottom)
                val cx = (left + right) * 0.5f
                val cy = (top + bottom) * 0.5f
                val areaRatio = bbox.width() * bbox.height()

                val smoothedClass = smoothClassLabel(i, rawClassName, score)
                val distanceM = depthEstimator.estimateDistance(smoothedClass, bbox.height())
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

            val preMs = (t1 - t0) / 1_000_000f
            val infMs = (t2 - t1) / 1_000_000f
            val postMs = (t3 - t2) / 1_000_000f
            val totalMs = (t3 - t0) / 1_000_000f

            lastTimings = DetectionTimings(preMs, infMs, postMs, totalMs)
            lastError = null
        } catch (e: Exception) {
            lastError = "TFLite inference: ${e.message}"
            Log.e(tag, "Inference error: ${e.message}", e)
        }

        return rawDetections
    }

    private fun smoothClassLabel(slot: Int, newClass: String, confidence: Float): String {
        val history = classHistoryMap.getOrPut(slot) { ArrayDeque(4) }
        if (history.size >= 4) history.removeFirst()
        history.addLast(Pair(newClass, confidence))

        val scoreMap = mutableMapOf<String, Float>()
        for ((cls, conf) in history) {
            scoreMap[cls] = (scoreMap[cls] ?: 0f) + conf
        }
        return scoreMap.maxByOrNull { it.value }?.key ?: newClass
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}