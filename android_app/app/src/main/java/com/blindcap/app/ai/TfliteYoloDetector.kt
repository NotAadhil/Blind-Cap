package com.blindcap.app.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.blindcap.app.engine.DepthEstimator
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.ArrayDeque
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

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

    // 320x320 processing resolution for ultra fast real-time 30+ FPS performance
    private val inputSize = 320
    private val confThreshold = 0.25f

    var activeDevice: String = "YOLO26n 320 (Initializing)"
    var lastInferenceMs: Float = 0f
    var lastError: String? = null

    // Pre-allocated direct FloatBuffer for NHWC input [1, 320, 320, 3]
    private val inputDirectBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val inputFloatBuffer: FloatBuffer = inputDirectBuffer.asFloatBuffer()
    private val intValues = IntArray(inputSize * inputSize)

    // Reusable scaled bitmap to avoid memory churn
    private var scaledBitmap: Bitmap? = null

    // Output shape for YOLO26n End-to-End: [1, 300, 6] -> [x1, y1, x2, y2, score, class_id]
    private val outputBuffer: Array<Array<FloatArray>> = Array(1) { Array(300) { FloatArray(6) } }

    private val classHistoryMap = mutableMapOf<Int, ArrayDeque<Pair<String, Float>>>()

    init {
        loadLabels()
        initInterpreter()
        printDiagnosticStartupBanner()
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

            // Try GPU Delegate first for maximum FPS on Pixel / Tensor
            val compatList = CompatibilityList()
            var initializedWithGpu = false

            if (compatList.isDelegateSupportedOnThisDevice) {
                try {
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    gpuDelegate = GpuDelegate(delegateOptions)
                    val gpuOptions = Interpreter.Options().apply {
                        addDelegate(gpuDelegate)
                    }
                    interpreter = Interpreter(modelBuffer, gpuOptions)
                    activeDevice = "YOLO26n 320 (GPU Accel)"
                    initializedWithGpu = true
                    Log.i(tag, "Initialized TFLite with GPU Delegate acceleration successfully")
                } catch (e: Exception) {
                    Log.w(tag, "GPU Delegate failed to initialize, falling back to CPU: ${e.message}")
                    gpuDelegate?.close()
                    gpuDelegate = null
                }
            }

            // Fallback to high performance 4-thread XNNPACK CPU
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

    private fun printDiagnosticStartupBanner() {
        val interp = interpreter ?: return
        try {
            val inputTensor = interp.getInputTensor(0)
            val outputTensor = interp.getOutputTensor(0)

            val inputShapeStr = inputTensor.shape().joinToString("x")
            val outputShapeStr = outputTensor.shape().joinToString("x")

            Log.i(tag, "============================================================")
            Log.i(tag, "YOLO26n TFLite MODEL DIAGNOSTIC")
            Log.i(tag, "File: $modelFileName")
            Log.i(tag, "Input shape: $inputShapeStr, Type: ${inputTensor.dataType()}")
            Log.i(tag, "Output shape: $outputShapeStr, Type: ${outputTensor.dataType()}")
            Log.i(tag, "Classes count: ${labels.size} (COCO official)")
            Log.i(tag, "Active Engine: $activeDevice")
            Log.i(tag, "Architecture: YOLO26n End-to-End (NMS-Free)")
            Log.i(tag, "Confidence threshold: $confThreshold")
            Log.i(tag, "============================================================")
        } catch (e: Exception) {
            Log.w(tag, "Could not print model metadata: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val interp = interpreter
        if (interp == null) {
            lastError = "Interpreter not initialized"
            return emptyList()
        }

        val startTime = SystemClock.elapsedRealtime()

        val resized = if (bitmap.width == inputSize && bitmap.height == inputSize) {
            bitmap
        } else {
            val target = scaledBitmap ?: Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true).also {
                scaledBitmap = it
            }
            if (target.width != inputSize || target.height != inputSize) {
                Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            } else {
                val canvas = android.graphics.Canvas(target)
                val srcRect = android.graphics.Rect(0, 0, bitmap.width, bitmap.height)
                val dstRect = android.graphics.Rect(0, 0, inputSize, inputSize)
                canvas.drawBitmap(bitmap, srcRect, dstRect, null)
                target
            }
        }

        // Fast NHWC FloatBuffer normalization
        synchronized(inputFloatBuffer) {
            inputFloatBuffer.rewind()
            resized.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

            val totalPixels = inputSize * inputSize
            val inv255 = 1.0f / 255.0f
            for (i in 0 until totalPixels) {
                val pixel = intValues[i]
                inputFloatBuffer.put(((pixel shr 16) and 0xFF) * inv255)
                inputFloatBuffer.put(((pixel shr 8) and 0xFF) * inv255)
                inputFloatBuffer.put((pixel and 0xFF) * inv255)
            }
            inputFloatBuffer.rewind()
        }

        val rawDetections = mutableListOf<Detection>()

        try {
            interp.run(inputDirectBuffer, outputBuffer)

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
            lastError = null
        } catch (e: Exception) {
            lastError = "TFLite inference: ${e.message}"
            Log.e(tag, "Inference error: ${e.message}", e)
        }

        lastInferenceMs = (SystemClock.elapsedRealtime() - startTime).toFloat()
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
        scaledBitmap?.recycle()
        scaledBitmap = null
    }
}