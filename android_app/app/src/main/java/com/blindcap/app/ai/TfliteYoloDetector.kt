package com.blindcap.app.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.blindcap.app.engine.DepthEstimator
import org.tensorflow.lite.Interpreter
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

    private val inputSize = 480
    private val confThreshold = 0.25f

    var activeDevice: String = "TFLite CPU (XNNPACK)"
    var lastInferenceMs: Float = 0f
    var lastError: String? = null

    // Pre-allocated direct FloatBuffer for NHWC input [1, 480, 480, 3]
    private val inputDirectBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val inputFloatBuffer: FloatBuffer = inputDirectBuffer.asFloatBuffer()
    private val intValues = IntArray(inputSize * inputSize)

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

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }

            interpreter = Interpreter(modelBuffer, options)
            activeDevice = "YOLO26n (TFLite XNNPACK 4T)"
            Log.i(tag, "TensorFlow Lite interpreter successfully initialized with ${modelBytes.size} bytes model")
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
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        }

        // Convert Bitmap to NHWC FloatBuffer [1, 480, 480, 3] normalized to [0.0, 1.0]
        synchronized(inputFloatBuffer) {
            inputFloatBuffer.rewind()
            resized.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

            val totalPixels = inputSize * inputSize
            for (i in 0 until totalPixels) {
                val pixel = intValues[i]
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                inputFloatBuffer.put(r)
                inputFloatBuffer.put(g)
                inputFloatBuffer.put(b)
            }
            inputFloatBuffer.rewind()
        }

        val rawDetections = mutableListOf<Detection>()

        try {
            interp.run(inputDirectBuffer, outputBuffer)

            val detections300 = outputBuffer[0]
            for (i in 0 until 300) {
                val row = detections300[i]
                val x1 = row[0] / inputSize.toFloat()
                val y1 = row[1] / inputSize.toFloat()
                val x2 = row[2] / inputSize.toFloat()
                val y2 = row[3] / inputSize.toFloat()
                val score = row[4]
                val classId = row[5].roundToInt()

                if (score >= confThreshold && classId in labels.indices) {
                    val rawClassName = labels[classId]
                    val left = max(0f, min(1f, x1))
                    val top = max(0f, min(1f, y1))
                    val right = max(0f, min(1f, x2))
                    val bottom = max(0f, min(1f, y2))

                    val bbox = RectF(left, top, right, bottom)
                    val cx = (left + right) / 2.0f
                    val cy = (top + bottom) / 2.0f
                    val areaRatio = bbox.width() * bbox.height()

                    val smoothedClass = smoothClassLabel(i, rawClassName, score)
                    val distanceM = depthEstimator.estimateDistance(smoothedClass, bbox.height())
                    val region = depthEstimator.classifyRegion(cx)

                    // Diagnostic logging for every verified detection
                    Log.d(tag, "DETECTION: class_id=$classId, class_name=$smoothedClass, confidence=%.3f, box=[%.2f, %.2f, %.2f, %.2f], dist=%.1fm".format(score, left, top, right, bottom, distanceM))

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
        val history = classHistoryMap.getOrPut(slot) { ArrayDeque(5) }
        if (history.size >= 5) history.removeFirst()
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