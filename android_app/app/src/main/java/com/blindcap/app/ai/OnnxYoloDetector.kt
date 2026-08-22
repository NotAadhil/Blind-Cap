package com.blindcap.app.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.blindcap.app.engine.DepthEstimator
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.ArrayDeque
import java.util.Collections
import kotlin.math.max
import kotlin.math.min

class OnnxYoloDetector(
    private val context: Context,
    private val depthEstimator: DepthEstimator,
    private val modelFileName: String = "yolo.onnx",
    private val labelsFileName: String = "labels.txt"
) {

    private val tag = "OnnxYoloDetector"
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val labels = mutableListOf<String>()

    private val inputSize = 480
    private val confThreshold = 0.25f

    var activeDevice: String = "CPU"
    var lastInferenceMs: Float = 0f

    private val classHistoryMap = mutableMapOf<Int, ArrayDeque<Pair<String, Float>>>()

    init {
        loadLabels()
        initSession()
    }

    private fun loadLabels() {
        try {
            val reader = BufferedReader(InputStreamReader(context.assets.open(labelsFileName)))
            reader.forEachLine { line ->
                val trimmed = line.trim()
                if (trimmed.isNotEmpty()) labels.add(trimmed)
            }
            reader.close()
            Log.i(tag, "Loaded ${labels.size} labels from assets")
        } catch (e: Exception) {
            Log.e(tag, "Error loading labels: ${e.message}")
        }
    }

    private fun initSession() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(4)
                try {
                    addNnapi()
                    activeDevice = "NNAPI (Mobile NPU/GPU)"
                } catch (e: Exception) {
                    activeDevice = "CPU (Multi-Thread)"
                }
            }

            val modelBytes = context.assets.open(modelFileName).readBytes()
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
            Log.i(tag, "ONNX Runtime session initialized successfully on $activeDevice")
        } catch (e: Exception) {
            Log.e(tag, "Failed to initialize ONNX session: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val session = ortSession ?: return emptyList()
        val env = ortEnv ?: return emptyList()

        val startTime = SystemClock.elapsedRealtime()

        val resized = if (bitmap.width == inputSize && bitmap.height == inputSize) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        }

        // Convert Bitmap to NCHW FloatBuffer [1, 3, 480, 480] normalized to [0, 1]
        val floatBuffer = FloatBuffer.allocate(1 * 3 * inputSize * inputSize)
        val intValues = IntArray(inputSize * inputSize)
        resized.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        // R plane
        for (i in 0 until inputSize * inputSize) {
            val pixel = intValues[i]
            floatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        }
        // G plane
        for (i in 0 until inputSize * inputSize) {
            val pixel = intValues[i]
            floatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        }
        // B plane
        for (i in 0 until inputSize * inputSize) {
            val pixel = intValues[i]
            floatBuffer.put((pixel and 0xFF) / 255.0f)
        }
        floatBuffer.rewind()

        val inputName = session.inputNames.iterator().next()
        val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)

        val rawDetections = mutableListOf<Detection>()

        try {
            val results = session.run(Collections.singletonMap(inputName, inputTensor))
            val outputTensor = results[0] as? OnnxTensor

            if (outputTensor != null) {
                val outputShape = outputTensor.info.shape
                // End-to-End shape [1, 300, 6]
                if (outputShape.size == 3 && outputShape[2] == 6L) {
                    val fb = outputTensor.floatBuffer
                    val numDetections = outputShape[1].toInt()

                    for (i in 0 until numDetections) {
                        val offset = i * 6
                        val x1 = fb.get(offset + 0) / inputSize
                        val y1 = fb.get(offset + 1) / inputSize
                        val x2 = fb.get(offset + 2) / inputSize
                        val y2 = fb.get(offset + 3) / inputSize
                        val score = fb.get(offset + 4)
                        val classId = fb.get(offset + 5).toInt()

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
                }
            }
            results.close()
        } catch (e: Exception) {
            Log.e(tag, "Inference error: ${e.message}")
        } finally {
            inputTensor.close()
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
        ortSession?.close()
        ortEnv?.close()
    }
}