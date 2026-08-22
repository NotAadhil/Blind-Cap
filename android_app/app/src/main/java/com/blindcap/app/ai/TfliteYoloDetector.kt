package com.blindcap.app.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.blindcap.app.engine.DepthEstimator
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.util.ArrayDeque
import kotlin.math.max
import kotlin.math.min

class TfliteYoloDetector(
    private val context: Context,
    private val depthEstimator: DepthEstimator,
    private val modelFileName: String = "yolo.tflite",
    private val labelsFileName: String = "labels.txt"
) {

    private val tag = "TfliteYoloDetector"
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private val labels = mutableListOf<String>()

    private val inputSize = 480
    private val confThreshold = 0.28f
    private val iouThreshold = 0.45f

    var activeDevice: String = "CPU"
    var lastInferenceMs: Float = 0f

    // Confidence-weighted majority class smoothing buffer per track slot
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
            Log.i(tag, "Loaded ${labels.size} labels from assets")
        } catch (e: Exception) {
            Log.e(tag, "Error loading labels: ${e.message}")
        }
    }

    private fun initInterpreter() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, modelFileName)
            val options = Interpreter.Options()

            // 1. Try GPU Delegate
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                interpreter = Interpreter(modelBuffer, options)
                activeDevice = "GPU (Mobile GPU)"
                Log.i(tag, "TFLite initialized on GPU Delegate")
                return
            } catch (e: Exception) {
                Log.w(tag, "GPU Delegate failed, falling back to multi-threaded CPU: ${e.message}")
            }

            // 2. Multi-threaded CPU Fallback
            options.setNumThreads(4)
            interpreter = Interpreter(modelBuffer, options)
            activeDevice = "CPU (4 Threads)"
            Log.i(tag, "TFLite initialized on multi-threaded CPU")
        } catch (e: Exception) {
            Log.e(tag, "Failed to initialize TFLite interpreter: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val tflite = interpreter ?: return emptyList()

        val t0 = System.nanoTime()

        // 1. Preprocess: Resize bitmap to 480x480 with bilinear interpolation
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resized)

        // Normalize to [0.0, 1.0]
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        val processedImage = imageProcessor.process(tensorImage)

        // 2. Run Inference
        // Output tensor shape: [1, 300, 6] for End-to-End or [1, 84, 4725] for anchor format
        val outputShape = tflite.getOutputTensor(0).shape()
        val detections = mutableListOf<Detection>()

        if (outputShape.size == 3 && outputShape[2] == 6) {
            // Format A: End-to-End [1, N, 6] (x1, y1, x2, y2, score, classId)
            val outputBuffer = Array(1) { Array(outputShape[1]) { FloatArray(6) } }
            tflite.run(processedImage.buffer, outputBuffer)

            val rawDetections = mutableListOf<Detection>()
            for (row in outputBuffer[0]) {
                val score = row[4]
                if (score < confThreshold) continue
                val classId = row[5].toInt().coerceIn(0, max(0, labels.size - 1))
                val rawClassName = labels.getOrElse(classId) { "object" }

                val x1Norm = (row[0] / inputSize).coerceIn(0f, 1f)
                val y1Norm = (row[1] / inputSize).coerceIn(0f, 1f)
                val x2Norm = (row[2] / inputSize).coerceIn(0f, 1f)
                val y2Norm = (row[3] / inputSize).coerceIn(0f, 1f)

                if (x2Norm <= x1Norm || y2Norm <= y1Norm) continue

                val rect = RectF(x1Norm, y1Norm, x2Norm, y2Norm)
                val cx = (x1Norm + x2Norm) / 2f
                val cy = (y1Norm + y2Norm) / 2f
                val areaRatio = (x2Norm - x1Norm) * (y2Norm - y1Norm)
                val region = depthEstimator.classifyRegion(cx)
                val distance = depthEstimator.estimateDistance(rawClassName, y2Norm - y1Norm)

                rawDetections.add(
                    Detection(
                        className = rawClassName,
                        classId = classId,
                        confidence = score,
                        bbox = rect,
                        center = Pair(cx, cy),
                        areaRatio = areaRatio,
                        region = region,
                        estimatedDistanceM = distance
                    )
                )
            }
            detections.addAll(applyClassAwareNms(rawDetections))
        } else if (outputShape.size == 3 && outputShape[1] >= 84) {
            // Format B: Anchor Grid [1, 84, N] (cx, cy, w, h, class_probs[80])
            val numAnchors = outputShape[2]
            val outputBuffer = Array(1) { Array(outputShape[1]) { FloatArray(numAnchors) } }
            tflite.run(processedImage.buffer, outputBuffer)

            val rawDetections = mutableListOf<Detection>()
            for (col in 0 until numAnchors) {
                var maxScore = 0f
                var bestClassId = -1

                for (c in 4 until outputShape[1]) {
                    val score = outputBuffer[0][c][col]
                    if (score > maxScore) {
                        maxScore = score
                        bestClassId = c - 4
                    }
                }

                if (maxScore < confThreshold || bestClassId < 0) continue

                val cxPx = outputBuffer[0][0][col]
                val cyPx = outputBuffer[0][1][col]
                val wPx = outputBuffer[0][2][col]
                val hPx = outputBuffer[0][3][col]

                val x1Norm = ((cxPx - wPx / 2f) / inputSize).coerceIn(0f, 1f)
                val y1Norm = ((cyPx - hPx / 2f) / inputSize).coerceIn(0f, 1f)
                val x2Norm = ((cxPx + wPx / 2f) / inputSize).coerceIn(0f, 1f)
                val y2Norm = ((cyPx + hPx / 2f) / inputSize).coerceIn(0f, 1f)

                if (x2Norm <= x1Norm || y2Norm <= y1Norm) continue

                val rawClassName = labels.getOrElse(bestClassId) { "object" }
                val rect = RectF(x1Norm, y1Norm, x2Norm, y2Norm)
                val cx = (x1Norm + x2Norm) / 2f
                val cy = (y1Norm + y2Norm) / 2f
                val areaRatio = (x2Norm - x1Norm) * (y2Norm - y1Norm)
                val region = depthEstimator.classifyRegion(cx)
                val distance = depthEstimator.estimateDistance(rawClassName, y2Norm - y1Norm)

                rawDetections.add(
                    Detection(
                        className = rawClassName,
                        classId = bestClassId,
                        confidence = maxScore,
                        bbox = rect,
                        center = Pair(cx, cy),
                        areaRatio = areaRatio,
                        region = region,
                        estimatedDistanceM = distance
                    )
                )
            }
            detections.addAll(applyClassAwareNms(rawDetections))
        }

        lastInferenceMs = (System.nanoTime() - t0) / 1_000_000.0f
        return detections
    }

    private fun applyClassAwareNms(candidates: List<Detection>): List<Detection> {
        val sorted = candidates.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()

        for (cand in sorted) {
            var shouldKeep = true
            for (sel in selected) {
                // Class-aware: only suppress if same class! Overlapping different classes are preserved.
                if (sel.classId == cand.classId) {
                    val iou = calculateIoU(sel.bbox, cand.bbox)
                    if (iou > iouThreshold) {
                        shouldKeep = false
                        break
                    }
                }
            }
            if (shouldKeep) {
                selected.add(cand)
            }
        }
        return selected
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        if (interRight <= interLeft || interBottom <= interTop) return 0.0f
        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val unionArea = (a.width() * a.height()) + (b.width() * b.height()) - interArea
        return if (unionArea <= 0.0f) 0.0f else interArea / unionArea
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
    }
}
