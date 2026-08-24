package com.blindcap.app.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class FaceContact(
    val id: String,
    val name: String,
    val embedding: FloatArray,
    val enrolledTimestamp: Long
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as FaceContact
        return id == other.id
    }

    override fun hashCode(): Int = id.hashCode()
}

data class RecognizedFace(
    val name: String?,
    val isKnown: Boolean,
    val confidence: Float,
    val bbox: RectF, // Normalized [0..1]
    val isFacingUser: Boolean,
    val headEulerY: Float,
    val trackingId: Int?
)

class FaceRecognitionManager(
    private val context: Context,
    private val modelFileName: String = "mobilefacenet.tflite"
) {

    private val tag = "FaceRecognitionMgr"
    private val registryFileName = "faces_registry.json"
    
    // Calibrated Cosine Similarity Threshold for 192D MobileFaceNet:
    // Random different faces: -0.10 to +0.25
    // Same person under varying lighting/poses: 0.40 to 0.75
    // Optimal operating point: 0.44f
    private val cosineSimilarityThreshold = 0.44f

    private var faceDetector: FaceDetector? = null
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private var inputSize = 112 // MobileFaceNet input dimension
    private var embeddingDim = 192 // Standard output dimension

    // Diagnostic Telemetry
    @Volatile
    var lastDiagnostic: String = "Initializing..."
        private set

    // In-memory contact cache
    private val registeredContacts = mutableListOf<FaceContact>()

    // Reusable buffers to eliminate per-frame garbage collection
    private lateinit var inputByteBuffer: ByteBuffer
    private lateinit var outputEmbeddingBuffer: Array<FloatArray>
    private val cropFaceBitmap: Bitmap = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888)
    private val cropCanvas: Canvas = Canvas(cropFaceBitmap)
    private val cropPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

    var isInitialized: Boolean = false
        private set

    init {
        initFaceDetector()
        initMobileFaceNet()
        loadRegisteredContacts()
    }

    private fun initFaceDetector() {
        try {
            val options = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setMinFaceSize(0.12f)
                .enableTracking()
                .build()

            faceDetector = FaceDetection.getClient(options)
            lastDiagnostic = "ML Kit Face Detector Ready"
            Log.i(tag, "Initialized ML Kit Fast Face Detector successfully")
        } catch (e: Exception) {
            lastDiagnostic = "ML Kit Init Error: ${e.message}"
            Log.e(tag, "Failed to initialize ML Kit Face Detector: ${e.message}", e)
        }
    }

    private fun initMobileFaceNet() {
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
                initializedWithGpu = true
                Log.i(tag, "MobileFaceNet initialized with GPU Delegate")
            } catch (e: Exception) {
                Log.w(tag, "MobileFaceNet GPU init failed, using XNNPACK CPU: ${e.message}")
                gpuDelegate?.close()
                gpuDelegate = null
            }

            if (!initializedWithGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(2)
                    setUseXNNPACK(true)
                }
                interpreter = Interpreter(modelBuffer, cpuOptions)
                Log.i(tag, "MobileFaceNet initialized with XNNPACK CPU")
            }

            val interp = interpreter ?: return

            val inputShape = interp.getInputTensor(0).shape()
            val outputShape = interp.getOutputTensor(0).shape()

            inputSize = if (inputShape.size >= 3) inputShape[1] else 112
            embeddingDim = if (outputShape.size >= 2) outputShape[1] else 192

            inputByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            outputEmbeddingBuffer = Array(1) { FloatArray(embeddingDim) }
            isInitialized = true
            lastDiagnostic = "MobileFaceNet Ready (${registeredContacts.size} contacts)"
            Log.i(tag, "MobileFaceNet ready (inputSize=$inputSize, embeddingDim=$embeddingDim)")

        } catch (e: Exception) {
            lastDiagnostic = "MobileFaceNet Init Error: ${e.message}"
            Log.e(tag, "Failed to initialize MobileFaceNet: ${e.message}", e)
        }
    }

    /**
     * Run real-time face detection and match against registered contacts.
     * Supports Dual Strategy: Primary ML Kit Face Detector + Fallback YOLO Person Head Cropping.
     */
    fun detectAndRecognizeFaces(
        bitmap: Bitmap,
        personDetections: List<Detection> = emptyList()
    ): List<RecognizedFace> {
        val recognizedList = mutableListOf<RecognizedFace>()
        val bmpW = bitmap.width.toFloat()
        val bmpH = bitmap.height.toFloat()

        if (bmpW <= 0 || bmpH <= 0) return emptyList()

        // Strategy A: Primary ML Kit Face Detection
        var mlKitSucceeded = false
        val detector = faceDetector
        if (detector != null) {
            try {
                val inputImage = InputImage.fromBitmap(bitmap, 0)
                val task = detector.process(inputImage)
                val faces: List<Face> = Tasks.await(task, 250, TimeUnit.MILLISECONDS)
                mlKitSucceeded = true

                for (face in faces) {
                    val bounds = face.boundingBox

                    val left = max(0f, bounds.left / bmpW)
                    val top = max(0f, bounds.top / bmpH)
                    val right = min(1f, bounds.right / bmpW)
                    val bottom = min(1f, bounds.bottom / bmpH)

                    if (right <= left || bottom <= top) continue

                    val normBbox = RectF(left, top, right, bottom)
                    val eulerY = face.headEulerAngleY
                    val isFacingUser = abs(eulerY) <= 22f

                    val embedding = extractFaceEmbedding(bitmap, bounds)
                    if (embedding != null) {
                        val match = findBestContactMatch(embedding)
                        if (match != null && match.second >= cosineSimilarityThreshold) {
                            recognizedList.add(
                                RecognizedFace(
                                    name = match.first.name,
                                    isKnown = true,
                                    confidence = match.second,
                                    bbox = normBbox,
                                    isFacingUser = isFacingUser,
                                    headEulerY = eulerY,
                                    trackingId = face.trackingId
                                )
                            )
                        } else {
                            recognizedList.add(
                                RecognizedFace(
                                    name = null,
                                    isKnown = false,
                                    confidence = match?.second ?: 0f,
                                    bbox = normBbox,
                                    isFacingUser = isFacingUser,
                                    headEulerY = eulerY,
                                    trackingId = face.trackingId
                                )
                            )
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(tag, "ML Kit detection step skipped/timed out: ${e.message}")
            }
        }

        // Strategy B: Fallback Anatomical Head Extraction from YOLO Person Bounding Boxes
        // If ML Kit found 0 faces but YOLO detected persons, extract the anatomical head region (top 35% of person box)
        if (recognizedList.isEmpty() && personDetections.isNotEmpty()) {
            for (det in personDetections) {
                if (!det.className.equals("person", ignoreCase = true)) continue

                val pBox = det.bbox
                val headTop = pBox.top
                val headBottom = pBox.top + (pBox.height() * 0.35f)
                val headLeft = pBox.left
                val headRight = pBox.right

                val pixelBounds = Rect(
                    (headLeft * bmpW).toInt().coerceIn(0, bitmap.width - 1),
                    (headTop * bmpH).toInt().coerceIn(0, bitmap.height - 1),
                    (headRight * bmpW).toInt().coerceIn(1, bitmap.width),
                    (headBottom * bmpH).toInt().coerceIn(1, bitmap.height)
                )

                if (pixelBounds.width() > 20 && pixelBounds.height() > 20) {
                    val embedding = extractFaceEmbedding(bitmap, pixelBounds)
                    if (embedding != null) {
                        val match = findBestContactMatch(embedding)
                        val normBbox = RectF(headLeft, headTop, headRight, headBottom)
                        if (match != null && match.second >= cosineSimilarityThreshold) {
                            recognizedList.add(
                                RecognizedFace(
                                    name = match.first.name,
                                    isKnown = true,
                                    confidence = match.second,
                                    bbox = normBbox,
                                    isFacingUser = true,
                                    headEulerY = 0f,
                                    trackingId = null
                                )
                            )
                        }
                    }
                }
            }
        }

        lastDiagnostic = if (recognizedList.isNotEmpty()) {
            val known = recognizedList.firstOrNull { it.isKnown }
            if (known != null) "Identified: ${known.name} (${String.format("%.2f", known.confidence)})"
            else "Face Detected (Unknown)"
        } else {
            "Scanning (${registeredContacts.size} contacts)"
        }

        return recognizedList
    }

    /**
     * Extract 192D normalized facial embedding from a square cropped face.
     * Synchronized to ensure thread-safe TFLite inference.
     */
    @Synchronized
    private fun extractFaceEmbedding(bitmap: Bitmap, faceBounds: Rect): FloatArray? {
        val interp = interpreter ?: return null

        try {
            // Square crop with 25% margin to preserve facial proportions without distortion
            val cx = faceBounds.centerX()
            val cy = faceBounds.centerY()
            val faceSize = max(faceBounds.width(), faceBounds.height()) * 1.25f
            val half = (faceSize / 2f).toInt()

            val srcLeft = max(0, cx - half)
            val srcTop = max(0, cy - half)
            val srcRight = min(bitmap.width, cx + half)
            val srcBottom = min(bitmap.height, cy + half)

            if (srcRight <= srcLeft || srcBottom <= srcTop) return null

            val srcRect = Rect(srcLeft, srcTop, srcRight, srcBottom)
            val dstRect = Rect(0, 0, inputSize, inputSize)

            cropCanvas.drawBitmap(bitmap, srcRect, dstRect, cropPaint)

            // Preprocess into Float32 ByteBuffer: (pixel - 127.5) / 128.0
            inputByteBuffer.rewind()
            val intValues = IntArray(inputSize * inputSize)
            cropFaceBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

            for (pixel in intValues) {
                val r = ((pixel shr 16) and 0xFF)
                val g = ((pixel shr 8) and 0xFF)
                val b = (pixel and 0xFF)

                inputByteBuffer.putFloat((r - 127.5f) / 128.0f)
                inputByteBuffer.putFloat((g - 127.5f) / 128.0f)
                inputByteBuffer.putFloat((b - 127.5f) / 128.0f)
            }

            // Run TFLite inference
            interp.run(inputByteBuffer, outputEmbeddingBuffer)

            // L2-Normalize output embedding vector
            val rawVector = outputEmbeddingBuffer[0]
            var sumSquares = 0f
            for (v in rawVector) {
                sumSquares += v * v
            }
            val norm = sqrt(sumSquares).coerceAtLeast(1e-6f)
            val normalizedVector = FloatArray(rawVector.size)
            for (i in rawVector.indices) {
                normalizedVector[i] = rawVector[i] / norm
            }

            return normalizedVector

        } catch (e: Exception) {
            Log.e(tag, "Face embedding extraction error: ${e.message}")
            return null
        }
    }

    /**
     * Compare query embedding against registered contacts using Cosine Similarity.
     */
    private fun findBestContactMatch(queryEmbedding: FloatArray): Pair<FaceContact, Float>? {
        if (registeredContacts.isEmpty()) return null

        var bestContact: FaceContact? = null
        var maxSimilarity = -1.0f

        for (contact in registeredContacts) {
            val similarity = computeCosineSimilarity(queryEmbedding, contact.embedding)
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity
                bestContact = contact
            }
        }

        return if (bestContact != null) Pair(bestContact, maxSimilarity) else null
    }

    /**
     * Cosine similarity between two unit vectors = dot product.
     */
    private fun computeCosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0f
        val len = min(v1.size, v2.size)
        for (i in 0 until len) {
            dot += v1[i] * v2[i]
        }
        return dot
    }

    /**
     * Register a new face contact from a captured bitmap.
     * Executes fully off the UI thread.
     */
    @Synchronized
    fun registerFaceFromBitmap(name: String, bitmap: Bitmap): Result<FaceContact> {
        val trimmedName = name.trim()
        if (trimmedName.isEmpty()) {
            return Result.failure(IllegalArgumentException("Contact name cannot be empty"))
        }

        try {
            var faceBounds: Rect? = null

            // 1. Try ML Kit Face Detection
            val detector = faceDetector
            if (detector != null) {
                try {
                    val inputImage = InputImage.fromBitmap(bitmap, 0)
                    val task = detector.process(inputImage)
                    val faces: List<Face> = Tasks.await(task, 2, TimeUnit.SECONDS)
                    if (faces.isNotEmpty()) {
                        val primary = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                        faceBounds = primary?.boundingBox
                    }
                } catch (e: Exception) {
                    Log.w(tag, "ML Kit enrollment scan error: ${e.message}")
                }
            }

            // 2. Fallback: Center crop if ML Kit was not available
            if (faceBounds == null) {
                val cx = bitmap.width / 2
                val cy = bitmap.height / 2
                val size = (min(bitmap.width, bitmap.height) * 0.5f).toInt()
                faceBounds = Rect(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
            }

            val embedding = extractFaceEmbedding(bitmap, faceBounds)
                ?: return Result.failure(Exception("Failed to extract facial features. Please ensure good lighting."))

            val contact = FaceContact(
                id = UUID.randomUUID().toString(),
                name = trimmedName,
                embedding = embedding,
                enrolledTimestamp = System.currentTimeMillis()
            )

            registeredContacts.removeAll { it.name.equals(trimmedName, ignoreCase = true) }
            registeredContacts.add(contact)
            saveRegisteredContacts()

            lastDiagnostic = "Enrolled: $trimmedName (${registeredContacts.size} contacts)"
            Log.i(tag, "Successfully enrolled face contact for: $trimmedName")
            return Result.success(contact)

        } catch (e: Exception) {
            Log.e(tag, "Enrollment failed: ${e.message}", e)
            return Result.failure(e)
        }
    }

    fun getRegisteredContacts(): List<FaceContact> = registeredContacts.toList()

    fun deleteContact(id: String): Boolean {
        val removed = registeredContacts.removeAll { it.id == id }
        if (removed) {
            saveRegisteredContacts()
        }
        return removed
    }

    fun clearAllContacts() {
        registeredContacts.clear()
        saveRegisteredContacts()
    }

    private fun loadRegisteredContacts() {
        try {
            val file = File(context.filesDir, registryFileName)
            if (!file.exists()) return

            val jsonText = file.readText()
            val jsonArray = JSONArray(jsonText)

            registeredContacts.clear()
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val id = obj.optString("id", UUID.randomUUID().toString())
                val name = obj.getString("name")
                val timestamp = obj.optLong("timestamp", System.currentTimeMillis())
                val embArray = obj.getJSONArray("embedding")
                val embedding = FloatArray(embArray.length()) { j ->
                    embArray.getDouble(j).toFloat()
                }

                registeredContacts.add(FaceContact(id, name, embedding, timestamp))
            }
            Log.i(tag, "Loaded ${registeredContacts.size} registered face contacts")
        } catch (e: Exception) {
            Log.e(tag, "Error loading face contacts: ${e.message}", e)
        }
    }

    private fun saveRegisteredContacts() {
        try {
            val jsonArray = JSONArray()
            for (contact in registeredContacts) {
                val obj = JSONObject().apply {
                    put("id", contact.id)
                    put("name", contact.name)
                    put("timestamp", contact.enrolledTimestamp)
                    val embArray = JSONArray()
                    for (v in contact.embedding) {
                        embArray.put(v.toDouble())
                    }
                    put("embedding", embArray)
                }
                jsonArray.put(obj)
            }

            val file = File(context.filesDir, registryFileName)
            file.writeText(jsonArray.toString(2))
            Log.i(tag, "Saved ${registeredContacts.size} face contacts")
        } catch (e: Exception) {
            Log.e(tag, "Error saving face contacts: ${e.message}", e)
        }
    }

    fun close() {
        faceDetector?.close()
        interpreter?.close()
        gpuDelegate?.close()
        cropFaceBitmap.recycle()
    }
}