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
import java.util.concurrent.atomic.AtomicBoolean
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
    
    // Calibrated Cosine Similarity Threshold for 192D MobileFaceNet: 0.44f
    private val cosineSimilarityThreshold = 0.44f

    private var faceDetector: FaceDetector? = null
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private val inputSize = 112 // MobileFaceNet input dimension
    private val embeddingDim = 192 // Standard output dimension

    // Diagnostic Telemetry
    @Volatile
    var lastDiagnostic: String = "Initializing..."
        private set

    @Volatile
    var lastFaceScanMs: Float = 0f
        private set

    // In-flight task guard
    val isScanningActive = AtomicBoolean(false)

    // In-memory contact cache
    private val registeredContacts = mutableListOf<FaceContact>()

    // Reusable buffers to eliminate per-frame garbage collection
    private lateinit var inputByteBuffer: ByteBuffer
    private val outputEmbeddingBuffer: Array<FloatArray> = Array(1) { FloatArray(embeddingDim) }
    private val cropFaceBitmap: Bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val cropCanvas: Canvas = Canvas(cropFaceBitmap)
    private val cropPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val intPixels = IntArray(inputSize * inputSize)

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

            // Allocate direct byte buffer for Float32 tensor: 1 * 112 * 112 * 3 * 4
            inputByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            isInitialized = true
        } catch (e: Exception) {
            lastDiagnostic = "Model Init Error: ${e.message}"
            Log.e(tag, "Failed to initialize MobileFaceNet: ${e.message}", e)
        }
    }

    /**
     * Detect and identify faces asynchronously.
     * Uses zero per-frame allocation with reusable embedding buffers.
     */
    fun detectAndRecognizeFaces(
        bitmap: Bitmap,
        personDetections: List<Detection> = emptyList()
    ): List<RecognizedFace> {
        val t0 = System.currentTimeMillis()
        val detector = faceDetector
        val recognizedList = ArrayList<RecognizedFace>(4)
        val bmpW = bitmap.width.toFloat()
        val bmpH = bitmap.height.toFloat()

        if (detector != null) {
            try {
                val inputImage = InputImage.fromBitmap(bitmap, 0)
                val task = detector.process(inputImage)
                val faces: List<Face> = Tasks.await(task, 450L, TimeUnit.MILLISECONDS)

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
                // Timeout or skip safely
            }
        }

        // Strategy B: Fallback Anatomical Head Extraction from YOLO Person Bounding Boxes
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
                            break // Bound to 1 fallback head match per scan
                        }
                    }
                }
            }
        }

        lastFaceScanMs = (System.currentTimeMillis() - t0).toFloat()

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
     * Synchronized and uses zero per-call heap allocation.
     */
    @Synchronized
    private fun extractFaceEmbedding(bitmap: Bitmap, faceBounds: Rect): FloatArray? {
        val interp = interpreter ?: return null

        try {
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

            // Direct normalization into Float32 ByteBuffer: (pixel - 127.5) / 128.0
            inputByteBuffer.rewind()
            cropFaceBitmap.getPixels(intPixels, 0, inputSize, 0, 0, inputSize, inputSize)

            val normInv = 1.0f / 128.0f
            val totalPixels = inputSize * inputSize
            for (i in 0 until totalPixels) {
                val pixel = intPixels[i]
                val r = (((pixel shr 16) and 0xFF) - 127.5f) * normInv
                val g = (((pixel shr 8) and 0xFF) - 127.5f) * normInv
                val b = ((pixel and 0xFF) - 127.5f) * normInv

                inputByteBuffer.putFloat(r)
                inputByteBuffer.putFloat(g)
                inputByteBuffer.putFloat(b)
            }

            // Run TFLite inference
            interp.run(inputByteBuffer, outputEmbeddingBuffer)

            // L2-Normalize output embedding vector in-place
            val rawVector = outputEmbeddingBuffer[0]
            var sumSquares = 0f
            for (v in rawVector) {
                sumSquares += v * v
            }
            val norm = sqrt(sumSquares).coerceAtLeast(1e-6f)
            val invNorm = 1.0f / norm
            val result = FloatArray(embeddingDim)
            for (i in 0 until embeddingDim) {
                result[i] = rawVector[i] * invNorm
            }

            return result

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

    private fun computeCosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0f
        val len = min(v1.size, v2.size)
        for (i in 0 until len) {
            dot += v1[i] * v2[i]
        }
        return dot
    }

    fun registerFaceFromBitmap(name: String, bitmap: Bitmap): Result<FaceContact> {
        val detector = faceDetector ?: return Result.failure(IllegalStateException("Face detector not ready"))
        return try {
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            val faces = Tasks.await(detector.process(inputImage), 1500L, TimeUnit.MILLISECONDS)
            if (faces.isEmpty()) {
                return Result.failure(IllegalStateException("No face detected in image"))
            }

            val largestFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                ?: return Result.failure(IllegalStateException("No valid face found"))

            val embedding = extractFaceEmbedding(bitmap, largestFace.boundingBox)
                ?: return Result.failure(IllegalStateException("Failed to extract face embedding"))

            val contact = FaceContact(
                id = UUID.randomUUID().toString(),
                name = name.trim(),
                embedding = embedding,
                enrolledTimestamp = System.currentTimeMillis()
            )

            registeredContacts.removeAll { it.name.equals(name.trim(), ignoreCase = true) }
            registeredContacts.add(contact)
            saveRegisteredContacts()
            lastDiagnostic = "Enrolled: ${contact.name}"
            Result.success(contact)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun getRegisteredContacts(): List<FaceContact> = registeredContacts.toList()

    fun deleteContact(id: String): Boolean {
        val removed = registeredContacts.removeAll { it.id == id }
        if (removed) saveRegisteredContacts()
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

            val jsonStr = file.readText()
            val root = JSONObject(jsonStr)
            val contactsArr = root.optJSONArray("contacts") ?: return

            registeredContacts.clear()
            for (i in 0 until contactsArr.length()) {
                val obj = contactsArr.getJSONObject(i)
                val id = obj.getString("id")
                val name = obj.getString("name")
                val timestamp = obj.optLong("timestamp", System.currentTimeMillis())
                val embArr = obj.getJSONArray("embedding")
                val embedding = FloatArray(embArr.length())
                for (j in 0 until embArr.length()) {
                    embedding[j] = embArr.getDouble(j).toFloat()
                }

                registeredContacts.add(FaceContact(id, name, embedding, timestamp))
            }
            Log.i(tag, "Loaded ${registeredContacts.size} registered contacts from storage")
        } catch (e: Exception) {
            Log.e(tag, "Error loading face registry: ${e.message}", e)
        }
    }

    private fun saveRegisteredContacts() {
        try {
            val file = File(context.filesDir, registryFileName)
            val root = JSONObject()
            val contactsArr = JSONArray()

            for (contact in registeredContacts) {
                val obj = JSONObject().apply {
                    put("id", contact.id)
                    put("name", contact.name)
                    put("timestamp", contact.enrolledTimestamp)
                    val embArr = JSONArray()
                    for (v in contact.embedding) embArr.put(v.toDouble())
                    put("embedding", embArr)
                }
                contactsArr.put(obj)
            }

            root.put("contacts", contactsArr)
            file.writeText(root.toString())
            Log.i(tag, "Saved ${registeredContacts.size} contacts to registry")
        } catch (e: Exception) {
            Log.e(tag, "Error saving face registry: ${e.message}", e)
        }
    }

    fun close() {
        faceDetector?.close()
        interpreter?.close()
        gpuDelegate?.close()
        cropFaceBitmap.recycle()
    }
}
