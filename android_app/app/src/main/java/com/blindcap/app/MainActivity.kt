package com.blindcap.app

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.GestureDetector
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.FaceContact
import com.blindcap.app.ai.FaceRecognitionManager
import com.blindcap.app.ai.RecognizedFace
import com.blindcap.app.ai.TfliteYoloDetector
import com.blindcap.app.databinding.ActivityMainBinding
import com.blindcap.app.engine.DecisionEngine
import com.blindcap.app.engine.DepthEstimator
import com.blindcap.app.net.MjpegStreamReader
import com.blindcap.app.ocr.OcrManager
import com.blindcap.app.speech.TtsManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

enum class VideoInputSource {
    PHONE_CAMERA,
    ESP32_CAM
}

class MainActivity : AppCompatActivity() {

    private val tag = "BlindCapMainActivity"
    private val cameraPermissionCode = 1001
    private val prefsName = "BlindCapPrefs"
    private val keyStreamUrl = "esp32_stream_url"
    private val defaultStreamUrl = "http://192.168.4.1:81/stream"

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var faceExecutor: ExecutorService
    private lateinit var prefs: SharedPreferences

    private lateinit var depthEstimator: DepthEstimator
    private lateinit var detector: TfliteYoloDetector
    private lateinit var faceRecognitionManager: FaceRecognitionManager
    private lateinit var decisionEngine: DecisionEngine
    private lateinit var ttsManager: TtsManager
    private lateinit var ocrManager: OcrManager
    private lateinit var mjpegStreamReader: MjpegStreamReader

    private var currentSource = VideoInputSource.PHONE_CAMERA
    private var cameraProvider: ProcessCameraProvider? = null

    // Decoupled Frame Transfer
    private val latestFrameRef = AtomicReference<Bitmap?>(null)
    private val isAiRunning = AtomicBoolean(true)
    private var aiWorkerThread: Thread? = null

    // Asynchronous Face Recognition State (Runs parallel to YOLO, never blocks inference)
    private val isFaceScanning = AtomicBoolean(false)
    private val activeRecognizedFaces = AtomicReference<List<RecognizedFace>>(emptyList())
    private var lastFaceScanTime = 0L

    // OCR Request State & Debounce Guard
    private val isOcrProcessing = AtomicBoolean(false)
    private var lastOcrRequestTime = 0L

    // FPS Measurement Instrumentation
    private var cameraFrameCount = 0
    private var lastCameraFpsTime = SystemClock.elapsedRealtime()
    private var currentCameraFps = 0f

    private var aiFrameCount = 0
    private var lastAiFpsTime = SystemClock.elapsedRealtime()
    private var currentAiFps = 0f

    @Volatile
    private var latestBitmap: Bitmap? = null
    private var currentDetections: List<Detection> = emptyList()

    private lateinit var gestureDetector: GestureDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        cameraExecutor = Executors.newSingleThreadExecutor()
        faceExecutor = Executors.newSingleThreadExecutor()

        depthEstimator = DepthEstimator()
        detector = TfliteYoloDetector(this, depthEstimator)
        faceRecognitionManager = FaceRecognitionManager(this)
        decisionEngine = DecisionEngine()
        ocrManager = OcrManager()

        mjpegStreamReader = MjpegStreamReader(
            onFrameReceived = { bitmap ->
                if (currentSource == VideoInputSource.ESP32_CAM) {
                    measureCameraFps()
                    runOnUiThread {
                        binding.esp32StreamView.setImageBitmap(bitmap)
                    }
                    if (latestFrameRef.get() == null) {
                        latestFrameRef.set(bitmap)
                    }
                }
            },
            onStatusChanged = { status ->
                runOnUiThread {
                    binding.txtStreamStatus.text = status
                    binding.txtStreamStatus.visibility = if (status.isNotEmpty()) View.VISIBLE else View.GONE
                }
            }
        )

        ttsManager = TtsManager(this) {
            ttsManager.speak("Blind Cap ready.", priority = 50, severity = "INFO")
        }

        setupGestures()
        setupActionButtons()
        setupTopBarControls()
        startAiWorkerLoop()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                cameraPermissionCode
            )
        }
    }

    private fun startAiWorkerLoop() {
        isAiRunning.set(true)
        aiWorkerThread = Thread({
            while (isAiRunning.get()) {
                val frame = latestFrameRef.getAndSet(null)
                if (frame != null) {
                    processAiFrame(frame)
                    measureAiFps()
                } else {
                    try {
                        Thread.sleep(2)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }
        }, "BlindCap-AI-Worker").apply {
            priority = Thread.NORM_PRIORITY + 2
            isDaemon = true
            start()
        }
    }

    private fun processAiFrame(bitmap: Bitmap) {
        latestBitmap = bitmap

        // 1. Hardware Object Detection (YOLO26n / YOLOv8 - runs at full speed without waiting)
        val detections = detector.detect(bitmap)
        currentDetections = detections

        // 2. Asynchronous Face Recognition Dispatch (Parallel non-blocking worker with Dual Strategy)
        val now = SystemClock.elapsedRealtime()
        val hasPerson = detections.any { it.className.equals("person", ignoreCase = true) }

        if ((hasPerson || (now - lastFaceScanTime >= 400L)) && !isFaceScanning.get()) {
            lastFaceScanTime = now
            if (isFaceScanning.compareAndSet(false, true)) {
                // Pass immutable frame snapshot and detections to background face worker
                faceExecutor.execute {
                    try {
                        val faces = faceRecognitionManager.detectAndRecognizeFaces(bitmap, detections)
                        activeRecognizedFaces.set(faces)
                    } catch (e: Exception) {
                        Log.e(tag, "Face scan error: ${e.message}", e)
                    } finally {
                        isFaceScanning.set(false)
                    }
                }
            }
        }

        // Get latest identified faces instantly (0ms lock-free read)
        val faces = activeRecognizedFaces.get()

        // 3. Decision Engine with Scale-Invariant Tracking & Face Integration
        val event = decisionEngine.evaluate(detections, faces)

        // 4. Dispatch Speech if warranted
        if (event.warningText != null) {
            ttsManager.speak(
                text = event.warningText,
                priority = event.speakPriority,
                severity = event.severity
            )
        }

        // 5. Update UI Overlay with Telemetry
        val sourceLabel = if (currentSource == VideoInputSource.PHONE_CAMERA) "Phone" else "ESP32"
        val ttsStatus = if (ttsManager.isSpeaking) "SPEAKING" else "SILENT"
        val faceDiag = faceRecognitionManager.lastDiagnostic

        runOnUiThread {
            binding.overlayView.cameraFps = currentCameraFps
            binding.overlayView.aiFps = currentAiFps
            binding.overlayView.timings = detector.lastTimings
            binding.overlayView.activeDevice = "${detector.activeDevice} [$sourceLabel]"
            binding.overlayView.ttsStatus = ttsStatus
            binding.overlayView.errorMessage = detector.lastError
            binding.overlayView.faceDiagnostic = faceDiag
            binding.overlayView.updateResults(detections, event, faces)
        }
    }

    private fun measureCameraFps() {
        cameraFrameCount++
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - lastCameraFpsTime
        if (elapsed >= 1000L) {
            currentCameraFps = (cameraFrameCount * 1000f) / elapsed
            cameraFrameCount = 0
            lastCameraFpsTime = now
        }
    }

    private fun measureAiFps() {
        aiFrameCount++
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - lastAiFpsTime
        if (elapsed >= 1000L) {
            currentAiFps = (aiFrameCount * 1000f) / elapsed
            aiFrameCount = 0
            lastAiFpsTime = now
        }
    }

    private fun setupTopBarControls() {
        updateSourceUi()

        binding.btnSourceToggle.setOnClickListener {
            if (currentSource == VideoInputSource.PHONE_CAMERA) {
                switchToEsp32Cam()
            } else {
                switchToPhoneCamera()
            }
        }

        binding.btnStreamSettings.setOnClickListener {
            showStreamSettingsDialog()
        }

        binding.btnFaces.setOnClickListener {
            showFaceContactsDialog()
        }
    }

    private fun updateSourceUi() {
        if (currentSource == VideoInputSource.PHONE_CAMERA) {
            binding.btnSourceToggle.text = "Source: Phone"
            binding.btnSourceToggle.setBackgroundColor(0xFF333333.toInt())
            binding.viewFinder.visibility = View.VISIBLE
            binding.esp32StreamView.visibility = View.GONE
            binding.txtStreamStatus.text = ""
            binding.txtStreamStatus.visibility = View.GONE
        } else {
            binding.btnSourceToggle.text = "Source: ESP32"
            binding.btnSourceToggle.setBackgroundColor(0xFF994400.toInt())
            binding.viewFinder.visibility = View.GONE
            binding.esp32StreamView.visibility = View.VISIBLE
        }
    }

    private fun switchToEsp32Cam() {
        currentSource = VideoInputSource.ESP32_CAM
        updateSourceUi()

        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {}

        decisionEngine.reset()
        val url = prefs.getString(keyStreamUrl, defaultStreamUrl) ?: defaultStreamUrl
        mjpegStreamReader.start(url)
        ttsManager.speak("Switched to external ESP 32 camera stream.", priority = 60, severity = "INFO")
        Toast.makeText(this, "Connecting to ESP32: $url", Toast.LENGTH_SHORT).show()
    }

    private fun switchToPhoneCamera() {
        currentSource = VideoInputSource.PHONE_CAMERA
        updateSourceUi()

        mjpegStreamReader.stop()
        decisionEngine.reset()

        startCamera()
        ttsManager.speak("Switched to phone camera.", priority = 60, severity = "INFO")
    }

    private fun showStreamSettingsDialog() {
        val currentUrl = prefs.getString(keyStreamUrl, defaultStreamUrl) ?: defaultStreamUrl
        val input = EditText(this).apply {
            setText(currentUrl)
            setSelection(text.length)
            hint = "http://192.168.4.1:81/stream"
        }

        AlertDialog.Builder(this)
            .setTitle("ESP32-CAM Stream URL")
            .setMessage("Enter the MJPEG stream URL of your ESP32-CAM (e.g. http://192.168.4.1:81/stream):")
            .setView(input)
            .setPositiveButton("Save & Connect") { _, _ ->
                val newUrl = input.text.toString().trim()
                if (newUrl.isNotEmpty()) {
                    prefs.edit().putString(keyStreamUrl, newUrl).apply()
                    Toast.makeText(this, "Saved: $newUrl", Toast.LENGTH_SHORT).show()
                    if (currentSource == VideoInputSource.ESP32_CAM) {
                        mjpegStreamReader.start(newUrl)
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showFaceContactsDialog() {
        val contacts = faceRecognitionManager.getRegisteredContacts()

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 30, 50, 10)
        }

        val headerText = TextView(this).apply {
            text = "Registered Face Contacts (${contacts.size}):"
            textSize = 15f
            setTypeface(null, android.graphics.Typeface.BOLD)
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 0, 0, 12)
        }
        layout.addView(headerText)

        if (contacts.isEmpty()) {
            val emptyText = TextView(this).apply {
                text = "No contacts enrolled yet.\nTap 'Enroll New Face' to save a friend or family member."
                textSize = 13f
                setTextColor(0xFF999999.toInt())
                setPadding(0, 10, 0, 20)
            }
            layout.addView(emptyText)
        } else {
            val dateFormat = SimpleDateFormat("MMM d, yyyy", Locale.getDefault())
            for (contact in contacts) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(0, 8, 0, 8)
                    gravity = android.view.Gravity.CENTER_VERTICAL
                }

                val info = TextView(this).apply {
                    text = "👤 ${contact.name} (${dateFormat.format(Date(contact.enrolledTimestamp))})"
                    textSize = 14f
                    setTextColor(0xFF00FFCC.toInt())
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                }

                val delBtn = android.widget.Button(this).apply {
                    text = "Delete"
                    textSize = 11f
                    setBackgroundColor(0xFF882222.toInt())
                    setTextColor(0xFFFFFFFF.toInt())
                    setOnClickListener {
                        faceRecognitionManager.deleteContact(contact.id)
                        Toast.makeText(this@MainActivity, "Deleted ${contact.name}", Toast.LENGTH_SHORT).show()
                        showFaceContactsDialog()
                    }
                }

                row.addView(info)
                row.addView(delBtn)
                layout.addView(row)
            }
        }

        AlertDialog.Builder(this)
            .setTitle("Face Identification")
            .setView(layout)
            .setPositiveButton("Enroll New Face") { _, _ ->
                promptEnrollNewFace()
            }
            .setNeutralButton("Clear All") { _, _ ->
                faceRecognitionManager.clearAllContacts()
                Toast.makeText(this, "Cleared all face contacts", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun promptEnrollNewFace() {
        val bitmap = latestBitmap
        if (bitmap == null) {
            Toast.makeText(this, "Camera not ready. Point camera at face first.", Toast.LENGTH_LONG).show()
            return
        }

        val input = EditText(this).apply {
            hint = "Enter person's name (e.g. Mom, John, Doctor)"
            isSingleLine = true
        }

        AlertDialog.Builder(this)
            .setTitle("Enroll Face")
            .setMessage("Point camera directly at your friend or family member's face and enter their name:")
            .setView(input)
            .setPositiveButton("Save Contact") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isNotEmpty()) {
                    val frameToEnroll = latestBitmap ?: bitmap
                    Toast.makeText(this, "Scanning and registering face for $name...", Toast.LENGTH_SHORT).show()

                    // Execute on dedicated face worker thread (never on UI thread)
                    faceExecutor.execute {
                        val result = faceRecognitionManager.registerFaceFromBitmap(name, frameToEnroll)
                        runOnUiThread {
                            if (result.isSuccess) {
                                Toast.makeText(this@MainActivity, "Enrolled: $name successfully!", Toast.LENGTH_LONG).show()
                                ttsManager.speak("Face registered for $name.", priority = 60, severity = "INFO")
                            } else {
                                val err = result.exceptionOrNull()?.message ?: "Could not detect face"
                                Toast.makeText(this@MainActivity, "Enrollment failed: $err", Toast.LENGTH_LONG).show()
                                ttsManager.speak("Could not find face. Please ensure good lighting and try again.", priority = 60, severity = "INFO")
                            }
                        }
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun setupActionButtons() {
        binding.btnOcr.setOnClickListener {
            triggerOcrReading()
        }

        binding.btnScene.setOnClickListener {
            triggerSceneSummary()
        }
    }

    private fun setupGestures() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (ttsManager.isSpeaking) {
                    stopSpeech()
                    return true
                }
                return false
            }

            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                triggerSceneSummary()
                return true
            }

            override fun onLongPress(e: MotionEvent) {
                stopSpeech()
            }
        })

        binding.overlayView.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }
    }

    private fun stopSpeech() {
        ttsManager.stop()
        Toast.makeText(this, "Speech stopped", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        isAiRunning.set(false)
        aiWorkerThread?.interrupt()
        faceExecutor.shutdown()
        mjpegStreamReader.stop()
        cameraExecutor.shutdown()
        detector.close()
        faceRecognitionManager.close()
        ocrManager.close()
        ttsManager.shutdown()
    }

    private fun startCamera() {
        if (currentSource != VideoInputSource.PHONE_CAMERA) return

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            val provider = cameraProvider ?: return@addListener

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            // High-Performance 480x480 analysis resolution
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(480, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processImageProxy(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                provider.unbindAll()
                if (currentSource == VideoInputSource.PHONE_CAMERA) {
                    provider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                }
            } catch (exc: Exception) {
                Log.e(tag, "Camera binding failed: ${exc.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun processImageProxy(imageProxy: ImageProxy) {
        if (currentSource != VideoInputSource.PHONE_CAMERA) {
            imageProxy.close()
            return
        }

        measureCameraFps()

        // Critical optimization: If AI worker already has a frame pending, DO NOT do any bitmap conversion!
        if (latestFrameRef.get() != null) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                latestFrameRef.set(bitmap)
            }
        } catch (e: Exception) {
            Log.e(tag, "Error extracting image frame: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Thread-safe snapshot creation with proper rotation.
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val rawBitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            if (rotationDegrees != 0) {
                val matrix = Matrix().apply {
                    postRotate(rotationDegrees.toFloat())
                }
                Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            } else {
                rawBitmap
            }
        } catch (e: Exception) {
            Log.e(tag, "toBitmap error: ${e.message}")
            null
        }
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (event.action == KeyEvent.ACTION_DOWN) {
            when (event.keyCode) {
                KeyEvent.KEYCODE_VOLUME_UP -> {
                    if (ttsManager.isOcrActive || ttsManager.isSpeaking) {
                        stopSpeech()
                    } else {
                        triggerOcrReading()
                    }
                    return true
                }
                KeyEvent.KEYCODE_VOLUME_DOWN -> {
                    if (ttsManager.isSpeaking) {
                        stopSpeech()
                    } else {
                        triggerSceneSummary()
                    }
                    return true
                }
            }
        }
        return super.dispatchKeyEvent(event)
    }

    private fun triggerOcrReading() {
        val now = SystemClock.elapsedRealtime()

        if (now - lastOcrRequestTime < 1500L) {
            return
        }

        if (!isOcrProcessing.compareAndSet(false, true)) {
            return
        }

        if (ttsManager.isOcrActive) {
            isOcrProcessing.set(false)
            return
        }

        lastOcrRequestTime = now
        val bitmap = latestBitmap
        if (bitmap == null) {
            isOcrProcessing.set(false)
            ttsManager.speak("Video source initializing. Please wait.", priority = 70, severity = "INFO")
            return
        }

        ttsManager.speak("Reading text...", priority = 80, severity = "INFO")

        lifecycleScope.launch {
            try {
                val resultText = ocrManager.extractText(bitmap)
                withContext(Dispatchers.Main) {
                    ttsManager.startOcrReading(resultText)
                    Toast.makeText(this@MainActivity, resultText, Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e(tag, "OCR processing error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    ttsManager.speak("Could not read text.", priority = 70, severity = "INFO")
                }
            } finally {
                delay(600L)
                isOcrProcessing.set(false)
            }
        }
    }

    private fun triggerSceneSummary() {
        val summary = decisionEngine.getFullSceneSummary(currentDetections)
        ttsManager.speak(summary, priority = 80, severity = "INFO")
        Toast.makeText(this, summary, Toast.LENGTH_LONG).show()
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == cameraPermissionCode) {
            if (allPermissionsGranted()) {
                if (currentSource == VideoInputSource.PHONE_CAMERA) {
                    startCamera()
                }
            } else {
                Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
}