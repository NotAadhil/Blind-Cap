package com.blindcap.app

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.GestureDetector
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.widget.EditText
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
import com.blindcap.app.ai.TfliteYoloDetector
import com.blindcap.app.databinding.ActivityMainBinding
import com.blindcap.app.engine.DecisionEngine
import com.blindcap.app.engine.DepthEstimator
import com.blindcap.app.net.MjpegStreamReader
import com.blindcap.app.ocr.OcrManager
import com.blindcap.app.speech.TtsManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
    private lateinit var prefs: SharedPreferences

    private lateinit var depthEstimator: DepthEstimator
    private lateinit var detector: TfliteYoloDetector
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

    // FPS Measurement Instrumentation
    private var cameraFrameCount = 0
    private var lastCameraFpsTime = SystemClock.elapsedRealtime()
    private var currentCameraFps = 0f

    private var aiFrameCount = 0
    private var lastAiFpsTime = SystemClock.elapsedRealtime()
    private var currentAiFps = 0f

    private var latestBitmap: Bitmap? = null
    private var currentDetections: List<Detection> = emptyList()

    private lateinit var gestureDetector: GestureDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        cameraExecutor = Executors.newSingleThreadExecutor()

        depthEstimator = DepthEstimator()
        detector = TfliteYoloDetector(this, depthEstimator)
        decisionEngine = DecisionEngine()
        ocrManager = OcrManager()

        mjpegStreamReader = MjpegStreamReader(
            onFrameReceived = { bitmap ->
                if (currentSource == VideoInputSource.ESP32_CAM) {
                    measureCameraFps()
                    runOnUiThread {
                        binding.esp32StreamView.setImageBitmap(bitmap)
                    }
                    latestFrameRef.set(bitmap)
                }
            },
            onStatusChanged = { status ->
                runOnUiThread {
                    binding.txtStreamStatus.text = status
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
                        Thread.sleep(8) // Short sleep to yield CPU if no new frame
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

        // 1. Hardware Inference & Pre/Post Processing
        val detections = detector.detect(bitmap)
        currentDetections = detections

        // 2. Decision Engine with Scale-Invariant Tracking & Zero-Repeat State Machine
        val event = decisionEngine.evaluate(detections)

        // 3. Dispatch Speech if warranted
        if (event.warningText != null) {
            ttsManager.speak(
                text = event.warningText,
                priority = event.speakPriority,
                severity = event.severity
            )
        }

        // 4. Update UI Overlay with Performance Breakdown
        val sourceLabel = if (currentSource == VideoInputSource.PHONE_CAMERA) "Phone" else "ESP32"
        val ttsStatus = if (ttsManager.isSpeaking) "SPEAKING" else "SILENT"

        runOnUiThread {
            binding.overlayView.cameraFps = currentCameraFps
            binding.overlayView.aiFps = currentAiFps
            binding.overlayView.timings = detector.lastTimings
            binding.overlayView.activeDevice = "${detector.activeDevice} [$sourceLabel]"
            binding.overlayView.ttsStatus = ttsStatus
            binding.overlayView.errorMessage = detector.lastError
            binding.overlayView.updateResults(detections, event)
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
    }

    private fun updateSourceUi() {
        if (currentSource == VideoInputSource.PHONE_CAMERA) {
            binding.btnSourceToggle.text = "Source: Phone Cam"
            binding.btnSourceToggle.setBackgroundColor(0xFF333333.toInt())
            binding.viewFinder.visibility = View.VISIBLE
            binding.esp32StreamView.visibility = View.GONE
            binding.txtStreamStatus.text = ""
        } else {
            binding.btnSourceToggle.text = "Source: ESP32-CAM"
            binding.btnSourceToggle.setBackgroundColor(0xFF994400.toInt())
            binding.viewFinder.visibility = View.GONE
            binding.esp32StreamView.visibility = View.VISIBLE
        }
    }

    private fun switchToEsp32Cam() {
        currentSource = VideoInputSource.ESP32_CAM
        updateSourceUi()

        // Unbind phone camera to conserve battery
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

        // Stop ESP32 stream
        mjpegStreamReader.stop()
        decisionEngine.reset()

        // Rebind Phone Camera
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
            .setMessage("Enter the MJPEG stream URL of your ESP32-CAM (e.g. http://192.168.4.1:81/stream or http://192.168.1.50/stream):")
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

    private fun setupGestures() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                ttsManager.repeatLast()
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                ttsManager.isMuted = !ttsManager.isMuted
                val status = if (ttsManager.isMuted) "Speech muted." else "Speech active."
                Toast.makeText(this@MainActivity, status, Toast.LENGTH_SHORT).show()
                if (!ttsManager.isMuted) {
                    ttsManager.speak("Speech active.", priority = 60, severity = "INFO")
                }
                return true
            }

            override fun onLongPress(e: MotionEvent) {
                triggerOcrReading()
            }
        })

        binding.root.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }
    }

    private fun setupActionButtons() {
        binding.btnOcr.setOnClickListener {
            triggerOcrReading()
        }
        binding.btnScene.setOnClickListener {
            triggerSceneSummary()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isAiRunning.set(false)
        aiWorkerThread?.interrupt()
        mjpegStreamReader.stop()
        cameraExecutor.shutdown()
        detector.close()
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

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
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

        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                // Non-blocking handoff: set latest frame atomically for the AI worker thread
                latestFrameRef.set(bitmap)
            }
        } catch (e: Exception) {
            Log.e(tag, "Error extracting image frame: ${e.message}")
        } finally {
            // Immediately close imageProxy so CameraX buffer pool is NEVER starved!
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            if (rotationDegrees != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotationDegrees.toFloat())
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
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
                    triggerOcrReading()
                    return true
                }
                KeyEvent.KEYCODE_VOLUME_DOWN -> {
                    triggerSceneSummary()
                    return true
                }
            }
        }
        return super.dispatchKeyEvent(event)
    }

    private fun triggerOcrReading() {
        val bitmap = latestBitmap
        if (bitmap == null) {
            ttsManager.speak("Video source initializing. Please wait.", priority = 70, severity = "INFO")
            return
        }
        ttsManager.speak("Reading text...", priority = 75, severity = "INFO")
        lifecycleScope.launch {
            val resultText = ocrManager.extractText(bitmap)
            withContext(Dispatchers.Main) {
                ttsManager.speak(resultText, priority = 75, severity = "INFO")
                Toast.makeText(this@MainActivity, resultText, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun triggerSceneSummary() {
        val summary = decisionEngine.getFullSceneSummary(currentDetections)
        ttsManager.speak(summary, priority = 75, severity = "INFO")
        Toast.makeText(this@MainActivity, summary, Toast.LENGTH_LONG).show()
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