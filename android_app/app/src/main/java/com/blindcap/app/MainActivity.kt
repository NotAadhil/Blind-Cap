package com.blindcap.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.GestureDetector
import android.view.KeyEvent
import android.view.MotionEvent
import android.widget.Toast
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
import com.blindcap.app.ocr.OcrManager
import com.blindcap.app.speech.TtsManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private val tag = "BlindCapMainActivity"
    private val cameraPermissionCode = 1001

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var depthEstimator: DepthEstimator
    private lateinit var detector: TfliteYoloDetector
    private lateinit var decisionEngine: DecisionEngine
    private lateinit var ttsManager: TtsManager
    private lateinit var ocrManager: OcrManager

    private val isAnalyzing = AtomicBoolean(false)
    private var lastFrameTime = 0L
    private var latestBitmap: Bitmap? = null
    private var currentDetections: List<Detection> = emptyList()

    private lateinit var gestureDetector: GestureDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        depthEstimator = DepthEstimator()
        detector = TfliteYoloDetector(this, depthEstimator)
        decisionEngine = DecisionEngine()
        ocrManager = OcrManager()

        ttsManager = TtsManager(this) {
            ttsManager.speak("Blind Cap ready.", priority = 50, severity = "INFO")
        }

        setupGestures()

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

    private fun setupGestures() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                ttsManager.repeatLast()
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                ttsManager.isMuted = !ttsManager.isMuted
                val status = if (ttsManager.isMuted) "Speech muted." else "Speech unmuted."
                Toast.makeText(this@MainActivity, status, Toast.LENGTH_SHORT).show()
                if (!ttsManager.isMuted) {
                    ttsManager.speak("Speech active.", priority = 60, severity = "INFO")
                }
                return true
            }
        })

        binding.root.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detector.close()
        ttsManager.shutdown()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processImageProxy(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (exc: Exception) {
                Log.e(tag, "Camera binding failed: ${exc.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun processImageProxy(imageProxy: ImageProxy) {
        if (!isAnalyzing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                latestBitmap = bitmap

                // 1. Run ML Kit Object Detection (synchronous via latch, always works on Pixel)
                val detections = detector.detect(bitmap)
                currentDetections = detections

                // 2. Evaluate Decision Engine
                val event = decisionEngine.evaluate(detections)

                // 3. Dispatch Speech if warranted
                if (event.warningText != null) {
                    ttsManager.speak(
                        text = event.warningText,
                        priority = event.speakPriority,
                        severity = event.severity
                    )
                }

                // 4. Update UI Overlay
                val now = SystemClock.elapsedRealtime()
                val dt = (now - lastFrameTime).coerceAtLeast(1)
                lastFrameTime = now
                val cameraFps = 1000f / dt
                val inferenceFps = if (detector.lastInferenceMs > 0) 1000f / detector.lastInferenceMs else 0f

                runOnUiThread {
                    binding.overlayView.cameraFps = cameraFps
                    binding.overlayView.inferenceFps = inferenceFps
                    binding.overlayView.inferenceMs = detector.lastInferenceMs
                    binding.overlayView.activeDevice = detector.activeDevice
                    binding.overlayView.errorMessage = detector.lastError
                    binding.overlayView.updateResults(detections, event)
                }
            }
        } catch (e: Exception) {
            Log.e(tag, "Error in image processing: ${e.message}")
        } finally {
            imageProxy.close()
            isAnalyzing.set(false)
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
            Log.e(tag, "toBitmap failed: ${e.message}, trying YUV fallback")
            try {
                yuvToBitmapFallback(imageProxy)
            } catch (e2: Exception) {
                Log.e(tag, "YUV fallback failed: ${e2.message}")
                null
            }
        }
    }

    private fun yuvToBitmapFallback(imageProxy: ImageProxy): Bitmap? {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(
            nv21,
            android.graphics.ImageFormat.NV21,
            imageProxy.width,
            imageProxy.height,
            null
        )
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
        val imageBytes = out.toByteArray()
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        return if (rotationDegrees != 0 && bitmap != null) {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                triggerOcrReading()
                return true
            }
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                triggerSceneSummary()
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun triggerOcrReading() {
        val bitmap = latestBitmap ?: return
        ttsManager.speak("Reading text...", priority = 70, severity = "INFO")
        lifecycleScope.launch {
            val resultText = ocrManager.extractText(bitmap)
            withContext(Dispatchers.Main) {
                ttsManager.speak(resultText, priority = 70, severity = "INFO")
            }
        }
    }

    private fun triggerSceneSummary() {
        val summary = decisionEngine.getFullSceneSummary(currentDetections)
        ttsManager.speak(summary, priority = 70, severity = "INFO")
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
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
}