package com.blindcap.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.GestureDetector
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
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
import com.blindcap.app.ai.ColorDetector
import com.blindcap.app.ai.CurrencyDetector
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.FaceRecognitionManager
import com.blindcap.app.ai.RecognizedFace
import com.blindcap.app.ai.TfliteYoloDetector
import com.blindcap.app.barcode.BarcodeScannerManager
import com.blindcap.app.databinding.ActivityMainBinding
import com.blindcap.app.engine.DecisionEngine
import com.blindcap.app.engine.DepthEstimator
import com.blindcap.app.engine.HazardEvent
import com.blindcap.app.engine.RecognitionState
import com.blindcap.app.haptics.HapticManager
import com.blindcap.app.net.MjpegStreamReader
import com.blindcap.app.ocr.OcrManager
import com.blindcap.app.safety.SosManager
import com.blindcap.app.speech.TtsManager
import com.blindcap.app.speech.VoiceCommand
import com.blindcap.app.speech.VoiceCommandManager
import com.blindcap.app.speech.VoiceCommandType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
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

enum class AppDetectionMode {
    COMBINED,
    OBJECT_DETECTION_ONLY,
    OCR_ONLY,
    CURRENCY_ONLY,
    COLOR_ONLY,
    BARCODE_ONLY
}

class MainActivity : AppCompatActivity() {

    private val tag = "OculusMainActivity"
    private val appPermissionsCode = 1001
    private val prefsName = "OculusPrefs"

    // Preference keys
    private val keySeparateModes = "pref_separate_modes"
    private val keyEnableCurrency = "pref_enable_currency_mode"
    private val keyEnableColor = "pref_enable_color_mode"
    private val keyEnableBarcode = "pref_enable_barcode_mode"
    private val keyVoiceCommands = "pref_voice_commands"
    private val keyShowDebugHud = "pref_show_debug_hud"
    private val keySavedMode = "pref_saved_mode"

    private lateinit var binding: ActivityMainBinding
    private lateinit var prefs: SharedPreferences

    // Single source of truth for active application mode
    private var currentMode = AppDetectionMode.COMBINED

    private var currentSource = VideoInputSource.PHONE_CAMERA
    private var cameraProvider: ProcessCameraProvider? = null

    // Dedicated asynchronous pipelines
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var faceExecutor: ExecutorService

    // AI & Sensor Engines
    private lateinit var detector: TfliteYoloDetector
    private lateinit var depthEstimator: DepthEstimator
    private lateinit var faceRecognitionManager: FaceRecognitionManager
    private lateinit var decisionEngine: DecisionEngine
    private lateinit var ocrManager: OcrManager
    private lateinit var ttsManager: TtsManager
    private lateinit var mjpegStreamReader: MjpegStreamReader
    private lateinit var hapticManager: HapticManager
    private lateinit var currencyDetector: CurrencyDetector
    private lateinit var colorDetector: ColorDetector
    private lateinit var barcodeScannerManager: BarcodeScannerManager
    private lateinit var sosManager: SosManager
    private lateinit var voiceCommandManager: VoiceCommandManager

    // Decoupled Frame Transfer
    private val latestFrameRef = AtomicReference<Bitmap?>()
    private val isAiRunning = AtomicBoolean(false)
    private var aiWorkerThread: Thread? = null

    // Asynchronous Face Recognition State
    private val isFaceScanning = AtomicBoolean(false)
    private val activeRecognizedFaces = AtomicReference<List<RecognizedFace>>(emptyList())
    private var lastFaceScanTime = 0L

    // Tool Processing States & Debounce Guards
    private val isOcrProcessing = AtomicBoolean(false)
    private var lastOcrRequestTime = 0L
    private val isToolBusy = AtomicBoolean(false)

    // Volume button tracking
    private val mainHandler = Handler(Looper.getMainLooper())
    private var isVolumeUpLongPressTriggered = false
    private var volumeUpLongPressRunnable: Runnable? = null
    private var volumeDownClickCount = 0
    private var lastVolumeDownClickTime = 0L

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

        hapticManager = HapticManager(this)
        depthEstimator = DepthEstimator()
        detector = TfliteYoloDetector(this, depthEstimator)
        faceRecognitionManager = FaceRecognitionManager(this)
        decisionEngine = DecisionEngine()
        ocrManager = OcrManager()
        currencyDetector = CurrencyDetector()
        colorDetector = ColorDetector()
        barcodeScannerManager = BarcodeScannerManager()

        ttsManager = TtsManager(this) {
            ttsManager.speak("Oculus AI ready.", priority = 50, severity = "INFO")
        }

        sosManager = SosManager(this, ttsManager, hapticManager)

        voiceCommandManager = VoiceCommandManager(
            context = this,
            onCommandReceived = { command ->
                handleVoiceCommand(command)
            },
            onStatusChanged = { isListening, msg ->
                runOnUiThread {
                    if (isListening) {
                        binding.txtModeStatus.text = "🎤 $msg"
                    } else {
                        updateCarouselUi()
                    }
                }
            }
        )

        // Apply saved debug HUD preference
        binding.overlayView.showDebugHud = prefs.getBoolean(keyShowDebugHud, false)

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
                    binding.txtSourceBadge.text = "ESP32: $status"
                }
            }
        )

        setupGestures()
        setupTopBarControls()
        setupBottomControls()
        startAiWorkerLoop()

        val isSeparateEnabled = prefs.getBoolean(keySeparateModes, false)
        if (isSeparateEnabled) {
            val savedMode = prefs.getString(keySavedMode, "OBJECT")
            val targetMode = when (savedMode) {
                "OCR" -> AppDetectionMode.OCR_ONLY
                "CURRENCY" -> AppDetectionMode.CURRENCY_ONLY
                "COLOR" -> AppDetectionMode.COLOR_ONLY
                "BARCODE" -> AppDetectionMode.BARCODE_ONLY
                else -> AppDetectionMode.OBJECT_DETECTION_ONLY
            }
            setDetectionMode(targetMode, announce = false)
            showStartupModeDialog()
        } else {
            setDetectionMode(AppDetectionMode.COMBINED, announce = false)
        }

        requestRequiredPermissions()
    }

    override fun onResume() {
        super.onResume()
        // Refresh preferences that might have changed in SettingsActivity
        binding.overlayView.showDebugHud = prefs.getBoolean(keyShowDebugHud, false)
        val isSeparate = prefs.getBoolean(keySeparateModes, false)
        if (!isSeparate && currentMode != AppDetectionMode.COMBINED) {
            setDetectionMode(AppDetectionMode.COMBINED, announce = false)
        } else if (isSeparate && currentMode == AppDetectionMode.COMBINED) {
            setDetectionMode(AppDetectionMode.OBJECT_DETECTION_ONLY, announce = false)
        }
        updateCarouselUi()
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.SEND_SMS,
            Manifest.permission.RECORD_AUDIO
        )
        val needed = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), appPermissionsCode)
        } else {
            startCamera()
        }
    }

    private fun startAiWorkerLoop() {
        isAiRunning.set(true)
        aiWorkerThread = Thread({
            while (isAiRunning.get()) {
                val frame = latestFrameRef.getAndSet(null)
                if (frame != null) {
                    latestBitmap = frame

                    // When specialized on-demand mode is active, YOLO and Face inference are completely suspended
                    if (currentMode != AppDetectionMode.COMBINED && currentMode != AppDetectionMode.OBJECT_DETECTION_ONLY) {
                        if (currentDetections.isNotEmpty()) {
                            currentDetections = emptyList()
                            runOnUiThread {
                                binding.overlayView.updateResults(emptyList(), HazardEvent(), emptyList())
                            }
                        }
                        try {
                            Thread.sleep(25)
                        } catch (_: InterruptedException) {
                            break
                        }
                    } else {
                        processAiFrame(frame)
                        measureAiFps()
                    }
                } else {
                    try {
                        Thread.sleep(2)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }
        }, "Oculus-AI-Worker").apply {
            priority = Thread.NORM_PRIORITY + 2
            isDaemon = true
            start()
        }
    }

    private fun processAiFrame(bitmap: Bitmap) {
        val detections = detector.detect(bitmap)
        currentDetections = detections

        val now = SystemClock.elapsedRealtime()
        val hasPersons = detections.any { it.className.equals("person", ignoreCase = true) }

        if (!hasPersons) {
            activeRecognizedFaces.set(emptyList())
        }

        val hasUnannouncedPerson = hasPersons && (
            decisionEngine.trackedObjects.isEmpty() ||
            decisionEngine.trackedObjects.values.any {
                (it.className.equals("person", ignoreCase = true) || it.faceName != null) &&
                it.framesMissing == 0 &&
                (it.recognitionState != RecognitionState.ANNOUNCED || now - it.lastFaceSeenTimeMs > 3000L)
            }
        )

        if ((now - lastFaceScanTime >= 220L) && hasUnannouncedPerson && !isFaceScanning.get()) {
            if (isFaceScanning.compareAndSet(false, true)) {
                lastFaceScanTime = now
                faceExecutor.execute {
                    try {
                        val faces = faceRecognitionManager.detectAndRecognizeFaces(bitmap)
                        activeRecognizedFaces.set(faces)
                    } catch (e: Exception) {
                        Log.e(tag, "Face scan error: ${e.message}", e)
                    } finally {
                        isFaceScanning.set(false)
                    }
                }
            }
        }

        val faces = if (hasPersons && (now - lastFaceScanTime <= 400L)) {
            activeRecognizedFaces.get()
        } else {
            emptyList()
        }

        val event = decisionEngine.evaluate(detections, faces)

        // Haptic directional feedback
        if (event.hazardDetected) {
            when (event.severity) {
                "CRITICAL" -> hapticManager.vibrateDangerStop()
                "WARNING" -> hapticManager.vibrateObstacleLeft()
                else -> {}
            }
        }

        // Dispatch speech if warranted
        if (event.warningText != null) {
            ttsManager.speak(
                text = event.warningText,
                priority = event.speakPriority,
                severity = event.severity
            )
        }

        // Update UI Overlay
        val sourceLabel = if (currentSource == VideoInputSource.PHONE_CAMERA) "Phone" else "ESP32"
        val ttsStatus = if (ttsManager.isSpeaking) "SPEAKING" else "SILENT"
        val faceDiag = faceRecognitionManager.lastDiagnostic
        val faceScanTimeMs = faceRecognitionManager.lastFaceScanMs
        val registeredNames = faceRecognitionManager.getRegisteredContacts().map { it.name.lowercase() }.toSet()
        val liveObservations = event.faceObservations

        runOnUiThread {
            binding.overlayView.cameraFps = currentCameraFps
            binding.overlayView.aiFps = currentAiFps
            binding.overlayView.timings = detector.lastTimings
            binding.overlayView.activeDevice = "${detector.activeDevice} [$sourceLabel]"
            binding.overlayView.ttsStatus = ttsStatus
            binding.overlayView.errorMessage = detector.lastError
            binding.overlayView.faceDiagnostic = faceDiag
            binding.overlayView.faceScanMs = faceScanTimeMs
            binding.overlayView.registeredContactNames = registeredNames
            binding.overlayView.updateResults(detections, event, liveObservations)
        }
    }

    /**
     * Mode Controller (Single Source of Truth)
     * Switches processing consumers cleanly WITHOUT restarting the camera or stream.
     */
    private fun setDetectionMode(newMode: AppDetectionMode, announce: Boolean = true) {
        currentMode = newMode
        decisionEngine.reset()
        hapticManager.vibrateClick()

        when (newMode) {
            AppDetectionMode.OCR_ONLY -> {
                prefs.edit().putString(keySavedMode, "OCR").apply()
                currentDetections = emptyList()
                activeRecognizedFaces.set(emptyList())
                runOnUiThread {
                    binding.overlayView.updateResults(emptyList(), HazardEvent(), emptyList())
                }
                if (announce) {
                    ttsManager.speak("OCR reading mode. Tap shutter or volume up to read text.", priority = 60, severity = "INFO")
                }
            }
            AppDetectionMode.CURRENCY_ONLY -> {
                prefs.edit().putString(keySavedMode, "CURRENCY").apply()
                currentDetections = emptyList()
                activeRecognizedFaces.set(emptyList())
                runOnUiThread {
                    binding.overlayView.updateResults(emptyList(), HazardEvent(), emptyList())
                }
                if (announce) {
                    ttsManager.speak("Banknote reader mode. Tap shutter or volume up to identify Indian Rupee note.", priority = 60, severity = "INFO")
                }
            }
            AppDetectionMode.COLOR_ONLY -> {
                prefs.edit().putString(keySavedMode, "COLOR").apply()
                currentDetections = emptyList()
                activeRecognizedFaces.set(emptyList())
                runOnUiThread {
                    binding.overlayView.updateResults(emptyList(), HazardEvent(), emptyList())
                }
                if (announce) {
                    ttsManager.speak("Color detector mode. Center object inside reticle.", priority = 60, severity = "INFO")
                }
            }
            AppDetectionMode.BARCODE_ONLY -> {
                prefs.edit().putString(keySavedMode, "BARCODE").apply()
                currentDetections = emptyList()
                activeRecognizedFaces.set(emptyList())
                runOnUiThread {
                    binding.overlayView.updateResults(emptyList(), HazardEvent(), emptyList())
                }
                if (announce) {
                    ttsManager.speak("Barcode scanner mode. Align barcode or QR code.", priority = 60, severity = "INFO")
                }
            }
            AppDetectionMode.OBJECT_DETECTION_ONLY -> {
                prefs.edit().putString(keySavedMode, "OBJECT").apply()
                if (announce) {
                    ttsManager.speak("Object detection mode active.", priority = 60, severity = "INFO")
                }
            }
            AppDetectionMode.COMBINED -> {
                if (announce) {
                    ttsManager.speak("Combined mode active.", priority = 60, severity = "INFO")
                }
            }
        }

        runOnUiThread {
            updateCarouselUi()
        }
    }

    private fun updateCarouselUi() {
        val isSeparate = prefs.getBoolean(keySeparateModes, false)
        val enableCurrency = prefs.getBoolean(keyEnableCurrency, true)
        val enableColor = prefs.getBoolean(keyEnableColor, true)
        val enableBarcode = prefs.getBoolean(keyEnableBarcode, true)

        if (!isSeparate) {
            binding.btnPillCombined.visibility = View.VISIBLE
            binding.btnPillObjectNav.visibility = View.GONE
            binding.btnPillOcr.visibility = View.GONE
            binding.btnPillCurrency.visibility = View.GONE
            binding.btnPillColor.visibility = View.GONE
            binding.btnPillBarcode.visibility = View.GONE

            binding.btnPillCombined.setBackgroundResource(R.drawable.mode_pill_selected)
            binding.btnPillCombined.setTextColor(0xFF1F2421.toInt())
            binding.txtModeStatus.text = "COMBINED MODE"
        } else {
            binding.btnPillCombined.visibility = View.GONE
            binding.btnPillObjectNav.visibility = View.VISIBLE
            binding.btnPillOcr.visibility = View.VISIBLE

            binding.btnPillCurrency.visibility = if (enableCurrency) View.VISIBLE else View.GONE
            binding.btnPillColor.visibility = if (enableColor) View.VISIBLE else View.GONE
            binding.btnPillBarcode.visibility = if (enableBarcode) View.VISIBLE else View.GONE

            // Reset all pills
            val unselBg = R.drawable.mode_pill_unselected
            val unselCol = 0xFFFFFFFF.toInt()
            val selBg = R.drawable.mode_pill_selected
            val selCol = 0xFF1F2421.toInt()

            binding.btnPillObjectNav.setBackgroundResource(unselBg)
            binding.btnPillObjectNav.setTextColor(unselCol)
            binding.btnPillOcr.setBackgroundResource(unselBg)
            binding.btnPillOcr.setTextColor(unselCol)
            binding.btnPillCurrency.setBackgroundResource(unselBg)
            binding.btnPillCurrency.setTextColor(unselCol)
            binding.btnPillColor.setBackgroundResource(unselBg)
            binding.btnPillColor.setTextColor(unselCol)
            binding.btnPillBarcode.setBackgroundResource(unselBg)
            binding.btnPillBarcode.setTextColor(unselCol)

            when (currentMode) {
                AppDetectionMode.OBJECT_DETECTION_ONLY -> {
                    binding.btnPillObjectNav.setBackgroundResource(selBg)
                    binding.btnPillObjectNav.setTextColor(selCol)
                    binding.txtModeStatus.text = "OBJECT DETECTION"
                }
                AppDetectionMode.OCR_ONLY -> {
                    binding.btnPillOcr.setBackgroundResource(selBg)
                    binding.btnPillOcr.setTextColor(selCol)
                    binding.txtModeStatus.text = "OCR TEXT READER"
                }
                AppDetectionMode.CURRENCY_ONLY -> {
                    binding.btnPillCurrency.setBackgroundResource(selBg)
                    binding.btnPillCurrency.setTextColor(selCol)
                    binding.txtModeStatus.text = "BANKNOTE READER"
                }
                AppDetectionMode.COLOR_ONLY -> {
                    binding.btnPillColor.setBackgroundResource(selBg)
                    binding.btnPillColor.setTextColor(selCol)
                    binding.txtModeStatus.text = "COLOR DETECTOR"
                }
                AppDetectionMode.BARCODE_ONLY -> {
                    binding.btnPillBarcode.setBackgroundResource(selBg)
                    binding.btnPillBarcode.setTextColor(selCol)
                    binding.txtModeStatus.text = "BARCODE SCANNER"
                }
                AppDetectionMode.COMBINED -> {
                    binding.txtModeStatus.text = "COMBINED MODE"
                }
            }
        }
    }

    private fun showStartupModeDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_mode_select, null)
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .setCancelable(true)
            .create()

        val cardObjectNav = dialogView.findViewById<View>(R.id.cardOptionObjectNav)
        val cardOcrReader = dialogView.findViewById<View>(R.id.cardOptionOcrReader)

        cardObjectNav.setOnClickListener {
            setDetectionMode(AppDetectionMode.OBJECT_DETECTION_ONLY, announce = true)
            dialog.dismiss()
        }

        cardOcrReader.setOnClickListener {
            setDetectionMode(AppDetectionMode.OCR_ONLY, announce = true)
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun showToolsDialog() {
        hapticManager.vibrateClick()
        val dialogView = layoutInflater.inflate(R.layout.dialog_tools, null)
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .setCancelable(true)
            .create()

        val cardCurrency = dialogView.findViewById<View>(R.id.cardToolCurrency)
        val cardColor = dialogView.findViewById<View>(R.id.cardToolColor)
        val cardBarcode = dialogView.findViewById<View>(R.id.cardToolBarcode)
        val btnCancel = dialogView.findViewById<View>(R.id.btnCancelTools)

        cardCurrency.setOnClickListener {
            dialog.dismiss()
            triggerCurrencyScan()
        }

        cardColor.setOnClickListener {
            dialog.dismiss()
            triggerColorDetection()
        }

        cardBarcode.setOnClickListener {
            dialog.dismiss()
            triggerBarcodeScan()
        }

        btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun triggerCurrencyScan() {
        val bitmap = latestBitmap
        if (bitmap == null) {
            ttsManager.speak("Camera initializing. Please hold banknote in front of camera.", priority = 70, severity = "INFO")
            return
        }
        hapticManager.vibrateClick()
        ttsManager.speak("Scanning banknote...", priority = 75, severity = "INFO")

        lifecycleScope.launch(Dispatchers.Default) {
            val result = currencyDetector.detectCurrency(bitmap)
            withContext(Dispatchers.Main) {
                hapticManager.vibrateClick()
                ttsManager.speak(result.spokenText, priority = 85, severity = "INFO")
                Toast.makeText(this@MainActivity, result.spokenText, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun triggerColorDetection() {
        val bitmap = latestBitmap
        if (bitmap == null) {
            ttsManager.speak("Camera initializing. Point camera at colored surface.", priority = 70, severity = "INFO")
            return
        }
        hapticManager.vibrateClick()

        lifecycleScope.launch(Dispatchers.Default) {
            val result = colorDetector.detectColor(bitmap)
            withContext(Dispatchers.Main) {
                hapticManager.vibrateClick()
                ttsManager.speak(result.spokenDescription, priority = 85, severity = "INFO")
                Toast.makeText(this@MainActivity, result.spokenDescription, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun triggerBarcodeScan() {
        val bitmap = latestBitmap
        if (bitmap == null) {
            ttsManager.speak("Camera initializing. Align barcode in front of camera.", priority = 70, severity = "INFO")
            return
        }
        hapticManager.vibrateClick()
        ttsManager.speak("Scanning barcode or QR code...", priority = 75, severity = "INFO")

        lifecycleScope.launch(Dispatchers.Default) {
            val result = barcodeScannerManager.scanBitmap(bitmap)
            withContext(Dispatchers.Main) {
                if (result != null) {
                    hapticManager.vibrateClick()
                    ttsManager.speak(result.spokenText, priority = 85, severity = "INFO")
                    Toast.makeText(this@MainActivity, result.spokenText, Toast.LENGTH_LONG).show()
                } else {
                    ttsManager.speak("No barcode or QR code detected. Try holding closer.", priority = 75, severity = "INFO")
                    Toast.makeText(this@MainActivity, "No barcode detected", Toast.LENGTH_SHORT).show()
                }
            }
        }
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

    private fun handleVoiceCommand(command: VoiceCommand) {
        Log.i(tag, "Voice command received: type=${command.type}, text='${command.rawText}'")
        hapticManager.vibrateClick()

        when (command.type) {
            VoiceCommandType.COLOR_QUERY -> {
                triggerColorDetection()
            }
            VoiceCommandType.CURRENCY_QUERY -> {
                triggerCurrencyScan()
            }
            VoiceCommandType.SCENE_QUERY -> {
                triggerSceneSummary()
            }
            VoiceCommandType.OCR_QUERY -> {
                triggerOcrReading()
            }
            VoiceCommandType.BARCODE_QUERY -> {
                triggerBarcodeScan()
            }
            VoiceCommandType.SOS_QUERY -> {
                sosManager.triggerEmergencySos { success, msg ->
                    Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                }
            }
            VoiceCommandType.STOP_SPEECH -> {
                stopSpeech()
            }
            VoiceCommandType.UNKNOWN -> {
                ttsManager.speak("You can ask what color, what note, describe scene, or read text.", priority = 70, severity = "INFO")
            }
        }
    }

    private fun executePrimaryModeAction() {
        hapticManager.vibrateClick()
        when (currentMode) {
            AppDetectionMode.OCR_ONLY -> triggerOcrReading()
            AppDetectionMode.CURRENCY_ONLY -> triggerCurrencyScan()
            AppDetectionMode.COLOR_ONLY -> triggerColorDetection()
            AppDetectionMode.BARCODE_ONLY -> triggerBarcodeScan()
            AppDetectionMode.OBJECT_DETECTION_ONLY,
            AppDetectionMode.COMBINED -> triggerSceneSummary()
        }
    }

    private fun setupTopBarControls() {
        updateSourceUi()

        binding.btnSettings.setOnClickListener {
            hapticManager.vibrateClick()
            val intent = Intent(this, SettingsActivity::class.java)
            startActivity(intent)
        }
    }

    private fun setupBottomControls() {
        binding.btnPillCombined.setOnClickListener {
            setDetectionMode(AppDetectionMode.COMBINED)
        }

        binding.btnPillObjectNav.setOnClickListener {
            setDetectionMode(AppDetectionMode.OBJECT_DETECTION_ONLY)
        }

        binding.btnPillOcr.setOnClickListener {
            setDetectionMode(AppDetectionMode.OCR_ONLY)
        }

        binding.btnPillCurrency.setOnClickListener {
            setDetectionMode(AppDetectionMode.CURRENCY_ONLY)
        }

        binding.btnPillColor.setOnClickListener {
            setDetectionMode(AppDetectionMode.COLOR_ONLY)
        }

        binding.btnPillBarcode.setOnClickListener {
            setDetectionMode(AppDetectionMode.BARCODE_ONLY)
        }

        binding.btnShutter.setOnClickListener {
            executePrimaryModeAction()
        }

        binding.btnSceneSummary.setOnClickListener {
            hapticManager.vibrateClick()
            triggerSceneSummary()
        }

        binding.btnTools.setOnClickListener {
            showToolsDialog()
        }

        updateCarouselUi()
    }

    /**
     * CRITICAL FIX: Intercept Volume UP and Volume DOWN keys without calling super.
     * Prevents the Android system volume slider dialog from popping up on screen!
     */
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        val keyCode = event.keyCode
        val action = event.action

        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            val voiceEnabled = prefs.getBoolean(keyVoiceCommands, true)

            if (action == KeyEvent.ACTION_DOWN) {
                // If speech or OCR is currently speaking, stop speech immediately
                if (ttsManager.isOcrActive || ttsManager.isSpeaking) {
                    stopSpeech()
                    return true
                }

                if (event.repeatCount == 0) {
                    isVolumeUpLongPressTriggered = false
                    if (voiceEnabled) {
                        volumeUpLongPressRunnable = Runnable {
                            isVolumeUpLongPressTriggered = true
                            hapticManager.vibrateClick()
                            ttsManager.speak("Listening...", priority = 80, severity = "INFO")
                            voiceCommandManager.startListening()
                        }
                        mainHandler.postDelayed(volumeUpLongPressRunnable!!, 450L)
                    }
                }
            } else if (action == KeyEvent.ACTION_UP) {
                volumeUpLongPressRunnable?.let { mainHandler.removeCallbacks(it) }

                if (isVolumeUpLongPressTriggered) {
                    isVolumeUpLongPressTriggered = false
                    voiceCommandManager.stopListening()
                } else {
                    // Single short press: execute active mode primary action!
                    executePrimaryModeAction()
                }
            }
            return true
        }

        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            if (action == KeyEvent.ACTION_DOWN) {
                if (ttsManager.isSpeaking) {
                    stopSpeech()
                    return true
                }

                val now = SystemClock.elapsedRealtime()
                if (now - lastVolumeDownClickTime < 600L) {
                    volumeDownClickCount++
                } else {
                    volumeDownClickCount = 1
                }
                lastVolumeDownClickTime = now

                // 4-click Volume DOWN triggers Emergency SOS
                if (volumeDownClickCount >= 4) {
                    volumeDownClickCount = 0
                    sosManager.triggerEmergencySos { success, msg ->
                        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                    }
                    return true
                }

                triggerSceneSummary()
            }
            return true
        }

        return super.dispatchKeyEvent(event)
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

    private fun updateSourceUi() {
        if (currentSource == VideoInputSource.PHONE_CAMERA) {
            binding.txtSourceBadge.text = "📱 Phone"
            binding.viewFinder.visibility = View.VISIBLE
            binding.esp32StreamView.visibility = View.GONE
        } else {
            binding.txtSourceBadge.text = "🧢 ESP32"
            binding.viewFinder.visibility = View.GONE
            binding.esp32StreamView.visibility = View.VISIBLE
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
                if (sosManager.cancelSos()) {
                    return true
                }
                executePrimaryModeAction()
                return true
            }

            override fun onLongPress(e: MotionEvent) {
                triggerColorDetection()
            }
        })

        binding.overlayView.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }
    }

    private fun stopSpeech() {
        ttsManager.stop()
        hapticManager.vibrateClick()
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
        currencyDetector.close()
        barcodeScannerManager.close()
        voiceCommandManager.destroy()
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

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val rawBitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            if (rotationDegrees != 0) {
                val matrix = Matrix().apply {
                    postRotate(rotationDegrees.toFloat())
                }
                val rotated = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                rawBitmap.recycle()
                rotated
            } else {
                rawBitmap
            }
        } catch (e: Exception) {
            Log.e(tag, "toBitmap error: ${e.message}")
            null
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == appPermissionsCode) {
            val cameraGranted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED

            if (cameraGranted && currentSource == VideoInputSource.PHONE_CAMERA) {
                startCamera()
            }
        }
    }
}
