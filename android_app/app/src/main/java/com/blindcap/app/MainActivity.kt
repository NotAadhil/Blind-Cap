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
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
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
import com.blindcap.app.audio.SoundEffectManager
import com.blindcap.app.ai.ColorDetector
import com.blindcap.app.ai.CurrencyDetector
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.FaceContact
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
import com.blindcap.app.speech.TtsMode
import com.blindcap.app.speech.VoiceCommand
import com.blindcap.app.speech.VoiceCommandManager
import com.blindcap.app.speech.VoiceCommandType
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
    private val keyVideoSource = "pref_video_source"
    private val keyStreamUrl = "esp32_stream_url"
    private val defaultStreamUrl = "http://192.168.4.1:81/stream"

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

    // Active Face Enrollment Dialog reference for live status updates
    private var activeFaceDialog: AlertDialog? = null
    private var txtFaceAlignmentStatusRef: TextView? = null
    private var viewFaceStatusDotRef: View? = null

    // Tool Processing States & Debounce Guards
    private val isOcrProcessing = AtomicBoolean(false)
    private val isCurrencyScanning = AtomicBoolean(false)
    private val isColorScanning = AtomicBoolean(false)
    private val isBarcodeScanning = AtomicBoolean(false)
    private var lastOcrRequestTime = 0L

    // Button Debounce Timestamps
    private var lastVolumeUpTapTime = 0L
    private var lastVolumeDownTapTime = 0L
    private var lastShutterTapTime = 0L
    private var lastSceneSummaryTapTime = 0L
    private var lastModeSwitchTapTime = 0L

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
            ttsManager.speak("Oculus AI ready.", priority = 50, severity = "INFO", mode = TtsMode.SYSTEM)
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
                        binding.voiceAssistantOverlay.visibility = View.VISIBLE
                        binding.txtVoiceStatus.text = msg
                        binding.txtModeStatus.text = "$msg"
                    } else {
                        if (binding.layoutVoiceResult.visibility != View.VISIBLE) {
                            binding.voiceAssistantOverlay.visibility = View.GONE
                            updateCarouselUi()
                        }
                    }
                }
            }
        )

        binding.btnCancelVoice.setOnClickListener {
            SoundEffectManager.playVoiceCloseChime()
            voiceCommandManager.cancel()
            binding.voiceAssistantOverlay.visibility = View.GONE
            binding.layoutVoiceResult.visibility = View.GONE
            updateCarouselUi()
        }

        binding.chipVoiceColor.setOnClickListener {
            hapticManager.vibrateClick()
            handleVoiceCommand(VoiceCommand(VoiceCommandType.COLOR_QUERY, "what color is this"))
        }
        binding.chipVoiceCurrency.setOnClickListener {
            hapticManager.vibrateClick()
            handleVoiceCommand(VoiceCommand(VoiceCommandType.CURRENCY_QUERY, "what note is this"))
        }
        binding.chipVoiceScene.setOnClickListener {
            hapticManager.vibrateClick()
            handleVoiceCommand(VoiceCommand(VoiceCommandType.SCENE_QUERY, "describe scene"))
        }
        binding.chipVoiceOcr.setOnClickListener {
            hapticManager.vibrateClick()
            handleVoiceCommand(VoiceCommand(VoiceCommandType.OCR_QUERY, "read text"))
        }
        binding.chipVoiceBarcode.setOnClickListener {
            hapticManager.vibrateClick()
            handleVoiceCommand(VoiceCommand(VoiceCommandType.BARCODE_QUERY, "scan barcode"))
        }

        // Apply saved debug HUD preference
        binding.overlayView.showDebugHud = prefs.getBoolean(keyShowDebugHud, false)

        mjpegStreamReader = MjpegStreamReader(
            context = this,
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
                    if (currentSource == VideoInputSource.ESP32_CAM) {
                        binding.txtSourceBadge.text = "ESP32: $status"
                    }
                }
            }
        )

        setupGestures()
        setupTopBarControls()
        setupBottomControls()
        startAiWorkerLoop()

        // Initialize video source from saved preference
        val isPhone = prefs.getBoolean(keyVideoSource, true)
        val initialSource = if (isPhone) VideoInputSource.PHONE_CAMERA else VideoInputSource.ESP32_CAM
        setVideoInputSource(initialSource, announce = false)

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
        handleIncomingIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIncomingIntent(intent)
    }

    private fun handleIncomingIntent(intent: Intent?) {
        if (intent?.getBooleanExtra("open_face_enroll", false) == true) {
            mainHandler.postDelayed({
                showFaceEnrollDialog()
            }, 300L)
        }
    }

    override fun onResume() {
        super.onResume()
        // Refresh preferences
        binding.overlayView.showDebugHud = prefs.getBoolean(keyShowDebugHud, false)
        val isSeparate = prefs.getBoolean(keySeparateModes, false)
        if (!isSeparate && currentMode != AppDetectionMode.COMBINED) {
            setDetectionMode(AppDetectionMode.COMBINED, announce = false)
        } else if (isSeparate && currentMode == AppDetectionMode.COMBINED) {
            setDetectionMode(AppDetectionMode.OBJECT_DETECTION_ONLY, announce = false)
        }

        val isPhonePref = prefs.getBoolean(keyVideoSource, true)
        val targetSource = if (isPhonePref) VideoInputSource.PHONE_CAMERA else VideoInputSource.ESP32_CAM
        if (targetSource != currentSource) {
            setVideoInputSource(targetSource, announce = true)
        } else if (currentSource == VideoInputSource.ESP32_CAM) {
            // Refresh URL in case it changed in Settings
            val url = prefs.getString(keyStreamUrl, defaultStreamUrl) ?: defaultStreamUrl
            if (url != mjpegStreamReader.currentUrl) {
                mjpegStreamReader.start(url)
            }
        }

        updateCarouselUi()
    }

    private fun setVideoInputSource(source: VideoInputSource, announce: Boolean = true) {
        currentSource = source
        hapticManager.vibrateClick()

        if (source == VideoInputSource.ESP32_CAM) {
            prefs.edit().putBoolean(keyVideoSource, false).apply()
            binding.txtSourceBadge.text = "ESP32: Connecting..."
            binding.viewFinder.visibility = View.GONE
            binding.esp32StreamView.visibility = View.VISIBLE

            // Unbind phone camera preview to conserve power
            try {
                cameraProvider?.unbindAll()
            } catch (_: Exception) {}

            val streamUrl = prefs.getString(keyStreamUrl, defaultStreamUrl) ?: defaultStreamUrl
            mjpegStreamReader.start(streamUrl)

            if (announce) {
                ttsManager.speak("Switched to ESP32 smart cap camera.", priority = 70, severity = "INFO", mode = TtsMode.SYSTEM)
                Toast.makeText(this, "ESP32 Stream: $streamUrl", Toast.LENGTH_LONG).show()
            }
        } else {
            prefs.edit().putBoolean(keyVideoSource, true).apply()
            binding.txtSourceBadge.text = "Phone Camera"
            binding.esp32StreamView.visibility = View.GONE
            binding.viewFinder.visibility = View.VISIBLE

            mjpegStreamReader.stop()
            startCamera()

            if (announce) {
                ttsManager.speak("Switched to Phone camera.", priority = 70, severity = "INFO", mode = TtsMode.SYSTEM)
                Toast.makeText(this, "Phone Camera Active", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showEsp32ConfigDialog() {
        hapticManager.vibrateClick()
        val currentUrl = prefs.getString(keyStreamUrl, defaultStreamUrl) ?: defaultStreamUrl

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 16)
        }

        val msgTv = TextView(this).apply {
            text = "Enter your ESP32-CAM IP address or stream URL. The app will auto-probe all camera ports (81, 80, /stream, /capture):"
            setTextColor(0xFFCAC4D0.toInt())
            textSize = 13f
            setPadding(0, 0, 0, 16)
        }

        val input = EditText(this).apply {
            setText(currentUrl)
            setSelection(text.length)
            hint = "192.168.4.1 or http://192.168.4.1:81/stream"
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xFF2B2930.toInt())
            setPadding(24, 20, 24, 20)
        }

        // Quick Preset Buttons: 192.168.4.1 (AP) | 192.168.1.x (Router Wi-Fi)
        val presetLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 16, 0, 8)
        }

        val btnApPreset = Button(this).apply {
            text = "AP: 192.168.4.1"
            textSize = 11f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                setMargins(0, 0, 8, 0)
            }
            setOnClickListener {
                input.setText("http://192.168.4.1:81/stream")
                input.setSelection(input.text.length)
            }
        }

        val btnHomePreset = Button(this).apply {
            text = "Home Wi-Fi Auto"
            textSize = 11f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener {
                input.setText("http://192.168.1.184:81/stream")
                input.setSelection(input.text.length)
            }
        }

        presetLayout.addView(btnApPreset)
        presetLayout.addView(btnHomePreset)

        container.addView(msgTv)
        container.addView(input)
        container.addView(presetLayout)

        AlertDialog.Builder(this)
            .setTitle("ESP32-CAM Setup & Auto-Connect")
            .setView(container)
            .setPositiveButton("Connect & Stream") { _, _ ->
                val rawInput = input.text.toString().trim()
                val normalized = MjpegStreamReader.normalizeStreamUrl(rawInput)
                prefs.edit().putString(keyStreamUrl, normalized).apply()
                hapticManager.vibrateClick()
                Toast.makeText(this, "Connecting & auto-probing: $normalized", Toast.LENGTH_LONG).show()
                ttsManager.speak("Connecting to ESP32 smart cap.", priority = 75, severity = "INFO", mode = TtsMode.SYSTEM)
                setVideoInputSource(VideoInputSource.ESP32_CAM, announce = false)
            }
            .setNeutralButton("Use Phone Camera") { _, _ ->
                setVideoInputSource(VideoInputSource.PHONE_CAMERA, announce = true)
            }
            .setNegativeButton("Cancel", null)
            .show()
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
            if (currentSource == VideoInputSource.PHONE_CAMERA) {
                startCamera()
            }
        }
    }

    private fun startAiWorkerLoop() {
        isAiRunning.set(true)
        aiWorkerThread = Thread({
            while (isAiRunning.get()) {
                val frame = latestFrameRef.getAndSet(null)
                if (frame != null) {
                    latestBitmap = frame

                    // When specialized on-demand mode is active, YOLO and Face inference are suspended
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
        if (bitmap.isRecycled) return

        // Capture current generation ID before inference
        val genId = ttsManager.getGenerationId()

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

        // Asynchronously scan for faces
        if ((now - lastFaceScanTime >= 220L) && hasUnannouncedPerson && !isFaceScanning.get()) {
            if (isFaceScanning.compareAndSet(false, true)) {
                lastFaceScanTime = now
                faceExecutor.execute {
                    try {
                        if (!bitmap.isRecycled) {
                            val faces = faceRecognitionManager.detectAndRecognizeFaces(bitmap)
                            activeRecognizedFaces.set(faces)
                            updateFaceEnrollmentDialogStatus(faces.isNotEmpty())
                        }
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

        // Dispatch speech with mode tagging and generation token validation
        if (event.warningText != null) {
            ttsManager.speak(
                text = event.warningText,
                priority = event.speakPriority,
                severity = event.severity,
                mode = TtsMode.OBJECT_DETECTION,
                generationId = genId
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

    private fun updateFaceEnrollmentDialogStatus(hasFace: Boolean) {
        if (activeFaceDialog == null) return
        runOnUiThread {
            if (hasFace) {
                txtFaceAlignmentStatusRef?.text = "Face detected! Ready to enroll."
                txtFaceAlignmentStatusRef?.setTextColor(0xFF00FFCC.toInt())
                viewFaceStatusDotRef?.setBackgroundColor(0xFF00FFCC.toInt())
            } else {
                txtFaceAlignmentStatusRef?.text = "Looking for face in camera..."
                txtFaceAlignmentStatusRef?.setTextColor(0xFFFFD54F.toInt())
                viewFaceStatusDotRef?.setBackgroundColor(0xFFFFD54F.toInt())
            }
        }
    }

    /**
     * Dedicated Face Management & Enrollment Dialog
     * Keeps camera preview live and visible in the background.
     */
    private fun showFaceEnrollDialog() {
        hapticManager.vibrateClick()
        val dialogView = layoutInflater.inflate(R.layout.dialog_face_enroll, null)
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .setCancelable(true)
            .create()

        activeFaceDialog = dialog
        dialog.setOnDismissListener {
            activeFaceDialog = null
            txtFaceAlignmentStatusRef = null
            viewFaceStatusDotRef = null
        }

        val txtCount = dialogView.findViewById<TextView>(R.id.txtFaceLiveCount)
        val txtStatus = dialogView.findViewById<TextView>(R.id.txtFaceAlignmentStatus)
        val dotStatus = dialogView.findViewById<View>(R.id.viewFaceStatusDot)
        val editName = dialogView.findViewById<EditText>(R.id.editFaceName)
        val btnCapture = dialogView.findViewById<Button>(R.id.btnCaptureFace)
        val layoutContacts = dialogView.findViewById<LinearLayout>(R.id.layoutEnrolledContactsList)
        val btnClearAll = dialogView.findViewById<Button>(R.id.btnClearAllFaces)
        val btnClose = dialogView.findViewById<Button>(R.id.btnCloseFaceDialog)

        txtFaceAlignmentStatusRef = txtStatus
        viewFaceStatusDotRef = dotStatus

        fun refreshContactsUi() {
            val contacts = faceRecognitionManager.getRegisteredContacts()
            txtCount.text = "${contacts.size} Enrolled"
            layoutContacts.removeAllViews()

            if (contacts.isEmpty()) {
                val placeholder = TextView(this).apply {
                    text = "No registered faces yet."
                    setTextColor(0xFF938F99.toInt())
                    textSize = 13f
                    gravity = android.view.Gravity.CENTER
                    setPadding(0, 40, 0, 40)
                }
                layoutContacts.addView(placeholder)
            } else {
                val sdf = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault())
                for (contact in contacts) {
                    val row = LinearLayout(this).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = android.view.Gravity.CENTER_VERTICAL
                        setPadding(16, 12, 16, 12)
                        setBackgroundColor(0xFF2B2930.toInt())
                        val params = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).apply { setMargins(0, 0, 0, 8) }
                        layoutParams = params
                    }

                    val infoLayout = LinearLayout(this).apply {
                        orientation = LinearLayout.VERTICAL
                        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    }

                    val nameTv = TextView(this).apply {
                        text = contact.name
                        setTextColor(0xFFFFFFFF.toInt())
                        textSize = 15f
                        setTypeface(null, android.graphics.Typeface.BOLD)
                    }

                    val timeTv = TextView(this).apply {
                        text = "Enrolled: ${sdf.format(Date(contact.enrolledTimestamp))}"
                        setTextColor(0xFFCAC4D0.toInt())
                        textSize = 11f
                    }

                    infoLayout.addView(nameTv)
                    infoLayout.addView(timeTv)

                    val delBtn = ImageButton(this).apply {
                        setImageResource(android.R.drawable.ic_menu_delete)
                        setBackgroundColor(0x00000000)
                        setColorFilter(0xFFFFB4AB.toInt())
                        setOnClickListener {
                            faceRecognitionManager.deleteContact(contact.id)
                            hapticManager.vibrateClick()
                            ttsManager.speak("Deleted ${contact.name}.", priority = 70, severity = "INFO", mode = TtsMode.FACE)
                            refreshContactsUi()
                        }
                    }

                    row.addView(infoLayout)
                    row.addView(delBtn)
                    layoutContacts.addView(row)
                }
            }
        }

        refreshContactsUi()

        btnCapture.setOnClickListener {
            val name = editName.text.toString().trim()
            if (name.isBlank()) {
                Toast.makeText(this, "Please enter a name first", Toast.LENGTH_SHORT).show()
                ttsManager.speak("Please enter a name first.", priority = 70, severity = "INFO", mode = TtsMode.FACE)
                return@setOnClickListener
            }

            val bitmap = latestBitmap
            if (bitmap == null || bitmap.isRecycled) {
                Toast.makeText(this, "Camera not ready yet", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            hapticManager.vibrateClick()
            btnCapture.isEnabled = false
            btnCapture.text = "Processing Face..."

            val genId = ttsManager.getGenerationId()

            lifecycleScope.launch(Dispatchers.Default) {
                val result = faceRecognitionManager.registerFaceFromBitmap(name, bitmap)
                withContext(Dispatchers.Main) {
                    btnCapture.isEnabled = true
                    btnCapture.text = "Capture & Enroll Face"

                    result.onSuccess { contact ->
                        hapticManager.vibrateClick()
                        editName.setText("")
                        Toast.makeText(this@MainActivity, "Enrolled ${contact.name} successfully!", Toast.LENGTH_LONG).show()
                        ttsManager.speak("Enrolled ${contact.name} successfully.", priority = 80, severity = "INFO", mode = TtsMode.FACE, generationId = genId)
                        refreshContactsUi()
                    }.onFailure { err ->
                        val msg = err.message ?: "Could not detect face"
                        Toast.makeText(this@MainActivity, msg, Toast.LENGTH_LONG).show()
                        ttsManager.speak(msg, priority = 70, severity = "INFO", mode = TtsMode.FACE, generationId = genId)
                    }
                }
            }
        }

        btnClearAll.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Clear All Faces")
                .setMessage("Are you sure you want to delete all registered faces?")
                .setPositiveButton("Delete All") { _, _ ->
                    faceRecognitionManager.clearAllContacts()
                    hapticManager.vibrateClick()
                    ttsManager.speak("All registered faces deleted.", priority = 70, severity = "INFO", mode = TtsMode.FACE)
                    refreshContactsUi()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        btnClose.setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    /**
     * Mode Controller (Single Source of Truth)
     * Atomically switches TtsMode, invalidates old speech tokens, and clears pending speech.
     */
    private fun setDetectionMode(newMode: AppDetectionMode, announce: Boolean = true) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastModeSwitchTapTime < 250L) {
            return
        }
        lastModeSwitchTapTime = now

        currentMode = newMode
        decisionEngine.reset()
        hapticManager.vibrateClick()

        val targetTtsMode = when (newMode) {
            AppDetectionMode.OCR_ONLY -> TtsMode.OCR
            AppDetectionMode.CURRENCY_ONLY -> TtsMode.CURRENCY
            AppDetectionMode.COLOR_ONLY -> TtsMode.COLOR
            AppDetectionMode.BARCODE_ONLY -> TtsMode.BARCODE
            AppDetectionMode.OBJECT_DETECTION_ONLY,
            AppDetectionMode.COMBINED -> TtsMode.OBJECT_DETECTION
        }

        // Atomically switch TTS mode: cuts off old speech audio, increments generation ID, clears queue
        ttsManager.switchMode(targetTtsMode)

        when (newMode) {
            AppDetectionMode.OCR_ONLY -> {
                prefs.edit().putString(keySavedMode, "OCR").apply()
                currentDetections = emptyList()
                activeRecognizedFaces.set(emptyList())
                runOnUiThread {
                    binding.overlayView.updateResults(emptyList(), HazardEvent(), emptyList())
                }
                if (announce) {
                    ttsManager.speak("OCR reading mode. Tap shutter or volume up to read text.", priority = 60, severity = "INFO", mode = TtsMode.SYSTEM)
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
                    ttsManager.speak("Banknote reader mode. Tap shutter or volume up to identify Indian Rupee note.", priority = 60, severity = "INFO", mode = TtsMode.SYSTEM)
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
                    ttsManager.speak("Color detector mode. Center object inside reticle.", priority = 60, severity = "INFO", mode = TtsMode.SYSTEM)
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
                    ttsManager.speak("Barcode scanner mode. Align barcode or QR code in front of camera.", priority = 60, severity = "INFO", mode = TtsMode.SYSTEM)
                }
            }
            AppDetectionMode.OBJECT_DETECTION_ONLY -> {
                prefs.edit().putString(keySavedMode, "OBJECT").apply()
                if (announce) {
                    ttsManager.speak("Object detection mode active.", priority = 60, severity = "INFO", mode = TtsMode.SYSTEM)
                }
            }
            AppDetectionMode.COMBINED -> {
                if (announce) {
                    ttsManager.speak("Combined mode active.", priority = 60, severity = "INFO", mode = TtsMode.SYSTEM)
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

        val cardFace = dialogView.findViewById<View>(R.id.cardToolFaceEnroll)
        val cardCurrency = dialogView.findViewById<View>(R.id.cardToolCurrency)
        val cardColor = dialogView.findViewById<View>(R.id.cardToolColor)
        val cardBarcode = dialogView.findViewById<View>(R.id.cardToolBarcode)
        val btnCancel = dialogView.findViewById<View>(R.id.btnCancelTools)

        cardFace.setOnClickListener {
            dialog.dismiss()
            showFaceEnrollDialog()
        }

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
        if (!isCurrencyScanning.compareAndSet(false, true)) {
            return
        }

        val bitmap = latestBitmap
        if (bitmap == null || bitmap.isRecycled) {
            isCurrencyScanning.set(false)
            ttsManager.speak("Camera initializing. Please hold banknote in front of camera.", priority = 70, severity = "INFO", mode = TtsMode.CURRENCY)
            return
        }
        hapticManager.vibrateClick()
        ttsManager.speak("Scanning banknote...", priority = 75, severity = "INFO", mode = TtsMode.CURRENCY)

        val genId = ttsManager.getGenerationId()

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val result = currencyDetector.detectCurrency(bitmap)
                withContext(Dispatchers.Main) {
                    hapticManager.vibrateClick()
                    ttsManager.speak(result.spokenText, priority = 85, severity = "INFO", mode = TtsMode.CURRENCY, generationId = genId)
                    showOnScreenAnnouncement("💵", result.spokenText)
                }
            } finally {
                delay(400L)
                isCurrencyScanning.set(false)
            }
        }
    }

    private fun triggerColorDetection() {
        if (!isColorScanning.compareAndSet(false, true)) {
            return
        }

        val bitmap = latestBitmap
        if (bitmap == null || bitmap.isRecycled) {
            isColorScanning.set(false)
            ttsManager.speak("Camera initializing. Point camera at colored surface.", priority = 70, severity = "INFO", mode = TtsMode.COLOR)
            return
        }
        hapticManager.vibrateClick()

        val genId = ttsManager.getGenerationId()

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val result = colorDetector.detectColor(bitmap)
                withContext(Dispatchers.Main) {
                    hapticManager.vibrateClick()
                    ttsManager.speak(result.spokenDescription, priority = 85, severity = "INFO", mode = TtsMode.COLOR, generationId = genId)
                    showOnScreenAnnouncement("🎨", result.spokenDescription)
                }
            } finally {
                delay(400L)
                isColorScanning.set(false)
            }
        }
    }

    private fun triggerBarcodeScan() {
        if (!isBarcodeScanning.compareAndSet(false, true)) {
            return
        }

        val bitmap = latestBitmap
        if (bitmap == null || bitmap.isRecycled) {
            isBarcodeScanning.set(false)
            ttsManager.speak("Camera initializing. Align barcode in front of camera.", priority = 70, severity = "INFO", mode = TtsMode.BARCODE)
            return
        }
        hapticManager.vibrateClick()
        ttsManager.speak("Scanning barcode or QR code...", priority = 75, severity = "INFO", mode = TtsMode.BARCODE)

        val genId = ttsManager.getGenerationId()

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val result = barcodeScannerManager.scanBitmap(bitmap)
                withContext(Dispatchers.Main) {
                    if (result != null) {
                        hapticManager.vibrateClick()
                        ttsManager.speak(result.spokenText, priority = 85, severity = "INFO", mode = TtsMode.BARCODE, generationId = genId)
                        showOnScreenAnnouncement("💵", result.spokenText)
                    } else {
                        ttsManager.speak("No barcode or QR code detected. Hold steady.", priority = 75, severity = "INFO", mode = TtsMode.BARCODE, generationId = genId)
                        Toast.makeText(this@MainActivity, "No barcode detected", Toast.LENGTH_SHORT).show()
                    }
                }
            } finally {
                delay(400L)
                isBarcodeScanning.set(false)
            }
        }
    }

    private fun triggerOcrReading() {
        val now = SystemClock.elapsedRealtime()

        if (now - lastOcrRequestTime < 1200L) {
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
        if (bitmap == null || bitmap.isRecycled) {
            isOcrProcessing.set(false)
            ttsManager.speak("Video source initializing. Please wait.", priority = 70, severity = "INFO", mode = TtsMode.OCR)
            return
        }

        ttsManager.speak("Reading text...", priority = 80, severity = "INFO", mode = TtsMode.OCR)

        val genId = ttsManager.getGenerationId()

        lifecycleScope.launch {
            try {
                val resultText = ocrManager.extractText(bitmap)
                withContext(Dispatchers.Main) {
                    ttsManager.startOcrReading(resultText, generationId = genId)
                    showOnScreenAnnouncement("📖", resultText)
                }
            } catch (e: Exception) {
                Log.e(tag, "OCR processing error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    ttsManager.speak("Could not read text.", priority = 70, severity = "INFO", mode = TtsMode.OCR, generationId = genId)
                }
            } finally {
                delay(600L)
                isOcrProcessing.set(false)
            }
        }
    }

    private fun triggerSceneSummary() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastSceneSummaryTapTime < 600L) {
            return
        }
        lastSceneSummaryTapTime = now

        val summary = decisionEngine.getFullSceneSummary(currentDetections)
        ttsManager.speak(summary, priority = 80, severity = "INFO", mode = TtsMode.SYSTEM)
        showOnScreenAnnouncement("👁️", summary)
    }

    private var hideNotificationRunnable: Runnable? = null

    private fun showOnScreenAnnouncement(icon: String, message: String, durationMs: Long = 3500L) {
        runOnUiThread {
            hideNotificationRunnable?.let { mainHandler.removeCallbacks(it) }
            binding.txtNotificationIcon.text = icon
            binding.txtNotificationMessage.text = message
            binding.layoutNotificationBanner.alpha = 0f
            binding.layoutNotificationBanner.visibility = View.VISIBLE
            binding.layoutNotificationBanner.animate().alpha(1f).setDuration(200).start()

            val runnable = Runnable {
                binding.layoutNotificationBanner.animate().alpha(0f).setDuration(250).withEndAction {
                    binding.layoutNotificationBanner.visibility = View.GONE
                }.start()
            }
            hideNotificationRunnable = runnable
            mainHandler.postDelayed(runnable, durationMs)
        }
    }

    private fun showVoiceAssistantResult(title: String, body: String, speakText: String = body) {
        SoundEffectManager.playVoiceCloseChime()
        runOnUiThread {
            binding.voiceAssistantOverlay.visibility = View.VISIBLE
            binding.txtVoiceStatus.text = "Assistant Response"
            binding.layoutVoiceResult.visibility = View.VISIBLE
            binding.txtVoiceResultTitle.text = title
            binding.txtVoiceResultBody.text = body

            ttsManager.speak(
                text = speakText,
                priority = 90,
                severity = "INFO",
                mode = TtsMode.VOICE_ASSISTANT
            )

            // Auto-dismiss after 4.5 seconds
            mainHandler.postDelayed({
                if (binding.voiceAssistantOverlay.visibility == View.VISIBLE) {
                    binding.voiceAssistantOverlay.visibility = View.GONE
                    binding.layoutVoiceResult.visibility = View.GONE
                    updateCarouselUi()
                }
            }, 4500L)
        }
    }

    private fun handleVoiceCommand(command: VoiceCommand) {
        Log.i(tag, "Voice command received: type=${command.type}, text='${command.rawText}'")
        hapticManager.vibrateClick()

        runOnUiThread {
            binding.voiceAssistantOverlay.visibility = View.VISIBLE
            binding.txtVoiceStatus.text = "Processing..."
            binding.txtVoiceTranscription.text = "“${command.rawText}”"
        }

        when (command.type) {
            VoiceCommandType.COLOR_QUERY -> {
                val bitmap = latestBitmap
                if (bitmap == null || bitmap.isRecycled) {
                    showVoiceAssistantResult("Color Detector", "Camera initializing. Point camera at surface.")
                    return
                }
                lifecycleScope.launch(Dispatchers.Default) {
                    val result = colorDetector.detectColor(bitmap)
                    withContext(Dispatchers.Main) {
                        showVoiceAssistantResult("Color Detector", result.spokenDescription)
                    }
                }
            }
            VoiceCommandType.CURRENCY_QUERY -> {
                val bitmap = latestBitmap
                if (bitmap == null || bitmap.isRecycled) {
                    showVoiceAssistantResult("Banknote Reader", "Camera initializing. Hold banknote in front of camera.")
                    return
                }
                lifecycleScope.launch(Dispatchers.Default) {
                    val result = currencyDetector.detectCurrency(bitmap)
                    withContext(Dispatchers.Main) {
                        showVoiceAssistantResult("Banknote Reader", result.spokenText)
                    }
                }
            }
            VoiceCommandType.SCENE_QUERY -> {
                val summary = decisionEngine.getFullSceneSummary(currentDetections)
                showVoiceAssistantResult("Scene Summary", summary)
            }
            VoiceCommandType.OCR_QUERY -> {
                val bitmap = latestBitmap
                if (bitmap == null || bitmap.isRecycled) {
                    showVoiceAssistantResult("OCR Reader", "Camera initializing. Please wait.")
                    return
                }
                lifecycleScope.launch(Dispatchers.Default) {
                    try {
                        val text = ocrManager.extractText(bitmap)
                        withContext(Dispatchers.Main) {
                            showVoiceAssistantResult("OCR Text Reader", text)
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            showVoiceAssistantResult("OCR Reader", "Could not read text clearly.")
                        }
                    }
                }
            }
            VoiceCommandType.BARCODE_QUERY -> {
                val bitmap = latestBitmap
                if (bitmap == null || bitmap.isRecycled) {
                    showVoiceAssistantResult("Barcode Scanner", "Camera initializing. Align barcode.")
                    return
                }
                lifecycleScope.launch(Dispatchers.Default) {
                    val result = barcodeScannerManager.scanBitmap(bitmap)
                    withContext(Dispatchers.Main) {
                        val msg = result?.spokenText ?: "No barcode or QR code detected in view."
                        showVoiceAssistantResult("Barcode Scanner", msg)
                    }
                }
            }
            VoiceCommandType.FACE_QUERY -> {
                val known = activeRecognizedFaces.get().filter { it.isKnown }
                val msg = if (known.isNotEmpty()) {
                    val names = known.joinToString(", ") { it.name ?: "" }
                    "Identified: $names in front of you."
                } else {
                    "No registered faces recognized in camera view."
                }
                showVoiceAssistantResult("Face Recognition", msg)
            }
            VoiceCommandType.SOS_QUERY -> {
                binding.voiceAssistantOverlay.visibility = View.GONE
                sosManager.triggerEmergencySos { _, msg ->
                    Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                }
            }
            VoiceCommandType.STOP_SPEECH -> {
                binding.voiceAssistantOverlay.visibility = View.GONE
                stopSpeech()
            }
            VoiceCommandType.UNKNOWN -> {
                showVoiceAssistantResult(
                    "Voice Assistant",
                    "Ask: 'What color is this', 'What note is this', 'Describe scene', or 'Read text'."
                )
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
        // Fast 1-Click Video Source Toggle (Phone Camera <-> ESP32 Smart Cap)
        binding.txtSourceBadge.setOnClickListener {
            val nextSource = if (currentSource == VideoInputSource.PHONE_CAMERA) {
                VideoInputSource.ESP32_CAM
            } else {
                VideoInputSource.PHONE_CAMERA
            }
            setVideoInputSource(nextSource, announce = true)
        }

        // Long press opens IP / Stream URL dialog directly
        binding.txtSourceBadge.setOnLongClickListener {
            showEsp32ConfigDialog()
            true
        }

        binding.btnSettings.setOnClickListener {
            hapticManager.vibrateClick()
            val intent = Intent(this, SettingsActivity::class.java)
            startActivity(intent)
        }

        binding.btnFaceManager.setOnClickListener {
            showFaceEnrollDialog()
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

        binding.btnShutter.setOnTouchListener { v, event ->
            when (event.action) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    v.animate().scaleX(0.88f).scaleY(0.88f).setDuration(70).start()
                }
                android.view.MotionEvent.ACTION_UP, android.view.MotionEvent.ACTION_CANCEL -> {
                    v.animate().scaleX(1.0f).scaleY(1.0f).setDuration(120).start()
                    if (event.action == android.view.MotionEvent.ACTION_UP) {
                        val now = SystemClock.elapsedRealtime()
                        if (now - lastShutterTapTime >= 350L) {
                            lastShutterTapTime = now
                            executePrimaryModeAction()
                        }
                    }
                }
            }
            true
        }

        binding.btnSceneSummary.setOnClickListener {
            triggerSceneSummary()
        }

        binding.btnTools.setOnClickListener {
            showToolsDialog()
        }

        binding.txtModeStatus.setOnClickListener {
            hapticManager.vibrateClick()
            ttsManager.stop()
            voiceCommandManager.startListening()
        }

        updateCarouselUi()
    }

    /**
     * Intercept Volume UP and Volume DOWN physical buttons cleanly.
     * Prevents the Android system volume slider dialog from appearing.
     * Includes strict button debouncing and speech preemption.
     */
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        val keyCode = event.keyCode
        val action = event.action

        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            if (action == KeyEvent.ACTION_DOWN) {
                if (event.repeatCount == 0) {
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastVolumeUpTapTime < 250L) {
                        return true
                    }
                    lastVolumeUpTapTime = now

                    Log.i(tag, "Volume UP pressed -> Activating Voice Assistant")
                    ttsManager.stop()
                    hapticManager.vibrateClick()
                    SoundEffectManager.playVoiceOpenChime()

                    val voiceEnabled = prefs.getBoolean(keyVoiceCommands, true)
                    if (voiceEnabled) {
                        runOnUiThread {
                            binding.voiceAssistantOverlay.visibility = View.VISIBLE
                            binding.layoutVoiceResult.visibility = View.GONE
                            binding.txtVoiceStatus.text = "Listening... Speak now"
                            binding.txtVoiceTranscription.text = "Speak: 'What color is this', 'What note is this', 'Describe scene'..."
                            binding.txtModeStatus.text = "Listening..."
                        }
                        voiceCommandManager.startListening()
                    } else {
                        Toast.makeText(this, "Voice commands are disabled in Settings", Toast.LENGTH_SHORT).show()
                    }
                }
                return true
            } else if (action == KeyEvent.ACTION_UP) {
                // Dedicated to Voice Assistant — do not trigger capture or mode changes on Volume UP
                return true
            }
            return true
        }

        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            if (action == KeyEvent.ACTION_DOWN) {
                if (event.repeatCount == 0) {
                    if (ttsManager.isSpeaking) {
                        stopSpeech()
                        return true
                    }

                    val now = SystemClock.elapsedRealtime()
                    if (now - lastVolumeDownTapTime < 250L) {
                        return true
                    }
                    lastVolumeDownTapTime = now

                    if (now - lastVolumeDownClickTime < 600L) {
                        volumeDownClickCount++
                    } else {
                        volumeDownClickCount = 1
                    }
                    lastVolumeDownClickTime = now

                    // 4-click Volume DOWN triggers Emergency SOS
                    if (volumeDownClickCount >= 4) {
                        volumeDownClickCount = 0
                        sosManager.triggerEmergencySos { _, msg ->
                            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                        }
                        return true
                    }

                    triggerSceneSummary()
                }
                return true
            } else if (action == KeyEvent.ACTION_UP) {
                return true
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
