package com.blindcap.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.DetectionTimings
import com.blindcap.app.ai.RecognizedFace
import com.blindcap.app.engine.HazardEvent
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
    }

    private val facePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    private val smallTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 22f
        isAntiAlias = true
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }

    private val faceTextPaint = Paint().apply {
        textSize = 26f
        isAntiAlias = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    private val corridorPaint = Paint().apply {
        color = Color.argb(100, 255, 255, 255)
        strokeWidth = 2f
        style = Paint.Style.STROKE
    }

    private val hudBgPaint = Paint().apply {
        color = Color.argb(180, 20, 20, 20)
        style = Paint.Style.FILL
    }

    var cameraFps: Float = 0f
    var aiFps: Float = 0f
    var timings: DetectionTimings = DetectionTimings()
    var activeDevice: String = "CPU"
    var ttsStatus: String = "READY"
    var errorMessage: String? = null
    var faceDiagnostic: String = "Ready"

    private var detections: List<Detection> = emptyList()
    private var hazardEvent: HazardEvent? = null
    private var recognizedFaces: List<RecognizedFace> = emptyList()

    fun updateResults(
        newDetections: List<Detection>,
        newEvent: HazardEvent,
        newFaces: List<RecognizedFace> = emptyList()
    ) {
        detections = newDetections
        hazardEvent = newEvent
        recognizedFaces = newFaces
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        if (w <= 0 || h <= 0) return

        // 1. Draw Walking Corridor Boundaries (Center 40% of FOV: 0.30 to 0.70)
        canvas.drawLine(w * 0.30f, 0f, w * 0.30f, h, corridorPaint)
        canvas.drawLine(w * 0.70f, 0f, w * 0.70f, h, corridorPaint)

        // 2. Draw Object Detections
        for (det in detections) {
            val left = det.bbox.left * w
            val top = det.bbox.top * h
            val right = det.bbox.right * w
            val bottom = det.bbox.bottom * h

            // Color coding based on Hazard Level
            val isObstruction = hazardEvent?.allHazards?.any { it.className.equals(det.className, true) } == true
            boxPaint.color = when {
                isObstruction && det.areaRatio >= 0.15f -> Color.RED
                isObstruction -> Color.parseColor("#FFA500") // Orange
                det.region == "center" -> Color.YELLOW
                else -> Color.GREEN
            }

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val distStr = if (det.estimatedDistanceM > 0) " (%.1fm)".format(det.estimatedDistanceM) else ""
            val label = "${det.className.uppercase()} %.0f%%%s".format(det.confidence * 100, distStr)
            canvas.drawText(label, left + 4f, max(top - 8f, 25f), textPaint)
        }

        // 3. Draw Real-Time Face Recognition Overlay
        for (face in recognizedFaces) {
            val left = face.bbox.left * w
            val top = face.bbox.top * h
            val right = face.bbox.right * w
            val bottom = face.bbox.bottom * h

            if (face.isKnown) {
                facePaint.color = Color.parseColor("#00FFFF") // Cyan for known contact
                faceTextPaint.color = Color.parseColor("#00FFFF")
            } else {
                facePaint.color = Color.parseColor("#FFD700") // Gold for unknown face
                faceTextPaint.color = Color.parseColor("#FFD700")
            }

            canvas.drawRect(left, top, right, bottom, facePaint)

            val faceLabel = if (face.isKnown) {
                "👤 ${face.name?.uppercase()}"
            } else if (face.isFacingUser) {
                "LOOKING AT YOU"
            } else {
                "FACE"
            }
            canvas.drawText(faceLabel, left + 4f, max(top - 8f, 30f), faceTextPaint)
        }

        // 4. Draw Performance Diagnostic HUD (Top Left, below Top Bar)
        val hudWidth = 440f
        val hudHeight = 310f
        val startX = 20f
        val startY = 160f

        canvas.drawRoundRect(startX, startY, startX + hudWidth, startY + hudHeight, 16f, 16f, hudBgPaint)

        var yPos = startY + 34f
        val lineGap = 34f

        canvas.drawText("OCULUS AI PERFORMANCE", startX + 16f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("Camera: %.0f FPS  |  AI: %.1f FPS".format(cameraFps, aiFps), startX + 16f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("Latency: %.0fms (Pre:%.0f Inf:%.0f Post:%.0f)".format(
            timings.totalMs, timings.preprocessMs, timings.inferenceMs, timings.postprocessMs
        ), startX + 16f, yPos, smallTextPaint)
        yPos += lineGap
        canvas.drawText("Backend: $activeDevice", startX + 16f, yPos, smallTextPaint)
        yPos += lineGap
        val faceCountStr = if (recognizedFaces.isNotEmpty()) " | Faces: ${recognizedFaces.size}" else ""
        canvas.drawText("Objects: ${detections.size}$faceCountStr | TTS: $ttsStatus", startX + 16f, yPos, smallTextPaint)
        yPos += lineGap

        val truncDiag = if (faceDiagnostic.length > 28) faceDiagnostic.take(28) + "..." else faceDiagnostic
        canvas.drawText("Face AI: $truncDiag", startX + 16f, yPos, smallTextPaint)
        yPos += lineGap

        val statusText = errorMessage ?: (hazardEvent?.warningText ?: "Scene stable (silent)")
        val truncated = if (statusText.length > 28) statusText.take(28) + "..." else statusText
        canvas.drawText("Event: $truncated", startX + 16f, yPos, smallTextPaint)
    }
}