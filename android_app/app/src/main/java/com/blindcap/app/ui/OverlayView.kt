package com.blindcap.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.DetectionTimings
import com.blindcap.app.engine.HazardEvent
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var recognizedFaces: List<com.blindcap.app.ai.RecognizedFace> = emptyList()
    private var hazardEvent: HazardEvent? = null

    var cameraFps: Float = 0.0f
    var aiFps: Float = 0.0f
    var timings: DetectionTimings = DetectionTimings()
    var activeDevice: String = "Initializing..."
    var ttsStatus: String = "IDLE"
    var errorMessage: String? = null

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val facePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    private val faceTextPaint = Paint().apply {
        color = Color.parseColor("#00FFFF")
        textSize = 28f
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    private val smallTextPaint = Paint().apply {
        color = Color.parseColor("#E0E0E0")
        textSize = 26f
        isAntiAlias = true
    }

    private val hudBgPaint = Paint().apply {
        color = Color.parseColor("#C0101010")
        style = Paint.Style.FILL
    }

    private val corridorPaint = Paint().apply {
        color = Color.parseColor("#60FFFFFF")
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    fun updateResults(
        newDetections: List<Detection>,
        event: HazardEvent,
        faces: List<com.blindcap.app.ai.RecognizedFace> = emptyList()
    ) {
        this.detections = newDetections
        this.hazardEvent = event
        this.recognizedFaces = faces
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        // 1. Draw Walking Corridor Boundaries (Left: 35%, Right: 65%)
        val leftLineX = w * 0.35f
        val rightLineX = w * 0.65f
        canvas.drawLine(leftLineX, 0f, leftLineX, h, corridorPaint)
        canvas.drawLine(rightLineX, 0f, rightLineX, h, corridorPaint)

        // 2. Draw Object Bounding Boxes & Distance Labels
        for (det in detections) {
            val left = det.bbox.left * w
            val top = det.bbox.top * h
            val right = det.bbox.right * w
            val bottom = det.bbox.bottom * h

            boxPaint.color = when (det.region) {
                "center" -> if (det.estimatedDistanceM <= 1.8f) Color.RED else Color.parseColor("#FFA500")
                else -> Color.parseColor("#00FF88")
            }

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val distLabel = "%.1fm".format(det.estimatedDistanceM)
            val confLabel = "${(det.confidence * 100).toInt()}%"
            val labelText = "${det.className.uppercase()} $confLabel ($distLabel)"
            canvas.drawText(labelText, left + 4f, max(top - 10f, 35f), textPaint)
        }

        // 3. Draw Face Identification Overlays
        for (face in recognizedFaces) {
            val left = face.bbox.left * w
            val top = face.bbox.top * h
            val right = face.bbox.right * w
            val bottom = face.bbox.bottom * h

            if (face.isKnown) {
                facePaint.color = Color.parseColor("#00FFFF") // Cyan for known contact
                faceTextPaint.color = Color.parseColor("#00FFFF")
            } else {
                facePaint.color = Color.parseColor("#FFD700") // Gold/Yellow for unknown face
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
        val hudWidth = 420f
        val hudHeight = 280f
        val startX = 20f
        val startY = 160f

        canvas.drawRoundRect(startX, startY, startX + hudWidth, startY + hudHeight, 16f, 16f, hudBgPaint)

        var yPos = startY + 36f
        val lineGap = 34f

        canvas.drawText("BLIND CAP PERFORMANCE", startX + 16f, yPos, textPaint)
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

        val statusText = errorMessage ?: (hazardEvent?.warningText ?: "Scene stable (silent)")
        val truncated = if (statusText.length > 28) statusText.take(28) + "..." else statusText
        canvas.drawText("Event: $truncated", startX + 16f, yPos, smallTextPaint)
    }
}