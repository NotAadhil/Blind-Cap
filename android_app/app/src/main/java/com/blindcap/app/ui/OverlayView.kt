package com.blindcap.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.blindcap.app.R
import com.blindcap.app.ai.Detection
import com.blindcap.app.engine.HazardEvent
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var hazardEvent: HazardEvent? = null

    var cameraFps: Float = 60.0f
    var inferenceFps: Float = 20.0f
    var inferenceMs: Float = 50.0f
    var activeDevice: String = "GPU (Mobile NPU)"
    var ttsMode: String = "NORMAL"
    var errorMessage: String? = null

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 34f
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    private val hudBgPaint = Paint().apply {
        color = Color.parseColor("#B0151515")
        style = Paint.Style.FILL
    }

    private val corridorPaint = Paint().apply {
        color = Color.parseColor("#80FFFFFF")
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    fun updateResults(newDetections: List<Detection>, event: HazardEvent) {
        this.detections = newDetections
        this.hazardEvent = event
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        // 1. Draw Corridor Boundary Lines (Left: 35%, Right: 65%)
        val leftLineX = w * 0.35f
        val rightLineX = w * 0.65f
        canvas.drawLine(leftLineX, 0f, leftLineX, h, corridorPaint)
        canvas.drawLine(rightLineX, 0f, rightLineX, h, corridorPaint)

        // 2. Draw Object Bounding Boxes
        for (det in detections) {
            val left = det.bbox.left * w
            val top = det.bbox.top * h
            val right = det.bbox.right * w
            val bottom = det.bbox.bottom * h

            // Color-blind friendly severity colors
            boxPaint.color = when (det.region) {
                "center" -> if (det.estimatedDistanceM <= 1.8f) Color.RED else Color.parseColor("#FFA500")
                else -> Color.parseColor("#00FF88")
            }

            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Label above box
            val distLabel = "%.1fm".format(det.estimatedDistanceM)
            val confLabel = "${(det.confidence * 100).toInt()}%"
            val labelText = "${det.className.uppercase()} $confLabel ($distLabel)"
            canvas.drawText(labelText, left + 4f, max(top - 10f, 40f), textPaint)
        }

        // 3. Draw Semi-Transparent Accessibility HUD (Top Left)
        val hudWidth = 380f
        val hudHeight = 260f
        canvas.drawRoundRect(20f, 40f, 20f + hudWidth, 40f + hudHeight, 16f, 16f, hudBgPaint)

        var yPos = 80f
        val lineGap = 36f

        canvas.drawText("BLIND CAP MOBILE", 40f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("Video FPS: %.0f".format(cameraFps), 40f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("AI FPS: %.1f (%.0f ms)".format(inferenceFps, inferenceMs), 40f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("Engine: $activeDevice", 40f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("Objects in View: ${detections.size}", 40f, yPos, textPaint)
        yPos += lineGap
        val statusText = errorMessage ?: (hazardEvent?.warningText ?: "Path is clear.")
        canvas.drawText("Status: ${statusText.take(24)}", 40f, yPos, textPaint)
    }
}
