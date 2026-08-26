package com.blindcap.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import com.blindcap.app.ai.Detection
import com.blindcap.app.ai.DetectionTimings
import com.blindcap.app.ai.FaceObservation
import com.blindcap.app.engine.HazardEvent
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
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
        color = Color.parseColor("#00FFFF")
        textSize = 26f
        isAntiAlias = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    private val goldTextPaint = Paint().apply {
        color = Color.parseColor("#FFD700")
        textSize = 24f
        isAntiAlias = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    private val corridorPaint = Paint().apply {
        color = Color.argb(100, 255, 255, 255)
        strokeWidth = 2f
        style = Paint.Style.STROKE
    }

    private val hudBgPaint = Paint().apply {
        color = Color.argb(190, 18, 16, 14)
        style = Paint.Style.FILL
    }

    var cameraFps: Float = 0f
    var aiFps: Float = 0f
    var timings: DetectionTimings = DetectionTimings()
    var activeDevice: String = "CPU"
    var ttsStatus: String = "READY"
    var errorMessage: String? = null
    var faceDiagnostic: String = "Ready"
    var faceScanMs: Float = 0f
    var registeredContactNames: Set<String> = emptySet()

    private var latestRenderedFrameId: Long = 0L
    private var detections: List<Detection> = emptyList()
    private var hazardEvent: HazardEvent? = null
    private var faceObservations: List<FaceObservation> = emptyList()

    fun updateResults(
        newDetections: List<Detection>,
        newEvent: HazardEvent,
        newObservations: List<FaceObservation> = emptyList(),
        frameId: Long = 0L
    ) {
        if (frameId > 0L && frameId < latestRenderedFrameId) {
            return // Drop stale out-of-order asynchronous result
        }
        if (frameId > 0L) {
            latestRenderedFrameId = frameId
        }

        detections = newDetections.toList()
        hazardEvent = newEvent
        faceObservations = newObservations.toList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        if (w <= 0 || h <= 0) return

        val now = SystemClock.elapsedRealtime()

        // 1. Draw Walking Corridor Boundaries (Center 40% of FOV: 0.30 to 0.70)
        canvas.drawLine(w * 0.30f, 0f, w * 0.30f, h, corridorPaint)
        canvas.drawLine(w * 0.70f, 0f, w * 0.70f, h, corridorPaint)

        // 2. Draw Object Bounding Boxes (Live YOLO Detections of CURRENT frame)
        for (det in detections) {
            val left = (det.bbox.left * w).coerceIn(0f, w)
            val top = (det.bbox.top * h).coerceIn(0f, h)
            val right = (det.bbox.right * w).coerceIn(0f, w)
            val bottom = (det.bbox.bottom * h).coerceIn(0f, h)

            if (right <= left || bottom <= top) continue

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

        // 3. Draw Dynamic Real-Time Face Observations (Strict Freshness <= 300ms)
        val validObservations = faceObservations.filter { !it.isStale(now, maxAgeMs = 300L) }
        for (obs in validObservations) {
            val left = (obs.bbox.left * w).coerceIn(0f, w)
            val top = (obs.bbox.top * h).coerceIn(0f, h)
            val right = (obs.bbox.right * w).coerceIn(0f, w)
            val bottom = (obs.bbox.bottom * h).coerceIn(0f, h)

            if (right <= left || bottom <= top) continue

            if (obs.isKnown && obs.identity != null) {
                // Known Contact: Vibrant Cyan Box with Contact Name
                facePaint.color = Color.parseColor("#00FFFF")
                canvas.drawRect(left, top, right, bottom, facePaint)

                val label = "👤 ${obs.identity.uppercase()}"
                canvas.drawText(label, left + 4f, max(top - 8f, 25f), faceTextPaint)
            } else if (obs.isFacingUser) {
                // Unknown Face Looking Directly at User: Gold Box
                facePaint.color = Color.parseColor("#FFD700")
                canvas.drawRect(left, top, right, bottom, facePaint)
                canvas.drawText("LOOKING AT YOU", left + 4f, max(top - 8f, 25f), goldTextPaint)
            } else {
                // General Face Detection: Subtle White/Cyan Box
                facePaint.color = Color.argb(180, 0, 255, 255)
                canvas.drawRect(left, top, right, bottom, facePaint)
                canvas.drawText("FACE", left + 4f, max(top - 8f, 25f), smallTextPaint)
            }
        }

        // 4. Draw Performance Diagnostic HUD (Top Left)
        val hudWidth = 460f
        val hudHeight = 330f
        val startX = 20f
        val startY = 160f

        canvas.drawRoundRect(startX, startY, startX + hudWidth, startY + hudHeight, 16f, 16f, hudBgPaint)

        var yPos = startY + 34f
        val lineGap = 34f

        canvas.drawText("OCULUS AI TELEMETRY", startX + 16f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("Source: %.0f FPS  |  AI: %.1f FPS".format(cameraFps, aiFps), startX + 16f, yPos, textPaint)
        yPos += lineGap
        canvas.drawText("Lat: %.0fms (Pre:%.1f Inf:%.1f Post:%.1f)".format(
            timings.totalMs, timings.preprocessMs, timings.inferenceMs, timings.postprocessMs
        ), startX + 16f, yPos, smallTextPaint)
        yPos += lineGap
        canvas.drawText("P95 Latency: %.1fms  |  Backend: %s".format(timings.p95Ms, activeDevice), startX + 16f, yPos, smallTextPaint)
        yPos += lineGap
        val faceCountStr = if (validObservations.isNotEmpty()) " | Faces: ${validObservations.size}" else ""
        canvas.drawText("Objects: ${detections.size}$faceCountStr | TTS: $ttsStatus", startX + 16f, yPos, smallTextPaint)
        yPos += lineGap

        val faceMsStr = if (faceScanMs > 0f) " (%.0fms)".format(faceScanMs) else ""
        val truncDiag = if (faceDiagnostic.length > 22) faceDiagnostic.take(22) + "..." else faceDiagnostic
        canvas.drawText("Face AI: $truncDiag$faceMsStr", startX + 16f, yPos, smallTextPaint)
        yPos += lineGap

        val statusText = errorMessage ?: (hazardEvent?.warningText ?: "Path clear (silent)")
        val truncated = if (statusText.length > 28) statusText.take(28) + "..." else statusText
        canvas.drawText("Event: $truncated", startX + 16f, yPos, smallTextPaint)
    }
}
