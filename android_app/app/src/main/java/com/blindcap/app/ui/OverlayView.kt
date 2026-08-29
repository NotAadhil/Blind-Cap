package com.blindcap.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
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
        strokeWidth = 3.5f
        isAntiAlias = true
    }

    private val boxFillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val facePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val labelBgPaint = Paint().apply {
        color = Color.argb(200, 16, 20, 24)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 26f
        isAntiAlias = true
        isFakeBoldText = true
    }

    private val smallTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 20f
        isAntiAlias = true
    }

    private val faceTextPaint = Paint().apply {
        color = Color.parseColor("#00FFCC")
        textSize = 24f
        isAntiAlias = true
        isFakeBoldText = true
    }

    private val goldTextPaint = Paint().apply {
        color = Color.parseColor("#FFD54F")
        textSize = 22f
        isAntiAlias = true
        isFakeBoldText = true
    }

    // Subtle, modern dashed walking corridor guidelines (30% opacity, dashed)
    private val corridorPaint = Paint().apply {
        color = Color.argb(45, 255, 255, 255)
        strokeWidth = 2f
        style = Paint.Style.STROKE
        pathEffect = DashPathEffect(floatArrayOf(12f, 12f), 0f)
        isAntiAlias = true
    }

    private val hudBgPaint = Paint().apply {
        color = Color.argb(220, 14, 17, 21)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // Configurable HUD Visibility (Default: OFF for clean modern camera UI)
    var showDebugHud: Boolean = false
        set(value) {
            field = value
            invalidate()
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

    private var detections: List<Detection> = emptyList()
    private var hazardEvent: HazardEvent? = null
    private var faceObservations: List<FaceObservation> = emptyList()

    fun updateResults(
        newDetections: List<Detection>,
        newEvent: HazardEvent,
        newObservations: List<FaceObservation> = emptyList()
    ) {
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

        // 1. Draw Subtle Walking Corridor Guides (Only inside middle viewfinder area: 15% to 75% height)
        val corridorTop = h * 0.15f
        val corridorBottom = h * 0.74f
        canvas.drawLine(w * 0.30f, corridorTop, w * 0.30f, corridorBottom, corridorPaint)
        canvas.drawLine(w * 0.70f, corridorTop, w * 0.70f, corridorBottom, corridorPaint)

        // 2. Draw Modern Rounded Object Bounding Boxes
        for (det in detections) {
            val left = (det.bbox.left * w).coerceIn(0f, w)
            val top = (det.bbox.top * h).coerceIn(0f, h)
            val right = (det.bbox.right * w).coerceIn(0f, w)
            val bottom = (det.bbox.bottom * h).coerceIn(0f, h)

            if (right <= left || bottom <= top) continue

            val isObstruction = hazardEvent?.allHazards?.any { it.className.equals(det.className, true) } == true
            val strokeColor = when {
                isObstruction && det.areaRatio >= 0.15f -> Color.parseColor("#FF3B30") // Modern Red
                isObstruction -> Color.parseColor("#FF9500") // Modern Amber
                det.region == "center" -> Color.parseColor("#00FFCC") // Modern Turquoise
                else -> Color.parseColor("#34C759") // Modern Emerald
            }

            boxPaint.color = strokeColor
            val rect = RectF(left, top, right, bottom)
            canvas.drawRoundRect(rect, 14f, 14f, boxPaint)

            // Label Card with dark pill backdrop
            val distStr = if (det.estimatedDistanceM > 0) " (%.1fm)".format(det.estimatedDistanceM) else ""
            val label = "${det.className.uppercase()} %.0f%%%s".format(det.confidence * 100, distStr)
            val labelWidth = textPaint.measureText(label) + 20f
            val labelHeight = 36f
            val labelTop = max(top - labelHeight - 6f, 16f)

            val labelBgRect = RectF(left, labelTop, left + labelWidth, labelTop + labelHeight)
            canvas.drawRoundRect(labelBgRect, 10f, 10f, labelBgPaint)
            canvas.drawText(label, left + 10f, labelTop + 26f, textPaint)
        }

        // 3. Draw Modern Face Observation Cards
        val validObservations = faceObservations.filter { !it.isStale(now, maxAgeMs = 300L) }
        for (obs in validObservations) {
            val left = (obs.bbox.left * w).coerceIn(0f, w)
            val top = (obs.bbox.top * h).coerceIn(0f, h)
            val right = (obs.bbox.right * w).coerceIn(0f, w)
            val bottom = (obs.bbox.bottom * h).coerceIn(0f, h)

            if (right <= left || bottom <= top) continue

            val rect = RectF(left, top, right, bottom)
            if (obs.isKnown && obs.identity != null) {
                facePaint.color = Color.parseColor("#00FFCC")
                canvas.drawRoundRect(rect, 16f, 16f, facePaint)

                val label = "👤 ${obs.identity.uppercase()}"
                val labelWidth = faceTextPaint.measureText(label) + 20f
                val labelHeight = 36f
                val labelTop = max(top - labelHeight - 6f, 16f)
                val labelBgRect = RectF(left, labelTop, left + labelWidth, labelTop + labelHeight)
                canvas.drawRoundRect(labelBgRect, 10f, 10f, labelBgPaint)
                canvas.drawText(label, left + 10f, labelTop + 26f, faceTextPaint)
            } else if (obs.isFacingUser) {
                facePaint.color = Color.parseColor("#FFD54F")
                canvas.drawRoundRect(rect, 16f, 16f, facePaint)

                val label = "LOOKING AT YOU"
                val labelWidth = goldTextPaint.measureText(label) + 20f
                val labelHeight = 36f
                val labelTop = max(top - labelHeight - 6f, 16f)
                val labelBgRect = RectF(left, labelTop, left + labelWidth, labelTop + labelHeight)
                canvas.drawRoundRect(labelBgRect, 10f, 10f, labelBgPaint)
                canvas.drawText(label, left + 10f, labelTop + 26f, goldTextPaint)
            } else {
                facePaint.color = Color.argb(160, 255, 255, 255)
                canvas.drawRoundRect(rect, 16f, 16f, facePaint)

                val label = "FACE"
                val labelWidth = smallTextPaint.measureText(label) + 16f
                val labelHeight = 32f
                val labelTop = max(top - labelHeight - 6f, 16f)
                val labelBgRect = RectF(left, labelTop, left + labelWidth, labelTop + labelHeight)
                canvas.drawRoundRect(labelBgRect, 8f, 8f, labelBgPaint)
                canvas.drawText(label, left + 8f, labelTop + 22f, smallTextPaint)
            }
        }

        // 4. Draw Performance Telemetry HUD ONLY if enabled
        if (showDebugHud) {
            val hudWidth = 460f
            val hudHeight = 330f
            val startX = 24f
            val startY = 180f

            canvas.drawRoundRect(startX, startY, startX + hudWidth, startY + hudHeight, 20f, 20f, hudBgPaint)

            var yPos = startY + 34f
            val lineGap = 34f

            canvas.drawText("OCULUS AI TELEMETRY", startX + 18f, yPos, textPaint)
            yPos += lineGap
            canvas.drawText("Source: %.0f FPS  |  AI: %.1f FPS".format(cameraFps, aiFps), startX + 18f, yPos, textPaint)
            yPos += lineGap
            canvas.drawText("Lat: %.0fms (Pre:%.1f Inf:%.1f Post:%.1f)".format(
                timings.totalMs, timings.preprocessMs, timings.inferenceMs, timings.postprocessMs
            ), startX + 18f, yPos, smallTextPaint)
            yPos += lineGap
            canvas.drawText("P95 Latency: %.1fms  |  Backend: %s".format(timings.p95Ms, activeDevice), startX + 18f, yPos, smallTextPaint)
            yPos += lineGap
            val faceCountStr = if (validObservations.isNotEmpty()) " | Faces: ${validObservations.size}" else ""
            canvas.drawText("Objects: ${detections.size}$faceCountStr | TTS: $ttsStatus", startX + 18f, yPos, smallTextPaint)
            yPos += lineGap

            val faceMsStr = if (faceScanMs > 0f) " (%.0fms)".format(faceScanMs) else ""
            val truncDiag = if (faceDiagnostic.length > 22) faceDiagnostic.take(22) + "..." else faceDiagnostic
            canvas.drawText("Face AI: $truncDiag$faceMsStr", startX + 18f, yPos, smallTextPaint)
            yPos += lineGap

            val statusText = errorMessage ?: (hazardEvent?.warningText ?: "Path clear (silent)")
            val truncated = if (statusText.length > 28) statusText.take(28) + "..." else statusText
            canvas.drawText("Event: $truncated", startX + 18f, yPos, smallTextPaint)
        }
    }
}
