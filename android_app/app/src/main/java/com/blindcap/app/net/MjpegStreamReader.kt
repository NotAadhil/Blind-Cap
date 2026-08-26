package com.blindcap.app.net

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

class MjpegStreamReader(
    private val onFrameReceived: (Bitmap) -> Unit,
    private val onStatusChanged: (String) -> Unit
) {

    private val tag = "MjpegStreamReader"
    private val isRunning = AtomicBoolean(false)
    private var streamThread: Thread? = null

    fun start(streamUrl: String) {
        if (isRunning.getAndSet(true)) {
            stop()
            isRunning.set(true)
        }

        streamThread = Thread({
            runStreamLoop(streamUrl)
        }, "ESP32-CAM-Stream-Thread").apply {
            isDaemon = true
            start()
        }
    }

    fun stop() {
        isRunning.set(false)
        streamThread?.interrupt()
        streamThread = null
        onStatusChanged("Disconnected")
    }

    private fun runStreamLoop(streamUrl: String) {
        var retryDelayMs = 1000L

        while (isRunning.get()) {
            var connection: HttpURLConnection? = null
            var inputStream: BufferedInputStream? = null

            try {
                onStatusChanged("Connecting to $streamUrl...")
                Log.i(tag, "Connecting to ESP32-CAM at $streamUrl")

                val url = URL(streamUrl)
                connection = (url.openConnection() as HttpURLConnection).apply {
                    connectTimeout = 5000
                    readTimeout = 7000
                    useCaches = false
                    requestMethod = "GET"
                    setRequestProperty("User-Agent", "OculusAI-Android")
                    setRequestProperty("Connection", "keep-alive")
                }

                connection.connect()
                val responseCode = connection.responseCode
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    throw IllegalStateException("HTTP server returned code $responseCode")
                }

                inputStream = BufferedInputStream(connection.inputStream, 32768)
                onStatusChanged("Connected (Streaming)")
                Log.i(tag, "Connected to ESP32-CAM successfully")
                retryDelayMs = 1000L

                val frameBuffer = ByteArrayOutputStream(65536)
                var prevByte = -1
                var insideJpeg = false

                // Fast byte-by-byte scanner for JPEG SOI (0xFF, 0xD8) and EOI (0xFF, 0xD9)
                while (isRunning.get()) {
                    val currByte = inputStream.read()
                    if (currByte == -1) break

                    if (!insideJpeg) {
                        if (prevByte == 0xFF && currByte == 0xD8) {
                            insideJpeg = true
                            frameBuffer.reset()
                            frameBuffer.write(0xFF)
                            frameBuffer.write(0xD8)
                        }
                    } else {
                        frameBuffer.write(currByte)
                        if (prevByte == 0xFF && currByte == 0xD9) {
                            insideJpeg = false
                            val jpegData = frameBuffer.toByteArray()
                            if (jpegData.isNotEmpty()) {
                                try {
                                    val bitmap = BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)
                                    if (bitmap != null && isRunning.get()) {
                                        onFrameReceived(bitmap)
                                    }
                                } catch (e: Exception) {
                                    Log.w(tag, "Error decoding frame: ${e.message}")
                                }
                            }
                            frameBuffer.reset()
                        }
                    }
                    prevByte = currByte
                }

            } catch (e: Exception) {
                if (isRunning.get()) {
                    Log.w(tag, "Stream error: ${e.message}. Reconnecting in ${retryDelayMs}ms...")
                    onStatusChanged("Connection lost. Retrying in ${retryDelayMs / 1000}s...")
                    try {
                        Thread.sleep(retryDelayMs)
                        retryDelayMs = (retryDelayMs * 1.5).toLong().coerceAtMost(5000L)
                    } catch (ie: InterruptedException) {
                        break
                    }
                }
            } finally {
                try {
                    inputStream?.close()
                } catch (_: Exception) {}
                try {
                    connection?.disconnect()
                } catch (_: Exception) {}
            }
        }
    }
}