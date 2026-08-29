package com.blindcap.app.net

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.SystemClock
import android.util.Log
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

class MjpegStreamReader(
    private val context: Context,
    private val onFrameReceived: (Bitmap) -> Unit,
    private val onStatusChanged: (String) -> Unit
) {

    private val tag = "MjpegStreamReader"
    private val isRunning = AtomicBoolean(false)
    private var streamThread: Thread? = null

    @Volatile
    var currentUrl: String = "http://192.168.4.1:81/stream"
        private set

    companion object {
        fun normalizeStreamUrl(input: String): String {
            var trimmed = input.trim()
            if (trimmed.isEmpty()) return "http://192.168.4.1:81/stream"

            // Prepend http:// if missing
            if (!trimmed.startsWith("http://", ignoreCase = true) && !trimmed.startsWith("https://", ignoreCase = true)) {
                trimmed = "http://$trimmed"
            }

            val url = try {
                URL(trimmed)
            } catch (_: Exception) {
                return "http://192.168.4.1:81/stream"
            }

            val host = url.host
            val port = if (url.port != -1) url.port else 81
            val path = if (url.path.isNullOrEmpty() || url.path == "/") "/stream" else url.path

            return "http://$host:$port$path"
        }
    }

    fun start(streamUrl: String) {
        val normalized = normalizeStreamUrl(streamUrl)
        currentUrl = normalized

        if (isRunning.getAndSet(true)) {
            stop()
            isRunning.set(true)
        }

        Log.i(tag, "Starting MJPEG stream for: $normalized")
        streamThread = Thread({
            runStreamLoop(normalized)
        }, "ESP32-CAM-Stream-Thread").apply {
            priority = Thread.NORM_PRIORITY + 1
            isDaemon = true
            start()
        }
    }

    fun stop() {
        Log.i(tag, "Stopping MJPEG stream")
        isRunning.set(false)
        streamThread?.interrupt()
        streamThread = null
        onStatusChanged("Disconnected")
    }

    private fun runStreamLoop(targetUrl: String) {
        var retryDelayMs = 1000L

        while (isRunning.get()) {
            var connection: HttpURLConnection? = null
            var inputStream: BufferedInputStream? = null

            try {
                onStatusChanged("Connecting...")
                Log.i(tag, "Opening HTTP connection to $targetUrl")

                val url = URL(targetUrl)

                // Optional: Bind to Wi-Fi network on Android 10+ so traffic isn't routed to cell data
                bindToWifiNetworkIfAvailable()

                connection = (url.openConnection() as HttpURLConnection).apply {
                    connectTimeout = 4000
                    readTimeout = 6000
                    useCaches = false
                    requestMethod = "GET"
                    setRequestProperty("User-Agent", "OculusAI-Android")
                    setRequestProperty("Connection", "keep-alive")
                    setRequestProperty("Accept", "multipart/x-mixed-replace, image/jpeg, */*")
                }

                connection.connect()
                val responseCode = connection.responseCode
                if (responseCode != HttpURLConnection.HTTP_OK && responseCode != 206) {
                    throw IllegalStateException("HTTP server returned status $responseCode")
                }

                inputStream = BufferedInputStream(connection.inputStream, 32768)
                onStatusChanged("Connected")
                Log.i(tag, "SUCCESS: Connected to ESP32-CAM stream at $targetUrl")
                retryDelayMs = 1000L

                val frameStream = ByteArrayOutputStream(65536)
                val buffer = ByteArray(8192)
                var prevByte = 0
                var insideJpeg = false

                while (isRunning.get()) {
                    val bytesRead = inputStream.read(buffer, 0, buffer.size)
                    if (bytesRead <= 0) break

                    for (i in 0 until bytesRead) {
                        val currByte = buffer[i].toInt() and 0xFF

                        if (!insideJpeg) {
                            // Check for SOI: 0xFF 0xD8
                            if (prevByte == 0xFF && currByte == 0xD8) {
                                insideJpeg = true
                                frameStream.reset()
                                frameStream.write(0xFF)
                                frameStream.write(0xD8)
                            }
                        } else {
                            frameStream.write(currByte)
                            // Check for EOI: 0xFF 0xD9
                            if (prevByte == 0xFF && currByte == 0xD9) {
                                insideJpeg = false
                                val jpegBytes = frameStream.toByteArray()
                                if (jpegBytes.size > 200) {
                                    try {
                                        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                                        if (bitmap != null && isRunning.get()) {
                                            onFrameReceived(bitmap)
                                        }
                                    } catch (decEx: Exception) {
                                        Log.w(tag, "JPEG decode error: ${decEx.message}")
                                    }
                                }
                                frameStream.reset()
                            }
                        }
                        prevByte = currByte
                    }
                }

            } catch (e: Exception) {
                if (isRunning.get()) {
                    val msg = e.message ?: "Connection error"
                    Log.w(tag, "Stream error ($targetUrl): $msg. Reconnecting in ${retryDelayMs}ms...")
                    onStatusChanged("Retrying...")
                    try {
                        Thread.sleep(retryDelayMs)
                        retryDelayMs = (retryDelayMs * 1.5).toLong().coerceAtMost(4000L)
                    } catch (_: InterruptedException) {
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

    private fun bindToWifiNetworkIfAvailable() {
        try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return
            val networks = cm.allNetworks
            for (network in networks) {
                val caps = cm.getNetworkCapabilities(network) ?: continue
                if (caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) {
                    cm.bindProcessToNetwork(network)
                    break
                }
            }
        } catch (_: Exception) {}
    }
}
