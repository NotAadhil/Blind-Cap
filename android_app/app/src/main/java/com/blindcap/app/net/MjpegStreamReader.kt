package com.blindcap.app.net

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.ConnectivityManager
import android.net.Network
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

        Log.i(tag, "Starting ESP32-CAM stream reader for: $normalized")
        streamThread = Thread({
            runStreamLoop(normalized)
        }, "ESP32-CAM-Stream-Thread").apply {
            priority = Thread.NORM_PRIORITY + 2
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

    private fun findWifiNetwork(): Network? {
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return null
            for (network in cm.allNetworks) {
                val caps = cm.getNetworkCapabilities(network) ?: continue
                if (caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) {
                    return network
                }
            }
            null
        } catch (e: Exception) {
            Log.w(tag, "Error locating Wi-Fi network: ${e.message}")
            null
        }
    }

    private fun buildCandidateEndpoints(targetUrl: String): List<String> {
        val candidates = mutableListOf<String>()
        candidates.add(targetUrl)

        val host = try {
            URL(targetUrl).host
        } catch (_: Exception) {
            "192.168.4.1"
        }

        val fallbacks = listOf(
            "http://$host:81/stream",
            "http://$host:80/stream",
            "http://$host/stream",
            "http://$host:81/",
            "http://$host/capture",
            "http://$host:80/capture",
            "http://$host:8080/?action=stream",
            "http://$host:8080/stream"
        )

        for (fb in fallbacks) {
            if (!candidates.contains(fb)) {
                candidates.add(fb)
            }
        }
        return candidates
    }

    private fun openHttpConnection(targetUrl: String, timeoutMs: Int): HttpURLConnection {
        val url = URL(targetUrl)
        val wifiNet = findWifiNetwork()

        val connection = if (wifiNet != null) {
            Log.d(tag, "Binding connection directly to Wi-Fi adapter ($targetUrl)")
            wifiNet.openConnection(url) as HttpURLConnection
        } else {
            url.openConnection() as HttpURLConnection
        }

        connection.apply {
            connectTimeout = timeoutMs
            readTimeout = timeoutMs + 2000
            useCaches = false
            requestMethod = "GET"
            setRequestProperty("User-Agent", "OculusAI-Android")
            setRequestProperty("Connection", "keep-alive")
            setRequestProperty("Accept", "multipart/x-mixed-replace, image/jpeg, */*")
        }
        return connection
    }

    private fun runStreamLoop(primaryUrl: String) {
        val candidateEndpoints = buildCandidateEndpoints(primaryUrl)
        var activeWorkingUrl: String? = null
        var candidateIndex = 0
        var consecutiveFailures = 0

        while (isRunning.get()) {
            val targetUrl = activeWorkingUrl ?: candidateEndpoints[candidateIndex % candidateEndpoints.size]
            var connection: HttpURLConnection? = null
            var inputStream: BufferedInputStream? = null

            try {
                if (activeWorkingUrl == null) {
                    onStatusChanged("Probing ${URL(targetUrl).port.let { if (it != -1) ":$it" else "" }}${URL(targetUrl).path}...")
                } else {
                    onStatusChanged("Connecting...")
                }

                Log.i(tag, "Connecting to endpoint: $targetUrl")
                connection = openHttpConnection(targetUrl, timeoutMs = 3000)
                connection.connect()

                val responseCode = connection.responseCode
                val contentType = connection.contentType ?: ""

                if (responseCode != HttpURLConnection.HTTP_OK && responseCode != 206) {
                    throw IllegalStateException("HTTP error $responseCode from $targetUrl")
                }

                activeWorkingUrl = targetUrl
                currentUrl = targetUrl
                consecutiveFailures = 0
                onStatusChanged("Connected")
                Log.i(tag, "SUCCESS: Connected to ESP32-CAM stream at $targetUrl (Content-Type: $contentType)")

                inputStream = BufferedInputStream(connection.inputStream, 65536)

                // Check if the endpoint is sending single JPEG images or multipart MJPEG stream
                val isSingleJpeg = contentType.contains("image/jpeg", ignoreCase = true) && !contentType.contains("multipart", ignoreCase = true)

                if (isSingleJpeg) {
                    // Continuous Snapshot Polling Mode (for basic /capture endpoints)
                    val rawBytes = inputStream.readBytes()
                    if (rawBytes.size > 200) {
                        val bitmap = BitmapFactory.decodeByteArray(rawBytes, 0, rawBytes.size)
                        if (bitmap != null && isRunning.get()) {
                            onFrameReceived(bitmap)
                        }
                    }
                    // Loop directly to next capture
                    try {
                        Thread.sleep(30)
                    } catch (_: InterruptedException) {
                        break
                    }
                } else {
                    // Continuous Multipart MJPEG Byte Stream (SOI/EOI Extraction)
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
                                // Detect JPEG Start of Image (SOI): 0xFF 0xD8
                                if (prevByte == 0xFF && currByte == 0xD8) {
                                    insideJpeg = true
                                    frameStream.reset()
                                    frameStream.write(0xFF)
                                    frameStream.write(0xD8)
                                }
                            } else {
                                frameStream.write(currByte)
                                // Detect JPEG End of Image (EOI): 0xFF 0xD9
                                if (prevByte == 0xFF && currByte == 0xD9) {
                                    insideJpeg = false
                                    val jpegBytes = frameStream.toByteArray()
                                    if (jpegBytes.size > 300) {
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
                }

            } catch (e: Exception) {
                if (isRunning.get()) {
                    consecutiveFailures++
                    val msg = e.message ?: "Connection failure"
                    Log.w(tag, "Stream error on $targetUrl: $msg")

                    if (activeWorkingUrl != null && consecutiveFailures > 2) {
                        Log.i(tag, "Working URL failed repeatedly, falling back to endpoint discovery")
                        activeWorkingUrl = null
                    }

                    candidateIndex++
                    onStatusChanged("Searching...")
                    try {
                        Thread.sleep(600)
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
}
