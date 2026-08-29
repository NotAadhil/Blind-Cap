package com.blindcap.app.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.sin

object SoundEffectManager {
    private const val TAG = "SoundEffectManager"
    private val executor = Executors.newSingleThreadExecutor()
    private const val SAMPLE_RATE = 44100

    private val openChimeBuffer: ShortArray by lazy {
        // Musical ascending 3-tone arpeggio: D5 (587 Hz) -> A5 (880 Hz) -> D6 (1174 Hz)
        generateMusicalChime(
            frequencies = doubleArrayOf(587.33, 880.0, 1174.66),
            durationsMs = intArrayOf(55, 55, 110),
            volume = 0.85
        )
    }

    private val closeChimeBuffer: ShortArray by lazy {
        // Musical descending 2-tone resolution: A5 (880 Hz) -> D5 (587 Hz)
        generateMusicalChime(
            frequencies = doubleArrayOf(880.0, 587.33),
            durationsMs = intArrayOf(65, 120),
            volume = 0.75
        )
    }

    private fun generateMusicalChime(frequencies: DoubleArray, durationsMs: IntArray, volume: Double): ShortArray {
        var totalSamples = 0
        for (d in durationsMs) {
            totalSamples += (SAMPLE_RATE * d / 1000)
        }
        val buffer = ShortArray(totalSamples)
        var offset = 0

        for (i in frequencies.indices) {
            val freq = frequencies[i]
            val durationMs = durationsMs[i]
            val numSamples = (SAMPLE_RATE * durationMs / 1000)

            for (j in 0 until numSamples) {
                val time = j.toDouble() / SAMPLE_RATE
                // Rich harmonic timbre (fundamental + overtone)
                val rawSine = sin(2.0 * PI * freq * time) + 0.35 * sin(4.0 * PI * freq * time) + 0.15 * sin(6.0 * PI * freq * time)

                val progress = j.toDouble() / numSamples
                // Fast 8% linear attack, smooth exponential decay
                val envelope = when {
                    progress < 0.08 -> progress / 0.08
                    else -> (1.0 - progress) * (1.0 - progress)
                }

                val sample = (rawSine * envelope * volume * 28000.0).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                buffer[offset + j] = sample.toShort()
            }
            offset += numSamples
        }
        return buffer
    }

    fun playVoiceOpenChime() {
        executor.execute {
            try {
                playBuffer(openChimeBuffer)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to play voice open chime: ${e.message}")
            }
        }
    }

    fun playVoiceCloseChime() {
        executor.execute {
            try {
                playBuffer(closeChimeBuffer)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to play voice close chime: ${e.message}")
            }
        }
    }

    private fun playBuffer(buffer: ShortArray) {
        var audioTrack: AudioTrack? = null
        try {
            audioTrack = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(buffer.size * 2)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()

            audioTrack.write(buffer, 0, buffer.size)
            audioTrack.play()
            Thread.sleep((buffer.size * 1000L / SAMPLE_RATE) + 40L)
        } finally {
            try {
                audioTrack?.stop()
                audioTrack?.release()
            } catch (_: Exception) {}
        }
    }
}
