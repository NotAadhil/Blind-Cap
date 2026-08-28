package com.blindcap.app.haptics

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

class HapticManager(private val context: Context) {

    private val prefs: SharedPreferences = context.getSharedPreferences("OculusPrefs", Context.MODE_PRIVATE)
    private val keyHapticsEnabled = "pref_haptics_enabled"

    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
        vibratorManager?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    val isEnabled: Boolean
        get() = prefs.getBoolean(keyHapticsEnabled, true)

    fun vibrateClick() {
        if (!isEnabled || vibrator == null || !vibrator.hasVibrator()) return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(25L, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(25L)
        }
    }

    fun vibrateObstacleLeft() {
        if (!isEnabled || vibrator == null || !vibrator.hasVibrator()) return
        val pattern = longArrayOf(0, 40, 60, 40)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(pattern, -1)
        }
    }

    fun vibrateObstacleRight() {
        if (!isEnabled || vibrator == null || !vibrator.hasVibrator()) return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(50L, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(50L)
        }
    }

    fun vibrateDangerStop() {
        if (!isEnabled || vibrator == null || !vibrator.hasVibrator()) return
        val pattern = longArrayOf(0, 200, 100, 300)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(pattern, -1)
        }
    }

    fun vibrateSos() {
        if (vibrator == null || !vibrator.hasVibrator()) return
        // Morse code: ... --- ... (S: 100ms, O: 300ms)
        val pattern = longArrayOf(
            0, 100, 100, 100, 100, 100, 200,
            300, 100, 300, 100, 300, 200,
            100, 100, 100, 100, 100
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(pattern, -1)
        }
    }
}
