package com.blindcap.app.safety

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.location.Location
import android.os.Handler
import android.os.Looper
import android.telephony.SmsManager
import android.util.Log
import androidx.core.content.ContextCompat
import com.blindcap.app.haptics.HapticManager
import com.blindcap.app.speech.TtsManager
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

class SosManager(
    private val context: Context,
    private val ttsManager: TtsManager,
    private val hapticManager: HapticManager
) {

    private val tag = "OculusSosManager"
    private val prefs: SharedPreferences = context.getSharedPreferences("OculusPrefs", Context.MODE_PRIVATE)
    private val keyEmergencyContact = "pref_emergency_contact"

    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val mainHandler = Handler(Looper.getMainLooper())
    private var isCountdownActive = false
    private var countdownRunnable: Runnable? = null

    val emergencyContact: String
        get() = prefs.getString(keyEmergencyContact, "") ?: ""

    fun hasContact(): Boolean = emergencyContact.trim().isNotEmpty()

    fun triggerEmergencySos(onCompleted: (Boolean, String) -> Unit) {
        if (!hasContact()) {
            ttsManager.speak("Emergency contact not configured. Please set emergency phone number in settings.", priority = 95, severity = "CRITICAL")
            onCompleted(false, "Emergency contact not set")
            return
        }

        isCountdownActive = true
        hapticManager.vibrateSos()
        ttsManager.speak("Emergency SOS initiated. Sending alert in three seconds. Tap screen to cancel.", priority = 100, severity = "CRITICAL")

        countdownRunnable = Runnable {
            if (isCountdownActive) {
                isCountdownActive = false
                dispatchActualSos(onCompleted)
            }
        }
        mainHandler.postDelayed(countdownRunnable!!, 3500L)
    }

    fun cancelSos(): Boolean {
        if (isCountdownActive) {
            isCountdownActive = false
            countdownRunnable?.let { mainHandler.removeCallbacks(it) }
            ttsManager.speak("Emergency SOS cancelled.", priority = 90, severity = "INFO")
            hapticManager.vibrateClick()
            return true
        }
        return false
    }

    @SuppressLint("MissingPermission")
    private fun dispatchActualSos(onCompleted: (Boolean, String) -> Unit) {
        val hasLocationPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(
            context, Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED

        if (hasLocationPermission) {
            fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                .addOnSuccessListener { location: Location? ->
                    val coords = if (location != null) {
                        "https://maps.google.com/?q=${location.latitude},${location.longitude}"
                    } else {
                        "Location unavailable"
                    }
                    sendEmergencySms(coords, onCompleted)
                }
                .addOnFailureListener {
                    sendEmergencySms("Location unavailable", onCompleted)
                }
        } else {
            sendEmergencySms("Location permission not granted", onCompleted)
        }
    }

    private fun sendEmergencySms(locationText: String, onCompleted: (Boolean, String) -> Unit) {
        val contactNumber = emergencyContact.trim()
        val message = "EMERGENCY SOS: Oculus AI alert from BlindCap user. I need assistance. Current location: $locationText"

        val hasSmsPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.SEND_SMS
        ) == PackageManager.PERMISSION_GRANTED

        if (hasSmsPermission) {
            try {
                @Suppress("DEPRECATION")
                val smsManager = SmsManager.getDefault()
                smsManager.sendTextMessage(contactNumber, null, message, null, null)
                hapticManager.vibrateSos()
                ttsManager.speak("Emergency alert sent with location to emergency contact.", priority = 100, severity = "CRITICAL")
                onCompleted(true, "SOS sent to $contactNumber")
            } catch (e: Exception) {
                Log.e(tag, "SMS dispatch error: ${e.message}", e)
                ttsManager.speak("Could not send SMS. Please dial emergency services.", priority = 100, severity = "CRITICAL")
                onCompleted(false, "SMS failed: ${e.message}")
            }
        } else {
            ttsManager.speak("SMS permission not granted. Please enable SMS in app settings.", priority = 90, severity = "CRITICAL")
            onCompleted(false, "SEND_SMS permission missing")
        }
    }

    fun testSosAlert(onCompleted: (String) -> Unit) {
        hapticManager.vibrateSos()
        val contact = if (hasContact()) emergencyContact else "(Not configured)"
        val msg = "Test SOS simulated successfully. Target contact: $contact. No real SMS sent."
        ttsManager.speak("Test emergency SOS alert verified successfully.", priority = 75, severity = "INFO")
        onCompleted(msg)
    }
}
