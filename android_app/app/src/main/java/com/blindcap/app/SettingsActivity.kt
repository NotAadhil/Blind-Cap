package com.blindcap.app

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.blindcap.app.ai.FaceRecognitionManager
import com.blindcap.app.databinding.ActivitySettingsBinding
import com.blindcap.app.haptics.HapticManager
import com.blindcap.app.safety.SosManager
import com.blindcap.app.speech.TtsManager

class SettingsActivity : AppCompatActivity() {

    private val prefsName = "OculusPrefs"
    private val keySeparateModes = "pref_separate_modes"
    private val keyEnableCurrency = "pref_enable_currency_mode"
    private val keyEnableColor = "pref_enable_color_mode"
    private val keyEnableBarcode = "pref_enable_barcode_mode"
    private val keyVoiceCommands = "pref_voice_commands"
    private val keyShowDebugHud = "pref_show_debug_hud"
    private val keyHaptics = "pref_haptics_enabled"
    private val keyEmergencyContact = "pref_emergency_contact"
    private val keyStreamUrl = "esp32_stream_url"
    private val defaultStreamUrl = "http://192.168.4.1:81/stream"
    private val keyVideoSource = "pref_video_source"

    private lateinit var binding: ActivitySettingsBinding
    private lateinit var prefs: SharedPreferences
    private lateinit var ttsManager: TtsManager
    private lateinit var hapticManager: HapticManager
    private lateinit var sosManager: SosManager
    private lateinit var faceRecognitionManager: FaceRecognitionManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        hapticManager = HapticManager(this)
        ttsManager = TtsManager(this) {}
        sosManager = SosManager(this, ttsManager, hapticManager)
        faceRecognitionManager = FaceRecognitionManager(this)

        setupViews()
    }

    private fun setupViews() {
        binding.btnSettingsBack.setOnClickListener {
            hapticManager.vibrateClick()
            finish()
        }

        // Separate Modes
        binding.switchSeparateModes.isChecked = prefs.getBoolean(keySeparateModes, false)
        binding.switchSeparateModes.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(keySeparateModes, isChecked).apply()
            hapticManager.vibrateClick()
        }

        // Dedicated Mode Carousel Toggles
        binding.switchEnableCurrency.isChecked = prefs.getBoolean(keyEnableCurrency, true)
        binding.switchEnableCurrency.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(keyEnableCurrency, isChecked).apply()
            hapticManager.vibrateClick()
        }

        binding.switchEnableColor.isChecked = prefs.getBoolean(keyEnableColor, true)
        binding.switchEnableColor.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(keyEnableColor, isChecked).apply()
            hapticManager.vibrateClick()
        }

        binding.switchEnableBarcode.isChecked = prefs.getBoolean(keyEnableBarcode, true)
        binding.switchEnableBarcode.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(keyEnableBarcode, isChecked).apply()
            hapticManager.vibrateClick()
        }

        // Voice Commands
        binding.switchVoiceCommands.isChecked = prefs.getBoolean(keyVoiceCommands, true)
        binding.switchVoiceCommands.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(keyVoiceCommands, isChecked).apply()
            hapticManager.vibrateClick()
        }

        // Haptics
        binding.switchHaptics.isChecked = prefs.getBoolean(keyHaptics, true)
        binding.switchHaptics.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(keyHaptics, isChecked).apply()
            hapticManager.vibrateClick()
        }

        // Debug HUD
        binding.switchDebugHud.isChecked = prefs.getBoolean(keyShowDebugHud, false)
        binding.switchDebugHud.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(keyShowDebugHud, isChecked).apply()
            hapticManager.vibrateClick()
        }

        // Emergency Contact
        val savedContact = prefs.getString(keyEmergencyContact, "") ?: ""
        binding.editEmergencyContact.setText(savedContact)
        binding.btnSaveContact.setOnClickListener {
            val contact = binding.editEmergencyContact.text.toString().trim()
            prefs.edit().putString(keyEmergencyContact, contact).apply()
            hapticManager.vibrateClick()
            Toast.makeText(this, "Emergency contact saved: $contact", Toast.LENGTH_SHORT).show()
            ttsManager.speak("Emergency contact saved.", priority = 60, severity = "INFO")
        }

        // Test SOS
        binding.btnTestSos.setOnClickListener {
            sosManager.testSosAlert { msg ->
                Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            }
        }

        // Video Input Source
        val isPhone = prefs.getBoolean(keyVideoSource, true)
        binding.btnSettingsSourceToggle.text = if (isPhone) {
            "Video Input Source: Phone Camera"
        } else {
            "Video Input Source: ESP32 Smart Cap"
        }
        binding.btnSettingsSourceToggle.setOnClickListener {
            val current = prefs.getBoolean(keyVideoSource, true)
            val next = !current
            prefs.edit().putBoolean(keyVideoSource, next).apply()
            binding.btnSettingsSourceToggle.text = if (next) {
                "Video Input Source: Phone Camera"
            } else {
                "Video Input Source: ESP32 Smart Cap"
            }
            hapticManager.vibrateClick()
        }

        // ESP32 Stream URL
        binding.btnSettingsEsp32Url.setOnClickListener {
            showStreamUrlDialog()
        }

        // Face Contacts
        binding.btnSettingsManageFaces.setOnClickListener {
            showFaceContactsDialog()
        }
    }

    private fun showStreamUrlDialog() {
        val currentUrl = prefs.getString(keyStreamUrl, defaultStreamUrl) ?: defaultStreamUrl
        val input = EditText(this).apply {
            setText(currentUrl)
            setSelection(text.length)
            hint = "http://192.168.4.1:81/stream"
        }

        AlertDialog.Builder(this)
            .setTitle("ESP32-CAM Stream URL")
            .setMessage("Enter the MJPEG stream URL of your ESP32-CAM headwear:")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val newUrl = input.text.toString().trim()
                if (newUrl.isNotEmpty()) {
                    prefs.edit().putString(keyStreamUrl, newUrl).apply()
                    Toast.makeText(this, "Saved: $newUrl", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showFaceContactsDialog() {
        val contacts = faceRecognitionManager.getRegisteredContacts()
        val names = if (contacts.isEmpty()) {
            "No contacts enrolled yet.\n\nPoint camera at a face on the main screen to enroll."
        } else {
            contacts.mapIndexed { idx, c -> "${idx + 1}. ${c.name}" }.joinToString("\n")
        }

        val layout = TextView(this).apply {
            text = names
            textSize = 16f
            setPadding(40, 20, 40, 20)
        }

        AlertDialog.Builder(this)
            .setTitle("Registered Faces (${contacts.size})")
            .setView(layout)
            .setNeutralButton("Clear All") { _, _ ->
                faceRecognitionManager.clearAllContacts()
                Toast.makeText(this, "Cleared all face contacts", Toast.LENGTH_SHORT).show()
            }
            .setPositiveButton("Close", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        ttsManager.shutdown()
        faceRecognitionManager.close()
    }
}
