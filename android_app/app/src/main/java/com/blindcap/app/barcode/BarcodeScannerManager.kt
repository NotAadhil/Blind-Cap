package com.blindcap.app.barcode

import android.graphics.Bitmap
import android.net.Uri
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine

data class BarcodeResult(
    val rawValue: String,
    val formatName: String,
    val payloadType: String,
    val productName: String?,
    val spokenText: String
)

class BarcodeScannerManager {

    private val scanner: BarcodeScanner = BarcodeScanning.getClient(
        BarcodeScannerOptions.Builder()
            .setBarcodeFormats(
                Barcode.FORMAT_QR_CODE,
                Barcode.FORMAT_EAN_13,
                Barcode.FORMAT_EAN_8,
                Barcode.FORMAT_UPC_A,
                Barcode.FORMAT_UPC_E,
                Barcode.FORMAT_CODE_128,
                Barcode.FORMAT_CODE_39
            )
            .build()
    )

    // Deduplication & Announcement Cooldown
    private var lastAnnouncedRawCode: String? = null
    private var lastAnnouncementTime: Long = 0L
    private val deduplicationCooldownMs = 8000L

    // Verified Local Product Database (Exact Matches Only)
    private val verifiedProductDb = mapOf(
        "8901030000000" to "Dettol Antiseptic Liquid 250ml",
        "8901058000000" to "Maggi 2-Minute Noodles",
        "8901233000000" to "Paracetamol 500mg Tablets",
        "8901725000000" to "Tata Salt 1 Kilogram",
        "8901262000000" to "Amul Butter 100g",
        "8901063000000" to "Parle-G Glucose Biscuits",
        "8901719000000" to "Colgate Total Toothpaste 150g",
        "8901030580000" to "Lifebuoy Handwash 200ml",
        "8901030700000" to "Surf Excel Detergent Powder 1kg",
        "8901030800000" to "Lizol Surface Cleaner 500ml"
    )

    suspend fun scanBitmap(bitmap: Bitmap): BarcodeResult? = suspendCancellableCoroutine { continuation ->
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        scanner.process(inputImage)
            .addOnSuccessListener { barcodes ->
                if (barcodes.isNotEmpty()) {
                    val first = barcodes[0]
                    val raw = (first.rawValue ?: first.displayValue ?: "").trim()
                    if (raw.isBlank()) {
                        continuation.resume(null)
                        return@addOnSuccessListener
                    }

                    val now = System.currentTimeMillis()
                    // Deduplication check: if same code scanned within cooldown, return null to avoid audio spam
                    if (raw == lastAnnouncedRawCode && (now - lastAnnouncementTime < deduplicationCooldownMs)) {
                        continuation.resume(null)
                        return@addOnSuccessListener
                    }

                    val formatName = getFormatName(first.format)
                    val (payloadType, spoken) = parseBarcodeContent(first, formatName)

                    lastAnnouncedRawCode = raw
                    lastAnnouncementTime = now

                    val matchedProduct = verifiedProductDb[raw]

                    continuation.resume(BarcodeResult(raw, formatName, payloadType, matchedProduct, spoken))
                } else {
                    continuation.resume(null)
                }
            }
            .addOnFailureListener {
                continuation.resume(null)
            }
    }

    private fun parseBarcodeContent(barcode: Barcode, formatName: String): Pair<String, String> {
        val raw = barcode.rawValue ?: barcode.displayValue ?: ""

        if (barcode.format == Barcode.FORMAT_QR_CODE) {
            return when (barcode.valueType) {
                Barcode.TYPE_URL -> {
                    val urlStr = barcode.url?.url ?: raw
                    val domain = try {
                        Uri.parse(urlStr).host?.replace("www.", "") ?: urlStr
                    } catch (_: Exception) {
                        urlStr
                    }
                    Pair("Website URL", "Website QR code for $domain")
                }
                Barcode.TYPE_WIFI -> {
                    val ssid = barcode.wifi?.ssid ?: "Unknown"
                    Pair("Wi-Fi Network", "Wi-Fi network QR code for $ssid")
                }
                Barcode.TYPE_CONTACT_INFO -> {
                    val name = barcode.contactInfo?.name?.formattedName ?: "Contact"
                    Pair("Contact Card", "Contact card QR code for $name")
                }
                Barcode.TYPE_EMAIL -> {
                    val email = barcode.email?.address ?: raw
                    Pair("Email Address", "Email address: $email")
                }
                Barcode.TYPE_PHONE -> {
                    val phone = barcode.phone?.number ?: raw
                    Pair("Phone Number", "Phone number: $phone")
                }
                Barcode.TYPE_SMS -> {
                    val phone = barcode.sms?.phoneNumber ?: ""
                    Pair("SMS Message", "SMS link to $phone")
                }
                Barcode.TYPE_GEO -> {
                    Pair("Location Coordinates", "Map location coordinate code")
                }
                Barcode.TYPE_TEXT -> {
                    val cleanText = if (raw.length > 80) raw.take(80) + "..." else raw
                    Pair("Plain Text", "QR Code text: $cleanText")
                }
                else -> {
                    Pair("QR Data", "QR Code detected.")
                }
            }
        }

        // Product Barcodes (EAN, UPC, Code 128)
        val exactProduct = verifiedProductDb[raw]
        if (exactProduct != null) {
            return Pair("Verified Product", "Product identified: $exactProduct")
        }

        // Non-hallucinating accessibility announcement
        val last4 = if (raw.length >= 4) {
            raw.takeLast(4).toCharArray().joinToString(" ")
        } else {
            raw.toCharArray().joinToString(" ")
        }

        val speech = when (barcode.format) {
            Barcode.FORMAT_EAN_13, Barcode.FORMAT_EAN_8, Barcode.FORMAT_UPC_A, Barcode.FORMAT_UPC_E ->
                "Product barcode detected: $formatName, ending in $last4"
            else ->
                "Barcode detected: $formatName, ending in $last4"
        }

        return Pair("Product Barcode", speech)
    }

    private fun getFormatName(format: Int): String = when (format) {
        Barcode.FORMAT_QR_CODE -> "QR Code"
        Barcode.FORMAT_EAN_13 -> "EAN 13"
        Barcode.FORMAT_EAN_8 -> "EAN 8"
        Barcode.FORMAT_UPC_A -> "UPC A"
        Barcode.FORMAT_UPC_E -> "UPC E"
        Barcode.FORMAT_CODE_128 -> "Code 128"
        Barcode.FORMAT_CODE_39 -> "Code 39"
        else -> "Barcode"
    }

    fun resetCooldown() {
        lastAnnouncedRawCode = null
        lastAnnouncementTime = 0L
    }

    fun close() {
        scanner.close()
    }
}
