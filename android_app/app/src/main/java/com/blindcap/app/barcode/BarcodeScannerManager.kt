package com.blindcap.app.barcode

import android.graphics.Bitmap
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
    val productName: String?,
    val spokenText: String
)

class BarcodeScannerManager {

    private val scanner: BarcodeScanner = BarcodeScanning.getClient(
        BarcodeScannerOptions.Builder()
            .setBarcodeFormats(
                Barcode.FORMAT_QR_CODE,
                Barcode.FORMAT_EAN_13,
                Barcode.FORMAT_UPC_A,
                Barcode.FORMAT_CODE_128,
                Barcode.FORMAT_CODE_39
            )
            .build()
    )

    private val localProductDb = mapOf(
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
                    val raw = first.rawValue ?: first.displayValue ?: ""
                    val formatName = getFormatName(first.format)

                    val matchedProduct = matchProduct(raw)
                    val spoken = if (matchedProduct != null) {
                        "Product found: $matchedProduct"
                    } else if (first.format == Barcode.FORMAT_QR_CODE) {
                        "QR Code: $raw"
                    } else {
                        "Barcode detected: $raw"
                    }

                    continuation.resume(BarcodeResult(raw, formatName, matchedProduct, spoken))
                } else {
                    continuation.resume(null)
                }
            }
            .addOnFailureListener {
                continuation.resume(null)
            }
    }

    private fun matchProduct(barcode: String): String? {
        val clean = barcode.trim()
        localProductDb[clean]?.let { return it }
        for ((key, prod) in localProductDb) {
            if (clean.startsWith(key.substring(0, 6))) {
                return prod
            }
        }
        return null
    }

    private fun getFormatName(format: Int): String = when (format) {
        Barcode.FORMAT_QR_CODE -> "QR Code"
        Barcode.FORMAT_EAN_13 -> "EAN-13"
        Barcode.FORMAT_UPC_A -> "UPC-A"
        Barcode.FORMAT_CODE_128 -> "Code 128"
        Barcode.FORMAT_CODE_39 -> "Code 39"
        else -> "Barcode"
    }

    fun close() {
        scanner.close()
    }
}
