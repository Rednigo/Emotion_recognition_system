package com.example.emotionrecognition.utils

import android.content.Context
import android.util.Log

object ResourceChecker {
    private const val TAG = "ResourceChecker"

    fun checkAssets(context: Context): Boolean {
        return try {
            Log.d(TAG, "Checking assets...")

            val assetManager = context.assets
            val assetFiles = assetManager.list("") ?: emptyArray()

            Log.d(TAG, "Assets found: ${assetFiles.toList()}")

            // Check for MediaPipe model
            val hasMediaPipeModel = assetFiles.contains("face_landmarker.task")
            Log.d(TAG, "MediaPipe model (face_landmarker.task) present: $hasMediaPipeModel")

            if (hasMediaPipeModel) {
                // Check model file size
                val inputStream = assetManager.open("face_landmarker.task")
                val fileSize = inputStream.available()
                inputStream.close()

                Log.d(TAG, "MediaPipe model size: $fileSize bytes")

                if (fileSize < 1000) { // Model should be much larger than 1KB
                    Log.e(TAG, "MediaPipe model file seems too small, possibly corrupted")
                    return false
                }
            }

            hasMediaPipeModel
        } catch (e: Exception) {
            Log.e(TAG, "Error checking assets: ${e.message}", e)
            false
        }
    }

    fun listAllAssets(context: Context) {
        try {
            val assetManager = context.assets

            fun listAssetsRecursive(path: String, indent: String = "") {
                val files = assetManager.list(path) ?: return

                for (file in files) {
                    val fullPath = if (path.isEmpty()) file else "$path/$file"
                    Log.d(TAG, "$indent$file")

                    try {
                        // Try to list as directory
                        val subFiles = assetManager.list(fullPath)
                        if (subFiles != null && subFiles.isNotEmpty()) {
                            listAssetsRecursive(fullPath, "$indent  ")
                        }
                    } catch (e: Exception) {
                        // Not a directory, it's a file
                    }
                }
            }

            Log.d(TAG, "=== All Assets ===")
            listAssetsRecursive("")
            Log.d(TAG, "=== End Assets ===")

        } catch (e: Exception) {
            Log.e(TAG, "Error listing assets: ${e.message}")
        }
    }
}