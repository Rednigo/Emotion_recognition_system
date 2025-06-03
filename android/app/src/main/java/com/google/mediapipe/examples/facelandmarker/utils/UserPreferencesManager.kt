package com.google.mediapipe.examples.facelandmarker.utils

import android.content.Context
import android.content.SharedPreferences
import java.util.UUID

class UserPreferencesManager(context: Context) {
    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences("FaceLandmarkerPrefs", Context.MODE_PRIVATE)

    companion object {
        private const val KEY_USER_ID = "user_id"
        private const val KEY_PROCESSED_CONTENT = "processed_content_"
    }

    // Get or create user ID
    fun getUserId(): String {
        var userId = sharedPreferences.getString(KEY_USER_ID, null)
        if (userId == null) {
            userId = UUID.randomUUID().toString()
            sharedPreferences.edit().putString(KEY_USER_ID, userId).apply()
        }
        return userId
    }

    // Check if content has been processed
    fun isContentProcessed(contentId: String): Boolean {
        return sharedPreferences.getBoolean(KEY_PROCESSED_CONTENT + contentId, false)
    }

    // Mark content as processed
    fun markContentAsProcessed(contentId: String) {
        sharedPreferences.edit().putBoolean(KEY_PROCESSED_CONTENT + contentId, true).apply()
    }

    // Get all processed content IDs (optional, for cleanup)
    fun getProcessedContentIds(): Set<String> {
        return sharedPreferences.all.keys
            .filter { it.startsWith(KEY_PROCESSED_CONTENT) }
            .map { it.removePrefix(KEY_PROCESSED_CONTENT) }
            .toSet()
    }
}