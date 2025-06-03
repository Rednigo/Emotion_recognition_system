package com.google.mediapipe.examples.facelandmarker.emotion

import android.content.Context
import android.util.Log
import com.google.mediapipe.examples.facelandmarker.network.LandmarksRequest
import com.google.mediapipe.examples.facelandmarker.network.NetworkService
import com.google.mediapipe.examples.facelandmarker.utils.UserPreferencesManager
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlinx.coroutines.*
import java.util.UUID

class EmotionProcessor(private val context: Context) {

    private val userPreferencesManager = UserPreferencesManager(context)
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    companion object {
        private const val TAG = "EmotionProcessor"
    }

    fun processLandmarks(
        landmarks: List<NormalizedLandmark>,
        onSuccess: (String, Float) -> Unit = { _, _ -> },
        onError: (Exception) -> Unit = { }
    ) {
        val contentId = UUID.randomUUID().toString()

        // Check if this content has already been processed
        if (userPreferencesManager.isContentProcessed(contentId)) {
            Log.d(TAG, "Content $contentId already processed, skipping")
            return
        }

        // Convert landmarks to the format expected by the server
        val landmarksList = landmarks.map { landmark ->
            listOf(landmark.x(), landmark.y(), landmark.z())
        }

        val request = LandmarksRequest(
            user_id = userPreferencesManager.getUserId(),
            content_id = contentId,
            landmarks = landmarksList,
            timestamp_ms = System.currentTimeMillis()
        )

        scope.launch {
            try {
                Log.d(TAG, "Sending landmarks for analysis: $contentId")
                val response = NetworkService.emotionApi.analyzeLandmarks(request)

                // Mark content as processed
                userPreferencesManager.markContentAsProcessed(contentId)

                Log.d(TAG, "Emotion analysis result: ${response.emotion} (${response.confidence})")

                withContext(Dispatchers.Main) {
                    onSuccess(response.emotion, response.confidence)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error analyzing landmarks", e)

                // Retry once
                delay(1000)
                try {
                    val response = NetworkService.emotionApi.analyzeLandmarks(request)
                    userPreferencesManager.markContentAsProcessed(contentId)

                    withContext(Dispatchers.Main) {
                        onSuccess(response.emotion, response.confidence)
                    }
                } catch (retryException: Exception) {
                    Log.e(TAG, "Retry failed", retryException)
                    withContext(Dispatchers.Main) {
                        onError(retryException)
                    }
                }
            }
        }
    }

    fun cleanup() {
        scope.cancel()
    }
}