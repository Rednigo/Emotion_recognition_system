package com.example.emotionrecognition.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.emotionrecognition.MainActivity
import com.example.emotionrecognition.R
import com.google.gson.Gson
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.example.emotionrecognition.network.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*

class EmotionAnalysisService : LifecycleService() {

    companion object {
        private const val TAG = "EmotionAnalysisService"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "emotion_analysis_channel"
        private const val CHANNEL_NAME = "Emotion Analysis Service"

        // Intent actions
        const val ACTION_EMOTION_UPDATE = "com.example.emotionrecognition.EMOTION_UPDATE"
        const val EXTRA_EMOTION_NAME = "emotion_name"
        const val EXTRA_EMOTION_CONFIDENCE = "emotion_confidence"
        const val EXTRA_EMOTION_METRICS = "emotion_metrics"

        // MediaPipe
        private const val MP_FACE_LANDMARKER_TASK = "face_landmarker.task"

        // Налаштування як в офіційному зразку
        const val DEFAULT_FACE_DETECTION_CONFIDENCE = 0.5f
        const val DEFAULT_FACE_TRACKING_CONFIDENCE = 0.5f
        const val DEFAULT_FACE_PRESENCE_CONFIDENCE = 0.5f

        // Selected key landmarks (27 vertices)
        private val SELECTED_LANDMARKS = mapOf(
            0 to 61, 1 to 292, 2 to 0, 3 to 17, 4 to 50, 5 to 280, 6 to 48, 7 to 4, 8 to 289,
            9 to 206, 10 to 426, 11 to 133, 12 to 130, 13 to 159, 14 to 145, 15 to 362, 16 to 359,
            17 to 386, 18 to 374, 19 to 122, 20 to 351, 21 to 46, 22 to 105, 23 to 107, 24 to 276,
            25 to 334, 26 to 336
        )

        // Angular features (10 angles)
        private val ANGLE_TRIPLETS = listOf(
            Triple(2, 0, 3), Triple(0, 2, 1), Triple(6, 7, 8), Triple(9, 7, 10), Triple(0, 7, 1),
            Triple(1, 5, 8), Triple(1, 10, 8), Triple(13, 12, 14), Triple(21, 22, 23), Triple(6, 19, 23)
        )
    }

    private var faceLandmarker: FaceLandmarker? = null
    private val gson = Gson()
    private val isServiceActive = AtomicBoolean(true)
    private var noFaceFrameCount = 0

    // Налаштування MediaPipe
    private var minFaceDetectionConfidence = DEFAULT_FACE_DETECTION_CONFIDENCE
    private var minFaceTrackingConfidence = DEFAULT_FACE_TRACKING_CONFIDENCE
    private var minFacePresenceConfidence = DEFAULT_FACE_PRESENCE_CONFIDENCE

    // BroadcastReceiver для отримання зображень з MainActivity
    private val imageReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == MainActivity.ACTION_SEND_IMAGE && isServiceActive.get()) {
                try {
                    val imageBytes = intent.getByteArrayExtra(MainActivity.EXTRA_IMAGE_DATA)
                    val imageType = intent.getStringExtra("image_type") ?: "unknown"

                    if (imageBytes != null && imageBytes.isNotEmpty()) {
                        Log.d(TAG, "Received image for analysis: ${imageBytes.size} bytes [$imageType]")

                        val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                        if (bitmap != null) {
                            Log.d(TAG, "Successfully decoded bitmap: ${bitmap.width}x${bitmap.height}")

                            // Обробляємо як в офіційному зразку MediaPipe
                            processImageLikeMediaPipeExample(bitmap, imageType)
                        } else {
                            Log.e(TAG, "Failed to decode bitmap from byte array")
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing received image: ${e.message}", e)
                }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()

        Log.d(TAG, "Service onCreate() - Starting MediaPipe emotion analysis service")

        createNotificationChannel()
        val notification = createNotification("Ініціалізація MediaPipe...")
        startForeground(NOTIFICATION_ID, notification)

        // Реєстрація BroadcastReceiver
        val filter = IntentFilter(MainActivity.ACTION_SEND_IMAGE)
        LocalBroadcastManager.getInstance(this).registerReceiver(imageReceiver, filter)

        // Ініціалізація MediaPipe як в офіційному зразку
        setupFaceLandmarker()
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Service onDestroy() - Starting shutdown")

        isServiceActive.set(false)

        try {
            LocalBroadcastManager.getInstance(this).unregisterReceiver(imageReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Error unregistering image receiver: ${e.message}")
        }

        clearFaceLandmarker()
        Log.d(TAG, "Service onDestroy() - Shutdown completed")
    }

    // Метод ініціалізації як в офіційному зразку MediaPipe
    private fun setupFaceLandmarker() {
        try {
            Log.d(TAG, "Setting up FaceLandmarker like official MediaPipe example...")

            val baseOptionBuilder = BaseOptions.builder()
                .setDelegate(Delegate.CPU)
                .setModelAssetPath(MP_FACE_LANDMARKER_TASK)

            val optionsBuilder = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptionBuilder.build())
                .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
                .setMinTrackingConfidence(minFaceTrackingConfidence)
                .setMinFacePresenceConfidence(minFacePresenceConfidence)
                .setNumFaces(1)
                .setOutputFaceBlendshapes(false) // Відключаємо blendshapes
                .setRunningMode(RunningMode.LIVE_STREAM) // LIVE_STREAM як в офіційному зразку
                .setResultListener(this::returnLivestreamResult)
                .setErrorListener(this::returnLivestreamError)

            val options = optionsBuilder.build()
            faceLandmarker = FaceLandmarker.createFromOptions(this, options)

            if (faceLandmarker != null) {
                Log.d(TAG, "MediaPipe FaceLandmarker initialized successfully (LIVE_STREAM mode)")
                updateNotification("MediaPipe готовий до аналізу...")
            } else {
                Log.e(TAG, "FaceLandmarker creation returned null")
                updateNotification("Помилка створення FaceLandmarker")
                sendEmotionUpdate("error", 0f, mapOf<String, Any>("error" to "FaceLandmarker creation failed"))
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup MediaPipe FaceLandmarker", e)
            updateNotification("Помилка ініціалізації MediaPipe: ${e.message}")
            sendEmotionUpdate("error", 0f, mapOf<String, Any>("error" to (e.message ?: "Setup error")))
        }
    }

    private fun clearFaceLandmarker() {
        try {
            faceLandmarker?.close()
            faceLandmarker = null
            Log.d(TAG, "FaceLandmarker closed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing FaceLandmarker: ${e.message}")
        }
    }

    // Обробка зображення як в офіційному зразку MediaPipe
    private fun processImageLikeMediaPipeExample(bitmap: Bitmap, imageType: String) {
        if (!isServiceActive.get()) {
            Log.d(TAG, "Service inactive, skipping analysis")
            return
        }

        try {
            val landmarker = faceLandmarker
            if (landmarker == null) {
                Log.e(TAG, "FaceLandmarker not initialized")
                sendEmotionUpdate("error", 0f, mapOf<String, Any>("error" to "FaceLandmarker not initialized"))
                return
            }

            Log.d(TAG, "Processing image like MediaPipe example for [$imageType]...")

            // Застосовуємо трансформації як в офіційному зразку
            val processedBitmap = applyMediaPipeTransformations(bitmap, isFrontCamera = true)

            // Конвертуємо в MPImage
            val mpImage = BitmapImageBuilder(processedBitmap).build()
            val frameTime = SystemClock.uptimeMillis()

            Log.d(TAG, "Calling detectAsync with frameTime: $frameTime")

            // Використовуємо detectAsync як в офіційному зразку
            landmarker.detectAsync(mpImage, frameTime)

            // Результат прийде в returnLivestreamResult callback

        } catch (e: Exception) {
            Log.e(TAG, "Error processing image [$imageType]: ${e.message}", e)
            if (isServiceActive.get()) {
                sendEmotionUpdate("error", 0f, mapOf<String, Any>("error" to "Processing error: ${e.message}"))
                updateNotification("Помилка обробки: ${e.message}")
            }
        }
    }

    // Трансформації зображення як в офіційному зразку MediaPipe
    private fun applyMediaPipeTransformations(bitmap: Bitmap, isFrontCamera: Boolean): Bitmap {
        return try {
            Log.d(TAG, "Applying MediaPipe transformations...")

            val matrix = Matrix().apply {
                // Ротація (якщо потрібна) - в нашому випадку вже зроблена в MainActivity
                // postRotate(rotationDegrees.toFloat())

                // Flip для фронтальної камери як в офіційному зразку
                if (isFrontCamera) {
                    postScale(-1f, 1f, bitmap.width.toFloat(), bitmap.height.toFloat())
                    Log.d(TAG, "Applied front camera flip")
                }
            }

            val transformedBitmap = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )

            Log.d(TAG, "Transformations applied: ${bitmap.width}x${bitmap.height} -> ${transformedBitmap.width}x${transformedBitmap.height}")
            transformedBitmap

        } catch (e: Exception) {
            Log.e(TAG, "Error applying transformations: ${e.message}")
            bitmap
        }
    }

    // Callback для результатів LIVE_STREAM (як в офіційному зразку)
    private fun returnLivestreamResult(result: FaceLandmarkerResult, input: MPImage) {
        if (!isServiceActive.get()) {
            Log.d(TAG, "Service inactive, ignoring livestream result")
            return
        }

        Log.d(TAG, "🎉 MediaPipe livestream result received! Faces: ${result.faceLandmarks().size}")

        if (result.faceLandmarks().size > 0) {
            noFaceFrameCount = 0

            val finishTimeMs = SystemClock.uptimeMillis()
            val inferenceTime = finishTimeMs - result.timestampMs()

            Log.d(TAG, "✅ SUCCESS: Face detected! Processing landmarks... (inference: ${inferenceTime}ms)")

            try {
                val faceLandmarks = result.faceLandmarks()[0]
                Log.d(TAG, "Face landmarks count: ${faceLandmarks.size}")

                // Обробляємо landmarks для аналізу емоцій
                processLandmarksForEmotionAnalysis(faceLandmarks, inferenceTime)

            } catch (e: Exception) {
                Log.e(TAG, "Error processing landmarks: ${e.message}", e)
                sendEmotionUpdate("error", 0f, mapOf("error" to "Landmarks processing error: ${e.message}"))
            }
        } else {
            handleNoFaceDetected()
        }
    }

    // Callback для помилок LIVE_STREAM
    private fun returnLivestreamError(error: RuntimeException) {
        Log.e(TAG, "MediaPipe livestream error: ${error.message}")
        if (isServiceActive.get()) {
            sendEmotionUpdate("error", 0f, mapOf<String, Any>("error" to (error.message ?: "Unknown MediaPipe error")))
            updateNotification("Помилка MediaPipe: ${error.message}")
        }
    }

    private fun handleNoFaceDetected() {
        noFaceFrameCount++
        Log.d(TAG, "No faces detected (count: $noFaceFrameCount)")

        if (noFaceFrameCount % 5 == 0) {
            val suggestion = when {
                noFaceFrameCount > 20 -> "Перевірте освітлення та положення обличчя"
                noFaceFrameCount > 10 -> "Тримайте обличчя прямо перед камерою"
                else -> "Наведіться на камеру"
            }

            sendEmotionUpdate("neutral", 0.0f, mapOf<String, Any>(
                "status" to "no_face_detected",
                "attempt_count" to noFaceFrameCount,
                "suggestion" to suggestion
            ))
            updateNotification("Обличчя не виявлено - $suggestion")
        }
    }

    private fun processLandmarksForEmotionAnalysis(landmarks: List<NormalizedLandmark>, inferenceTime: Long) {
        try {
            // Extract 27 key landmarks
            val keyLandmarks = selectKeyLandmarks(landmarks)
            Log.d(TAG, "Selected ${keyLandmarks.size} key landmarks")

            // Calculate 10 angular features
            val angularFeatures = extractAngularFeatures(keyLandmarks)
            val validFeatures = angularFeatures.count { !it.isNaN() && it.isFinite() }
            Log.d(TAG, "Calculated angular features: valid $validFeatures/${angularFeatures.size}")

            // Prepare metrics map
            val metricsMap = HashMap<String, Any>()

            // Add angular features
            for (i in angularFeatures.indices) {
                metricsMap["angle_${i + 1}"] = angularFeatures[i]
            }

            // Add additional metrics for UI
            metricsMap["mediapipe_landmarks"] = landmarks.size
            metricsMap["angular_features"] = angularFeatures.toList()
            metricsMap["valid_features"] = validFeatures
            metricsMap["inference_time_ms"] = inferenceTime

            // Calculate display metrics
            val displayMetrics = calculateDisplayMetrics(landmarks)
            metricsMap.putAll(displayMetrics)

            // Emotion estimation
            val emotionAnalysis = estimateEmotionFromAngles(angularFeatures)
            Log.d(TAG, "Emotion analysis result: ${emotionAnalysis.first} with confidence ${emotionAnalysis.second}")

            // Send update
            sendEmotionUpdate(emotionAnalysis.first, emotionAnalysis.second, metricsMap)

            // Update notification
            updateNotification("Емоція: ${translateEmotion(emotionAnalysis.first)} (${(emotionAnalysis.second * 100).toInt()}%)")

            // Send to server if enough valid features
            if (validFeatures >= 8) {
                sendToServerForAnalysis(angularFeatures, metricsMap)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error in emotion analysis: ${e.message}", e)
            sendEmotionUpdate("error", 0f, mapOf<String, Any>("error" to "Emotion analysis error: ${e.message}"))
        }
    }

    private fun selectKeyLandmarks(allLandmarks: List<NormalizedLandmark>): List<FloatArray> {
        val keyLandmarks = mutableListOf<FloatArray>()

        for ((vertexId, mediapipeId) in SELECTED_LANDMARKS) {
            if (mediapipeId < allLandmarks.size) {
                val landmark = allLandmarks[mediapipeId]
                keyLandmarks.add(floatArrayOf(landmark.x(), landmark.y()))
            } else {
                keyLandmarks.add(floatArrayOf(0.0f, 0.0f))
            }
        }

        return keyLandmarks
    }

    private fun extractAngularFeatures(keyLandmarks: List<FloatArray>): FloatArray {
        val angularFeatures = FloatArray(10)

        for (i in ANGLE_TRIPLETS.indices) {
            val (v1, v2, v3) = ANGLE_TRIPLETS[i]

            if (v1 < keyLandmarks.size && v2 < keyLandmarks.size && v3 < keyLandmarks.size) {
                try {
                    val p1 = keyLandmarks[v1]
                    val p2 = keyLandmarks[v2]
                    val p3 = keyLandmarks[v3]

                    if (p1.all { it.isFinite() } && p2.all { it.isFinite() } && p3.all { it.isFinite() }) {
                        angularFeatures[i] = calculateAngleBetweenPoints(p1, p2, p3)

                        if (!angularFeatures[i].isFinite()) {
                            angularFeatures[i] = 0.0f
                        }
                    } else {
                        angularFeatures[i] = 0.0f
                    }
                } catch (e: Exception) {
                    angularFeatures[i] = 0.0f
                }
            } else {
                angularFeatures[i] = 0.0f
            }
        }

        return angularFeatures
    }

    private fun calculateAngleBetweenPoints(p1: FloatArray, p2: FloatArray, p3: FloatArray): Float {
        return try {
            if (p1.size < 2 || p2.size < 2 || p3.size < 2) return 0.0f

            val deltaY_beta = p3[1] - p2[1]
            val deltaX_beta = p3[0] - p2[0]
            val deltaY_alpha = p1[1] - p2[1]
            val deltaX_alpha = p1[0] - p2[0]

            if (abs(deltaX_beta) < 1e-6f && abs(deltaY_beta) < 1e-6f) return 0.0f
            if (abs(deltaX_alpha) < 1e-6f && abs(deltaY_alpha) < 1e-6f) return 0.0f

            val beta = atan2(deltaY_beta, deltaX_beta)
            val alpha = atan2(deltaY_alpha, deltaX_alpha)
            var theta = beta - alpha

            var thetaDegrees = Math.toDegrees(theta.toDouble()).toFloat()
            while (thetaDegrees < 0) thetaDegrees += 360f
            while (thetaDegrees >= 360f) thetaDegrees -= 360f

            if (!thetaDegrees.isFinite()) return 0.0f
            thetaDegrees
        } catch (e: Exception) {
            0.0f
        }
    }

    private fun calculateDisplayMetrics(landmarks: List<NormalizedLandmark>): Map<String, Float> {
        val metrics = HashMap<String, Float>()

        try {
            // Eye openness
            metrics["leftEyeOpenness"] = calculateEyeOpenness(landmarks, true)
            metrics["rightEyeOpenness"] = calculateEyeOpenness(landmarks, false)

            // Smile metric
            metrics["smileMetric"] = calculateSmileMetric(landmarks)

            // Eyebrow raise
            metrics["leftEyebrowRaise"] = calculateEyebrowRaise(landmarks, true)
            metrics["rightEyebrowRaise"] = calculateEyebrowRaise(landmarks, false)

            // Mouth openness
            metrics["mouthOpenness"] = calculateMouthOpenness(landmarks)

            // Head angles
            val headAngles = estimateHeadAngles(landmarks)
            metrics["headEulerAngleX"] = headAngles[0]
            metrics["headEulerAngleY"] = headAngles[1]
            metrics["headEulerAngleZ"] = headAngles[2]

        } catch (e: Exception) {
            Log.e(TAG, "Error calculating display metrics: ${e.message}")
        }

        return metrics
    }

    private fun estimateEmotionFromAngles(angles: FloatArray): Pair<String, Float> {
        val mouthAngle1 = angles[0]
        val mouthAngle2 = angles[1]
        val browAngle = angles[8]

        return when {
            mouthAngle1 > 180 && mouthAngle2 < 180 -> "happy" to 0.7f
            mouthAngle1 < 150 -> "sad" to 0.6f
            browAngle > 200 && mouthAngle2 > 200 -> "surprised" to 0.65f
            browAngle < 150 && mouthAngle1 < 180 -> "angry" to 0.6f
            else -> "neutral" to 0.5f
        }
    }

    private fun sendEmotionUpdate(emotion: String, confidence: Float, metrics: Map<String, Any>) {
        if (!isServiceActive.get()) return

        try {
            val intent = Intent(ACTION_EMOTION_UPDATE).apply {
                putExtra(EXTRA_EMOTION_NAME, emotion)
                putExtra(EXTRA_EMOTION_CONFIDENCE, confidence)
                putExtra(EXTRA_EMOTION_METRICS, gson.toJson(metrics))
            }

            LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
            Log.d(TAG, "Emotion update sent: $emotion (confidence: $confidence)")
        } catch (e: Exception) {
            Log.e(TAG, "Error sending emotion update: ${e.message}")
        }
    }

    private fun sendToServerForAnalysis(angularFeatures: FloatArray, localMetrics: Map<String, Any>) {
        if (!isServiceActive.get()) return

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val request = ApiClient.createRequest(
                    angularFeatures = angularFeatures,
                    additionalMetadata = mapOf(
                        "device" to android.os.Build.MODEL,
                        "timestamp" to System.currentTimeMillis()
                    )
                )

                val response = ApiClient.apiService.analyzeEmotion(request)

                if (response.isSuccessful) {
                    val emotionResponse = response.body()
                    if (emotionResponse != null) {
                        withContext(Dispatchers.Main) {
                            val serverMetrics = localMetrics.toMutableMap()
                            serverMetrics["server_emotion"] = emotionResponse.emotion
                            serverMetrics["server_confidence"] = emotionResponse.confidence
                            serverMetrics["emotion_scores"] = emotionResponse.emotionScores

                            sendEmotionUpdate(
                                emotionResponse.emotion,
                                emotionResponse.confidence,
                                serverMetrics
                            )
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error sending to server: ${e.message}")
            }
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, CHANNEL_NAME, NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Канал для сервісу аналізу емоцій"
        }
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Аналіз емоцій MediaPipe")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        if (!isServiceActive.get()) return
        try {
            val notification = createNotification(text)
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            Log.e(TAG, "Error updating notification: ${e.message}")
        }
    }

    private fun translateEmotion(emotion: String): String {
        return when(emotion.lowercase()) {
            "happy", "happiness" -> "Щасливий"
            "sad", "sadness" -> "Сумний"
            "angry", "anger" -> "Злий"
            "surprised", "surprise" -> "Здивований"
            "disgusted", "disgust" -> "Відраза"
            "fearful", "fear" -> "Страх"
            "neutral" -> "Нейтральний"
            "error" -> "Помилка"
            else -> emotion
        }
    }

    // Helper methods for display metrics
    private fun calculateEyeOpenness(landmarks: List<NormalizedLandmark>, isLeft: Boolean): Float {
        return try {
            val upperEyeLid = if (isLeft) 159 else 386
            val lowerEyeLid = if (isLeft) 145 else 374

            if (upperEyeLid >= landmarks.size || lowerEyeLid >= landmarks.size) return 0.5f

            val distance = calculateDistance(landmarks[upperEyeLid], landmarks[lowerEyeLid])
            (distance * 100).coerceIn(0f, 1f)
        } catch (e: Exception) { 0.5f }
    }

    private fun calculateSmileMetric(landmarks: List<NormalizedLandmark>): Float {
        return try {
            val mouthLeft = SELECTED_LANDMARKS[0] ?: return 0.5f
            val mouthRight = SELECTED_LANDMARKS[1] ?: return 0.5f
            val mouthTop = SELECTED_LANDMARKS[2] ?: return 0.5f
            val mouthBottom = SELECTED_LANDMARKS[3] ?: return 0.5f

            if (mouthLeft >= landmarks.size || mouthRight >= landmarks.size ||
                mouthTop >= landmarks.size || mouthBottom >= landmarks.size) return 0.5f

            val mouthCornerDistance = calculateDistance(landmarks[mouthLeft], landmarks[mouthRight])
            val mouthCenter = (landmarks[mouthTop].y() + landmarks[mouthBottom].y()) / 2
            val leftCornerHeight = mouthCenter - landmarks[mouthLeft].y()
            val rightCornerHeight = mouthCenter - landmarks[mouthRight].y()
            val smileCurvature = (leftCornerHeight + rightCornerHeight) / 2

            ((mouthCornerDistance * 2) + (smileCurvature * 10)).coerceIn(0f, 1f)
        } catch (e: Exception) { 0.5f }
    }

    private fun calculateEyebrowRaise(landmarks: List<NormalizedLandmark>, isLeft: Boolean): Float {
        return try {
            val eyebrowMiddle = if (isLeft) SELECTED_LANDMARKS[22] else SELECTED_LANDMARKS[25]
            val eyeTop = if (isLeft) SELECTED_LANDMARKS[13] else SELECTED_LANDMARKS[17]

            if (eyebrowMiddle == null || eyeTop == null ||
                eyebrowMiddle >= landmarks.size || eyeTop >= landmarks.size) return 0.5f

            val distance = landmarks[eyebrowMiddle].y() - landmarks[eyeTop].y()
            (distance * 10).coerceIn(0f, 1f)
        } catch (e: Exception) { 0.5f }
    }

    private fun calculateMouthOpenness(landmarks: List<NormalizedLandmark>): Float {
        return try {
            val mouthTop = SELECTED_LANDMARKS[2] ?: return 0.5f
            val mouthBottom = SELECTED_LANDMARKS[3] ?: return 0.5f

            if (mouthTop >= landmarks.size || mouthBottom >= landmarks.size) return 0.5f

            val distance = calculateDistance(landmarks[mouthTop], landmarks[mouthBottom])
            (distance * 20).coerceIn(0f, 1f)
        } catch (e: Exception) { 0.5f }
    }

    private fun calculateDistance(p1: NormalizedLandmark, p2: NormalizedLandmark): Float {
        return sqrt((p1.x() - p2.x()).pow(2) + (p1.y() - p2.y()).pow(2) + (p1.z() - p2.z()).pow(2))
    }

    private fun estimateHeadAngles(landmarks: List<NormalizedLandmark>): FloatArray {
        return try {
            val noseTip = SELECTED_LANDMARKS[7] ?: return floatArrayOf(0f, 0f, 0f)
            val leftEyeInner = SELECTED_LANDMARKS[11] ?: return floatArrayOf(0f, 0f, 0f)
            val rightEyeInner = SELECTED_LANDMARKS[15] ?: return floatArrayOf(0f, 0f, 0f)

            if (noseTip >= landmarks.size || leftEyeInner >= landmarks.size || rightEyeInner >= landmarks.size) {
                return floatArrayOf(0f, 0f, 0f)
            }

            val noseTipLandmark = landmarks[noseTip]
            val leftEye = landmarks[leftEyeInner]
            val rightEye = landmarks[rightEyeInner]

            val eyeCenter = (leftEye.x() + rightEye.x()) / 2
            val yaw = (noseTipLandmark.x() - eyeCenter) * 50

            val eyeHeight = (leftEye.y() + rightEye.y()) / 2
            val pitch = (noseTipLandmark.y() - eyeHeight) * 50

            val roll = (rightEye.y() - leftEye.y()) * 50

            floatArrayOf(pitch, yaw, roll)
        } catch (e: Exception) {
            floatArrayOf(0f, 0f, 0f)
        }
    }
}