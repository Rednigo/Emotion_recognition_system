package com.example.emotionrecognition.model

import com.google.gson.annotations.SerializedName

/**
 * Клас для зберігання метрик обличчя від MediaPipe Face Mesh
 */
data class FaceMetrics(
    @SerializedName("faceWidth")
    val faceWidth: Int,

    @SerializedName("faceHeight")
    val faceHeight: Int,

    @SerializedName("smileMetric")
    val smileMetric: Float,

    @SerializedName("leftEyeOpenness")
    val leftEyeOpenness: Float,

    @SerializedName("rightEyeOpenness")
    val rightEyeOpenness: Float,

    @SerializedName("mouthOpenness")
    val mouthOpenness: Float,

    @SerializedName("leftEyebrowRaise")
    val leftEyebrowRaise: Float,

    @SerializedName("rightEyebrowRaise")
    val rightEyebrowRaise: Float,

    @SerializedName("mediapipe_landmarks")
    val mediapipeLandmarks: Map<String, List<Float>>,

    @SerializedName("headEulerAngleX")
    val headEulerAngleX: Float,

    @SerializedName("headEulerAngleY")
    val headEulerAngleY: Float,

    @SerializedName("headEulerAngleZ")
    val headEulerAngleZ: Float
)

/**
 * Індекси ключових точок MediaPipe Face Mesh для швидкого доступу
 */
object MediaPipeLandmarks {
    // Очі
    const val LEFT_EYE_INNER = 133
    const val LEFT_EYE_OUTER = 33
    const val LEFT_EYE_TOP = 159
    const val LEFT_EYE_BOTTOM = 145
    const val LEFT_PUPIL = 468

    const val RIGHT_EYE_INNER = 362
    const val RIGHT_EYE_OUTER = 263
    const val RIGHT_EYE_TOP = 386
    const val RIGHT_EYE_BOTTOM = 374
    const val RIGHT_PUPIL = 473

    // Брови
    const val LEFT_EYEBROW_INNER = 46
    const val LEFT_EYEBROW_OUTER = 70
    const val LEFT_EYEBROW_TOP = 63

    const val RIGHT_EYEBROW_INNER = 276
    const val RIGHT_EYEBROW_OUTER = 300
    const val RIGHT_EYEBROW_TOP = 293

    // Ніс
    const val NOSE_TIP = 1
    const val NOSE_BOTTOM = 2
    const val NOSE_LEFT = 129
    const val NOSE_RIGHT = 358

    // Рот
    const val MOUTH_LEFT = 61
    const val MOUTH_RIGHT = 291
    const val MOUTH_TOP = 13
    const val MOUTH_BOTTOM = 14
    const val UPPER_LIP_TOP = 12
    const val LOWER_LIP_BOTTOM = 15

    // Контур обличчя
    const val CHIN = 152
    const val LEFT_CHEEK = 116
    const val RIGHT_CHEEK = 345
    const val FOREHEAD = 9
}

/**
 * Запит на аналіз метрик обличчя
 */
data class MetricsRequest(
    @SerializedName("metrics")
    val metrics: String
)

/**
 * Відповідь сервера на аналіз емоцій
 */
data class EmotionResponse(
    @SerializedName("emotion")
    val emotion: String,

    @SerializedName("confidence")
    val confidence: Float,

    @SerializedName("recommendation")
    val recommendation: String
)

/**
 * Розширена відповідь з детальним аналізом емоцій
 */
data class DetailedEmotionResponse(
    @SerializedName("primary_emotion")
    val primaryEmotion: EmotionResponse,

    @SerializedName("emotion_scores")
    val emotionScores: Map<String, Float>,

    @SerializedName("facial_expressions")
    val facialExpressions: Map<String, Float>,

    @SerializedName("timestamp")
    val timestamp: Long
)