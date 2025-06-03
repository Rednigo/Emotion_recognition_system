package com.example.emotionrecognition.utils

import com.example.emotionrecognition.model.MediaPipeLandmarks
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Допоміжний клас для аналізу емоцій на основі MediaPipe Face Mesh landmarks
 */
class MediaPipeEmotionAnalyzer {

    companion object {
        // Порогові значення для визначення емоцій
        private const val SMILE_THRESHOLD = 0.7f
        private const val SAD_THRESHOLD = 0.3f
        private const val SURPRISED_THRESHOLD = 0.6f
        private const val ANGRY_THRESHOLD = 0.3f
        private const val FEAR_THRESHOLD = 0.5f

        // Вагові коефіцієнти для різних метрик
        private const val MOUTH_WEIGHT = 0.4f
        private const val EYES_WEIGHT = 0.3f
        private const val EYEBROWS_WEIGHT = 0.3f
    }

    /**
     * Комплексний аналіз емоцій на основі всіх доступних метрик
     */
    fun analyzeEmotions(landmarks: List<NormalizedLandmark>): Map<String, Float> {
        val emotions = mutableMapOf<String, Float>()

        // Обчислення базових метрик
        val smileScore = calculateSmileScore(landmarks)
        val eyeMetrics = calculateEyeMetrics(landmarks)
        val eyebrowMetrics = calculateEyebrowMetrics(landmarks)
        val mouthMetrics = calculateMouthMetrics(landmarks)
        val symmetryScore = calculateFaceSymmetry(landmarks)

        // Щастя
        emotions["happy"] = calculateHappyScore(smileScore, eyeMetrics, mouthMetrics)

        // Сум
        emotions["sad"] = calculateSadScore(smileScore, eyeMetrics, eyebrowMetrics, mouthMetrics)

        // Здивування
        emotions["surprised"] = calculateSurprisedScore(eyeMetrics, eyebrowMetrics, mouthMetrics)

        // Злість
        emotions["angry"] = calculateAngryScore(eyebrowMetrics, mouthMetrics, symmetryScore)

        // Відраза
        emotions["disgusted"] = calculateDisgustedScore(eyebrowMetrics, mouthMetrics)

        // Страх
        emotions["fearful"] = calculateFearfulScore(eyeMetrics, eyebrowMetrics, mouthMetrics)

        // Нейтральний стан
        emotions["neutral"] = calculateNeutralScore(emotions)

        return normalizeEmotionScores(emotions)
    }

    /**
     * Розрахунок метрик посмішки
     */
    private fun calculateSmileScore(landmarks: List<NormalizedLandmark>): Float {
        // Відстань між кутами рота
        val mouthWidth = distance2D(
            landmarks[MediaPipeLandmarks.MOUTH_LEFT],
            landmarks[MediaPipeLandmarks.MOUTH_RIGHT]
        )

        // Піднесення кутів рота
        val mouthCenter = (landmarks[MediaPipeLandmarks.MOUTH_TOP].y() +
                landmarks[MediaPipeLandmarks.MOUTH_BOTTOM].y()) / 2

        val leftCornerLift = mouthCenter - landmarks[MediaPipeLandmarks.MOUTH_LEFT].y()
        val rightCornerLift = mouthCenter - landmarks[MediaPipeLandmarks.MOUTH_RIGHT].y()
        val averageCornerLift = (leftCornerLift + rightCornerLift) / 2

        // Кривизна верхньої губи
        val upperLipCurvature = calculateLipCurvature(landmarks, isUpper = true)

        // Комбінований показник посмішки
        val smileScore = (mouthWidth * 2) + (averageCornerLift * 10) + (upperLipCurvature * 5)

        return smileScore.coerceIn(0f, 1f)
    }

    /**
     * Розрахунок метрик очей
     */
    private fun calculateEyeMetrics(landmarks: List<NormalizedLandmark>): EyeMetrics {
        val leftEyeOpenness = calculateEyeAspectRatio(landmarks, isLeft = true)
        val rightEyeOpenness = calculateEyeAspectRatio(landmarks, isLeft = false)

        // Швидкість моргання (потребує історії для точного розрахунку)
        val blinkRate = 0f // Placeholder

        // Напрямок погляду
        val gazeDirection = calculateGazeDirection(landmarks)

        return EyeMetrics(leftEyeOpenness, rightEyeOpenness, blinkRate, gazeDirection)
    }

    /**
     * Розрахунок Eye Aspect Ratio (EAR)
     */
    private fun calculateEyeAspectRatio(landmarks: List<NormalizedLandmark>, isLeft: Boolean): Float {
        val indices = if (isLeft) {
            listOf(
                MediaPipeLandmarks.LEFT_EYE_TOP to MediaPipeLandmarks.LEFT_EYE_BOTTOM,
                159 to 145, // Додаткові точки для точності
                160 to 144
            )
        } else {
            listOf(
                MediaPipeLandmarks.RIGHT_EYE_TOP to MediaPipeLandmarks.RIGHT_EYE_BOTTOM,
                386 to 374, // Додаткові точки для точності
                387 to 373
            )
        }

        var verticalSum = 0f
        indices.forEach { (top, bottom) ->
            verticalSum += distance2D(landmarks[top], landmarks[bottom])
        }

        val horizontalDist = if (isLeft) {
            distance2D(landmarks[MediaPipeLandmarks.LEFT_EYE_INNER],
                landmarks[MediaPipeLandmarks.LEFT_EYE_OUTER])
        } else {
            distance2D(landmarks[MediaPipeLandmarks.RIGHT_EYE_INNER],
                landmarks[MediaPipeLandmarks.RIGHT_EYE_OUTER])
        }

        return (verticalSum / indices.size) / horizontalDist
    }

    /**
     * Розрахунок метрик брів
     */
    private fun calculateEyebrowMetrics(landmarks: List<NormalizedLandmark>): EyebrowMetrics {
        // Висота брів відносно очей
        val leftBrowHeight = landmarks[MediaPipeLandmarks.LEFT_EYEBROW_TOP].y() -
                landmarks[MediaPipeLandmarks.LEFT_EYE_TOP].y()

        val rightBrowHeight = landmarks[MediaPipeLandmarks.RIGHT_EYEBROW_TOP].y() -
                landmarks[MediaPipeLandmarks.RIGHT_EYE_TOP].y()

        // Нахиленість брів (frowning)
        val leftBrowSlope = calculateBrowSlope(landmarks, isLeft = true)
        val rightBrowSlope = calculateBrowSlope(landmarks, isLeft = false)

        // Відстань між бровами
        val browDistance = distance2D(
            landmarks[MediaPipeLandmarks.LEFT_EYEBROW_INNER],
            landmarks[MediaPipeLandmarks.RIGHT_EYEBROW_INNER]
        )

        return EyebrowMetrics(
            leftHeight = leftBrowHeight,
            rightHeight = rightBrowHeight,
            leftSlope = leftBrowSlope,
            rightSlope = rightBrowSlope,
            distance = browDistance
        )
    }

    /**
     * Розрахунок метрик рота
     */
    private fun calculateMouthMetrics(landmarks: List<NormalizedLandmark>): MouthMetrics {
        // Відкритість рота
        val mouthOpenness = distance2D(
            landmarks[MediaPipeLandmarks.MOUTH_TOP],
            landmarks[MediaPipeLandmarks.MOUTH_BOTTOM]
        )

        // Ширина рота
        val mouthWidth = distance2D(
            landmarks[MediaPipeLandmarks.MOUTH_LEFT],
            landmarks[MediaPipeLandmarks.MOUTH_RIGHT]
        )

        // Співвідношення
        val aspectRatio = mouthOpenness / mouthWidth

        // Асиметрія рота
        val leftSide = distance2D(
            landmarks[MediaPipeLandmarks.MOUTH_LEFT],
            landmarks[MediaPipeLandmarks.MOUTH_TOP]
        )
        val rightSide = distance2D(
            landmarks[MediaPipeLandmarks.MOUTH_RIGHT],
            landmarks[MediaPipeLandmarks.MOUTH_TOP]
        )
        val asymmetry = abs(leftSide - rightSide) / (leftSide + rightSide)

        return MouthMetrics(
            openness = mouthOpenness,
            width = mouthWidth,
            aspectRatio = aspectRatio,
            asymmetry = asymmetry
        )
    }

    /**
     * Розрахунок симетрії обличчя
     */
    private fun calculateFaceSymmetry(landmarks: List<NormalizedLandmark>): Float {
        // Порівняння лівої та правої сторони обличчя
        val symmetryPairs = listOf(
            MediaPipeLandmarks.LEFT_EYE_OUTER to MediaPipeLandmarks.RIGHT_EYE_OUTER,
            MediaPipeLandmarks.LEFT_CHEEK to MediaPipeLandmarks.RIGHT_CHEEK,
            MediaPipeLandmarks.MOUTH_LEFT to MediaPipeLandmarks.MOUTH_RIGHT
        )

        var totalAsymmetry = 0f
        val noseCenter = landmarks[MediaPipeLandmarks.NOSE_TIP]

        symmetryPairs.forEach { (left, right) ->
            val leftDist = distance2D(landmarks[left], noseCenter)
            val rightDist = distance2D(landmarks[right], noseCenter)
            totalAsymmetry += abs(leftDist - rightDist) / (leftDist + rightDist)
        }

        return 1f - (totalAsymmetry / symmetryPairs.size)
    }

    /**
     * Розрахунок емоції "щастя"
     */
    private fun calculateHappyScore(
        smileScore: Float,
        eyeMetrics: EyeMetrics,
        mouthMetrics: MouthMetrics
    ): Float {
        // Щастя = висока посмішка + відкриті очі + піднята верхня губа
        val eyeContribution = (eyeMetrics.leftOpenness + eyeMetrics.rightOpenness) / 2
        val mouthContribution = if (mouthMetrics.aspectRatio > 0.3f) 0.2f else 0f

        return (smileScore * 0.7f + eyeContribution * 0.2f + mouthContribution * 0.1f)
            .coerceIn(0f, 1f)
    }

    /**
     * Розрахунок емоції "сум"
     */
    private fun calculateSadScore(
        smileScore: Float,
        eyeMetrics: EyeMetrics,
        eyebrowMetrics: EyebrowMetrics,
        mouthMetrics: MouthMetrics
    ): Float {
        // Сум = опущені куточки рота + опущені брови + можливі сльози (вузькі очі)
        val mouthDownturn = 1f - smileScore
        val browDrop = 1f - ((eyebrowMetrics.leftHeight + eyebrowMetrics.rightHeight) / 2)
        val eyeNarrowing = 1f - ((eyeMetrics.leftOpenness + eyeMetrics.rightOpenness) / 2)

        return (mouthDownturn * 0.4f + browDrop * 0.3f + eyeNarrowing * 0.3f)
            .coerceIn(0f, 1f)
    }

    /**
     * Розрахунок емоції "здивування"
     */
    private fun calculateSurprisedScore(
        eyeMetrics: EyeMetrics,
        eyebrowMetrics: EyebrowMetrics,
        mouthMetrics: MouthMetrics
    ): Float {
        // Здивування = широко відкриті очі + підняті брови + відкритий рот
        val eyeWide = (eyeMetrics.leftOpenness + eyeMetrics.rightOpenness) / 2
        val browRaise = (eyebrowMetrics.leftHeight + eyebrowMetrics.rightHeight) / 2
        val mouthOpen = mouthMetrics.aspectRatio

        return (eyeWide * 0.3f + browRaise * 0.3f + mouthOpen * 0.4f)
            .coerceIn(0f, 1f)
    }

    /**
     * Розрахунок емоції "злість"
     */
    private fun calculateAngryScore(
        eyebrowMetrics: EyebrowMetrics,
        mouthMetrics: MouthMetrics,
        symmetryScore: Float
    ): Float {
        // Злість = зведені брови + стиснутий рот + напруження обличчя
        val browFurrow = 1f - eyebrowMetrics.distance
        val browSlope = (abs(eyebrowMetrics.leftSlope) + abs(eyebrowMetrics.rightSlope)) / 2
        val mouthTension = 1f - mouthMetrics.openness

        return (browFurrow * 0.4f + browSlope * 0.3f + mouthTension * 0.3f)
            .coerceIn(0f, 1f)
    }

    /**
     * Розрахунок емоції "відраза"
     */
    private fun calculateDisgustedScore(
        eyebrowMetrics: EyebrowMetrics,
        mouthMetrics: MouthMetrics
    ): Float {
        // Відраза = асиметрія рота + підняття верхньої губи + звуження носа
        val mouthAsymmetry = mouthMetrics.asymmetry
        val browAsymmetry = abs(eyebrowMetrics.leftHeight - eyebrowMetrics.rightHeight)

        return (mouthAsymmetry * 0.5f + browAsymmetry * 0.5f)
            .coerceIn(0f, 1f)
    }

    /**
     * Розрахунок емоції "страх"
     */
    private fun calculateFearfulScore(
        eyeMetrics: EyeMetrics,
        eyebrowMetrics: EyebrowMetrics,
        mouthMetrics: MouthMetrics
    ): Float {
        // Страх = широкі очі + підняті брови + напружений рот
        val eyeWide = (eyeMetrics.leftOpenness + eyeMetrics.rightOpenness) / 2
        val browRaise = (eyebrowMetrics.leftHeight + eyebrowMetrics.rightHeight) / 2
        val mouthTension = mouthMetrics.aspectRatio * 0.5f

        return (eyeWide * 0.4f + browRaise * 0.4f + mouthTension * 0.2f)
            .coerceIn(0f, 1f)
    }

    /**
     * Розрахунок нейтрального стану
     */
    private fun calculateNeutralScore(emotions: Map<String, Float>): Float {
        // Нейтральний = низькі показники всіх інших емоцій
        val maxOtherEmotion = emotions.values.maxOrNull() ?: 0f
        return (1f - maxOtherEmotion).coerceIn(0f, 1f)
    }

    /**
     * Нормалізація емоційних показників
     */
    private fun normalizeEmotionScores(emotions: MutableMap<String, Float>): Map<String, Float> {
        val sum = emotions.values.sum()
        if (sum > 0) {
            emotions.forEach { (key, value) ->
                emotions[key] = value / sum
            }
        }
        return emotions
    }

    /**
     * Допоміжні функції
     */
    private fun distance2D(p1: NormalizedLandmark, p2: NormalizedLandmark): Float {
        return sqrt((p1.x() - p2.x()).pow(2) + (p1.y() - p2.y()).pow(2))
    }

    private fun calculateBrowSlope(landmarks: List<NormalizedLandmark>, isLeft: Boolean): Float {
        val inner = if (isLeft) MediaPipeLandmarks.LEFT_EYEBROW_INNER else MediaPipeLandmarks.RIGHT_EYEBROW_INNER
        val outer = if (isLeft) MediaPipeLandmarks.LEFT_EYEBROW_OUTER else MediaPipeLandmarks.RIGHT_EYEBROW_OUTER

        val deltaY = landmarks[outer].y() - landmarks[inner].y()
        val deltaX = abs(landmarks[outer].x() - landmarks[inner].x())

        return if (deltaX > 0) deltaY / deltaX else 0f
    }

    private fun calculateLipCurvature(landmarks: List<NormalizedLandmark>, isUpper: Boolean): Float {
        // Спрощений розрахунок кривизни губи
        val centerIndex = if (isUpper) MediaPipeLandmarks.UPPER_LIP_TOP else MediaPipeLandmarks.LOWER_LIP_BOTTOM
        val leftIndex = MediaPipeLandmarks.MOUTH_LEFT
        val rightIndex = MediaPipeLandmarks.MOUTH_RIGHT

        val centerY = landmarks[centerIndex].y()
        val avgCornerY = (landmarks[leftIndex].y() + landmarks[rightIndex].y()) / 2

        return centerY - avgCornerY
    }

    private fun calculateGazeDirection(landmarks: List<NormalizedLandmark>): Pair<Float, Float> {
        // Спрощений розрахунок напрямку погляду
        // В реальності потребує додаткової моделі eye tracking
        return Pair(0f, 0f)
    }

    /**
     * Допоміжні класи для метрик
     */
    data class EyeMetrics(
        val leftOpenness: Float,
        val rightOpenness: Float,
        val blinkRate: Float,
        val gazeDirection: Pair<Float, Float>
    )

    data class EyebrowMetrics(
        val leftHeight: Float,
        val rightHeight: Float,
        val leftSlope: Float,
        val rightSlope: Float,
        val distance: Float
    )

    data class MouthMetrics(
        val openness: Float,
        val width: Float,
        val aspectRatio: Float,
        val asymmetry: Float
    )
}