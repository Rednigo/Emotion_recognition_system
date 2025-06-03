package com.example.emotionrecognition.model

/**
 * Клас для зберігання даних про розпізнану емоцію
 */
data class EmotionData(
    val name: String,         // Назва емоції (anger, contempt, disgust, fear, happiness, neutrality, sadness, surprise)
    val confidence: Float,    // Рівень впевненості (від 0.0 до 1.0)
    val rawMetrics: Map<String, Any>? = null // Сирі метрики
)

/**
 * Перелік доступних емоцій
 */
enum class EmotionType(val displayName: String) {
    ANGER("Гнів"),
    CONTEMPT("Презирство"),
    DISGUST("Відраза"),
    FEAR("Страх"),
    HAPPINESS("Радість"),
    NEUTRALITY("Нейтральність"),
    SADNESS("Сум"),
    SURPRISE("Здивування"),
    UNKNOWN("Невідомо")
}