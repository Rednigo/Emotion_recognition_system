package com.example.emotionrecognition.viewmodel

import androidx.lifecycle.ViewModel
import com.example.emotionrecognition.model.EmotionData
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * ViewModel для головного екрану додатку
 * Тепер використовується лише як допоміжний клас,
 * основне спілкування відбувається через LocalBroadcastManager
 */
class MainViewModel : ViewModel() {

    // StateFlow для зберігання даних про поточну емоцію
    private val _emotionState = MutableStateFlow<EmotionData?>(null)
    val emotionState: StateFlow<EmotionData?> = _emotionState.asStateFlow()

    /**
     * Оновлення даних про емоцію
     */
    fun updateEmotion(emotionData: EmotionData) {
        _emotionState.value = emotionData
    }

    /**
     * Скидання даних про емоцію
     */
    fun resetEmotion() {
        _emotionState.value = null
    }
}