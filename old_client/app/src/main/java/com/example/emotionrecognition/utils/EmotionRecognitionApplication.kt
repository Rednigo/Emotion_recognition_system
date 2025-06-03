package com.example.emotionrecognition

import android.app.Application
import androidx.camera.view.PreviewView

/**
 * Application клас для збереження глобального стану
 */
class EmotionRecognitionApplication : Application() {

    private var previewView: PreviewView? = null

    /**
     * Встановлення PreviewView для використання в сервісі
     */
    fun setPreviewView(preview: PreviewView) {
        this.previewView = preview
    }

    /**
     * Отримання PreviewView для використання в сервісі
     */
    fun getPreviewView(): PreviewView? = previewView

    /**
     * Очищення всіх ресурсів
     */
    fun clearAll() {
        previewView = null
    }

    override fun onCreate() {
        super.onCreate()
        android.util.Log.d("EmotionRecognitionApp", "Application created")
    }

    override fun onTerminate() {
        clearAll()
        super.onTerminate()
        android.util.Log.d("EmotionRecognitionApp", "Application terminated")
    }
}