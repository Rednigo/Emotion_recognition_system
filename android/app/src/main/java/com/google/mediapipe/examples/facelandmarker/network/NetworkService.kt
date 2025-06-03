package com.google.mediapipe.examples.facelandmarker.network

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import java.util.concurrent.TimeUnit

// Data classes for API communication
data class LandmarksRequest(
    val user_id: String,
    val content_id: String,
    val landmarks: List<List<Float>>,
    val timestamp_ms: Long
)

data class EmotionResponse(
    val user_id: String,
    val content_id: String,
    val emotion: String,
    val confidence: Float
)

data class FeedItem(
    val content_id: String,
    val type: String,
    val thumbnail_url: String,
    val media_url: String,
    val category: String?
)

data class LikeRequest(
    val user_id: String,
    val content_id: String
)

data class LikeResponse(
    val success: Boolean,
    val message: String?
)

// API Interfaces
interface EmotionAnalysisApi {
    @POST("analyze_landmarks")
    suspend fun analyzeLandmarks(@Body request: LandmarksRequest): EmotionResponse
}

interface RecommendationApi {
    @GET("recommendations")
    suspend fun getRecommendations(@Query("user_id") userId: String): List<FeedItem>

    @POST("recommendations/{content_id}/like")
    suspend fun likeContent(
        @Path("content_id") contentId: String,
        @Body request: LikeRequest
    ): LikeResponse
}

// Network Service Singleton
object NetworkService {
    private const val EMOTION_API_BASE_URL = "http://your-server-ip:8000/"
    private const val RECOMMENDATION_API_BASE_URL = "http://your-recommendation-service:8080/"

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val emotionRetrofit = Retrofit.Builder()
        .baseUrl(EMOTION_API_BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    private val recommendationRetrofit = Retrofit.Builder()
        .baseUrl(RECOMMENDATION_API_BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val emotionApi: EmotionAnalysisApi = emotionRetrofit.create(EmotionAnalysisApi::class.java)
    val recommendationApi: RecommendationApi = recommendationRetrofit.create(RecommendationApi::class.java)
}