package com.example.emotionrecognition.network

import com.example.emotionrecognition.model.EmotionResponse
import com.google.gson.annotations.SerializedName
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import java.util.concurrent.TimeUnit

/**
 * Request model for angular features
 */
data class AngularFeaturesRequest(
    @SerializedName("angular_features")
    val angularFeatures: List<Float>,

    @SerializedName("metadata")
    val metadata: Map<String, Any> = emptyMap()
)

/**
 * Enhanced emotion response model
 */
data class DetailedEmotionResponse(
    @SerializedName("emotion")
    val emotion: String,

    @SerializedName("confidence")
    val confidence: Float,

    @SerializedName("emotion_scores")
    val emotionScores: Map<String, Float>,

    @SerializedName("processing_time_ms")
    val processingTimeMs: Float,

    @SerializedName("metadata")
    val metadata: Map<String, Any>
)

/**
 * Model information response
 */
data class ModelInfoResponse(
    @SerializedName("model_loaded")
    val modelLoaded: Boolean,

    @SerializedName("model_type")
    val modelType: String?,

    @SerializedName("emotions")
    val emotions: List<String>,

    @SerializedName("expected_features")
    val expectedFeatures: Int,

    @SerializedName("feature_names")
    val featureNames: List<String>,

    @SerializedName("preprocessing")
    val preprocessing: Map<String, Any>,

    @SerializedName("metadata")
    val metadata: Map<String, Any>
)

/**
 * Health check response
 */
data class HealthResponse(
    @SerializedName("status")
    val status: String,

    @SerializedName("model_loaded")
    val modelLoaded: Boolean,

    @SerializedName("timestamp")
    val timestamp: String
)

/**
 * API interface for emotion recognition server
 */
interface EmotionApiService {
    @GET("health")
    suspend fun healthCheck(): Response<HealthResponse>

    @GET("model_info")
    suspend fun getModelInfo(): Response<ModelInfoResponse>

    @POST("analyze_emotion")
    suspend fun analyzeEmotion(@Body request: AngularFeaturesRequest): Response<DetailedEmotionResponse>

    @POST("batch_analyze")
    suspend fun batchAnalyzeEmotions(@Body requests: List<AngularFeaturesRequest>): Response<List<DetailedEmotionResponse>>
}

/**
 * API Client singleton
 */
object ApiClient {
    // For Android emulator, use 10.0.2.2 to access host machine's localhost
    // For real device, use actual server IP address
    private const val BASE_URL = "http://10.0.2.2:8000/"

    // For production, use your actual server URL
    // private const val BASE_URL = "https://your-emotion-api.com/"

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        .addInterceptor(loggingInterceptor)
        .addInterceptor { chain ->
            val original = chain.request()
            val request = original.newBuilder()
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .method(original.method, original.body)
                .build()
            chain.proceed(request)
        }
        .build()

    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(httpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val apiService: EmotionApiService = retrofit.create(EmotionApiService::class.java)

    /**
     * Helper method to create request from angular features
     */
    fun createRequest(
        angularFeatures: FloatArray,
        additionalMetadata: Map<String, Any> = emptyMap()
    ): AngularFeaturesRequest {
        return AngularFeaturesRequest(
            angularFeatures = angularFeatures.toList(),
            metadata = additionalMetadata
        )
    }
}