package com.example.emotionrecognition

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.emotionrecognition.model.EmotionData
import com.example.emotionrecognition.service.EmotionAnalysisService
import com.example.emotionrecognition.ui.theme.EmotionRecognitionTheme
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
        const val ACTION_SEND_IMAGE = "com.example.emotionrecognition.SEND_IMAGE"
        const val EXTRA_IMAGE_DATA = "image_data"
    }

    // Стан для збереження даних про емоції
    private var currentEmotion by mutableStateOf<EmotionData?>(null)
    private var isServiceRunning by mutableStateOf(false)
    private var lastError by mutableStateOf<String?>(null)
    private var showCameraPreview by mutableStateOf(false)
    private var isCameraActive by mutableStateOf(false)

    private val gson = Gson()

    // Camera components
    private lateinit var previewView: PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.POST_NOTIFICATIONS
    )

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.entries.all { it.value }
        if (allGranted) {
            startEmotionAnalysisService()
        } else {
            val deniedPermissions = permissions.entries
                .filter { !it.value }
                .map { it.key }

            Toast.makeText(
                this,
                "Необхідні дозволи: ${deniedPermissions.joinToString(", ")}",
                Toast.LENGTH_LONG
            ).show()

            lastError = "Відсутні дозволи: ${deniedPermissions.joinToString(", ")}"
        }
    }

    // BroadcastReceiver для отримання оновлень емоцій
    private val emotionUpdateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == EmotionAnalysisService.ACTION_EMOTION_UPDATE) {
                val emotionName = intent.getStringExtra(EmotionAnalysisService.EXTRA_EMOTION_NAME) ?: "neutral"
                val emotionConfidence = intent.getFloatExtra(EmotionAnalysisService.EXTRA_EMOTION_CONFIDENCE, 0f)
                val metricsJson = intent.getStringExtra(EmotionAnalysisService.EXTRA_EMOTION_METRICS) ?: "{}"

                Log.d(TAG, "Received emotion update: $emotionName, confidence: $emotionConfidence")

                // Перевірка на помилки
                if (emotionName == "error") {
                    lastError = "Помилка аналізу: $emotionConfidence"
                    currentEmotion = null
                    return
                }

                // Парсинг метрик
                val metricsType = object : TypeToken<Map<String, Any>>() {}.type
                val rawMetrics: Map<String, Any>? = try {
                    gson.fromJson<Map<String, Any>>(metricsJson, metricsType)
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing metrics: ${e.message}")
                    lastError = "Помилка парсингу метрик: ${e.message}"
                    null
                }

                // Очистка помилок при успішному оновленні
                if (emotionName != "error" && rawMetrics != null) {
                    lastError = null
                }

                // Оновлення стану UI
                currentEmotion = EmotionData(
                    name = emotionName,
                    confidence = emotionConfidence,
                    rawMetrics = rawMetrics
                )
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Ініціалізація camera компонентів
        previewView = PreviewView(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Реєстрація BroadcastReceiver
        val filter = IntentFilter(EmotionAnalysisService.ACTION_EMOTION_UPDATE)
        LocalBroadcastManager.getInstance(this).registerReceiver(emotionUpdateReceiver, filter)

        setContent {
            EmotionRecognitionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(
                        onAnalyzeClick = { checkPermissionsAndStartService() },
                        onStopClick = { stopEmotionAnalysisService() },
                        onExitClick = { finish() },
                        onToggleCameraPreview = { showCameraPreview = !showCameraPreview },
                        currentEmotion = currentEmotion,
                        isServiceRunning = isServiceRunning,
                        lastError = lastError,
                        showCameraPreview = showCameraPreview,
                        isCameraActive = isCameraActive,
                        previewView = previewView,
                        onClearError = { lastError = null }
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Відписка від BroadcastReceiver
        try {
            LocalBroadcastManager.getInstance(this).unregisterReceiver(emotionUpdateReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Error unregistering receiver: ${e.message}")
        }

        // Зупинка сервісу та камери
        stopEmotionAnalysisService()

        // Закриття camera executor
        try {
            cameraExecutor.shutdown()
        } catch (e: Exception) {
            Log.e(TAG, "Error shutting down camera executor: ${e.message}")
        }
    }

    private fun checkPermissionsAndStartService() {
        lastError = null
        permissionLauncher.launch(requiredPermissions)
    }

    private fun startEmotionAnalysisService() {
        try {
            // Спочатку запускаємо сервіс
            val serviceIntent = Intent(this, EmotionAnalysisService::class.java)
            startForegroundService(serviceIntent)
            isServiceRunning = true

            // Потім запускаємо камеру
            startCamera()

            Toast.makeText(
                this,
                "MediaPipe аналіз емоцій запущено",
                Toast.LENGTH_SHORT
            ).show()

            Log.d(TAG, "Service and camera started successfully")
        } catch (e: Exception) {
            lastError = "Помилка запуску сервісу: ${e.message}"
            isServiceRunning = false
            Log.e(TAG, "Error starting service: ${e.message}")
        }
    }

    private fun stopEmotionAnalysisService() {
        try {
            // Спочатку зупиняємо камеру
            stopCamera()

            // Потім зупиняємо сервіс
            val serviceIntent = Intent(this, EmotionAnalysisService::class.java)
            stopService(serviceIntent)
            isServiceRunning = false
            currentEmotion = null
            showCameraPreview = false

            Toast.makeText(
                this,
                "Аналіз емоцій зупинено",
                Toast.LENGTH_SHORT
            ).show()

            Log.d(TAG, "Service and camera stopped successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping service: ${e.message}")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                // Налаштування Preview
                val preview = Preview.Builder()
                    .setTargetResolution(Size(640, 480))
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // Налаштування ImageAnalysis як в офіційному зразку MediaPipe
                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // Як в офіційному зразку
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, MediaPipeImageAnalyzer())
                    }

                // Селектор фронтальної камери
                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                try {
                    // Звільнення попередніх use cases
                    cameraProvider?.unbindAll()

                    // Прив'язка use cases до камери
                    cameraProvider?.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalyzer
                    )

                    isCameraActive = true
                    showCameraPreview = true
                    Log.d(TAG, "Camera started successfully")

                } catch (exc: Exception) {
                    Log.e(TAG, "Use case binding failed", exc)
                    lastError = "Помилка прив'язки камери: ${exc.message}"
                }

            } catch (exc: Exception) {
                Log.e(TAG, "Camera initialization failed", exc)
                lastError = "Помилка ініціалізації камери: ${exc.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        try {
            cameraProvider?.unbindAll()
            isCameraActive = false
            Log.d(TAG, "Camera stopped successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping camera: ${e.message}")
        }
    }

    // MediaPipe-стиль ImageAnalyzer як в офіційному зразку
    private inner class MediaPipeImageAnalyzer : ImageAnalysis.Analyzer {
        private var lastAnalysisTime = 0L
        private val analysisInterval = 2000L

        override fun analyze(imageProxy: ImageProxy) {
            val currentTime = System.currentTimeMillis()

            if (currentTime - lastAnalysisTime < analysisInterval) {
                imageProxy.close()
                return
            }

            lastAnalysisTime = currentTime

            try {
                Log.d(TAG, "MediaPipe-style analysis: ${imageProxy.width}x${imageProxy.height}, format: ${imageProxy.format}")

                // Конвертація як в офіційному зразку MediaPipe
                val bitmap = convertImageProxyToBitmapMediaPipeStyle(imageProxy)

                if (bitmap != null) {
                    Log.d(TAG, "Converted to bitmap (MediaPipe style): ${bitmap.width}x${bitmap.height}")

                    // Застосовуємо ротацію як в офіційному зразку
                    val rotatedBitmap = applyMediaPipeRotation(bitmap, imageProxy.imageInfo.rotationDegrees)
                    Log.d(TAG, "Applied MediaPipe rotation: ${rotatedBitmap.width}x${rotatedBitmap.height}")

                    // Відправляємо обробленное зображення
                    sendImageToService(rotatedBitmap, "mediapipe_style")

                    // Очищення пам'яті
                    if (rotatedBitmap != bitmap) {
                        bitmap.recycle()
                    }
                } else {
                    Log.e(TAG, "Failed to convert ImageProxy to bitmap (MediaPipe style)")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in MediaPipe-style analysis: ${e.message}", e)
            } finally {
                imageProxy.close()
            }
        }
    }

    // Конвертація ImageProxy як в офіційному зразку MediaPipe
    private fun convertImageProxyToBitmapMediaPipeStyle(imageProxy: ImageProxy): Bitmap? {
        return try {
            // Створюємо Bitmap буфер як в офіційному зразку
            val bitmapBuffer = Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )

            // Копіюємо пікселі з ImageProxy як в офіційному зразку
            imageProxy.use {
                bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
            }

            Log.d(TAG, "MediaPipe-style conversion successful: ${bitmapBuffer.width}x${bitmapBuffer.height}")
            bitmapBuffer

        } catch (e: Exception) {
            Log.e(TAG, "Error in MediaPipe-style conversion: ${e.message}")

            // Fallback до старого методу якщо MediaPipe метод не працює
            Log.d(TAG, "Falling back to YUV conversion method...")
            imageProxyToBitmap(imageProxy)
        }
    }

    // Ротація як в офіційному зразку MediaPipe
    private fun applyMediaPipeRotation(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        return try {
            val matrix = Matrix().apply {
                // Ротація як в офіційному зразку MediaPipe
                postRotate(rotationDegrees.toFloat())

                // Flip для фронтальної камери буде зроблений в сервісі
                // як в офіційному зразку
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )

            Log.d(TAG, "MediaPipe rotation applied: ${rotationDegrees}° -> ${bitmap.width}x${bitmap.height} to ${rotatedBitmap.width}x${rotatedBitmap.height}")
            rotatedBitmap

        } catch (e: Exception) {
            Log.e(TAG, "Error in MediaPipe rotation: ${e.message}")
            bitmap
        }
    }

    private fun sendImageToService(bitmap: Bitmap, suffix: String = "rotated") {
        try {
            if (isServiceRunning) {
                // Зменшуємо розмір зображення для кращої роботи MediaPipe
                val targetSize = 640 // Оптимальний розмір для MediaPipe
                val scaledBitmap = scaleBitmapToTarget(bitmap, targetSize)

                // Конвертуємо bitmap в byte array для передачі
                val stream = java.io.ByteArrayOutputStream()
                scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 85, stream) // Збільшена якість
                val imageBytes = stream.toByteArray()

                // Відправляємо через broadcast з byte array
                val intent = Intent(ACTION_SEND_IMAGE).apply {
                    putExtra(EXTRA_IMAGE_DATA, imageBytes)
                    putExtra("image_width", scaledBitmap.width)
                    putExtra("image_height", scaledBitmap.height)
                    putExtra("original_width", bitmap.width)
                    putExtra("original_height", bitmap.height)
                    putExtra("image_type", suffix) // Додаємо тип зображення
                }

                LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
                Log.d(TAG, "Image sent to service: ${imageBytes.size} bytes, scaled ${scaledBitmap.width}x${scaledBitmap.height} (from ${bitmap.width}x${bitmap.height}) [$suffix]")

                // Очищаємо scaled bitmap
                if (scaledBitmap != bitmap) {
                    scaledBitmap.recycle()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending image to service: ${e.message}")
        }
    }

    private fun scaleBitmapToTarget(bitmap: Bitmap, targetSize: Int): Bitmap {
        return try {
            val currentWidth = bitmap.width
            val currentHeight = bitmap.height

            // Якщо зображення вже оптимального розміру
            if (currentWidth <= targetSize && currentHeight <= targetSize) {
                return bitmap
            }

            // Розраховуємо scale factor зберігаючи пропорції
            val scaleFactor = if (currentWidth > currentHeight) {
                targetSize.toFloat() / currentWidth
            } else {
                targetSize.toFloat() / currentHeight
            }

            val newWidth = (currentWidth * scaleFactor).toInt()
            val newHeight = (currentHeight * scaleFactor).toInt()

            Log.d(TAG, "Scaling bitmap from ${currentWidth}x${currentHeight} to ${newWidth}x${newHeight}")

            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        } catch (e: Exception) {
            Log.e(TAG, "Error scaling bitmap: ${e.message}")
            bitmap // Повертаємо оригінал при помилці
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val yBuffer = imageProxy.planes[0].buffer
            val uBuffer = imageProxy.planes[1].buffer
            val vBuffer = imageProxy.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            val uvPixelStride = imageProxy.planes[1].pixelStride
            if (uvPixelStride == 1) {
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + vSize, uSize)
            } else {
                var pos = ySize
                for (i in 0 until uSize / uvPixelStride) {
                    nv21[pos] = vBuffer.get(i * uvPixelStride)
                    nv21[pos + 1] = uBuffer.get(i * uvPixelStride)
                    pos += 2
                }
            }

            val yuvImage = android.graphics.YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                imageProxy.width,
                imageProxy.height,
                null
            )

            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height),
                100,
                out
            )

            val imageBytes = out.toByteArray()
            android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap: ${e.message}")
            null
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        return try {
            Log.d(TAG, "Rotating bitmap by $degrees degrees")

            val matrix = Matrix().apply {
                // Ротація навколо центру зображення
                postRotate(degrees, bitmap.width / 2f, bitmap.height / 2f)

                // Дзеркальне відображення для фронтальної камери
                // Застосовуємо тільки для корректних кутів
                when (degrees.toInt()) {
                    90, 270 -> {
                        postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
                        Log.d(TAG, "Applied horizontal flip for front camera")
                    }
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

            Log.d(TAG, "Rotation completed: ${bitmap.width}x${bitmap.height} -> ${rotatedBitmap.width}x${rotatedBitmap.height}")

            rotatedBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error rotating bitmap: ${e.message}")
            bitmap
        }
    }

    private fun rotateBitmapSimple(bitmap: Bitmap, degrees: Float): Bitmap {
        return try {
            val matrix = Matrix().apply {
                postRotate(degrees, bitmap.width / 2f, bitmap.height / 2f)
            }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            Log.e(TAG, "Error in simple rotation: ${e.message}")
            bitmap
        }
    }

    private fun flipBitmap(bitmap: Bitmap): Bitmap {
        return try {
            val matrix = Matrix().apply {
                postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            Log.e(TAG, "Error in flip: ${e.message}")
            bitmap
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    onAnalyzeClick: () -> Unit,
    onStopClick: () -> Unit,
    onExitClick: () -> Unit,
    onToggleCameraPreview: () -> Unit,
    currentEmotion: EmotionData?,
    isServiceRunning: Boolean,
    lastError: String?,
    showCameraPreview: Boolean,
    isCameraActive: Boolean,
    previewView: PreviewView,
    onClearError: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MediaPipe розпізнавання емоцій") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                ),
                actions = {
                    IconButton(
                        onClick = onToggleCameraPreview,
                        enabled = isServiceRunning && isCameraActive
                    ) {
                        Icon(
                            imageVector = androidx.compose.material.icons.Icons.Default.Camera,
                            contentDescription = if (showCameraPreview) "Приховати камеру" else "Показати камеру",
                            tint = if (showCameraPreview && isCameraActive)
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        // Додаємо LazyColumn для скролу
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            item {
                // Заголовок
                Text(
                    text = "Аналіз емоцій з MediaPipe Face Mesh",
                    style = MaterialTheme.typography.headlineSmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
            }

            // Відображення камери (якщо увімкнено)
            if (showCameraPreview && isCameraActive) {
                item {
                    CameraPreviewCard(previewView = previewView)
                }
            }

            // Відображення помилок
            lastError?.let { error ->
                item {
                    ErrorCard(
                        error = error,
                        onDismiss = onClearError
                    )
                }
            }

            // Відображення поточної емоції (якщо є)
            currentEmotion?.let { emotion ->
                item {
                    EmotionInfoCard(
                        emotion = emotion.name,
                        confidence = emotion.confidence
                    )
                }

                // Відображення метрик MediaPipe
                emotion.rawMetrics?.let { metrics ->
                    if (metrics.isNotEmpty()) {
                        item {
                            MediaPipeMetricsCard(metrics = metrics)
                        }
                    }
                }
            }

            // Показати стан сервісу якщо запущений, але немає даних
            if (isServiceRunning && currentEmotion == null && lastError == null) {
                item {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(32.dp),
                                strokeWidth = 3.dp
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = if (isCameraActive)
                                    "Аналізуємо зображення...\nТримайте обличчя в кадрі"
                                else
                                    "Ініціалізація камери...",
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            }

            item {
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Кнопки управління
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = onAnalyzeClick,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        ),
                        enabled = !isServiceRunning
                    ) {
                        Icon(
                            imageVector = androidx.compose.material.icons.Icons.Default.Face,
                            contentDescription = null,
                            modifier = Modifier.padding(end = 8.dp)
                        )
                        Text(text = if (isServiceRunning) "Запущено" else "Запустити")
                    }

                    Button(
                        onClick = onStopClick,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.error
                        ),
                        enabled = isServiceRunning
                    ) {
                        Text(text = "Зупинити")
                    }
                }
            }

            item {
                Spacer(modifier = Modifier.height(16.dp))
            }

            item {
                Button(
                    onClick = onExitClick,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.secondary
                    )
                ) {
                    Text(text = "Вихід")
                }
            }

            item {
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Статус аналізу
            item {
                StatusCard(
                    isServiceRunning = isServiceRunning,
                    isCameraActive = isCameraActive,
                    currentEmotion = currentEmotion,
                    lastError = lastError
                )
            }

            // Додатковий відступ знизу
            item {
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
}

@Composable
fun CameraPreviewCard(previewView: PreviewView) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(300.dp)
            .padding(vertical = 8.dp),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Камера (фронтальна)",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )

                Icon(
                    imageVector = androidx.compose.material.icons.Icons.Default.Camera,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Контейнер для PreviewView
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color.Black
                )
            ) {
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize()
                ) { view ->
                    // Налаштування PreviewView
                    view.scaleType = PreviewView.ScaleType.FILL_CENTER
                    view.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Тримайте обличчя в центрі кадру для кращого розпізнавання",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

@Composable
fun ErrorCard(
    error: String,
    onDismiss: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = androidx.compose.material.icons.Icons.Default.Warning,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onErrorContainer,
                modifier = Modifier.padding(end = 12.dp)
            )

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Помилка",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = error,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }

            TextButton(
                onClick = onDismiss
            ) {
                Text(
                    text = "OK",
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
        }
    }
}

@Composable
fun StatusCard(
    isServiceRunning: Boolean,
    isCameraActive: Boolean,
    currentEmotion: EmotionData?,
    lastError: String?
) {
    val statusText = when {
        lastError != null -> "Помилка: $lastError"
        isServiceRunning && isCameraActive && currentEmotion != null -> "Активний аналіз (468 точок обличчя)"
        isServiceRunning && isCameraActive -> "Камера активна, очікуємо обличчя..."
        isServiceRunning -> "Сервіс запущено, ініціалізація камери..."
        else -> "Готовий до аналізу"
    }

    val statusColor = when {
        lastError != null -> MaterialTheme.colorScheme.error
        isServiceRunning && isCameraActive && currentEmotion != null -> MaterialTheme.colorScheme.primary
        isServiceRunning && isCameraActive -> MaterialTheme.colorScheme.tertiary
        isServiceRunning -> MaterialTheme.colorScheme.secondary
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = statusColor.copy(alpha = 0.1f)
        )
    ) {
        Text(
            text = statusText,
            color = statusColor,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            textAlign = TextAlign.Center
        )
    }
}

// Решта функцій залишаються такими ж як в попередньому коді...
@Composable
fun EmotionInfoCard(emotion: String, confidence: Float) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = when(emotion.lowercase()) {
                "happy" -> Color(0xFF4CAF50).copy(alpha = 0.1f)
                "sad" -> Color(0xFF2196F3).copy(alpha = 0.1f)
                "angry" -> Color(0xFFF44336).copy(alpha = 0.1f)
                "surprised" -> Color(0xFFFF9800).copy(alpha = 0.1f)
                "disgusted" -> Color(0xFF9C27B0).copy(alpha = 0.1f)
                else -> MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Column(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth()
        ) {
            Text(
                text = "Виявлена емоція",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = translateEmotion(emotion),
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = when(emotion.lowercase()) {
                        "happy" -> Color(0xFF4CAF50)
                        "sad" -> Color(0xFF2196F3)
                        "angry" -> Color(0xFFF44336)
                        "surprised" -> Color(0xFFFF9800)
                        "disgusted" -> Color(0xFF9C27B0)
                        else -> MaterialTheme.colorScheme.onSurface
                    }
                )

                CircularProgressIndicator(
                    progress = confidence,
                    modifier = Modifier.size(48.dp),
                    strokeWidth = 4.dp
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            LinearProgressIndicator(
                progress = confidence,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
            )

            Text(
                text = "Впевненість: ${(confidence * 100).toInt()}%",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 4.dp)
            )
        }
    }
}

@Composable
fun MediaPipeMetricsCard(metrics: Map<String, Any>) {
    var showDetails by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth()
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "MediaPipe метрики",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )

                    val landmarksCount = (metrics["mediapipe_landmarks"] as? Number)?.toInt() ?: 0
                    Text(
                        text = if (landmarksCount > 0) "$landmarksCount facial landmarks" else "Landmarks не знайдено",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (landmarksCount > 0) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error
                    )
                }

                Button(
                    onClick = { showDetails = !showDetails },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer,
                        contentColor = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                ) {
                    Text(text = if (showDetails) "Сховати" else "Показати")
                }
            }

            if (showDetails) {
                Spacer(modifier = Modifier.height(12.dp))

                // Відображення основних метрик MediaPipe
                DisplayMediaPipeMetrics(metrics)

                // Додаткові метрики
                Spacer(modifier = Modifier.height(12.dp))

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp)
                    ) {
                        Text(
                            text = "Детальні метрики (${metrics.size} параметрів):",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        // Відображення всіх метрик у списку
                        LazyColumn(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp)
                        ) {
                            val keyValueList = metrics.entries
                                .filter {
                                    // Фільтруємо складні структури для простого відображення
                                    it.key != "mediapipe_landmarks" && it.key != "angular_features"
                                }
                                .map { "${formatMetricName(it.key)}: ${formatMetricValue(it.value)}" }

                            items(keyValueList) { item ->
                                Text(
                                    text = item,
                                    style = MaterialTheme.typography.bodySmall,
                                    modifier = Modifier.padding(vertical = 2.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DisplayMediaPipeMetrics(metrics: Map<String, Any>) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
                shape = RoundedCornerShape(8.dp)
            )
            .padding(12.dp)
    ) {
        // Метрики очей
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "Відкритість очей:", fontWeight = FontWeight.Medium)
            Row {
                metrics["leftEyeOpenness"]?.let {
                    Text(
                        text = "Л: ${((it as Number).toFloat() * 100).toInt()}%",
                        modifier = Modifier.padding(end = 8.dp)
                    )
                }
                metrics["rightEyeOpenness"]?.let {
                    Text(text = "П: ${((it as Number).toFloat() * 100).toInt()}%")
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Метрика посмішки
        metrics["smileMetric"]?.let {
            val smileValue = (it as Number).toFloat()
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "Посмішка:", fontWeight = FontWeight.Medium)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    LinearProgressIndicator(
                        progress = smileValue,
                        modifier = Modifier
                            .width(60.dp)
                            .height(4.dp)
                            .padding(end = 8.dp)
                    )
                    Text(text = "${(smileValue * 100).toInt()}%")
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Метрики брів
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "Підняття брів:", fontWeight = FontWeight.Medium)
            Row {
                metrics["leftEyebrowRaise"]?.let {
                    Text(
                        text = "Л: ${((it as Number).toFloat() * 100).toInt()}%",
                        modifier = Modifier.padding(end = 8.dp)
                    )
                }
                metrics["rightEyebrowRaise"]?.let {
                    Text(text = "П: ${((it as Number).toFloat() * 100).toInt()}%")
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Відкритість рота
        metrics["mouthOpenness"]?.let {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "Відкритість рота:", fontWeight = FontWeight.Medium)
                Text(text = "${((it as Number).toFloat() * 100).toInt()}%")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Кути повороту голови
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "Орієнтація голови:", fontWeight = FontWeight.Medium)
            Text(
                text = "X:${formatAngle(metrics["headEulerAngleX"])} Y:${formatAngle(metrics["headEulerAngleY"])} Z:${formatAngle(metrics["headEulerAngleZ"])}",
                style = MaterialTheme.typography.bodySmall
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Кількість landmarks
        val landmarksCount = (metrics["mediapipe_landmarks"] as? Number)?.toInt() ?: 0
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "Facial landmarks:", fontWeight = FontWeight.Medium)
            Text(
                text = if (landmarksCount > 0) "$landmarksCount точок" else "0 точок",
                color = if (landmarksCount > 0) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

// Переклад емоцій українською
private fun translateEmotion(emotion: String): String {
    return when(emotion.lowercase()) {
        "happy" -> "😊 Щасливий"
        "sad" -> "😢 Сумний"
        "angry" -> "😠 Злий"
        "surprised" -> "😲 Здивований"
        "disgusted" -> "🤢 Відраза"
        "neutral" -> "😐 Нейтральний"
        "error" -> "❌ Помилка"
        else -> emotion
    }
}

// Форматування назви метрики
private fun formatMetricName(name: String): String {
    return when(name) {
        "leftEyeOpenness" -> "Ліве око"
        "rightEyeOpenness" -> "Праве око"
        "smileMetric" -> "Посмішка"
        "leftEyebrowRaise" -> "Ліва брова"
        "rightEyebrowRaise" -> "Права брова"
        "mouthOpenness" -> "Рот"
        "faceWidth" -> "Ширина обличчя"
        "faceHeight" -> "Висота обличчя"
        "headEulerAngleX" -> "Нахил X"
        "headEulerAngleY" -> "Нахил Y"
        "headEulerAngleZ" -> "Нахил Z"
        "status" -> "Статус"
        "error" -> "Помилка"
        else -> name.replace("angle_", "Кут ")
    }
}

// Форматування значення метрики для відображення
private fun formatMetricValue(value: Any): String {
    return when (value) {
        is Float -> {
            if (value < 1.5f) {
                "${(value * 100).toInt()}%"
            } else {
                String.format("%.1f", value)
            }
        }
        is Double -> {
            if (value < 1.5) {
                "${(value * 100).toInt()}%"
            } else {
                String.format("%.1f", value)
            }
        }
        is Number -> value.toString()
        is List<*> -> "список [${value.size} елем.]"
        is Map<*, *> -> "мапа [${value.size} елем.]"
        else -> value.toString()
    }
}

// Форматування кута для відображення
private fun formatAngle(angle: Any?): String {
    if (angle == null) return "N/A"

    return when (angle) {
        is Float -> String.format("%.0f°", angle)
        is Double -> String.format("%.0f°", angle)
        is Number -> "${angle}°"
        else -> "N/A"
    }
}