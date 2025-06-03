# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# MediaPipe
-keep class com.google.mediapipe.** { *; }
-keep class com.google.mediapipe.tasks.vision.** { *; }

# Protobuf
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageLite {
  <fields>;
}

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# MediaPipe native libraries
-keep class com.google.mediapipe.framework.jni.** { *; }
-keep class com.google.mediapipe.modules.** { *; }
-keep class com.google.mediapipe.solutions.** { *; }

# Retrofit
-keepattributes Signature
-keepattributes Exceptions
-keepattributes *Annotation*

-keepclasseswithmembers class * {
    @retrofit2.http.* <methods>;
}

-keep class retrofit2.** { *; }
-keepclasseswithmembers interface * {
    @retrofit2.* <methods>;
}

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-dontwarn sun.misc.**
-keep class com.google.gson.stream.** { *; }

# Application classes that will be serialized/deserialized over Gson
-keep class com.example.emotionrecognition.model.** { <fields>; }
-keep class com.example.emotionrecognition.network.** { <fields>; }

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**
-keepnames class okhttp3.internal.publicsuffix.PublicSuffixDatabase

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# CameraX
-keep class androidx.camera.** { *; }
-keep interface androidx.camera.** { *; }

# Android
-keep class androidx.lifecycle.** { *; }
-keep class androidx.compose.** { *; }

# Keep generic type information
-keepattributes Signature
-keepattributes InnerClasses
-keepattributes EnclosingMethod

# Keep annotation
-keepattributes *Annotation*

# MediaPipe specific optimizations
-optimizations !code/simplification/arithmetic,!code/simplification/cast,!field/*,!class/merging/*
-optimizationpasses 5
-allowaccessmodification