# Keep ONNX Runtime classes
-keep class ai.onnxruntime.** { *; }
-keep class org.apache.commons.logging.** { *; }

# This is needed to prevent R8 from stripping away JNI calls
-keepclasseswithmembernames class * {
    native <methods>;
}