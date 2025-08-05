import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';

/// Manages the application's state using the ChangeNotifier pattern.
class AppState extends ChangeNotifier {
  File? _selectedImage;
  Uint8List? _depthMapImageBytes;
  bool _isProcessing = false;
  bool _isImageLoading = false;
  String _inferenceTime = '';
  String _selectedColorMap = 'Viridis';
  ThemeMode _themeMode = ThemeMode.system;
  dynamic _rawDepthMap;
  Uint8List? _liveDepthMapBytes;
  int? _originalWidth;
  int? _originalHeight;

  // Getters for the private state variables.
  File? get selectedImage => _selectedImage;
  Uint8List? get depthMapImageBytes => _depthMapImageBytes;
  bool get isProcessing => _isProcessing;
  bool get isImageLoading => _isImageLoading;
  String get inferenceTime => _inferenceTime;
  String get selectedColorMap => _selectedColorMap;
  ThemeMode get themeMode => _themeMode;
  dynamic get rawDepthMap => _rawDepthMap;
  Uint8List? get liveDepthMapBytes => _liveDepthMapBytes;
  int? get originalWidth => _originalWidth;
  int? get originalHeight => _originalHeight;

  /// Sets the selected image and resets related states.
  void setSelectedImage(File? image) {
    _selectedImage = image;
    _depthMapImageBytes = null;
    _liveDepthMapBytes = null; 
    _rawDepthMap = null;
    _inferenceTime = '';
    notifyListeners();
  }

  /// Sets the generated depth map image bytes for the static image view.
  void setDepthMap(Uint8List? bytes) {
    _depthMapImageBytes = bytes;
    notifyListeners();
  }

  /// Sets the generated depth map for the live camera view.
  void setLiveDepthMap(Uint8List? bytes) {
    _liveDepthMapBytes = bytes;
    notifyListeners();
  }

  /// Stores the raw output from the depth estimation model.
  void setRawDepthMap(dynamic data) {
    _rawDepthMap = data;
    notifyListeners();
  }

  /// Sets the processing state for the depth estimation.
  void setProcessing(bool processing) {
    _isProcessing = processing;
    notifyListeners();
  }

  /// Sets the loading state for the image picker.
  void setImageLoading(bool loading) {
    _isImageLoading = loading;
    notifyListeners();
  }

  /// Sets the inference time taken by the model.
  void setInferenceTime(String time) {
    _inferenceTime = time;
    notifyListeners();
  }

  /// Sets the selected color map for the depth map.
  void setSelectedColorMap(String colorMap) {
    _selectedColorMap = colorMap;
    notifyListeners();
  }

  void setOriginalDimensions(int width, int height) {
    _originalWidth = width;
    _originalHeight = height;
  }

  /// Toggles the theme between light and dark mode.
  void toggleTheme() {
    _themeMode = _themeMode == ThemeMode.dark
        ? ThemeMode.light
        : ThemeMode.dark;
    notifyListeners();
  }
}