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
  String _selectedColorMap = 'Grayscale';
  ThemeMode _themeMode = ThemeMode.system;

  // Getters for the private state variables.
  File? get selectedImage => _selectedImage;
  Uint8List? get depthMapImageBytes => _depthMapImageBytes;
  bool get isProcessing => _isProcessing;
  bool get isImageLoading => _isImageLoading;
  String get inferenceTime => _inferenceTime;
  String get selectedColorMap => _selectedColorMap;
  ThemeMode get themeMode => _themeMode;

  /// Sets the selected image and resets related states.
  void setSelectedImage(File? image) {
    _selectedImage = image;
    _depthMapImageBytes = null;
    _inferenceTime = '';
    notifyListeners();
  }

  /// Sets the generated depth map image bytes.
  void setDepthMap(Uint8List? bytes) {
    _depthMapImageBytes = bytes;
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

  /// Toggles the theme between light and dark mode.
  void toggleTheme() {
    _themeMode = _themeMode == ThemeMode.dark
        ? ThemeMode.light
        : ThemeMode.dark;
    notifyListeners();
  }
}
