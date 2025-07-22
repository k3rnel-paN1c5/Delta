import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';

class AppState extends ChangeNotifier {
  File? _selectedImage;
  Uint8List? _depthMapImageBytes;
  bool _isProcessing = false;
  String _inferenceTime = '';
  String _selectedColorMap = 'Grayscale';
  ThemeMode _themeMode = ThemeMode.system; 

  File? get selectedImage => _selectedImage;
  Uint8List? get depthMapImageBytes => _depthMapImageBytes;
  bool get isProcessing => _isProcessing;
  String get inferenceTime => _inferenceTime;
  String get selectedColorMap => _selectedColorMap;
  ThemeMode get themeMode => _themeMode; 

  void setSelectedImage(File? image) {
    _selectedImage = image;
    _depthMapImageBytes = null;
    _inferenceTime = '';
    notifyListeners();
  }

  void setDepthMap(Uint8List? bytes) {
    _depthMapImageBytes = bytes;
    notifyListeners();
  }

  void setProcessing(bool processing) {
    _isProcessing = processing;
    notifyListeners();
  }

  void setInferenceTime(String time) {
    _inferenceTime = time;
    notifyListeners();
  }

  void setSelectedColorMap(String colorMap) {
    _selectedColorMap = colorMap;
    notifyListeners();
  }

  void toggleTheme() {
    _themeMode =
        _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    notifyListeners();
  }
}