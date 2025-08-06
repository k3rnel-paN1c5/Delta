class AppConfig {
  // Model Configuration
  static const String modelAssetPath = 'assets/Delta.onnx';
  static const int modelInputHeight = 384;
  static const int modelInputWidth = 384;

  // UI & State Defaults
  static const List<String> availableColorMaps = ['Grayscale', 'Viridis'];
  static const String defaultColorMap = 'Viridis';
}