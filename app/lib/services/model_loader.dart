import 'package:logging/logging.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelLoader {
  Interpreter? _interpreter;
  final Logger _logger = Logger('ModelLoader');

  Future<Interpreter?> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/Delta.tflite');
      _logger.info('Model loaded successfully!');
      _logger.info('Input Tensors: ${_interpreter?.getInputTensors()}');
      _logger.info('Output Tensors: ${_interpreter?.getOutputTensors()}');
      return _interpreter;
    } catch (e) {
      _logger.severe('Failed to load model: $e');
      return null;
    }
  }

  void close() {
    _interpreter?.close();
  }
}