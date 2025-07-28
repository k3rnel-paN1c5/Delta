import 'dart:io';
import 'package:flutter/services.dart';
import 'package:logging/logging.dart';
import 'package:onnxruntime/onnxruntime.dart';

class ModelLoader {
  OrtSession? _session;
  final Logger _logger = Logger('ModelLoader');

  Future<OrtSession?> loadModel() async {
    try {
      final sessionOptions = OrtSessionOptions();
      
      // Enable hardware acceleration based on the platform.
      if (Platform.isIOS) {
        _logger.info("Using CoreML Execution Provider for iOS.");
        sessionOptions.appendCoreMLProvider(CoreMLFlags.onlyEnableDeviceWithANE); 
      } else if (Platform.isAndroid) {
        _logger.info("Using NNAPI Execution Provider for Android.");

        sessionOptions.appendNnapiProvider(NnapiFlags.useFp16);
      }


      const assetFileName = 'assets/Delta.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(bytes, sessionOptions);

      _logger.info('Model loaded successfully!');
      _logger.info('Input Names: ${_session?.inputNames}');
      _logger.info('Output Names: ${_session?.outputNames}');
      return _session;
    } catch (e) {
      _logger.severe('Failed to load model: $e');
      return null;
    }
  }

  void close() {
    _session?.release();
  }
}