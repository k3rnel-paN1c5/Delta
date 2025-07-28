import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img_lib;
import 'package:onnxruntime/onnxruntime.dart';

import '../utils/color_maps.dart';

class DepthEstimator {
  final OrtSession _session;
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  DepthEstimator(this._session);

  Future<Map<String, dynamic>> runDepthEstimation(
    File imageFile,
    String colorMap,
  ) async {
    final originalImage = img_lib.decodeImage(await imageFile.readAsBytes());
    if (originalImage == null) {
      throw Exception('Failed to decode image.');
    }

    final inputShapes = [1, 3, 384, 384];
    final inputHeight = inputShapes[2];
    final inputWidth = inputShapes[3];

    final resizedImage = img_lib.copyResize(
      originalImage,
      width: inputWidth,
      height: inputHeight,
    );

    final input = _preProcess(resizedImage, inputHeight, inputWidth);
    final inputOrt = OrtValueTensor.createTensorWithDataList(input, inputShapes);
    final inputs = {'input': inputOrt};

    final runOptions = OrtRunOptions();
    final stopwatch = Stopwatch()..start();
    final outputs = await _session.runAsync(runOptions, inputs);
    stopwatch.stop();

    final output = outputs?[0]?.value;
    final outputShape = [1, 1, 384, 384];

    if (output == null) {
      throw Exception('Failed to get model output.');
    }
    
    final depthMapBytes = _postProcess(output, outputShape, colorMap);

    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) => element?.release());

    return {
      'depthMap': depthMapBytes,
      'inferenceTime': stopwatch.elapsedMilliseconds,
    };
  }

  Float32List _preProcess(
    img_lib.Image image,
    int height,
    int width,
  ) {
    final input = Float32List(1 * 3 * height * width);
    int bufferIndex = 0;

    // NCHW format
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          final pixel = image.getPixel(x, y);
          final r = pixel.r;
          final g = pixel.g;
          final b = pixel.b;
          final pixelChannels = [r, g, b];
          
          // Normalize and standardize the pixel values and ensure they are floats
          input[bufferIndex++] = ((pixelChannels[c] / 255.0) - _mean[c]) / _std[c];
        }
      }
    }
    return input;
  }

  Uint8List _postProcess(
    dynamic output,
    List<int> outputShape,
    String colorMap,
  ) {
    final outputHeight = outputShape[2];
    final outputWidth = outputShape[3];

    double minDepth = double.maxFinite;
    double maxDepth = double.negativeInfinity;

    final outputList = output as List<List<List<List<double>>>>;

    // First pass: find min and max depth
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        final depthValue = outputList[0][0][y][x];
        if (depthValue < minDepth) minDepth = depthValue;
        if (depthValue > maxDepth) maxDepth = depthValue;
      }
    }
    // maxDepth =1.0 ;
    // minDepth = 0.0;
    final double depthRange = maxDepth - minDepth;

    final depthImage = img_lib.Image(width: outputWidth, height: outputHeight);

    // Second pass: normalize and apply color map
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        final depthValue = outputList[0][0][y][x];
        final normalizedDepth = (depthValue - minDepth) / depthRange;
        
        if (colorMap == 'Viridis') {
          final color = getViridisColor(normalizedDepth);
          depthImage.setPixelRgb(x, y, color.r*255, color.g*255, color.b*255);
        } else {
          final pixelValue = (normalizedDepth * 255).round();
          depthImage.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
        }
      }
    }
    return Uint8List.fromList(img_lib.encodePng(depthImage));
  }
}