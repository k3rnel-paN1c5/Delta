import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../utils/color_maps.dart';

class DepthEstimator {
  final Interpreter _interpreter;
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  DepthEstimator(this._interpreter);

  Future<Map<String, dynamic>> runDepthEstimation(
    File imageFile,
    String colorMap,
  ) async {
    final originalImage = img_lib.decodeImage(await imageFile.readAsBytes());
    if (originalImage == null) {
      throw Exception('Failed to decode image.');
    }

    final inputTensor = _interpreter.getInputTensors()[0];
    final inputShape = inputTensor.shape;
    final inputHeight = inputShape[1];
    final inputWidth = inputShape[2];

    final resizedImage = img_lib.copyResize(
      originalImage,
      width: inputWidth,
      height: inputHeight,
    );

    final input = _preProcess(resizedImage, inputHeight, inputWidth);

    final outputTensor = _interpreter.getOutputTensors()[0];
    final outputShape = outputTensor.shape;
    final output = List.filled(
      outputShape.reduce((a, b) => a * b),
      0.0,
    ).reshape(outputShape);

    final stopwatch = Stopwatch()..start();
    _interpreter.run(input, output);
    stopwatch.stop();

    final depthMapBytes = _postProcess(output, outputShape, colorMap);

    return {
      'depthMap': depthMapBytes,
      'inferenceTime': stopwatch.elapsedMilliseconds,
    };
  }

  List<List<List<List<double>>>> _preProcess(
    img_lib.Image image,
    int height,
    int width,
  ) {
    return List.generate(1, (batch) {
      return List.generate(height, (y) {
        return List.generate(width, (x) {
          final pixel = image.getPixel(x, y);
          final r = pixel.r;
          final g = pixel.g;
          final b = pixel.b;
          return [
            (r / 255.0 - _mean[0]) / _std[0],
            (g / 255.0 - _mean[1]) / _std[1],
            (b / 255.0 - _mean[2]) / _std[2],
          ];
        });
      });
    });
  }

  Uint8List _postProcess(
    dynamic output,
    List<int> outputShape,
    String colorMap,
  ) {
    final outputHeight = outputShape[1];
    final outputWidth = outputShape[2];

    double minDepth = double.maxFinite;
    double maxDepth = double.negativeInfinity;

    // First pass: find min and max depth, ignoring non-finite values
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        final depthValue = (output[0][y][x][0] as num).toDouble();
        if (depthValue.isFinite) {
          if (depthValue < minDepth) minDepth = depthValue;
          if (depthValue > maxDepth) maxDepth = depthValue;
        }
      }
    }

    // Handle case where no finite depth values were found
    if (minDepth == double.maxFinite) {
      minDepth = 0.0;
      maxDepth = 0.0;
    }

    final double depthRange = maxDepth - minDepth;

    final depthImage = img_lib.Image(width: outputWidth, height: outputHeight);

    // Second pass: normalize and apply color map
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        final depthValue = (output[0][y][x][0] as num).toDouble();

        // Default to a black pixel for non-finite values
        if (!depthValue.isFinite) {
          depthImage.setPixelRgb(x, y, 0, 0, 0);
          continue;
        }

        // Normalize depth. Clamp to ensure it's always between 0.0 and 1.0.
        final double normalizedDepth = depthRange > 0
            ? ((depthValue - minDepth) / depthRange).clamp(0.0, 1.0)
            : 0.0;

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
