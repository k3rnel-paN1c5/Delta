import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img_lib;
import 'package:onnxruntime/onnxruntime.dart';

import '../utils/color_maps.dart';

/// Handles the depth estimation process using the ONNX model.
class DepthEstimator {
  final OrtSession _session;
  // Mean and standard deviation values for image normalization.
  static const List<double> _mean = [
    // ImageNet Normalization
    0.485,
    0.456,
    0.406,
  ];
  static const List<double> _std = [
    // ImageNet Normalization
    0.229,
    0.224,
    0.225,
  ];

  DepthEstimator(this._session);

  /// Runs the depth estimation on the given image file.
  Future<Map<String, dynamic>> runDepthEstimation(
    File imageFile,
    String colorMap,
  ) async {
    final imageBytes = await imageFile.readAsBytes();
    final originalImage = img_lib.decodeImage(imageBytes);
    if (originalImage == null) {
      throw Exception('Failed to decode image.');
    }

    // [Batch, Channels, H, W]
    final inputShapes = [1, 3, 384, 384];
    final inputHeight = inputShapes[2];
    final inputWidth = inputShapes[3];

    // Resize the image to the model's input size.
    final resizedImage = img_lib.copyResize(
      originalImage,
      width: inputWidth,
      height: inputHeight,
    );

    // Pre-process the image.
    final input = _preProcess(resizedImage, inputHeight, inputWidth);
    final inputOrt = OrtValueTensor.createTensorWithDataList(
      input,
      inputShapes,
    );
    final inputs = {'input': inputOrt};

    final runOptions = OrtRunOptions();
    final stopwatch = Stopwatch()..start();
    // Run the model.
    final outputs = await _session.runAsync(runOptions, inputs);
    stopwatch.stop();

    final output = outputs?[0]?.value;
    // Output shhape : [Batch, Channel, H, W]
    final outputShape = [1, 1, 384, 384];

    if (output == null) {
      throw Exception('Failed to get model output.');
    }

    // Post-process the model's output to get the depth map.
    final depthMapBytes = _postProcess(output, outputShape, colorMap);

    // Release the resources.
    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) => element?.release());

    return {
      'depthMap': depthMapBytes,
      'inferenceTime': stopwatch.elapsedMilliseconds,
    };
  }

  /// Pre-processes the image by normalizing and standardizing it.
  Float32List _preProcess(img_lib.Image image, int height, int width) {
    final input = Float32List(1 * 3 * height * width);
    int bufferIndex = 0;

    // Convert the image to NCHW format.
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          final pixel = image.getPixel(x, y);
          final r = pixel.r;
          final g = pixel.g;
          final b = pixel.b;
          final pixelChannels = [r, g, b];

          // Normalize and standardize the pixel values.
          input[bufferIndex++] =
              ((pixelChannels[c] / 255.0) - _mean[c]) / _std[c];
        }
      }
    }
    return input;
  }

  /// Post-processes the model's output to create a depth map image.
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

    // Find the min and max depth values in the output.
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        final depthValue = outputList[0][0][y][x];
        if (depthValue < minDepth) minDepth = depthValue;
        if (depthValue > maxDepth) maxDepth = depthValue;
      }
    }

    final double depthRange = max(maxDepth - minDepth, double.minPositive); // avoid divison by 0

    final depthImage = img_lib.Image(width: outputWidth, height: outputHeight);

    // Normalize the depth values and apply the selected color map.
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        final depthValue = outputList[0][0][y][x];
        final normalizedDepth = (depthValue - minDepth) / depthRange;

        if (colorMap == 'Viridis') {
          final color = getViridisColor(normalizedDepth);
          depthImage.setPixelRgb(
            x,
            y,
            color.r * 255,
            color.g * 255,
            color.b * 255,
          );
        } else {
          final pixelValue = (normalizedDepth * 255).round();
          depthImage.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
        }
      }
    }
    return Uint8List.fromList(img_lib.encodePng(depthImage));
  }
}
