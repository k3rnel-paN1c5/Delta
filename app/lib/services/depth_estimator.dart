import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img_lib;
import 'package:onnxruntime/onnxruntime.dart';

import '../utils/color_maps.dart';

///* A top-level function to run the image pre-processing in an isolate.
///
/// This function decodes, resizes, and normalizes the image data, preparing
/// it for the ONNX model. It's designed to be called with `compute()` to avoid
/// blocking the main UI thread.
///
/// [imagePath] is the path to the image file.
/// Returns a [Float32List] which is the processed input for the model.
Future<Float32List> _processImageForModel(String imagePath) async {
  final imageBytes = await File(imagePath).readAsBytes();
  final originalImage = img_lib.decodeImage(imageBytes);
  if (originalImage == null) {
    throw Exception('Failed to decode image.');
  }

  // Model input dimensions
  const inputHeight = 384;
  const inputWidth = 384;

  // Resize the image to the model's input size.
  final resizedImage = img_lib.copyResize(
    originalImage,
    width: inputWidth,
    height: inputHeight,
  );

  // Normalize and standardize the image, converting it to NCHW format.
  const mean = [0.485, 0.456, 0.406]; // ImageNet mean
  const std = [0.229, 0.224, 0.225]; // ImageNet std
  const channels = [img_lib.Channel.red, img_lib.Channel.green, img_lib.Channel.blue];
  final input = Float32List(1 * 3 * inputHeight * inputWidth);
  int bufferIndex = 0;

  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < inputHeight; ++y) {
      for (int x = 0; x < inputWidth; ++x) {
        final pixel = resizedImage.getPixel(x, y);
        // Normalize and standardize the pixel values.
        input[bufferIndex++] =
            (((pixel.getChannel(channels[c]) / 255.0) - mean[c]) / std[c]);
      }
    }
  }
  return input;
}

///* A record to hold parameters for the color map isolate function.
/// This is necessary because `compute` only accepts a single argument.
typedef ApplyColorMapParams = (
  List<List<List<List<double>>>> output,
  String colorMap
);

///* A top-level function to apply color mapping to the model's output in an isolate.
///
/// This function processes the raw depth data from the model, normalizes it,
/// and applies a color map to generate a visual representation. It is designed
/// to be run with `compute()` to prevent UI freezes.
///
/// [params] is a record containing the raw model output and the selected color map name.
/// Returns a [Uint8List] containing the bytes of the generated PNG image.
Uint8List _applyColorMapIsolate(ApplyColorMapParams params) {
  final output = params.$1;
  final colorMap = params.$2;

  // Output shape: [Batch, Channel, H, W]
  const outputHeight = 384;
  const outputWidth = 384;

  double minDepth = double.maxFinite;
  double maxDepth = double.negativeInfinity;

  // Find the min and max depth values in the output.
  for (int y = 0; y < outputHeight; y++) {
    for (int x = 0; x < outputWidth; x++) {
      final depthValue = output[0][0][y][x];
      if (depthValue < minDepth) minDepth = depthValue;
      if (depthValue > maxDepth) maxDepth = depthValue;
    }
  }

  // Avoid division by zero.
  final double depthRange = max(maxDepth - minDepth, double.minPositive);
  final depthImage = img_lib.Image(width: outputWidth, height: outputHeight);

  // Normalize the depth values and apply the selected color map.
  for (int y = 0; y < outputHeight; y++) {
    for (int x = 0; x < outputWidth; x++) {
      final depthValue = output[0][0][y][x];
      final normalizedDepth = (depthValue - minDepth) / depthRange;

      if (colorMap == 'Viridis') {
        final color = getViridisColor(normalizedDepth);
        depthImage.setPixelRgb(
          x,
          y,
          (color.r * 255).round(),
          (color.g * 255).round(),
          (color.b * 255).round(),
        );
      } else {
        final pixelValue = (normalizedDepth * 255).round();
        depthImage.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
      }
    }
  }
  return Uint8List.fromList(img_lib.encodePng(depthImage));
}


/// Handles the depth estimation process using the ONNX model.
///
/// This class encapsulates the logic for running the model and processing
/// its output. Heavy computations are offloaded to background isolates.
class DepthEstimator {
  final OrtSession _session;

  DepthEstimator(this._session);

  /// Pre-processes the image and runs the depth estimation model.
  /// The heavy image processing is offloaded to a separate isolate.
  Future<Map<String, dynamic>> runDepthEstimation(File imageFile) async {
    // [Batch, Channels, H, W]
    final inputShapes = [1, 3, 384, 384];

    // Offload image decoding, resizing, and normalization to an isolate.
    final input = await compute(_processImageForModel, imageFile.path);

    final inputOrt = OrtValueTensor.createTensorWithDataList(
      input,
      inputShapes,
    );
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();
    final stopwatch = Stopwatch()..start();

    // Run the model asynchronously.
    final outputs = await _session.runAsync(runOptions, inputs);
    stopwatch.stop();

    final outputValue = outputs?[0]?.value;
    if (outputValue == null) {
      throw Exception('Failed to get model output.');
    }

    // Release native resources.
    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) => element?.release());

    return {
      'rawDepthMap': outputValue,
      'inferenceTime': stopwatch.elapsedMilliseconds,
    };
  }

  /// Post-processes the model's output to create a depth map image.
  /// This heavy computation is offloaded to a separate isolate.
  Future<Uint8List> applyColorMap(
    List<List<List<List<double>>>> output,
    String colorMap,
  ) async {
    // Use a record to pass multiple arguments to the isolate function.
    final params = (output, colorMap);
    // Offload the color map application to an isolate.
    return await compute(_applyColorMapIsolate, params);
  }
}
