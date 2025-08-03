import 'dart:io';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img_lib;
import 'package:onnxruntime/onnxruntime.dart';

import '../utils/color_maps.dart';

/// A top-level function to run the image pre-processing in an isolate.
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
  const channels = [
    img_lib.Channel.red,
    img_lib.Channel.green,
    img_lib.Channel.blue
  ];
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

/// A record to hold parameters for the color map isolate function.
/// This is necessary because `compute` only accepts a single argument.
typedef ApplyColorMapParams = (
  List<List<List<List<double>>>> output,
  String colorMap
);

/// A top-level function to apply color mapping to the model's output in an isolate.
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

/// A top-level function to process a single camera frame in an isolate.
///
/// This function converts a [CameraImage] from YUV420 format to a resized
/// and normalized `Float32List` suitable for the ONNX model.
Future<Float32List> _processCameraFrame(CameraImage image) async {
  const modelInputWidth = 384;
  const modelInputHeight = 384;

  img_lib.Image rgbImage;

  if (Platform.isAndroid) {
    // Handle Android's YUV_420_888 format
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final yBuffer = yPlane.bytes;
    final uBuffer = uPlane.bytes;
    final vBuffer = vPlane.bytes;

    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel!;

    rgbImage = img_lib.Image(width: image.width, height: image.height);

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final int yIndex = y * yRowStride + x;
        final int uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;

        final yValue = yBuffer[yIndex];
        final uValue = uBuffer[uvIndex];
        final vValue = vBuffer[uvIndex];

        final r = (yValue + 1.402 * (vValue - 128)).round();
        final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).round();
        final b = (yValue + 1.772 * (uValue - 128)).round();

        rgbImage.setPixelRgba(x, y, r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255), 255);
      }
    }
  } else if (Platform.isIOS) {
    // Handle iOS's BGRA format
    final plane = image.planes[0];
    final bytes = plane.bytes;

    rgbImage = img_lib.Image(width: image.width, height: image.height);
    
    // The byte data is interleaved in BGRA format
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final int index = y * plane.bytesPerRow + x * 4; // 4 bytes per pixel (B, G, R, A)
        
        final b = bytes[index];
        final g = bytes[index + 1];
        final r = bytes[index + 2];
        // final a = bytes[index + 3]; // Alpha is not needed for the model

        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }
  } else {
    throw Exception('Unsupported platform for camera processing');
  }

  // --- The rest of the processing is the same for both platforms ---

  // Resize the RGB image to the model's expected input size
  final resizedImage = img_lib.copyResize(rgbImage, width: modelInputWidth, height: modelInputHeight);

  // Normalize the image data to feed into the model
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  final input = Float32List(1 * 3 * modelInputHeight * modelInputWidth);
  int bufferIndex = 0;

  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < modelInputHeight; ++y) {
      for (int x = 0; x < modelInputWidth; ++x) {
        final pixel = resizedImage.getPixel(x, y);
        switch (c) {
          case 0: // Red channel
            input[bufferIndex++] = ((pixel.r / 255.0) - mean[c]) / std[c];
            break;
          case 1: // Green channel
            input[bufferIndex++] = ((pixel.g / 255.0) - mean[c]) / std[c];
            break;
          case 2: // Blue channel
            input[bufferIndex++] = ((pixel.b / 255.0) - mean[c]) / std[c];
            break;
        }
      }
    }
  }

  return input;
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

  /// Pre-processes a camera frame and runs the depth estimation model.
  /// The heavy image processing is offloaded to a separate isolate.
  Future<Map<String, dynamic>> runDepthEstimationOnFrame(
      CameraImage image) async {
    final inputShape = [1, 3, 384, 384];

    // Offload image conversion, resizing, and normalization to an isolate.
    final input = await compute(_processCameraFrame, image);

    final inputOrt = OrtValueTensor.createTensorWithDataList(input, inputShape);

    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();

    final outputs = await _session.runAsync(runOptions, inputs);


    final outputValue = outputs?[0]?.value;
    if (outputValue == null) {
      throw Exception('Failed to get model output.');
    }

    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) => element?.release());

    return {
      'rawDepthMap': outputValue,
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