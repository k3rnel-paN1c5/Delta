import 'dart:io';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img_lib;
import 'package:onnxruntime/onnxruntime.dart';

import '../config/app_config.dart';
import '../utils/color_maps.dart';

typedef ProcessImageParams = (Uint8List imageBytes, int inputHeight, int inputWidth);

Future<Map<String, dynamic>> _processImageForModel(
    ProcessImageParams params) async {
  final imageBytes = params.$1;
  final inputHeight = params.$2;
  final inputWidth = params.$3;

  final originalImage = img_lib.decodeImage(imageBytes);
  if (originalImage == null) throw Exception('Failed to decode image.');

  // The image is now guaranteed to be a square, so we can just resize it
  // to the model's input dimensions.
  final croppedImage = img_lib.copyResize(
    originalImage,
    width: inputWidth,
    height: inputHeight,
  );

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  final input = Float32List(1 * 3 * inputHeight * inputWidth);
  int bufferIndex = 0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < inputHeight; ++y) {
      for (int x = 0; x < inputWidth; ++x) {
        final pixel = croppedImage.getPixel(x, y);
        switch (c) {
          case 0:
            input[bufferIndex++] = ((pixel.r / 255.0) - mean[c]) / std[c];
            break;
          case 1:
            input[bufferIndex++] = ((pixel.g / 255.0) - mean[c]) / std[c];
            break;
          case 2:
            input[bufferIndex++] = ((pixel.b / 255.0) - mean[c]) / std[c];
            break;
        }
      }
    }
  }

  return {
    'input': input,
    'originalWidth': originalImage.width,
    'originalHeight': originalImage.height,
  };
}


typedef ApplyColorMapParams = (
  List<List<List<List<double>>>> output,
  String colorMap,
  int originalWidth,
  int originalHeight
);

// --- THIS IS THE CORRECTED FUNCTION ---
/// A top-level function to apply color mapping to the model's output in an isolate.
///
/// This version correctly resizes the raw depth data using bilinear interpolation
/// without losing precision, directly creating the final colored image.
Uint8List _applyColorMapIsolate(ApplyColorMapParams params) {
  final modelOutput = params.$1[0][0]; // Shape [H, W]
  final colorMapName = params.$2;
  final finalWidth = params.$3;
  final finalHeight = params.$4;

  final modelHeight = modelOutput.length;
  final modelWidth = modelOutput[0].length;

  // 1. Find the min and max depth values for normalization
  double minDepth = double.maxFinite;
  double maxDepth = double.negativeInfinity;
  for (int y = 0; y < modelHeight; y++) {
    for (int x = 0; x < modelWidth; x++) {
      final depthValue = modelOutput[y][x];
      if (depthValue < minDepth) minDepth = depthValue;
      if (depthValue > maxDepth) maxDepth = depthValue;
    }
  }
  final double depthRange = max(maxDepth - minDepth, 1e-6);

  // 2. Create the final image with the target dimensions
  final finalImage = img_lib.Image(width: finalWidth, height: finalHeight);

  // 3. Iterate over each pixel of the final image
  for (int y = 0; y < finalHeight; y++) {
    for (int x = 0; x < finalWidth; x++) {
      // 4. Calculate the corresponding (floating point) coordinates in the small model output
      final double srcX = (x / finalWidth) * modelWidth;
      final double srcY = (y / finalHeight) * modelHeight;

      // 5. Perform bilinear interpolation
      final x1 = srcX.floor();
      final y1 = srcY.floor();
      final x2 = min(x1 + 1, modelWidth - 1);
      final y2 = min(y1 + 1, modelHeight - 1);

      final double xWeight = srcX - x1;
      final double yWeight = srcY - y1;

      final p11 = modelOutput[y1][x1];
      final p12 = modelOutput[y2][x1];
      final p21 = modelOutput[y1][x2];
      final p22 = modelOutput[y2][x2];

      final double interpolatedDepth = (p11 * (1 - xWeight) * (1 - yWeight)) +
                                       (p21 * xWeight * (1 - yWeight)) +
                                       (p12 * (1 - xWeight) * yWeight) +
                                       (p22 * xWeight * yWeight);

      // 6. Normalize the interpolated value and apply the color map
      final normalizedDepth = (interpolatedDepth - minDepth) / depthRange;
      
      if (colorMapName == 'Viridis') {
        final color = getViridisColor(normalizedDepth);
        finalImage.setPixelRgb(
          x,
          y,
          (color.r * 255).round(),
          (color.g * 255).round(),
          (color.b * 255).round(),
        );
      } else {
        final pixelValue = (normalizedDepth * 255).round();
        finalImage.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
      }
    }
  }
  return Uint8List.fromList(img_lib.encodePng(finalImage));
}


typedef ProcessFrameParams = (CameraImage image, int inputHeight, int inputWidth);
Future<Map<String, dynamic>> _processCameraFrame(
    ProcessFrameParams params) async {
  final image = params.$1;
  final modelInputHeight = params.$2;
  final modelInputWidth = params.$3;

  // --- Convert CameraImage to img_lib.Image ---
  img_lib.Image rgbImage;
  if (Platform.isAndroid) {
    // This conversion logic for YUV_420_888 is complex but correct.
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
        final int uvIndex =
            (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;
        final yValue = yBuffer[yIndex];
        final uValue = uBuffer[uvIndex];
        final vValue = vBuffer[uvIndex];
        final r = (yValue + 1.402 * (vValue - 128)).round();
        final g = (yValue -
                0.344136 * (uValue - 128) -
                0.714136 * (vValue - 128))
            .round();
        final b = (yValue + 1.772 * (uValue - 128)).round();
        rgbImage.setPixelRgba(
            x, y, r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255), 255);
      }
    }
  } else if (Platform.isIOS) {
    // Correct conversion for BGRA8888 on iOS.
    final plane = image.planes[0];
    final bytes = plane.bytes;
    rgbImage = img_lib.Image(width: image.width, height: image.height);
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final int index = y * plane.bytesPerRow + x * 4;
        final b = bytes[index];
        final g = bytes[index + 1];
        final r = bytes[index + 2];
        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }
  } else {
    throw Exception('Unsupported platform for camera processing');
  }

  // --- Center-Crop the image to a square ---
  final int cropSize = min(rgbImage.width, rgbImage.height);
  final int cropX = (rgbImage.width - cropSize) ~/ 2;
  final int cropY = (rgbImage.height - cropSize) ~/ 2;

  final img_lib.Image croppedImage = img_lib.copyCrop(
    rgbImage,
    x: cropX,
    y: cropY,
    width: cropSize,
    height: cropSize,
  );

  // --- Resize the square-cropped image to the model's input size ---
  final resizedImage = img_lib.copyResize(
    croppedImage,
    width: modelInputWidth,
    height: modelInputHeight,
  );

  // --- Normalize and prepare the tensor for the model ---
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  final input = Float32List(1 * 3 * modelInputHeight * modelInputWidth);
  int bufferIndex = 0;

  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < modelInputHeight; ++y) {
      for (int x = 0; x < modelInputWidth; ++x) {
        final pixel = resizedImage.getPixel(x, y);
        switch (c) {
          case 0:
            input[bufferIndex++] = ((pixel.r / 255.0) - mean[c]) / std[c];
            break;
          case 1:
            input[bufferIndex++] = ((pixel.g / 255.0) - mean[c]) / std[c];
            break;
          case 2:
            input[bufferIndex++] = ((pixel.b / 255.0) - mean[c]) / std[c];
            break;
        }
      }
    }
  }

  return {
    'input': input,
    // The original width/height is now the size of the square crop
    'originalWidth': cropSize,
    'originalHeight': cropSize,
  };
}


/// Handles the depth estimation process using the ONNX model.
class DepthEstimator {
    // ... (This class is CORRECT and UNCHANGED)
  final OrtSession _session;
  final _inputHeight = AppConfig.modelInputHeight;
  final _inputWidth = AppConfig.modelInputWidth;

  DepthEstimator(this._session);

  Future<Map<String, dynamic>> runDepthEstimation(Uint8List imageBytes) async {
    final inputShapes = [1, 3, _inputHeight, _inputWidth];
    final processParams = (imageBytes, _inputHeight, _inputWidth);

    final processedResult = await compute(_processImageForModel, processParams);
    final input = processedResult['input'] as Float32List;
    final originalWidth = processedResult['originalWidth'] as int;
    final originalHeight = processedResult['originalHeight'] as int;

    final inputOrt = OrtValueTensor.createTensorWithDataList(input, inputShapes);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();
    final stopwatch = Stopwatch()..start();

    final outputs = await _session.runAsync(runOptions, inputs);
    stopwatch.stop();

    final outputValue = outputs?[0]?.value;
    if (outputValue == null) throw Exception('Failed to get model output.');

    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) => element?.release());

    return {
      'rawDepthMap': outputValue,
      'inferenceTime': stopwatch.elapsedMilliseconds,
      'originalWidth': originalWidth,
      'originalHeight': originalHeight,
    };
  }

  Future<Map<String, dynamic>> runDepthEstimationOnFrame(
      CameraImage image) async {
    final inputShape = [1, 3, _inputHeight, _inputWidth];
    final processParams = (image, _inputHeight, _inputWidth);

    final processedResult = await compute(_processCameraFrame, processParams);
    final input = processedResult['input'] as Float32List;
    final originalWidth = processedResult['originalWidth'] as int;
    final originalHeight = processedResult['originalHeight'] as int;


    final inputOrt = OrtValueTensor.createTensorWithDataList(input, inputShape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(runOptions, inputs);
    final outputValue = outputs?[0]?.value;

    if (outputValue == null) throw Exception('Failed to get model output.');

    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) => element?.release());

    return {
      'rawDepthMap': outputValue,
      'originalWidth': originalWidth,
      'originalHeight': originalHeight,
    };
  }

  Future<Uint8List> applyColorMap(
    List<List<List<List<double>>>> output,
    String colorMap,
    int originalWidth,
    int originalHeight,
  ) async {
    final params = (output, colorMap, originalWidth, originalHeight);
    return await compute(_applyColorMapIsolate, params);
  }
}