import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img_lib;

/// A top-level function to be run in an isolate for processing CameraImage.
///
/// Converts a [CameraImage] to a square-cropped `img_lib.Image` and encodes
/// it as a JPG `Uint8List`. Returns null if conversion fails.
Future<Uint8List?> processCameraImage(CameraImage image) async {
  img_lib.Image? rgbImage;

  // --- Platform-specific YUV/BGRA to RGB conversion ---
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
  }

  if (rgbImage == null) {
    return null;
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

  // --- Encode the cropped image to JPG bytes ---
  return Uint8List.fromList(img_lib.encodeJpg(croppedImage));
}