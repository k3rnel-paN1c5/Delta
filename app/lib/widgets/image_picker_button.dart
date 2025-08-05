import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img_lib;
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';

/// A button that shows options to pick an image from the gallery or camera.
class ImagePickerButton extends StatelessWidget {
  final bool enabled; // Add an 'enabled' flag
  const ImagePickerButton({super.key, this.enabled = true});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: enabled ? () => _showImageSourceDialog(context) : null,
      icon: const Icon(Icons.photo_library),
      label: const Text('Select Image'),
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 15),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        textStyle: const TextStyle(fontSize: 18),
        disabledBackgroundColor: Theme.of(context).primaryColor.withAlpha(125),
        disabledForegroundColor: Colors.white70,
      ),
    );
  }

  /// Shows a dialog to choose between gallery and camera.
  Future<void> _showImageSourceDialog(BuildContext context) async {
    showModalBottomSheet(
      context: context,
      builder: (BuildContext context) {
        return SafeArea(
          child: Wrap(
            children: <Widget>[
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Photo Library'),
                onTap: () {
                  _pickImage(context, ImageSource.gallery);
                  Navigator.of(context).pop();
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_camera),
                title: const Text('Camera'),
                onTap: () {
                  _pickImage(context, ImageSource.camera);
                  Navigator.of(context).pop();
                },
              ),
            ],
          ),
        );
      },
    );
  }

  /// Opens the image picker with the specified source.
  Future<void> _pickImage(BuildContext context, ImageSource source) async {
    final appState = Provider.of<AppState>(context, listen: false);

    try {
      appState.setImageLoading(true);
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: source);

      if (image != null) {
        final imageBytes = await image.readAsBytes();
        final img_lib.Image? originalImage = img_lib.decodeImage(imageBytes);

        if (originalImage != null) {
          final int cropSize = min(originalImage.width, originalImage.height);
          final int cropX = (originalImage.width - cropSize) ~/ 2;
          final int cropY = (originalImage.height - cropSize) ~/ 2;

          final img_lib.Image croppedImage = img_lib.copyCrop(
            originalImage,
            x: cropX,
            y: cropY,
            width: cropSize,
            height: cropSize,
          );

          final Directory tempDir = Directory.systemTemp;
          final String tempPath =
              '${tempDir.path}/cropped_${DateTime.now().millisecondsSinceEpoch}.jpg';
          final File croppedFile = File(tempPath)
            ..writeAsBytesSync(img_lib.encodeJpg(croppedImage));

          appState.setSelectedImage(croppedFile);
        } else {
          // If decoding fails, fall back to the original image file
          appState.setSelectedImage(File(image.path));
        }
      }
    } finally {
      // Ensure the loading indicator is turned off, even if the user cancels.
      appState.setImageLoading(false);
    }
  }
}