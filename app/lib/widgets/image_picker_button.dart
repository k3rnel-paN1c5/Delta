import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'dart:io';

import '../models/app_state.dart';

/// A button that shows options to pick an image from the gallery or camera.
class ImagePickerButton extends StatelessWidget {
  const ImagePickerButton({super.key});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: () => _showImageSourceDialog(context),
      icon: const Icon(Icons.photo_library),
      label: const Text('Select Image'),
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 15),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        textStyle: const TextStyle(fontSize: 18),
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
        appState.setSelectedImage(File(image.path));
      }
    } finally {
      // Ensure the loading indicator is turned off, even if the user cancels.
      appState.setImageLoading(false);
    }
  }
}
