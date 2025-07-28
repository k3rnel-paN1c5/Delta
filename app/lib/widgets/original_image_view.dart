import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';

/// A widget to display the original selected image.
/// Shows a loading indicator while an image is being picked.
class OriginalImageView extends StatelessWidget {
  const OriginalImageView({super.key});

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);
    return Card(
      elevation: 4,
      clipBehavior: Clip.antiAlias, // Helps with rounding the image corners
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Container(
        height: 384,
        width: 384,
        alignment: Alignment.center,
        // Show a loading indicator if an image is being loaded.
        child: appState.isImageLoading
            ? const CircularProgressIndicator()
            : appState.selectedImage == null
            ? Text(
                'No image selected',
                style: TextStyle(color: Colors.grey[600]),
              )
            : Image.file(
                appState.selectedImage!,
                fit: BoxFit.cover,
                width: double.infinity,
                height: double.infinity,
                errorBuilder: (context, error, stackTrace) =>
                    const Center(child: Text('Error loading image')),
              ),
      ),
    );
  }
}
