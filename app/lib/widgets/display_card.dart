import 'dart:typed_data';
import 'package:flutter/material.dart';

/// A reusable card to display an image from memory.
///
/// Handles showing a loading indicator, an empty state message, and can
/// display a child widget overlaid on the image.
class DisplayCard extends StatelessWidget {
  /// The image data to display. If null, the [emptyMessage] is shown.
  final Uint8List? imageData;

  /// The message to display when [imageData] is null and [isLoading] is false.
  final String emptyMessage;

  /// If true, a [CircularProgressIndicator] is shown.
  final bool isLoading;

  /// An optional widget to display at the bottom of the card, typically
  /// for text like inference time.
  final Widget? overlayChild;

  const DisplayCard({
    super.key,
    required this.imageData,
    required this.emptyMessage,
    this.isLoading = false,
    this.overlayChild,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      clipBehavior: Clip.antiAlias, // Helps with rounding the image corners
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Container(
        height: 384,
        width: 384,
        alignment: Alignment.center,
        child: isLoading
            ? const CircularProgressIndicator()
            : imageData == null
                ? Text(
                    emptyMessage,
                    style: TextStyle(color: Colors.grey[600]),
                  )
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Expanded(
                        child: Image.memory(
                          imageData!,
                          fit: BoxFit.cover,
                          width: double.infinity,
                          height: double.infinity,
                          errorBuilder: (context, error, stackTrace) =>
                              const Text('Error loading image'),
                        ),
                      ),
                      if (overlayChild != null) overlayChild!,
                    ],
                  ),
      ),
    );
  }
}