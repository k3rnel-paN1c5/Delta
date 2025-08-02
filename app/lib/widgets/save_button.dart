import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';

/// A button to save the generated depth map.
class SaveButton extends StatelessWidget {
  final VoidCallback onPressed;

  const SaveButton({super.key, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);
    return ElevatedButton.icon(
      // Disable the button if no depth map is available.
      onPressed: appState.depthMapImageBytes == null ? null : onPressed,
      icon: const Icon(Icons.save_alt),
      label: const Text('Save Depth Map'),
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 15),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        textStyle: const TextStyle(fontSize: 18),
        backgroundColor: Colors.indigo, // A different color to distinguish
        foregroundColor: Colors.white,
      ),
    );
  }
}