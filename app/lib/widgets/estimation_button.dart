import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';

/// A button to trigger the depth estimation process.
class EstimationButton extends StatelessWidget {
  final VoidCallback onPressed;

  const EstimationButton({super.key, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);
    return ElevatedButton.icon(
      // Disable the button while processing OR if no image is selected.
      onPressed: appState.isProcessing || appState.selectedImage == null
          ? null
          : onPressed,
      icon: appState.isProcessing
          ? const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(
                color: Colors.white,
                strokeWidth: 2,
              ),
            )
          : const Icon(Icons.auto_awesome),
      label: Text(
        appState.isProcessing ? 'Processing...' : 'Run Depth Estimation',
      ),
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 15),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        textStyle: const TextStyle(fontSize: 18),
        backgroundColor: Colors.teal,
      ),
    );
  }
}
