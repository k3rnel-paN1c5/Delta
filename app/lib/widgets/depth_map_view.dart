import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';

class DepthMapView extends StatelessWidget {
  const DepthMapView({super.key});

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Container(
        height: 300,
        alignment: Alignment.center,
        child: appState.isProcessing
            ? const CircularProgressIndicator()
            : appState.depthMapImageBytes == null
                ? Text(
                    appState.selectedImage == null
                        ? 'Select an image to see depth map'
                        : 'No depth map generated yet',
                    style: TextStyle(color: Colors.grey[600]),
                  )
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Expanded(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.memory(
                            appState.depthMapImageBytes!,
                            fit: BoxFit.contain,
                            errorBuilder: (context, error, stackTrace) =>
                                const Text('Error loading depth map'),
                          ),
                        ),
                      ),
                      if (appState.inferenceTime.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Text(
                            appState.inferenceTime,
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        ),
                    ],
                  ),
      ),
    );
  }
}