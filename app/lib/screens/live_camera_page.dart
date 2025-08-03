import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';
import '../services/depth_estimator.dart';

class LiveCameraPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const LiveCameraPage({super.key, required this.cameras});

  @override
  // ignore: library_private_types_in_public_api
  _LiveCameraPageState createState() => _LiveCameraPageState();
}

class _LiveCameraPageState extends State<LiveCameraPage> {
  late CameraController _controller;
  bool _isProcessingFrame = false;

  @override
  void initState() {
    super.initState();
    if (widget.cameras.isEmpty) {
      // Handle case where no cameras are available
      return;
    }

    _controller = CameraController(widget.cameras[0], ResolutionPreset.high);
    _controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      // Clear any old depth map from the state when the page loads
      Provider.of<AppState>(context, listen: false).setLiveDepthMap(null);

      // Start streaming images
      _controller.startImageStream(_processCameraImage);
      setState(() {});
    });
  }

  @override
  void dispose() {
    _controller.stopImageStream();
    _controller.dispose();
    super.dispose();
  }

  void _processCameraImage(CameraImage image) {
    // If a frame is already being processed, skip this one.
    if (_isProcessingFrame) {
      return;
    }

    // Set the flag to true and update the UI to show the indicator
    setState(() {
      _isProcessingFrame = true;
    });

    final appState = Provider.of<AppState>(context, listen: false);
    final depthEstimator = Provider.of<DepthEstimator>(context, listen: false);

    // Run the entire processing pipeline in a non-blocking way
    Future<void> runPipeline() async {
      try {
        final result = await depthEstimator.runDepthEstimationOnFrame(image);
        final depthMapBytes = await depthEstimator.applyColorMap(
          result['rawDepthMap'],
          appState.selectedColorMap,
        );

        // If the widget is still mounted, update the state with the new depth map
        if (mounted) {
          appState.setLiveDepthMap(depthMapBytes);
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error processing frame: ${e.toString()}')),
          );
        }
      } finally {
        // Unset the flag in the setState callback to ensure UI updates
        // after the flag is changed.
        if (mounted) {
          setState(() {
            _isProcessingFrame = false;
          });
        }
      }
    }

    // Execute the pipeline
    runPipeline();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Live Depth Estimation')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          Consumer<AppState>(
            builder: (context, appState, child) {
              if (appState.liveDepthMapBytes == null) {
                return const SizedBox.shrink();
              }
              return Image.memory(
                appState.liveDepthMapBytes!,
                fit: BoxFit.cover,
                gaplessPlayback: true, 
              );
            },
          ),
        ],
      ),
    );
  }
}
