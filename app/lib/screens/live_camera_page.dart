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

    _controller = CameraController(widget.cameras[0], ResolutionPreset.medium,
        enableAudio: false);
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
    if (_isProcessingFrame) {
      return;
    }

    _isProcessingFrame = true;

    final appState = Provider.of<AppState>(context, listen: false);
    final depthEstimator = Provider.of<DepthEstimator>(context, listen: false);

    Future<void> runPipeline() async {
      try {
        final result = await depthEstimator.runDepthEstimationOnFrame(image);
        final int originalWidth = result['originalWidth'];
        final int originalHeight = result['originalHeight'];

        final depthMapBytes = await depthEstimator.applyColorMap(
          result['rawDepthMap'],
          appState.selectedColorMap,
          originalWidth,
          originalHeight,
        );

        if (mounted) {
          appState.setLiveDepthMap(depthMapBytes);
        }
      } catch (e) {
        debugPrint('Error processing frame: ${e.toString()}');
      } finally {
        if (mounted) {
          _isProcessingFrame = false;
        }
      }
    }

    runPipeline();
  }

  /// Helper widget to build section titles, similar to HomePage.
  Widget _buildSectionTitle(BuildContext context, String title) {
    return Text(
      title,
      style: Theme.of(context).textTheme.headlineSmall,
      textAlign: TextAlign.center,
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        body: const Center(child: CircularProgressIndicator()),
        appBar: AppBar(title: const Text('Live Depth Estimation')),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Live Depth Estimation')),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildSectionTitle(context, 'Live Camera Feed'),
              const SizedBox(height: 16),
              Card(
                elevation: 4,
                clipBehavior: Clip.antiAlias, // This crops the content
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
                child: SizedBox(
                  width: 384,
                  height: 384,
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      // Use the preview size for correct aspect ratio
                      width: _controller.value.previewSize?.height ?? 1,
                      height: _controller.value.previewSize?.width ?? 1,
                      child: CameraPreview(_controller),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 32),
              _buildSectionTitle(context, 'Live Depth Map'),
              const SizedBox(height: 16),
              Card(
                elevation: 4,
                clipBehavior: Clip.antiAlias,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
                child: Container(
                  width: 384,
                  height: 384,
                  alignment: Alignment.center,
                  child: Consumer<AppState>(
                    builder: (context, appState, child) {
                      if (appState.liveDepthMapBytes == null) {
                        return Text(
                          'Waiting for depth map...',
                          style: TextStyle(color: Colors.grey[600]),
                        );
                      }
                      return Image.memory(
                        appState.liveDepthMapBytes!,
                        fit: BoxFit.cover,
                        gaplessPlayback:
                            true, // Reduces flicker between frames
                      );
                    },
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}