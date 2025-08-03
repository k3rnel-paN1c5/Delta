import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';
import '../services/depth_estimator.dart';
import '../services/model_loader.dart';
import '../widgets/color_map_dropdown.dart';
import '../widgets/depth_map_view.dart';
import '../widgets/estimation_button.dart';
import '../widgets/image_picker_button.dart';
import '../widgets/original_image_view.dart';
import '../widgets/save_button.dart';
import 'live_camera_page.dart';

/// The main home page of the application.
class DepthEstimationHomePage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const DepthEstimationHomePage({super.key, required this.cameras});

  @override
  State<DepthEstimationHomePage> createState() =>
      _DepthEstimationHomePageState();
}

class _DepthEstimationHomePageState extends State<DepthEstimationHomePage> {
  final ModelLoader _modelLoader = ModelLoader();
  late final DepthEstimator _depthEstimator;
  bool _isModelLoading = true; // Start with loading state
  bool _isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    OrtEnv.instance.init();
    // Load the model in an isolate after the first frame is rendered.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _modelLoader.loadModel().then((session) {
        if (session != null) {
          setState(() {
            _depthEstimator = DepthEstimator(session);
            _isModelLoaded = true;
            _isModelLoading = false; // Finished loading
          });
        } else if (mounted) {
          setState(() {
            _isModelLoading = false; // Finished loading (with error)
          });
          _showErrorDialog(
            'Failed to load AI model. Please ensure the model file is correct and accessible.',
          );
        }
      });
    });
  }

  @override
  void dispose() {
    _modelLoader.close();
    OrtEnv.instance.release();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);
    final bool isReadyForEstimation =
        _isModelLoaded && !appState.isProcessing && appState.selectedImage != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Depth Estimation Demo'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(
              appState.themeMode == ThemeMode.dark
                  ? Icons.light_mode
                  : Icons.dark_mode,
            ),
            onPressed: () => appState.toggleTheme(),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              if (_isModelLoading)
                const Card(
                  elevation: 2,
                  child: Padding(
                    padding: EdgeInsets.all(16.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(width: 20),
                        Text("Loading AI Model..."),
                      ],
                    ),
                  ),
                ),
              _buildSectionTitle(context, 'Original Image'),
              const SizedBox(height: 16),
              ImagePickerButton(enabled: !_isModelLoading),
              const SizedBox(height: 16),
              const OriginalImageView(),
              const SizedBox(height: 32),
              _buildSectionTitle(context, 'Depth Map'),
              const SizedBox(height: 16),
              ColorMapDropdown(
                onColorMapChanged: (_) {
                  // If a depth map exists, re-apply the color map.
                  if (appState.rawDepthMap != null) {
                    _applyColorMapAndUpdateView();
                  }
                },
              ),
              const SizedBox(height: 16),
              const DepthMapView(),
              const SizedBox(height: 32),
              EstimationButton(
                onPressed:
                    isReadyForEstimation ? () => _runDepthEstimation(context) : null,
              ),
              const SizedBox(height: 16),
              SaveButton(onPressed: _saveDepthMap)
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // Navigate to the LiveCameraPage, passing the available cameras.
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => Provider.value(
                value: _depthEstimator,
                child: LiveCameraPage(cameras: widget.cameras),
              ),
            ),
          );
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }

  /// Helper widget to build section titles.
  Widget _buildSectionTitle(BuildContext context, String title) {
    return Text(
      title,
      style: Theme.of(context).textTheme.headlineSmall,
      textAlign: TextAlign.center,
    );
  }

  /// Runs the depth estimation process.
  /// The heavy lifting is performed in a background isolate via DepthEstimator.
  void _runDepthEstimation(BuildContext context) async {
    if (!_isModelLoaded) {
      _showSnackbar('Model is not ready yet. Please wait.');
      return;
    }

    final appState = Provider.of<AppState>(context, listen: false);

    if (appState.selectedImage == null) {
      _showSnackbar('Please select an image first.');
      return;
    }

    appState.setProcessing(true);

    try {
      final result = await _depthEstimator.runDepthEstimation(
        appState.selectedImage!,
      );
      appState.setRawDepthMap(result['rawDepthMap']);
      appState.setInferenceTime(
        'Inference Time: ${result['inferenceTime']} ms',
      );
      // After getting the raw map, apply the colorization.
      _applyColorMapAndUpdateView();
    } catch (e) {
      if (mounted) {
        _showErrorDialog('An error occurred during depth estimation: $e');
      }
    } finally {
      if (mounted) {
        appState.setProcessing(false);
      }
    }
  }

  /// Applies the current color map to the raw depth data and updates the view.
  /// The color mapping is a heavy operation and is run in a background isolate.
  void _applyColorMapAndUpdateView() async {
    final appState = Provider.of<AppState>(context, listen: false);
    if (appState.rawDepthMap == null) {
      return;
    }

    // The applyColorMap method now runs its logic in an isolate.
    final depthMapBytes = await _depthEstimator.applyColorMap(
      appState.rawDepthMap,
      appState.selectedColorMap,
    );

    if (mounted) {
      appState.setDepthMap(depthMapBytes);
    }
  }

  /// Saves the generated depth map to the device's gallery.
  void _saveDepthMap() async {
    final appState = Provider.of<AppState>(context, listen: false);
    if (appState.depthMapImageBytes == null) {
      _showSnackbar('No depth map to save.');
      return;
    }

    final status = await Permission.storage.request();

    if (status.isGranted) {
      try {
        final result = await ImageGallerySaver.saveImage(
          appState.depthMapImageBytes!,
          quality: 100,
          name: 'depth_map_${DateTime.now().millisecondsSinceEpoch}',
        );

        if (mounted && result['isSuccess']) {
          _showSnackbar('Depth map saved to gallery!');
        } else if (mounted) {
          _showSnackbar('Failed to save depth map.');
        }
      } catch (e) {
        if (mounted) {
          _showErrorDialog('An error occurred while saving: $e');
        }
      }
    } else {
      _showSnackbar('Storage permission is required to save the image.');
      if (status.isPermanentlyDenied) {
        openAppSettings();
      }
    }
  }

  /// Shows a snackbar with a message.
  void _showSnackbar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), behavior: SnackBarBehavior.floating),
    );
  }

  /// Shows an error dialog with a message.
  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Error'),
          content: Text(message),
          actions: <Widget>[
            TextButton(
              child: const Text('OK'),
              onPressed: () => Navigator.of(context).pop(),
            ),
          ],
        );
      },
    );
  }
}