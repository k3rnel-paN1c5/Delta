import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';
import '../services/depth_estimator.dart';
import '../services/model_loader.dart';
import '../widgets/depth_map_view.dart';
import '../widgets/estimation_button.dart';
import '../widgets/image_picker_button.dart';

class DepthEstimationHomePage extends StatefulWidget {
  const DepthEstimationHomePage({super.key});

  @override
  State<DepthEstimationHomePage> createState() => _DepthEstimationHomePageState();
}

class _DepthEstimationHomePageState extends State<DepthEstimationHomePage> {
  final ModelLoader _modelLoader = ModelLoader();
  late final DepthEstimator _depthEstimator;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _modelLoader.loadModel().then((interpreter) {
        if (interpreter != null) {
          _depthEstimator = DepthEstimator(interpreter);
        } else if (mounted) {
          _showErrorDialog(
              'Failed to load AI model. Please ensure the model file is correct and accessible.');
        }
      });
    });
  }

  @override
  void dispose() {
    _modelLoader.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Depth Estimation Demo'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(appState.themeMode == ThemeMode.dark
                ? Icons.light_mode
                : Icons.dark_mode),
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
              _buildSectionTitle(context, 'Original Image'),
              const SizedBox(height: 16),
              const ImagePickerButton(),
              const SizedBox(height: 16),
              const OriginalImageView(),
              const SizedBox(height: 32),
              _buildSectionTitle(context, 'Depth Map'),
              const SizedBox(height: 16),
              const ColorMapDropdown(),
              const SizedBox(height: 16),
              const DepthMapView(),
              const SizedBox(height: 32),
              EstimationButton(
                onPressed: () => _runDepthEstimation(context),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSectionTitle(BuildContext context, String title) {
    return Text(
      title,
      style: Theme.of(context).textTheme.headlineSmall,
      textAlign: TextAlign.center,
    );
  }

  void _runDepthEstimation(BuildContext context) async {
    final appState = Provider.of<AppState>(context, listen: false);

    if (appState.selectedImage == null) {
      _showSnackbar('Please select an image first.');
      return;
    }

    appState.setProcessing(true);

    try {
      final result = await _depthEstimator.runDepthEstimation(
        appState.selectedImage!,
        appState.selectedColorMap,
      );
      appState.setDepthMap(result['depthMap']);
      appState.setInferenceTime(
          'Inference Time: ${result['inferenceTime']} ms');
    } catch (e) {
      _showErrorDialog('An error occurred during depth estimation: $e');
    } finally {
      appState.setProcessing(false);
    }
  }

  void _showSnackbar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

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
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }
}

class OriginalImageView extends StatelessWidget {
  const OriginalImageView({super.key});

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
        child: appState.selectedImage == null
            ? Text(
                'No image selected',
                style: TextStyle(color: Colors.grey[600]),
              )
            : ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  appState.selectedImage!,
                  fit: BoxFit.contain,
                  errorBuilder: (context, error, stackTrace) =>
                      const Text('Error loading image'),
                ),
              ),
      ),
    );
  }
}

class ColorMapDropdown extends StatelessWidget {
  const ColorMapDropdown({super.key});

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12.0),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(10.0),
        border: Border.all(
          color: Theme.of(context).dividerColor,
        ),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: appState.selectedColorMap,
          isExpanded: true,
          onChanged: (String? newValue) {
            if (newValue != null) {
              appState.setSelectedColorMap(newValue);
            }
          },
          items: <String>['Grayscale', 'Viridis']
              .map<DropdownMenuItem<String>>((String value) {
            return DropdownMenuItem<String>(
              value: value,
              child: Text(value),
            );
          }).toList(),
        ),
      ),
    );
  }
}