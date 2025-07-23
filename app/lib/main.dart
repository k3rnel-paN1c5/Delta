import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib;
import 'package:logging/logging.dart';
final Logger _logger = Logger('ModelLoader');

void main() {
  runApp(const DepthEstimationApp());
}

class DepthEstimationApp extends StatelessWidget {
  const DepthEstimationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Depth Estimation Demo',
      theme: ThemeData(
        primarySwatch: Colors.blueGrey,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Inter', 
      ),
      home: const DepthEstimationHomePage(),
    );
  }
}

class DepthEstimationHomePage extends StatefulWidget {
  const DepthEstimationHomePage({super.key});

  @override
  State<DepthEstimationHomePage> createState() => _DepthEstimationHomePageState();
}

class _DepthEstimationHomePageState extends State<DepthEstimationHomePage> {
  File? _selectedImage; // Stores the original image picked by the user
  Uint8List? _depthMapImageBytes; // Stores the bytes of the processed depth map image
  Interpreter? _interpreter; // TensorFlow Lite interpreter
  bool _isProcessing = false; // To show loading indicator
  double? _depthMapDisplayWidth;
  double? _depthMapDisplayHeight;
  String _inferenceTime = '';

  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];
  
  @override
  void initState() {
    super.initState();
    _loadModel(); // Load the TFLite model when the app starts
  }

  // Function to load the TensorFlow Lite model
  Future<void> _loadModel() async {
    try {
      // Replace 'assets/depth_model.tflite' with the actual path to your model file.
      // Make sure you've added the model to your pubspec.yaml assets section.
      _interpreter = await Interpreter.fromAsset('assets/Delta.tflite');
      _logger.info('Model loaded successfully!'); // Use info level for successful operations
      _logger.info('Input Tensors: ${_interpreter?.getInputTensors()}');
      _logger.info('Output Tensors: ${_interpreter?.getOutputTensors()}');
    } catch (e) {
      _logger.severe('Failed to load model: $e');
      // Optionally show an error dialog to the user
      _showErrorDialog('Failed to load AI model. Please ensure the model file is correct and accessible.');
    }
  }

  // Function to pick an image from the gallery
  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _inferenceTime = ''; // Clear previous inference time
        _depthMapImageBytes = null; // Clear previous depth map
        _depthMapDisplayWidth = null; // Clear previous dimensions
        _depthMapDisplayHeight = null; // Clear previous dimensions
      });
      // Automatically run depth estimation after picking an image
      _runDepthEstimation();
    }
  }

  // Placeholder function for running depth estimation using the loaded model
  Future<void> _runDepthEstimation() async {
    if (_selectedImage == null) {
      _showSnackbar('Please select an image first.');
      return;
    }
    if (_interpreter == null) {
      _showSnackbar('AI model not loaded. Cannot perform depth estimation.');
      return;
    }

    setState(() {
      _isProcessing = true; // Start showing loading indicator
      _inferenceTime = '';
    });

    try {
      // 1. Pre-process the image:
      //    - Read image bytes
      //    - Decode image
      //    - Resize to model's input size (e.g., 256x256, 512x512)
      //    - Normalize pixel values (e.g., to 0-1 or -1 to 1)
      //    - Convert to a format compatible with the model (e.g., Float32List, Int32List)

      final img_lib.Image? originalImage = img_lib.decodeImage(_selectedImage!.readAsBytesSync());
      if (originalImage == null) {
        _showErrorDialog('Failed to decode image.');
        setState(() { _isProcessing = false; });
        return;
      }
      // Get model input shape and type dynamically
      final inputTensor = _interpreter!.getInputTensors()[0];
      final inputShape = inputTensor.shape;
      final inputType = inputTensor.type;

      if (inputShape.length != 4) {
        _showErrorDialog('Unexpected model input shape. Expected [1, height, width, channels]. Got $inputShape');
        setState(() { _isProcessing = false; });
        return;
      }

      final int inputHeight = inputShape[1];
      final int inputWidth = inputShape[2];

      // Resize the image to the model's expected input dimensions
      final img_lib.Image resizedImage = img_lib.copyResize(
        originalImage,
        width: inputWidth,
        height: inputHeight,
      );

      
      var input = List.generate(1, (i) => List.generate(inputHeight, (j) => List.generate(inputWidth, (k) => List.filled(3, 0.0))));

      if (inputType == TensorType.float32) {
        // Example for a float32 model (common for deep learning models)
        input = List.generate(1, (batch) {
          return List.generate(inputHeight, (y) {
            return List.generate(inputWidth, (x) {
              final pixel = resizedImage.getPixel(x, y);
              final r = pixel.r;
              final g = pixel.g;
              final b = pixel.b;
              return [
                (r / 255.0 - _mean[0]) / _std[0],
                (g / 255.0 - _mean[1]) / _std[1],
                (b / 255.0 - _mean[2]) / _std[2],
              ];
            });
          });
        });
      } else if (inputType == TensorType.uint8) {
        // Example for a uint8 model
        input = List.generate(1, (batch) {
          return List.generate(inputHeight, (y) {
            return List.generate(inputWidth, (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                pixel.r/1.0,
                pixel.g/1.0,
                pixel.b/1.0,
              ];
            });
          });
        });
      } else {
        _showErrorDialog('Unsupported input tensor type: $inputType');
        setState(() { _isProcessing = false; });
        return;
      }

      // 2. Run inference:
      //    - Define output tensor structure based on your model's output
      //    - Run the interpreter
      final outputTensors = _interpreter!.getOutputTensors();
      if (outputTensors.isEmpty) {
        _showErrorDialog('Model has no output tensors defined.');
        setState(() { _isProcessing = false; });
        return;
      }

      final outputTensor = outputTensors[0];
      final outputShape = outputTensor.shape;
      final outputType = outputTensor.type;

      // Initialize output buffer based on the expected output shape and type
      // This assumes a single output tensor. Adjust if your model has multiple outputs.
      dynamic output;
      if (outputType == TensorType.float32) {
        output = List.filled(outputShape.reduce((a, b) => a * b), 0.0).reshape(outputShape);
      } else if (outputType == TensorType.uint8) {
        output = List.filled(outputShape.reduce((a, b) => a * b), 0).reshape(outputShape);
      } else {
        _showErrorDialog('Unsupported output tensor type: $outputType');
        setState(() { _isProcessing = false; });
        return;
      }
      final Stopwatch stopwatch = Stopwatch()..start();

      _interpreter!.run(input, output);

      stopwatch.stop();
      // 3. Post-process the output:
      //    - Convert the raw model output (e.g., depth values) into a visual image (e.g., grayscale depth map)
      //    - Normalize depth values to 0-255 for image display
      //    - Create an image from the processed data

      // Example post-processing for a grayscale depth map:
      // Assuming output is [1, height, width, 1] for a single channel depth map
      if (outputShape.length == 4 && outputShape[3] == 1) {
        final int outputHeight = outputShape[1];
        final int outputWidth = outputShape[2];

        // Find min and max depth values for normalization
        double minDepth = double.infinity;
        double maxDepth = double.negativeInfinity;

        for (int y = 0; y < outputHeight; y++) {
          for (int x = 0; x < outputWidth; x++) {
            final depthValue = (output[0][y][x][0] as num).toDouble();
            if (depthValue < minDepth) minDepth = depthValue;
            if (depthValue > maxDepth) maxDepth = depthValue;
          }
        }

        final depthImage = img_lib.Image(width: outputWidth, height:outputHeight);
        for (int y = 0; y < outputHeight; y++) {
          for (int x = 0; x < outputWidth; x++) {
            final depthValue = (output[0][y][x][0] as num).toDouble();
            // Normalize depth to 0-255 and convert to grayscale pixel
            final normalizedDepth = (depthValue - minDepth) / (maxDepth - minDepth);
            final pixelValue = (normalizedDepth * 255).round().clamp(0, 255);
            depthImage.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
          }
        }
        // Calculate aspect ratio and determine display dimensions
        final double aspectRatio = outputWidth / outputHeight;
        // You might want to cap the height or width to prevent it from being too large on small screens.
        // For example, if you want it to take up a maximum of 80% of the screen width:
        final double screenWidth = MediaQuery.of(context).size.width - 32; // Account for padding
        double displayWidth = screenWidth;
        double displayHeight = screenWidth / aspectRatio;

        // Ensure the display height doesn't exceed a reasonable maximum, or adjust as needed
        if (displayHeight > 400) { // Example max height
          displayHeight = 400;
          displayWidth = displayHeight * aspectRatio;
        }
        setState(() {
          _depthMapImageBytes = Uint8List.fromList(img_lib.encodePng(depthImage));
          _inferenceTime = 'Inference Time: ${stopwatch.elapsedMilliseconds} ms';
          _depthMapDisplayWidth = displayWidth;
          _depthMapDisplayHeight = displayHeight;
        });
      } else {
        _showErrorDialog('Unexpected model output shape for depth map. Expected [1, height, width, 1]. Got $outputShape');
      }
    } catch (e) {
      _logger.severe('Error during depth estimation: $e');
      _showErrorDialog('An error occurred during depth estimation: $e');
    } finally {
      setState(() {
        _isProcessing = false; // Stop loading indicator
      });
    }
  }

  // Helper to show a SnackBar message
  void _showSnackbar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  // Helper to show an error dialog
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

  @override
  void dispose() {
    _interpreter?.close(); // Close the interpreter when the widget is disposed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Depth Estimation Demo'),
        centerTitle: true,
      ),
      body: SingleChildScrollView( // Allows content to scroll if it overflows
        child: Padding(
          padding: const EdgeInsets.all(40.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              // Section for Original Image
              Text(
                'Original Image',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
               // Button to pick image
              ElevatedButton.icon(
                onPressed: _pickImage,
                icon: const Icon(Icons.photo_library),
                label: const Text('Pick Image from Gallery'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(fontSize: 18),
                ),
              ),
              const SizedBox(height: 10),
              Container(
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blueGrey.shade200),
                ),
                height: 300, // Fixed height for image display
                alignment: Alignment.center,
                child: _selectedImage == null
                    ? Text(
                        'No image selected',
                        style: TextStyle(color: Colors.grey[600]),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: Image.file(
                          _selectedImage!,
                          fit: BoxFit.contain, // Adjusts image to fit container
                          errorBuilder: (context, error, stackTrace) =>
                              const Text('Error loading image'),
                        ),
                      ),
              ),
              const SizedBox(height: 20),

              // Section for Depth Map
              Text(
                'Depth Map',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Container(
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blueGrey.shade200),
                ),
                width: _depthMapDisplayWidth,
                height: _depthMapDisplayHeight,
                alignment: Alignment.center,
                child: _isProcessing
                    ? const CircularProgressIndicator() // Show loading indicator
                    : _depthMapImageBytes == null
                        ? Text(
                            _selectedImage == null ? 'Select an image to see depth map' : 'No depth map generated yet',
                            style: TextStyle(color: Colors.grey[600]),
                          )
                        : ClipRRect(
                            borderRadius: BorderRadius.circular(10),
                            child: Image.memory(
                              _depthMapImageBytes!,
                              fit: BoxFit.contain,
                              errorBuilder: (context, error, stackTrace) =>
                                  const Text('Error loading depth map'),
                            ),
                          ),
              ),
              if (_inferenceTime.isNotEmpty)
                Padding( // Add padding for better spacing
                  padding: const EdgeInsets.only(top: 10.0),
                  child: Text(
                    _inferenceTime,
                    style: Theme.of(context).textTheme.titleMedium,
                    textAlign: TextAlign.center,
                  ),
                ),
              const SizedBox(height: 30),

             
              const SizedBox(height: 10),
              // Button to run estimation (optional, as it runs automatically after pick)
              // This button can be useful for re-running if needed.
              ElevatedButton.icon(
                onPressed: _isProcessing ? null : _runDepthEstimation, // Disable when processing
                icon: _isProcessing
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      )
                    : const Icon(Icons.auto_awesome),
                label: Text(_isProcessing ? 'Processing...' : 'Run Depth Estimation'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(fontSize: 18),
                  backgroundColor: Colors.teal, // A different color for distinction
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
