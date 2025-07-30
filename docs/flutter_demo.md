# Depth Estimation Demo App Documentation

## **Application Overview**

This Flutter application serves as a demonstration for a lightweight, mobile-optimized depth estimation model. It provides users with an intuitive interface to select an image from their device's gallery or camera. Once an image is chosen, the application runs a pre-trained ONNX model to generate a depth map. The resulting depth information is visualized as an image, and users have the option to apply different color maps (like Grayscale or Viridis) to better interpret the output.

The application is architecturally sound, featuring a clear separation of concerns, robust state management, and a focus on performance by offloading heavy computations to background isolates. It also includes both light and dark themes for an enhanced user experience.

---

## **Project Structure & File-by-File Documentation**

### **`lib/main.dart`**

This file is the main entry point for the application.

* **`main()`**: The primary function that launches the app. It wraps the entire application in a `ChangeNotifierProvider`, which makes the `AppState` class available to all widgets in the tree. This is the foundation of the app's state management.
* **`DepthEstimationApp`**: The root widget.
    * It builds the `MaterialApp`, which is the core component for a Flutter app.
    * **Theming**: It defines detailed `ThemeData` for both **light** and **dark** modes, including colors, fonts (`Inter`), and component-specific themes like `AppBarTheme`. The `themeMode` is dynamically linked to the `AppState`, allowing for seamless theme switching.
    * **Home**: It sets `DepthEstimationHomePage` as the initial screen of the application.

---

### **`lib/models/app_state.dart`**

This file defines the central state management for the application.

* **`AppState` class**: This class extends `ChangeNotifier` and acts as the single source of truth for the application's state.
    * **State Properties**: It manages all dynamic data, including:
        * `_selectedImage`: The original image file chosen by the user.
        * `_depthMapImageBytes`: The generated depth map image, stored as bytes.
        * `_rawDepthMap`: The raw numerical output from the ONNX model.
        * `_isProcessing` & `_isImageLoading`: Boolean flags to control UI state (e.g., showing loading indicators).
        * `_inferenceTime`: A string to display the model's processing time.
        * `_selectedColorMap`: The currently active color map ('Grayscale' or 'Viridis').
        * `_themeMode`: The current `ThemeMode` (light, dark, or system).
    * **State Modifiers**: It provides public methods (`setSelectedImage`, `setProcessing`, etc.) to modify the state. Crucially, each of these methods calls `notifyListeners()`, which signals to all listening widgets that they need to rebuild with the new state.

---

### **`lib/services/model_loader.dart`**

This service handles the loading and configuration of the ONNX model.

* **`ModelLoader` class**: Encapsulates all logic related to the ONNX runtime session.
    * **`loadModel()` method**:
        * Asynchronously loads the ONNX model file (`assets/Delta.onnx`) from the app's assets.
        * Configures platform-specific hardware acceleration for optimal performance:
            * **CoreML** on iOS.
            * **NNAPI** on Android.
        * Initializes and returns an `OrtSession` object, which is used to run the model.
        * Includes logging to provide insights into the model's input/output structure during development.
    * **`close()` method**: Releases the native resources held by the `OrtSession` to prevent memory leaks. This is called when the home page is disposed.

---

### **`lib/services/depth_estimator.dart`**

This service contains the core logic for performing depth estimation and processing the results. It is highly optimized for performance.

* **Isolate Functions**: To avoid freezing the UI thread during heavy computations, this file defines two top-level functions designed to be run in background isolates via `compute()`:
    * **`_processImageForModel()`**: Takes an image path, decodes it, resizes it to the required 384x384 dimensions, and normalizes it according to the model's requirements (ImageNet mean/std). This returns a `Float32List` ready to be fed into the model.
    * **`_applyColorMapIsolate()`**: Takes the raw numerical output from the model, normalizes the depth values to a 0-1 range, and applies the selected color map (Grayscale or Viridis) pixel by pixel to generate a `Uint8List` representing the final PNG image.
* **`DepthEstimator` class**: Orchestrates the entire estimation pipeline.
    * **`runDepthEstimation()` method**:
        1.  Calls `_processImageForModel` in an isolate.
        2.  Runs the model asynchronously with the processed input.
        3.  Times the inference process using a `Stopwatch`.
        4.  Returns a map containing the raw model output (`rawDepthMap`) and the `inferenceTime`.
    * **`applyColorMap()` method**:
        1.  Calls `_applyColorMapIsolate` in an isolate.
        2.  Returns the final, colorized depth map image bytes.

---

### **`lib/screens/home_page.dart`**

This file defines the main UI screen of the application.

* **`DepthEstimationHomePage`**: A `StatefulWidget` that constructs the main view and interacts with the services and state.
    * **`initState()`**: Initializes the `OrtEnv` and asynchronously loads the model using `ModelLoader`. The model loading is deferred until after the first frame to ensure a smooth app startup.
    * **`build()`**: Constructs the UI by assembling the various widgets (`ImagePickerButton`, `OriginalImageView`, `DepthMapView`, etc.). It uses a `SingleChildScrollView` to prevent overflow on smaller screens. It also includes the theme-toggle `IconButton` in the `AppBar`.
    * **`_runDepthEstimation()`**: The core logic triggered by the user. It orchestrates the calls to the `DepthEstimator` and updates the `AppState` with the results. It handles UI states like showing loading indicators and displaying errors.
    * **`_applyColorMapAndUpdateView()`**: A helper function to regenerate the depth map view when the color map is changed.
    * **Error Handling**: Includes helper methods (`_showSnackbar`, `_showErrorDialog`) to provide clear feedback to the user in case of failures.

---

### **`lib/widgets/` (Directory)**

This directory contains all the reusable UI components, promoting a clean and modular codebase.

* **`image_picker_button.dart`**: A styled `ElevatedButton` that, when pressed, presents a modal bottom sheet with "Photo Library" and "Camera" options for selecting an image.
* **`original_image_view.dart`**: A styled `Card` that displays the user-selected image. It shows a loading indicator while the image is being picked.
* **`depth_map_view.dart`**: A `Card` that displays the final depth map image. It intelligently shows a loading indicator during processing or a placeholder text if no image has been processed yet. It also displays the inference time.
* **`estimation_button.dart`**: The main action button ("Run Depth Estimation"). Its state (enabled/disabled, text, and icon) is dynamically controlled by the `AppState`. It shows a `CircularProgressIndicator` during processing.
* **`color_map_dropdown.dart`**: A styled `DropdownButton` that allows the user to switch between 'Grayscale' and 'Viridis' color maps. Changing the value here triggers a regeneration of the depth map image.

---

### **`lib/utils/color_maps.dart`**

This utility file provides color mapping functions.

* **`getViridisColor()`**: A function that implements the **Viridis color map**. It takes a normalized `double` value (0.0 to 1.0) and returns a `Color`. It works by finding the position of the value within a predefined list of Viridis colors and linearly interpolating (`Color.lerp`) between the two nearest colors to produce a smooth gradient.

---

### **`test/` (Directory)**

This directory contains the unit tests for the application, demonstrating a commitment to code quality and robustness.

* **`models/app_state_test.dart`**: Contains unit tests for the `AppState` class. It verifies that the initial state is correct and that the state modification methods work as expected (e.g., setting an image resets the depth map).
* **`utils/color_maps_test.dart`**: Contains unit tests for the `getViridisColor` function. It checks edge cases (0.0 and 1.0) and verifies that the color interpolation logic is correct for mid-range values.