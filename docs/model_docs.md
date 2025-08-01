# Depth Estimation via Knowledge Distillation: Project Documentation

## **Project Overview** 🧠

This project implements a **knowledge distillation** framework for training a lightweight, efficient student model for the task of **monocular depth estimation**. The core idea is to transfer the knowledge from a large, powerful, pre-trained "teacher" model (like `DepthAnythingV2`) to a smaller, mobile-friendly "student" model (`MobileViT` with a custom decoder). This allows the student to achieve high accuracy without the computational overhead of the teacher.

The architecture is highly **modular and extensible**, employing design patterns like factories and strategies to facilitate easy experimentation with different models, datasets, and loss functions. The distillation process is comprehensive, leveraging a combination of pixel-wise, feature-level, and attention-based losses to ensure the student learns a rich and detailed representation of depth. The project also includes a robust testing suite and utilities for model deployment, such as ONNX conversion.

---

## **Project Structure & File-by-File Documentation**

### **`train.py`**

This is the main entry point for training the student model.

* **`main()` function**: Orchestrates the entire training pipeline.
    1.  **Configuration**: Loads hyperparameters from `config.py`.
    2.  **Factory Instantiation**: Uses the various factories (`data`, `models`, `criterions`, `utils`) to create all necessary components:
        * Data loaders (for training and validation).
        * Teacher and student models.
        * The combined distillation loss function.
        * Optimizer and learning rate scheduler.
    3.  **Training Loop**: Iterates through the training data for a specified number of epochs.
        * For each batch, it performs a forward pass through both the teacher and student models.
        * Calculates the combined distillation loss.
        * Performs backpropagation and updates the student model's weights.
    4.  **Evaluation**: After each epoch, it runs the an evaluation loop to measure the model's performance on the validation set.
    5.  **Checkpointing**: Saves the best-performing model based on validation metrics.

---

### **`evaluate.py`**

This script is responsible for evaluating a trained model's performance.

* **`evaluate()` function**:
    * Sets the model to evaluation mode (`model.eval()`).
    * Iterates through the validation or test dataset without computing gradients (`torch.no_grad()`).
    * For each batch, it passes the input image through the student model to get a depth prediction.
    * **Metrics Calculation**: It computes a set of standard depth estimation metrics (e.g., RMSE, AbsRel, δ1) using the functions from `utils/metrics.py`.
    * Returns a dictionary of aggregated metrics, which is used to track performance and save the best model during training.

---

### **`config.py`**

This file serves as the centralized configuration hub for the entire project. It defines all hyperparameters and settings in a clean, organized `Config` class, including:

* **`Paths`**: Directories for data, and saving models.
* **`Data`**: Image dimensions, data transformation settings,  batch size, and number of workers.
* **`Model`**: Names of the teacher (`DepthAnythingV2`) and student (`MobileViT-XS`) models, and decoder settings.
* **`Distillation`**: Weights for each component of the combined loss function (SILog, gradient, feature, attention).
* **`Training`**: Number of epochs, learning rate, optimizer type, and scheduler settings.

---

### **`models/` (Directory)** 🛠️

This directory contains all model-related code, including the teacher, student, and the factory for creating them.

* **`factory.py`**: Implements the **Model Factory** pattern.
    * **`model_factory()`**: A single function that takes the `Config` object as input and returns the instantiated teacher and student models. It handles the logic for selecting and initializing the correct models based on the configuration.
* **`teacher_model.py`**:
    * **`TeacherWrapper` class**: A wrapper around the Hugging Face `DepthAnythingV2` model. This is a crucial abstraction that simplifies the interaction with the complex teacher model.
        * It handles the loading of the pre-trained model and processor from Hugging Face.
        * The `forward()` method processes the input image and extracts both the final depth prediction and the intermediate hidden features, which are necessary for feature and attention distillation. It also reshapes the ViT features to be spatially compatible with the student's CNN-like features.
* **`student_model.py`**:
    * **`StudentDepthModel` class**: The main student model.
        * It uses a pre-trained `MobileViT` as its encoder to extract features from the input image.
        * It passes these features to a custom `MiniDPT` decoder (`models/decoder.py`) to predict the final depth map.
* **`decoder.py`**:
    * **`MiniDPT` class**: A lightweight decoder inspired by the DPT (Dense Prediction Transformer) architecture. It's composed of several `FeatureFusionBlock` and `UpsampleBlock` modules.
    * **`FeatureFusionBlock`**: Combines features from different levels of the encoder, allowing the model to integrate both high-level semantic information and low-level spatial details.
    * **`UpsampleBlock`**: Progressively upsamples the fused features to restore the original image resolution, resulting in a dense depth map.

---

### **`criterions/` (Directory)** 📉

This directory defines the loss functions used for knowledge distillation. It cleverly uses a **Strategy Pattern**.

* **`factory.py`**: Implements the **Criterion Factory**.
    * **`criterion_factory()`**: Takes the `Config` and returns the appropriate loss calculation strategy (e.g., `CombinedDistillationLoss`).
* **`criterion.py`**:
    * **`Criterion` class**: An abstract base class that defines the interface for all loss strategies. It has a single `calculate()` method. This allows the main training loop to be agnostic to the specific loss function being used.
* **`combined_distillation_loss.py`**:
    * **`CombinedDistillationLoss` class**: The core distillation loss. Its `calculate()` method computes a weighted sum of several individual loss components:
        1.  **`SILogLoss`**: Measures the scale-invariant error between the student's and teacher's depth predictions.
        2.  **`GradientLoss`**: Encourages the student to replicate the fine-grained details and edges present in the teacher's prediction.
        3.  **Feature Matching Loss (`L1Loss`)**: Pushes the student's intermediate features to be similar to the teacher's, ensuring representational alignment.
        4.  **Attention Matching Loss**: A custom loss that encourages the student's attention maps to mimic the teacher's, forcing it to "look at" the same important regions in the image.
* **`simple_distillation_loss.py`**:
    * **`SimpleDistillationLoss` class**: a simple distillation loss. Its `calculate()` method computes the MSE for  the  pixel-wise depth values.
---

### **`datasets/` (Directory)** 📊

This module handles all data loading and preprocessing.

* **`factory.py`**: Implements the **Data Factory**.
    * **`data_factory()`**: Creates and returns the training and validation `DataLoader` objects based on the dataset name specified in the config.
* **`data_loader.py`**:
    * **`UnlabeledImageDataset` class**: A custom `torch.utils.data.Dataset` for the a custom raw images dataset.
        * It handles finding the image and applying the transformation on the image.
        * The `__getitem__()` method loads an image, applies necessary augmentations and transformations (like resizing and normalization), and returns it.
### **`data/` (Directory)** 📊

This directory contains the data along with a utility script.

* **`raw/` (Directory)**: contains raw unlabled images that I took with mobile camera for the sake of this project
* **`name_images.sh`**:
    * **`UnlabeledImageDataset` class**: A custom `torch.utils.data.Dataset` for the a custom raw images dataset.
        * A shell script used to name images in a directory with numbers (image1.JPG, image2.JPG, ...)

---

### **`utils/` (Directory)** ⚙️

This directory contains various helper functions and utility classes.

* **`factory.py`**: A factory for creating optimizers and learning rate schedulers (e.g., `AdamW`, `CosineAnnealingLR`).
* **`metrics.py`**: Contains functions to compute standard depth estimation evaluation metrics (`rmse`, `abs_rel`, etc.). These are used in `evaluate.py`.
* **`transforms.py`**: Provides the transforms for training and evaluation.
* **`visuals.py`**: Has two functions, one for applying a color map to a depth map, and one to plot the image along with its depth maps.
* **`pth2onnx.py`**: A utility script to convert the trained student model (saved as a `.pth` file) into the **ONNX (Open Neural Network Exchange)** format. This is a critical step for deploying the model on mobile devices or in web browsers.

---

### **`tests/` (Directory)** ✅

This directory demonstrates a strong commitment to code quality by providing a comprehensive testing suite.

* **Unit Tests**: Contains tests for individual, isolated components.
    * `test_criterion.py`: Tests the individual loss components with known inputs to ensure they produce the correct outputs.
    * `test_metrics.py`: Verifies the correctness of the evaluation metric calculations.
    * It also includes tests for model components and utilities to ensure they behave as expected.
* **Integration Tests**:
    * `test_training_pipeline.py`: A crucial test that runs a single step of the entire training pipeline with mock data. It ensures that all components (data loading, model forward pass, loss calculation, and optimizer step) work together correctly. This helps catch bugs that unit tests might miss.