# Delta: A Lightweight, High-Quality Monocular Depth Estimation Project

**Delta** is a complete, end-to-end computer vision project for monocular depth estimation. It demonstrates how to leverage **knowledge distillation** to train a lightweight, efficient "student" model from a large, state-of-the-art "teacher" model. The final distilled model is deployed in a cross-platform **Flutter application** that can run inference on-device.

The core of this project is a **MobileVit-based student model** that learns to predict depth from a single RGB image by imitating the behavior of the powerful **DPT (Dense Prediction Transformer)** model. This approach creates a final model that is both accurate and fast enough for mobile applications.

## ✨ Key Features

  * **Teacher-Student Architecture**: Uses the `depth_anything_v2` model as a teacher to ensure high-quality depth predictions.
  * **Efficient Student Model**: Employs a custom DPT-style decoder on a **MobileViT** backbone, making it small and fast.
  * **Advanced Knowledge Distillation**: Implements a combined distillation loss function, usin pixel-wise, gradient matching, scale-invaraint, and feature-based losses to effectively transfer knowledge from the teacher's output and intermediate layers.
  * **End-to-End Pipeline**: Provides a full workflow from model training and evaluation to exporting for mobile deployment.
  * **Model Export**: Includes scripts to convert the trained PyTorch model to **ONNX** for mobile use.
  * **Cross-Platform Mobile App**: A Flutter app for Android and iOS that lets you pick an image from your gallery and visualize the generated depth map in real-time.

-----

## 📂 Project Structure

The repository is organized into two main parts: the `model` and the `app`.

```
delta/
├── app/         # Flutter mobile application
│   ├── lib/     # Core Dart application logic
│   ├── assets/  # Location for the .onnx model
│   └── ...
└── model/       # PyTorch model, training, and conversion scripts
    ├── config/  # configurations for the pipeline
    ├── criterions/ # Knowledge distillation loss functions
    ├── data/ # Custom data used for training and evaluating
    ├── datasets/ # Dataset loaders used for loading and preproccessing raw data 
    ├── models/  # Teacher and student model definitions
    ├── notebooks/  # Jupyter notebooks for experimentation
    ├── scripts/    # Training, inference, and evaluation scripts
    ├── utils/      # Utility scripts, including pth -> onnx converter
    ├── requrements.txt    # requirments for the python environment
    └── ...
```

-----

## 🧠 Model & Training

The core of the project is the knowledge distillation process, which trains the student model.

### Teacher Model

The teacher is the **Depth-AnythingV2** model, a powerful Vision Transformer-based model fine-tuned for monocular depth estimation on the MiDaS dataset. It provides the high-quality depth predictions that the student model learns to replicate.

### Student Model

The student model is designed to be efficient. It uses a **MobileViTV2** encoder, pre-trained on ImageNet, to extract features. These features are then passed to a custom, lightweight **DPT-style decoder** (`MiniDPT`) which upsamples them to produce the final depth map.

### Distillation

The training process minimizes a **`CombinedDistillationLoss`**, which is a weighted sum of four components:

1.  **Response-Based Loss**: An **L1 loss** between the teacher's final depth map and the student's prediction.
2.  **Feature-Based Loss**: A **cosine similarity** loss between the feature maps extracted from an intermediate layer of the teacher and the student. This encourages the student to learn the teacher's internal feature representation.

-----

## 🚀 Getting Started

### 1\. Model Training & Conversion

To train the model yourself or run inference, navigate to the `model` directory.

**Prerequisites:**

  * Python 3.8+
  * PyTorch
  * CUDA-enabled GPU (recommended for training)

**Installation:**

```bash
# Navigate to the model directory
cd model

# Install all required Python packages
pip install -r requirements.txt
```

**Training:**
The training process is handled by the `scripts/train.py` script. You will need to provide paths to your dataset and configure the training parameters in a config file (see `config/config.py`).

**Inference & Export:**

1.  **Run Inference**: Use `scripts/infer.py` to test the trained model on sample images.
2.  **Export to ONNX**: Use the `utils/pth2onnx.py` script to convert your trained PyTorch model (`.pth`) into the ONNX format.
3.  **Convert to TFLite**: Use the official TensorFlow ONNX converter to convert the `.onnx` file to a `.tflite` file. This is the final model that will be used in the mobile app.
    ```bash
    # Example conversion command
    pip install onnx-tf
    onnx-tf convert -i model.onnx -o model.tflite
    ```

### 2\. Running the Flutter App

**Prerequisites:**

  * [Flutter SDK](https://docs.flutter.dev/get-started/install) installed on your machine.
  * An Android or iOS device/emulator.

**Setup & Run:**

1.  **Place the Model**: Copy your generated `model.tflite` file and place it inside the `app/assets/` directory.
2.  **Get Dependencies**:
    ```bash
    # Navigate to the app directory
    cd app

    # Install Flutter dependencies
    flutter pub get
    ```
3.  **Run the App**:
    ```bash
    # Launch the application
    flutter run
    ```

The app will launch, allowing you to select an image and view the resulting depth map rendered with a "viridis" colormap.

-----

## 📜 License

This project is licensed under the **MIT License**. You can find the license file [here](https://www.google.com/search?q=./LICENSE).

@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
