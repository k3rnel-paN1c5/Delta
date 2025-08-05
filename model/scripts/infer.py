"""
This script performs inference with a trained student depth estimation model on a single image or a directory of images.

Key functionalities include:
1.  **Model Loading**: Loads a trained student model from a specified checkpoint file (.pth).
2.  **Image Preprocessing**: Applies the same evaluation transformations used during training to the input image(s).
3.  **Inference**: Runs the model on the preprocessed image tensor to generate a predicted depth map.
4.  **Visualization and Saving**: Applies a colormap to the grayscale depth map for better visualization and saves the result as a PNG image in a specified output directory.

The script is executed via the command line and requires paths to the trained model and the input image or directory. The output path is optional.
"""

import torch
import torch.nn.functional as F
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config
from utils.transforms import get_eval_transforms
from models.student_model import StudentDepthModel
from utils.crop_aspect_ratio import crop_to_aspect_ratio
from utils.visuals import apply_color_map


def infer(args):
    """
    Performs inference on a single image or a directory of images.
    """
    device = config.DEVICE
    print(f"Using device: {device}")

    # --- 1. Load the trained student model ---
    print("Loading the Student Depth Model...")
    model = StudentDepthModel(pretrained=True).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Define image transformations ---
    input_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    transform = get_eval_transforms(input_size)
    target_aspect_ratio = config.IMG_WIDTH / config.IMG_HEIGHT

    # --- 3. Get list of images to process ---
    if os.path.isdir(args.input_path):
        image_paths = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
    else:
        image_paths = [args.input_path]

    if not image_paths:
        print("No images found at the specified path.")
        return

    # --- 4. Create output directory if it doesn't exist ---
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output will be saved to: {args.output_path}")

    # --- 5. Process each image ---
    with torch.no_grad():
        for img_path in image_paths:
            print(f"Processing: {img_path}")
            try:
                image = Image.open(img_path).convert("RGB")
                
                cropped_image = crop_to_aspect_ratio(image, target_aspect_ratio)
                cropped_size = cropped_image.size # Returns (width, height)
                
                # original_size = image.size # Returns (width, height)
                input_tensor = transform(cropped_image).unsqueeze(0).to(device)

                # --- 6. Run inference ---
                predicted_depth, _ = model(input_tensor)
                
                resized_depth = F.interpolate(
                    predicted_depth,
                    size=(cropped_size[1], cropped_size[0]),
                    mode='bilinear',
                    align_corners=False
                )
                
                predicted_depth_numpy = resized_depth.squeeze().cpu().numpy()

                # --- 7. Visualize and save the output ---
                output_filename = (
                    os.path.splitext(os.path.basename(img_path))[0] + "_depth.png"
                )
                output_filepath = os.path.join(args.output_path, output_filename)

                colored_depth = apply_color_map(predicted_depth_numpy, cmap="viridis")
                plt.imsave(output_filepath, colored_depth, cmap="viridis")
                print(f"Saved depth map to: {output_filepath}")

            except Exception as e:
                print(f"Could not process {img_path}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for Student Depth Estimation Model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained student model (.pth file)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to an input image or a directory of images",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../output",
        help="Directory to save the output depth maps",
    )

    args = parser.parse_args()
    infer(args)
