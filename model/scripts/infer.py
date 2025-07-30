import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Assuming your model classes are in the parent directory under 'models'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.student_model import StudentDepthModel
from utils.visuals import apply_color_map

def infer(args):
    """
    Performs inference on a single image or a directory of images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load the trained student model ---
    print("Loading the Student Depth Model...")
    model = StudentDepthModel(encoder_name='mobilevit_xs', pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Define image transformations ---
    input_size = (384, 384)
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. Get list of images to process ---
    if os.path.isdir(args.input_path):
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith(('png', 'jpg', 'jpeg'))]
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
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)

                # --- 6. Run inference ---
                predicted_depth, _ = model(input_tensor)
                predicted_depth = predicted_depth.squeeze().cpu().numpy()

                # --- 7. Visualize and save the output ---
                output_filename = os.path.splitext(os.path.basename(img_path))[0] + "_depth.png"
                output_filepath = os.path.join(args.output_path, output_filename)

                colored_depth = apply_color_map(predicted_depth)
                plt.imsave(output_filepath, colored_depth, cmap='inferno')
                print(f"Saved depth map to: {output_filepath}")

            except Exception as e:
                print(f"Could not process {img_path}. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for Student Depth Estimation Model")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained student model (.pth file)")
    parser.add_argument('--input-path', type=str, required=True, help="Path to an input image or a directory of images")
    parser.add_argument('--output-path', type=str, default='../output', help="Directory to save the output depth maps")

    args = parser.parse_args()
    infer(args)