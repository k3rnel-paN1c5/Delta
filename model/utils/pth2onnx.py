import torch
import os
import sys
import argparse

from models.student_model import StudentDepthModel
from config import config

# to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def convert_to_onnx(trained_model_path: str, output_dir: str, verbose: bool):
    """
    Converts a trained PyTorch model (.pth) to the ONNX format.

    Args:
        trained_model_path (str): The path to the saved .pth model file.
        output_dir (str): The directory to save the exported ONNX file.
        verbose (bool): If True, prints a detailed description of the export process.
    """
    if not os.path.exists(trained_model_path):
        print(f"Error: Model file not found at '{trained_model_path}'")
        return

    # --- Setup ---
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # --- Load Model ---
    print("Loading the student model...")
    model = StudentDepthModel().to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Create Dummy Input ---
    input_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    dummy_input = torch.randn(1, 3, *input_size, requires_grad=True).to(device)

    # --- Prepare Output Path ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a descriptive output filename based on the input model
    base_name = os.path.basename(trained_model_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f"{file_name_without_ext}.onnx")

    # --- Export the model ---
    print(f"Exporting model to {output_path}...")
    try:
        torch.onnx.export(model,
                          dummy_input,
                          output_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},
                                        'output' : {0 : 'batch_size'}},
                          verbose=verbose)
        print(f"\nModel successfully exported to ONNX format! ✅")
        print(f"Saved at: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during ONNX export: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to ONNX format.")
    
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Path to the trained PyTorch model (.pth file)."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="exports", 
        help="Directory to save the exported .onnx file. Defaults to 'exports/' in the project root."
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output during the ONNX export process."
    )

    args = parser.parse_args()

    convert_to_onnx(args.model_path, args.output_dir, args.verbose)

