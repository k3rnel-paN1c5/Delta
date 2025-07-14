import torch
from torchvision import transforms
from transformers import AutoModelForDepthEstimation
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

from models.teacher_model import DepthModel
from models.student_model import StudentModel
from criterions import DepthDistillationLoss
from utils import get_eval_transforms
from utils import plot_depth_comparison

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    print("Loading teacher model...")
    teacher_base = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    teacher_model = DepthModel(teacher_base).to(device)
    teacher_model.eval()

    print(f"Loading student checkpoint from {args.checkpoint_path}...")
    student_model = StudentModel().to(device)
    student_model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    student_model.eval()

    criterion = DepthDistillationLoss()

    # --- Prepare Image ---
    transform = get_eval_transforms(input_size=(384, 384))
    
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # --- Get Predictions ---
    with torch.no_grad():
        teacher_depth_tensor = teacher_model(input_tensor)
        student_depth_tensor = student_model(input_tensor)
        
        loss = criterion(student_depth_tensor, teacher_depth_tensor)
        print(f"Evaluation Loss (Student vs. Teacher): {loss.item():.4f}")

        # Convert to numpy for visualization
        teacher_depth = teacher_depth_tensor.squeeze().cpu().numpy()
        student_depth = student_depth_tensor.squeeze().cpu().numpy()

    # --- Visualize ---
    plot_depth_comparison(image, teacher_depth, student_depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained student model.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the evaluation image.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the student model checkpoint (.pth file).')
    args = parser.parse_args()
    evaluate(args)



# python scripts/evaluate.py --image_path data/test.jpg --checkpoint_path checkpoints/student_epoch_15.pth