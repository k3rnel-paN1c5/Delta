import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Assuming your model and utility classes are in the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.student_model import StudentDepthModel
from models.teacher_model import TeacherWrapper
from datasets.data_loader import UnlabeledImageDataset
from utils.metrics import compute_depth_metrics
from config import config
from utils.transforms import get_eval_transforms

def evaluate(args):
    """
    Evaluates the student model against the teacher model on a given dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load the models ---
    print("Loading models...")
    student_model = StudentDepthModel(encoder_name='mobilevit_xs', pretrained=False).to(device)
    student_model.load_state_dict(torch.load(args.model_path, map_location=device))
    student_model.eval()

    teacher_model = TeacherWrapper().to(device)
    teacher_model.eval()
    print("Models loaded successfully.")

    # --- 2. Set up the dataset and dataloader ---
    input_size = (384, 384)
    eval_transform = get_eval_transforms(input_size)
    eval_dataset = UnlabeledImageDataset(root_dir=args.dataset_path, transform=eval_transform, resize_size=input_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating on {len(eval_dataset)} images.")

    # --- 3. Initialize metrics ---
    total_metrics = {
        'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0,
        'rmse_log': 0.0, 'a1': 0.0, 'a2': 0.0, 'a3': 0.0
    }
    num_samples = 0

    # --- 4. Evaluation loop ---
    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc="Evaluating")
        for images in progress_bar:
            images = images.to(device)

            # Get predictions
            teacher_depth, _ = teacher_model(images)
            student_depth, _ = student_model(images)

            # Compute metrics for the batch
            metrics = compute_depth_metrics(student_depth, teacher_depth)
            
            # Accumulate metrics
            batch_size = images.size(0)
            for key in total_metrics:
                total_metrics[key] += metrics[key] * batch_size
            num_samples += batch_size

    # --- 5. Calculate and print average metrics ---
    avg_metrics = {key: total / num_samples for key, total in total_metrics.items()}
    
    print("\n--- Evaluation Results ---")
    print(f"Absolute Relative Difference (AbsRel): {avg_metrics['abs_rel']:.4f}")
    print(f"Squared Relative Difference (SqRel): {avg_metrics['sq_rel']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {avg_metrics['rmse']:.4f}")
    print(f"Root Mean Squared Log Error (RMSELog): {avg_metrics['rmse_log']:.4f}")
    print(f"Delta1 (a1): {avg_metrics['a1']:.4f}")
    print(f"Delta2 (a2): {avg_metrics['a2']:.4f}")
    print(f"Delta3 (a3): {avg_metrics['a3']:.4f}")
    print("--------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script for Student Depth Estimation Model")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained student model (.pth file)")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the evaluation dataset directory")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for evaluation")

    args = parser.parse_args()
    evaluate(args)