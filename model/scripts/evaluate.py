"""Evaluation script for the depth estimation model.

This script is responsible for evaluating the performance of a trained student
model on a validation or test dataset. It provides quantitative metrics to
assess the model's accuracy and is used during the training process to track
progress and identify the best-performing model checkpoint.

The main `evaluate` function performs the following steps:
    1.  **Sets Model to Evaluation Mode**: It ensures the model is in `eval()`
        mode, which disables layers like dropout and batch normalization that
        behave differently during training and inference.
    2.  **Disables Gradient Computation**: It uses the `torch.no_grad()` context
        manager to prevent the computation of gradients, which saves memory and
        computation time during inference.
    3.  **Iterates Through Data**: It loops through the specified dataset
        (e.g., validation set).
    4.  **Inference**: For each batch of images, it performs a forward pass through
        the student model to obtain a depth prediction.
    5.  **Metrics Calculation**: It computes a standard set of depth estimation
        metrics by comparing the student's prediction to the teacher's prediction
        (or ground truth if available). These metrics are calculated using the
        utility functions in `utils/metrics.py`.
    6.  **Aggregates and Returns Metrics**: It aggregates the metrics over all
        batches and returns a dictionary containing the final, averaged scores
        (e.g., RMSE, AbsRel, a1-a3).
"""

import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Union

import time
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_model import StudentDepthModel
from models.teacher_model import TeacherWrapper
from datasets.data_loader import UnlabeledImageDataset
from utils.metrics import compute_depth_metrics
from config import config
from utils.transforms import get_eval_transforms



def evaluate(
    student_path: torch.nn.Module,
    dataset_path: str,
    batch_size: int
    ) -> Dict[str, float]:
    """
    Evaluates the student model's performance on the validation set.

    This function iterates through the validation dataset, computes depth
    predictions from both the student and teacher models, and calculates
    a set of standard depth estimation metrics.

    Args:
        student_path (str): The path to the student model (pth) to be evaluated.
        dataset_path (str): The path to the data for the validation set.
        batch_size (int): the size of the batch.

    Returns:
        Dict[str, float]: A dictionary containing the aggregated evaluation
                          metrics over the entire validation dataset.
    """
    device = config.DEVICE
    print(f"Using device: {device}")

    # --- 1. Load the models ---
    print("Loading models...")
    student_model = StudentDepthModel(pretrained=False).to(device)
    student_model.load_state_dict(torch.load(student_path, map_location=device))
    student_model.eval()

    teacher_model = TeacherWrapper(
        selected_features_indices=config.TEACHER_FEATURE_INDICES
    ).to(device)
    teacher_model.eval()
    print("Models loaded successfully.")

    # --- 2. Set up the dataset and dataloader ---
    input_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    target_aspect_ratio = config.IMG_WIDTH / config.IMG_HEIGHT

    eval_transform = get_eval_transforms(target_aspect_ratio, input_size)

    
    eval_dataset = UnlabeledImageDataset(
        root_dir=dataset_path, transform=eval_transform, resize_size=input_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    print(f"Evaluating on {len(eval_dataset)} images.")

    # --- 3. Initialize metrics ---
    total_metrics: Dict[str, float] = {
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "rmse": 0.0,
        "rmse_log": 0.0,
        "a1": 0.0,
        "a2": 0.0,
        "a3": 0.0,
    }
    num_samples = 0
    total_student_time = 0.0
    total_teacher_time = 0.0

    # --- 4. Evaluation loop ---
    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc="Evaluating")
        for images in progress_bar:
            images = images.to(device)
            batch_size = images.size(0)

            # Get predictions
            start_time = time.time()
            teacher_depth, _ = teacher_model(images)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            total_teacher_time += (end_time - start_time)

            start_time = time.time()
            student_depth, _ = student_model(images)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            total_student_time += (end_time - start_time)

            # Compute metrics for the batch
            metrics: Dict[str, float] = compute_depth_metrics(student_depth, teacher_depth)

            # Accumulate metrics
            batch_size: int = images.size(0)
            for key in total_metrics:
                total_metrics[key] += metrics[key] * batch_size
            num_samples += batch_size

    # --- 5. Calculate and print average metrics ---
    avg_metrics: Dict[str, float] = {key: total / num_samples for key, total in total_metrics.items()} 
    avg_student_time = total_student_time / num_samples
    avg_teacher_time = total_teacher_time / num_samples

    print("\n--- Evaluation Results ---")
    print(f"Absolute Relative Difference (AbsRel): {avg_metrics['abs_rel']:.4f}")
    print(f"Squared Relative Difference (SqRel): {avg_metrics['sq_rel']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {avg_metrics['rmse']:.4f}")
    print(f"Root Mean Squared Log Error (RMSELog): {avg_metrics['rmse_log']:.4f}")
    print(f"Delta1 (a1): {avg_metrics['a1']:.4f}")
    print(f"Delta2 (a2): {avg_metrics['a2']:.4f}")
    print(f"Delta3 (a3): {avg_metrics['a3']:.4f}")
    
    print("--------------------------\n")
    print("--- Inference Time ---")
    print(f"Average Student Inference Time: {avg_student_time:.4f} seconds/photo")
    print(f"Average Teacher Inference Time: {avg_teacher_time:.4f} seconds/photo")
    print("----------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for Student Depth Estimation Model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained student model (.pth file)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the evaluation dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for evaluation"
    )
    args = parser.parse_args()
    student_path: str = args.model_path
    dataset: str = args.dataset_path
    batch: int = args.batch_size
    evaluate(student_path, dataset, batch)
