"""This module provides functions for computing standard depth estimation metrics.

The `compute_depth_metrics` function calculates a set of common metrics used to
evaluate the performance of depth estimation models, including Absolute Relative
Difference (AbsRel), Squared Relative Difference (SqRel), Root Mean Squared Error
(RMSE), and threshold-based accuracy metrics (a1, a2, a3).
"""

import torch
from typing import Dict


def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Computes standard depth estimation metrics for a batch of predictions.

    Args:
        pred: The predicted depth map, with shape (B, 1, H, W).
        target: The ground truth or teacher depth map, with shape (B, 1, H, W).

    Returns:
        A dictionary containing the computed metrics:
        - 'abs_rel': Absolute Relative Difference
        - 'sq_rel': Squared Relative Difference
        - 'rmse': Root Mean Squared Error
        - 'rmse_log': Root Mean Squared Logarithmic Error
        - 'a1': Delta1 accuracy
        - 'a2': Delta2 accuracy
        - 'a3': Delta3 accuracy
    """
    # Create a mask for valid depth values (greater than a small epsilon)
    valid_mask: torch.Tensor = (target > 1e-3) & (pred > 1e-3)

    # Handle the edge case where there are no valid pixels
    if valid_mask.sum() == 0:
        return {
            "abs_rel": float("nan"),
            "sq_rel": float("nan"),
            "rmse": float("nan"),
            "rmse_log": float("nan"),
            "a1": float("nan"),
            "a2": float("nan"),
            "a3": float("nan"),
        }

    # Apply mask to predictions and target
    pred_valid: torch.Tensor = pred[valid_mask]
    target_valid: torch.Tensor = target[valid_mask]

    # --- 1. Error metrics ---
    # Absolute Relative Difference
    abs_rel: torch.Tensor = torch.mean(
        torch.abs(pred_valid - target_valid) / target_valid
    )

    # Squared Relative Difference
    sq_rel: torch.Tensor = torch.mean(((pred_valid - target_valid) ** 2) / target_valid)

    # Root Mean Squared Error
    rmse: torch.Tensor = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))

    # Root Mean Squared Logarithmic Error
    rmse_log: torch.Tensor = torch.sqrt(
        torch.mean((torch.log(pred_valid) - torch.log(target_valid)) ** 2)
    )

    # --- 2. Accuracy metrics (Delta thresholds) ---
    thresh: torch.Tensor = torch.max(
        (target_valid / pred_valid), (pred_valid / target_valid)
    )
    a1: torch.Tensor = (thresh < 1.25).float().mean()
    a2: torch.Tensor = (thresh < 1.25**2).float().mean()
    a3: torch.Tensor = (thresh < 1.25**3).float().mean()

    return {
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "a1": a1.item(),
        "a2": a2.item(),
        "a3": a3.item(),
    }
