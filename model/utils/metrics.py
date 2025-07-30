import torch

def compute_depth_metrics(pred, target):
    """
    Computes standard depth estimation metrics for a batch of predictions.
    
    Args:
        pred (torch.Tensor): The predicted depth map (B, 1, H, W).
        target (torch.Tensor): The ground truth or teacher depth map (B, 1, H, W).

    Returns:
        A dictionary containing the computed metrics.
    """
    # Create a mask for valid depth values (greater than a small epsilon)
    valid_mask = (target > 1e-3) & (pred > 1e-3)

    # Apply mask to predictions and target
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]

    # --- 1. Error metrics ---
    # Absolute Relative Difference
    abs_rel = torch.mean(torch.abs(pred_valid - target_valid) / target_valid)
    
    # Squared Relative Difference
    sq_rel = torch.mean(((pred_valid - target_valid) ** 2) / target_valid)
    
    # Root Mean Squared Error
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))
    
    # Root Mean Squared Logarithmic Error
    rmse_log = torch.sqrt(torch.mean((torch.log(pred_valid) - torch.log(target_valid)) ** 2))

    # --- 2. Accuracy metrics (Delta thresholds) ---
    thresh = torch.max((target_valid / pred_valid), (pred_valid / target_valid))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item()
    }