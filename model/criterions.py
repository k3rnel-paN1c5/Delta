import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class DepthDistillationLoss(nn.Module):
    """
    Custom loss function for depth distillation.
    Using Pixel wise MSE Loss, SSIM, Gradient MAE Loss, and Scale-Invariant MSE Loss.
    """
    def __init__(self, lambda_depth=0.7, lambda_si=1.0, lambda_grad=1.0, lambda_ssim=1.0, lambda_smooth=0.2, window_size=11):
        super().__init__()
        self.lambda_depth = lambda_depth   # Weight for depth map MSE loss
        self.lambda_si = lambda_si         # Weight for Scale-Invariant MSE loss
        self.lambda_grad = lambda_grad     # Weight for Gradient loss
        self.lambda_ssim = lambda_ssim     # Weight for SSIM loss
        self.lambda_smooth = lambda_smooth # Weight for smoothness regularizer

        self.mse_depth_loss = nn.MSELoss()  # Mean Squared Error for depth maps
        self.l1_loss = nn.L1Loss()          # L1 Loss for gradients

        # Initialize the SSIM calculation from torchmetrics
        if self.lambda_ssim > 0:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=window_size)

    def forward(self, student_outputs, teacher_outputs):
        student_depth = student_outputs
        teacher_depth = teacher_outputs

        # Ensure tensors have a batch dimension
        if student_depth.dim() == 3:
            student_depth = student_depth.unsqueeze(0)
        if teacher_depth.dim() == 3:
            teacher_depth = teacher_depth.unsqueeze(0)

        # Ensure tensors have a channel dimension for SSIM
        if student_depth.dim() == 3: # Assuming (Batch, Height, Width)
            student_depth = student_depth.unsqueeze(1) # Becomes (Batch, Channel, Height, Width)
        if teacher_depth.dim() == 3: # Assuming (Batch, Height, Width)
            teacher_depth = teacher_depth.unsqueeze(1) # Becomes (Batch, Channel, Height, Width)


        total_loss = torch.tensor(0.0, device=student_depth.device)

        # 1. MSE Depth Loss
        if self.lambda_depth > 0:
            loss_depth = self.mse_depth_loss(student_depth, teacher_depth)
            total_loss += self.lambda_depth * loss_depth

        # 2. Scale-Invariant MSE Loss
        if self.lambda_si > 0:
            diff = student_depth - teacher_depth
            loss_si = torch.mean(diff**2) - torch.mean(diff)**2
            total_loss += self.lambda_si * loss_si

        # 3. Gradient Loss (using L1 on gradients)
        if self.lambda_grad > 0:
            student_grad_x = torch.abs(student_depth[:, :, :, :-1] - student_depth[:, :, :, 1:])
            student_grad_y = torch.abs(student_depth[:, :, :-1, :] - student_depth[:, :, 1:, :])
            teacher_grad_x = torch.abs(teacher_depth[:, :, :, :-1] - teacher_depth[:, :, :, 1:])
            teacher_grad_y = torch.abs(teacher_depth[:, :, :-1, :] - teacher_depth[:, :, 1:, :])

            loss_grad = self.l1_loss(student_grad_x, teacher_grad_x) + self.l1_loss(student_grad_y, teacher_grad_y)
            total_loss += self.lambda_grad * loss_grad

        # 4.Smoothness Loss (Regularizer)
        if self.lambda_smooth > 0:
            # Penalizes the L1 norm of the student's depth gradients
            loss_smooth = torch.mean(student_grad_x) + torch.mean(student_grad_y)
            total_loss += self.lambda_smooth * loss_smooth

        # 5. SSIM Loss
        if self.lambda_ssim > 0:
            # Move ssim module to the same device as the tensors
            self.ssim.to(student_depth.device)

            # The torchmetrics SSIM implementation returns a value between -1 and 1.
            # A value of 1 indicates perfect similarity.
            # To use it as a loss, we subtract it from 1.
            d_ssim = self.ssim(student_depth, teacher_depth)
            loss_ssim = (1 - d_ssim) / 2 # Normalize to be between 0 and 1
            total_loss += self.lambda_ssim * loss_ssim

        return total_loss