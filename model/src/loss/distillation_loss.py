import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mul(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class DistillationLoss(nn.Module):
    """Calculates the L1 loss between the student's and teacher's predictions."""
    def __init__(self, lambda_depth=1.0, lambda_si=1.0, lambda_grad=1.0, lambda_ssim=1.0):
        super().__init__()
        self.lambda_depth = lambda_depth # Weight for depth map MSE loss
        self.lambda_si = lambda_si # Weight for Scale-Invariant MSE loss
        self.lambda_grad = lambda_grad # Weight for Gradient loss
        self.lambda_ssim = lambda_ssim # Weight for SSIM loss

        self.mse_depth_loss = nn.MSELoss() # Mean Squared Error for depth maps
        self.l1_loss = nn.L1Loss() # L1 Loss for gradients

    def forward(self, student_outputs, teacher_outputs):
        student_depth = student_outputs
        teacher_depth = teacher_outputs

        # Ensure tensors have a batch dimension (unsqueeze if necessary)
        if student_depth.dim() == 3:
            student_depth = student_depth.unsqueeze(0)
        if teacher_depth.dim() == 3:
            teacher_depth = teacher_depth.unsqueeze(0)


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
            # Calculate gradients
            # Adjusted slicing for potential 4D tensors (Batch, Channels, Height, Width)
            # Assumes channels dimension is 1 for depth maps
            student_grad_x = torch.abs(student_depth[:, :, :, :-1] - student_depth[:, :, :, 1:])
            student_grad_y = torch.abs(student_depth[:, :, :-1, :] - student_depth[:, :, 1:, :])
            teacher_grad_x = torch.abs(teacher_depth[:, :, :, :-1] - teacher_depth[:, :, :, 1:])
            teacher_grad_y = torch.abs(teacher_depth[:, :, :-1, :] - teacher_depth[:, :, 1:, :])

            loss_grad = self.l1_loss(student_grad_x, teacher_grad_x) + self.l1_loss(student_grad_y, teacher_grad_y)
            total_loss += self.lambda_grad * loss_grad

        # 4. SSIM Loss
        if self.lambda_ssim > 0:
            # SSIM is typically calculated on normalized images, but for depth maps,
            # we can apply it directly or normalize based on min/max depth in the batch.
            # For simplicity, we'll apply it directly here.
            # You might want to experiment with normalizing depth maps before calculating SSIM.
            # Ensure tensors have a channel dimension for SSIM if it's missing
            ssim_student_depth = student_depth
            ssim_teacher_depth = teacher_depth

            if ssim_student_depth.dim() == 3: # Assuming Batch, Height, Width
                 ssim_student_depth = ssim_student_depth.unsqueeze(1) # Add channel dimension
            if ssim_teacher_depth.dim() == 3: # Assuming Batch, Height, Width
                 ssim_teacher_depth = ssim_teacher_depth.unsqueeze(1) # Add channel dimension

            # If the tensors are 4D but the channel dimension is > 1,
            # you might need to handle that based on how your depth maps are structured.
            # Assuming depth maps are single channel, we proceed if the channel dim is 1 or was just added.


            loss_ssim = 1 - ssim(ssim_student_depth, ssim_teacher_depth)
            total_loss += self.lambda_ssim * loss_ssim


        return total_loss