"""
This module defines a combined distillation loss strategy.

The `CombinedDistillationLoss` class implements a combined loss calculation that
combines four loss components:
    1.  Scale-Invariant Log (SILog) Loss: Measures the overall accuracy of the
        predicted depth map.
    2.  Gradient Matching Loss (L1): Enforces that the student's depth map
        has similar edges and fine details as the teacher's.
    3.  Feature Matching Loss (L1): Encourages the student's intermediate
        feature representations to be similar to the teacher's.
    4.  Attention Matching Loss (L2): Encourages the student to focus on the
        same spatial regions of the image as the teacher.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .criterion import LossStrategy
from typing import List


def compute_depth_gradients(depth_map: torch.Tensor) -> torch.Tensor:
    """
    Computes the image gradients (dy, dx) for a batch of depth maps.

    This is done by applying Sobel filters to the depth map. The gradients
    are used to compute a loss that encourages the student model to preserve
    edges and fine details from the teacher's prediction.

    Args:
        depth_map (torch.Tensor): A batch of single-channel depth maps.

    Returns:
        torch.Tensor: A tensor containing the absolute gradients in the y and x
                      directions, concatenated along the channel dimension.
    """
    # Create Sobel filters for GPU computation
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=depth_map.device,
    ).view(1, 1, 3, 3)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=depth_map.device,
    ).view(1, 1, 3, 3)

    # Apply filters using depthwise convolution
    padded_map = F.pad(depth_map, (1, 1, 1, 1), mode="replicate")

    grad_y = F.conv2d(padded_map, sobel_y, padding=0)
    grad_x = F.conv2d(padded_map, sobel_x, padding=0)

    # Return the absolute gradients, stacked along the channel dimension
    return torch.cat([grad_y.abs(), grad_x.abs()], dim=1)


class CombinedDistillationLoss(LossStrategy):
    """
    A comprehensive loss function strategy for knowledge distillation in depth estimation.

    This loss function combines four different components to train the student
    model effectively:
    1.  Scale-Invariant Log (SILog) Loss: Measures the overall accuracy of the
        predicted depth map.
    2.  Gradient Matching Loss (L1): Enforces that the student's depth map
        has similar edges and fine details as the teacher's.
    3.  Feature Matching Loss (L1): Encourages the student's intermediate
        feature representations to be similar to the teacher's.
    4.  Attention Matching Loss (L2): Encourages the student to focus on the
        same spatial regions of the image as the teacher.
    """

    def __init__(self):
        self.projection_convs = None
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def _initialize_projections(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        device: torch.device,
    ):
        """
        Dynamically creates projection layers to match the channel counts of the
        student and teacher features. This is necessary because the student and
        teacher models may have different numbers of channels in their
        intermediate feature maps.
        """
        self.projection_convs = nn.ModuleList()
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_chan, t_chan = s_feat.shape[1], t_feat.shape[1]
            if s_chan != t_chan:
                # Create a 1x1 convolution to project student channels to teacher channels
                proj = nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False)
            else:
                proj = nn.Identity()
            self.projection_convs.append(proj)

    def _compute_attention_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Computes a spatial attention map from a feature map by summarizing
        across the channel dimension. This provides a simple way to capture
        which spatial regions the model is focusing on.
        """
        return torch.mean(torch.abs(feature_map), dim=1, keepdim=True)

    def calculate(
        self,
        student_depth,
        teacher_depth,
        student_attns,
        teacher_attns,
        student_features,
        teacher_features,
        **kwargs
    ):
        super().__init__()
        self.lambda_silog = kwargs["lambda_silog"]
        self.lambda_grad = kwargs["lambda_grad"]
        self.lambda_feat = kwargs["lambda_feat"]
        self.lambda_attn = kwargs["lambda_attn"]
        self.alpha = kwargs["alpha"]

        self.device = student_depth.device

        # Initialize projection layers on the first pass
        if self.projection_convs is None:
            self._initialize_projections(
                student_features, teacher_features, self.device
            )

        # --- 1. SILog Depth Loss ---
        valid_mask = (student_depth > 1e-8) & (teacher_depth > 1e-8)
        log_diff = torch.log(student_depth[valid_mask]) - torch.log(
            teacher_depth[valid_mask]
        )
        num_pixels = log_diff.numel()
        silog_loss = (
            torch.sum(log_diff**2) / num_pixels
            - self.alpha * (torch.sum(log_diff) ** 2) / (num_pixels**2)
            if num_pixels > 0
            else torch.tensor(0.0, device=self.device)
        )

        # --- 2. Gradient Matching Loss ---
        student_grads = compute_depth_gradients(student_depth)
        teacher_grads = compute_depth_gradients(teacher_depth)
        grad_loss = self.l1_loss(student_grads, teacher_grads)
        
        # --- 3. Feature & Attention Matching Loss ---
        feature_loss = torch.tensor(0.0, device=self.device)
        attention_loss = torch.tensor(0.0, device=self.device)

        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # Project the student feature to match the teacher's channel dimension
            s_feat_projected = self.projection_convs[i](s_feat)

            # Interpolate if spatial sizes don't match (essential for ViT vs CNN features)
            if s_feat_projected.shape[2:] != t_feat.shape[2:]:
                s_feat_resized = F.interpolate(
                    s_feat_projected,
                    size=t_feat.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                s_feat_resized = s_feat_projected

            feature_loss += self.l1_loss(s_feat_resized, t_feat)

            # Calculate the attention map loss
            s_attn = self._compute_attention_map(s_feat_resized)
            t_attn = self._compute_attention_map(t_feat)
            attention_loss += self.l2_loss(s_attn, t_attn)

        # --- 4. Combine All Losses ---
        total_loss = (
            (self.lambda_silog * silog_loss)
            + (self.lambda_grad * grad_loss)
            + (self.lambda_feat * feature_loss)
            + (self.lambda_attn * attention_loss)
        )

        return total_loss
