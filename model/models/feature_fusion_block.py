import torch.nn as nn
import torch.nn.functional as F
from Typing import Tuple, List

class FeatureFusionBlock(nn.Module):
    """
    Fuses features from a higher-level decoder stage with features
    from a lower-level encoder stage (skip connection).
    """
    def __init__(self, channels: int):
        super().__init__()
        # Using a depthwise separable convolution for efficiency
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, higher_level_features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        # Assumes higher_level_features have already been upsampled to match skip_features' spatial dimensions
        fused_features = torch.cat([higher_level_features, skip_features], dim=1)
        return self.conv(fused_features)