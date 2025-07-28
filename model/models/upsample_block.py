import torch.nn as nn
import torch.nn.functional as F
from Typing import Tuple, List

class UpsampleBlock(nn.Module):
    """
    A simple block that upsamples the feature map and applies a convolution.
    This is used to increase the spatial resolution of the features.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Using a depthwise separable convolution for efficiency
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))