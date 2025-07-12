import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A standard convolutional block with Conv2d, BatchNorm, and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class UpsampleBlock(nn.Module):
    """An upsampling block using bilinear upsampling followed by a ConvBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv_block(self.upsample(x))
