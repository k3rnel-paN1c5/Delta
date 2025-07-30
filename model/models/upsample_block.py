import torch
import torch.nn as nn

class UpsampleBlock(nn.Module):
    """
    A building block for the decoder that upsamples feature maps and refines them.

    This block first increases the spatial resolution of the input feature map by a
    factor of 2 using bilinear interpolation. It then applies a series of
    convolutional layers to refine the upsampled features. For efficiency,
    it uses depthwise separable convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the UpsampleBlock.

        Args:
            in_channels (int): The number of channels in the input feature map.
            out_channels (int): The number of channels in the output feature map.
        """
        super().__init__()
        
        # Upsampling layer to increase spatial resolution
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolutional layers to refine the upsampled features
        self.conv = nn.Sequential(
            # First 3x3 convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
             # Second 3x3 convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpsampleBlock.

        Args:
            x (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: The upsampled and refined feature map.
        """
        # Apply upsampling and then the convolutional layers
        upsampled_features = self.upsample(x)
        return self.conv(upsampled_features)
