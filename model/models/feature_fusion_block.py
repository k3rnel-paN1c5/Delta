import torch
import torch.nn as nn

class FeatureFusionBlock(nn.Module):
    """
    A block that fuses features from two different sources.

    This block is used to combine features from a higher-level (more abstract)
    decoder stage with features from a lower-level (more detailed) encoder stage
    via a skip connection. The features are concatenated along the channel
    dimension and then refined using a series of convolutional layers.
    """
    def __init__(self, channels: int):
        """
        Initializes the FeatureFusionBlock.

        Args:
            channels (int): The number of channels in each of the input feature maps.
                            The output will also have this many channels.
        """
        super().__init__()
        
        # Convolutional layers to process the fused features
        self.conv = nn.Sequential(
            # The input to this conv layer has 2 * channels because we concatenate two feature maps
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            # Another conv layer to further refine the features
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, higher_level_features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeatureFusionBlock.

        Args:
            higher_level_features (torch.Tensor): The feature map from the previous,
                                                  higher-level decoder stage. It is
                                                  assumed to have been upsampled to match
                                                  the spatial dimensions of `skip_features`.
            skip_features (torch.Tensor): The feature map from the corresponding
                                          encoder stage (skip connection).

        Returns:
            torch.Tensor: The fused and refined feature map.
        """
        # Concatenate the two feature maps along the channel dimension
        fused_features = torch.cat([higher_level_features, skip_features], dim=1)
        # Process the fused features with the convolutional layers
        return self.conv(fused_features)
