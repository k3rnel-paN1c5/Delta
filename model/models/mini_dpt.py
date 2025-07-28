import torch.nn as nn
import torch.nn.functional as F
from Typing import Tuple, List

class MiniDPT(nn.Module):
    """
    A lightweight, DPT-like decoder designed for a MobileViT backbone.

    It takes features from three different stages of the encoder and progressively
    fuses them to produce a single-channel depth map.
    """
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        """
        Args:
            encoder_channels (List[int]): A list of channel counts for the features
                                          extracted from the encoder, from lowest level
                                          to highest level.
                                          e.g., [64, 128, 256]
            decoder_channels (List[int]): A list of channel counts for the decoder stages.
                                          The length should be the same as encoder_channels.
                                          e.g., [32, 64, 128]
        """
        super().__init__()

        if len(encoder_channels) != len(decoder_channels):
            raise ValueError("Encoder and decoder channel lists must have the same length.")

        # Reverse for processing from high-level to low-level
        encoder_channels = encoder_channels[::-1] # Now [256, 128, 64]
        decoder_channels = decoder_channels[::-1] # Now [128, 64, 32]

        # 1. Initial 1x1 convolutions to project encoder features to decoder dimensions
        self.projection_convs = nn.ModuleList()
        for i in range(len(encoder_channels)):
            self.projection_convs.append(nn.Sequential(
                nn.Conv2d(encoder_channels[i], decoder_channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels[i]),
                nn.ReLU(inplace=True),
            ))

        # 2. Upsampling and Fusion blocks
        self.upsample_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        for i in range(len(decoder_channels) - 1):
            # Upsample from the current decoder channel count to the next (lower) one
            self.upsample_blocks.append(UpsampleBlock(decoder_channels[i], decoder_channels[i+1]))
            # Fusion block takes the upsampled features and the projected skip connection
            self.fusion_blocks.append(FeatureFusionBlock(decoder_channels[i+1]))

        # 3. Final prediction head to produce the single-channel depth map
        self.prediction_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[-1] // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_features (List[torch.Tensor]): List of feature maps from the encoder,
                                                   ordered from low-level to high-level.

        Returns:
            torch.Tensor: The final depth map.
        """
        # Reverse the features to process from high-level to low-level
        features = encoder_features[::-1]

        # Project all feature levels to the decoder's channel dimensions
        projected_features = [self.projection_convs[i](features[i]) for i in range(len(features))]

        # Start with the highest-level feature map
        current_features = projected_features[0]

        # Iteratively upsample and fuse with lower-level skip connections
        for i in range(len(self.fusion_blocks)):
            upsampled = self.upsample_blocks[i](current_features)
            skip_connection = projected_features[i+1]
            current_features = self.fusion_blocks[i](upsampled, skip_connection)

        # Generate final prediction
        return self.prediction_head(current_features)
