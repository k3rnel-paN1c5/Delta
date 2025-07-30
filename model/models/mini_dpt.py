import torch
import torch.nn as nn
from typing import Tuple, List

from .upsample_block import UpsampleBlock
from .feature_fusion_block import FeatureFusionBlock

class MiniDPT(nn.Module):
    """
    A lightweight, DPT-inspired decoder for monocular depth estimation.

    This decoder takes a list of feature maps from an encoder at different
    spatial resolutions and progressively fuses them to generate a high-resolution
    depth map. The architecture is inspired by the Dense Prediction Transformer (DPT)
    but is simplified for use with a lightweight backbone like MobileViT.
    """
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        """
        Initializes the MiniDPT decoder.

        Args:
            encoder_channels (List[int]): A list of the number of channels for each
                                          feature map extracted from the encoder.
                                          The list should be ordered from the lowest
                                          level (largest spatial resolution) to the
                                          highest level (smallest spatial resolution).
                                          Example: [64, 128, 256, 512]
            decoder_channels (List[int]): A list of the number of channels for each
                                          stage of the decoder. The length of this
                                          list must be the same as `encoder_channels`.
                                          Example: [256, 128, 96, 64]
        """
        super().__init__()

        if len(encoder_channels) != len(decoder_channels):
            raise ValueError("Encoder and decoder channel lists must have the same length.")

        # Reverse for processing from high-level to low-level
        encoder_channels = encoder_channels[::-1] 
        decoder_channels = decoder_channels[::-1]

        # 1. Projection Convolutions
        # These 1x1 convolutions project the encoder features to the number of
        # channels specified for the decoder.
        self.projection_convs = nn.ModuleList()
        for i in range(len(encoder_channels)):
            self.projection_convs.append(nn.Sequential(
                nn.Conv2d(encoder_channels[i], decoder_channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels[i]),
                nn.ReLU(inplace=True),
            ))

        # 2. Upsampling and Fusion Blocks
        # These blocks are used to upsample the features from a higher decoder
        # level and fuse them with the projected features from the corresponding
        # encoder level (skip connection).
        self.upsample_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        for i in range(len(decoder_channels) - 1):
            # Upsample from the current decoder channel count to the next (lower) one
            self.upsample_blocks.append(UpsampleBlock(decoder_channels[i], decoder_channels[i+1]))
            # Fusion block takes the upsampled features and the projected skip connection
            self.fusion_blocks.append(FeatureFusionBlock(decoder_channels[i+1]))

        # 3. Prediction Head
        # This final part of the decoder takes the fused features from the last
        # stage and produces the final single-channel depth map.
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
        Forward pass of the MiniDPT decoder.

        Args:
            encoder_features (List[torch.Tensor]): A list of feature maps from the
                                                   encoder, ordered from the lowest
                                                   level to the highest level.

        Returns:
            torch.Tensor: The final predicted depth map.
        """
        
        # Reverse the features to process from the highest level to the lowest
        features = encoder_features[::-1]

        # Project all encoder features to the decoder's channel dimensions
        projected_features = [self.projection_convs[i](features[i]) for i in range(len(features))]

        # Start with the highest-level (most abstract) feature map
        current_features = projected_features[0]

        # Iteratively upsample and fuse with lower-level skip connections
        for i in range(len(self.fusion_blocks)):
            upsampled = self.upsample_blocks[i](current_features)
            skip_connection = projected_features[i+1]
            current_features = self.fusion_blocks[i](upsampled, skip_connection)

        # Generate final prediction using the prediction head
        return self.prediction_head(current_features)
