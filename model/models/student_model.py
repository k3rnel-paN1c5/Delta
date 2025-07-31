import timm
import torch
import torch.nn as nn
from typing import Tuple, List

from .mini_dpt import MiniDPT


class StudentDepthModel(nn.Module):
    """
    The student model for monocular depth estimation.

    This model consists of a lightweight, pre-trained encoder (e.g., MobileViT)
    and a custom lightweight decoder (MiniDPT). It is designed to be trained
    efficiently, making it suitable for deployment on resource-constrained
    devices. The training is done via knowledge distillation from a larger,
    more powerful teacher model.
    """

    def __init__(
        self,
        feature_indices: Tuple[int, ...] = (0, 1, 2, 3),
        decoder_channels: Tuple[int, ...] = (64, 128, 160, 256),
        pretrained: bool = True,
    ):
        """
        Initializes the StudentDepthModel.

        Args:
            encoder_name (str): The name of the encoder model to use from the `timm`
                                library.
            feature_indices (Tuple[int, ...]): A tuple of indices specifying which
                                               feature maps to extract from the encoder.
            decoder_channels (Tuple[int, ...]): A tuple of channel counts for the
                                                decoder stages.
            pretrained (bool): Whether to load pre-trained weights for the encoder.
        """
        super().__init__()
        if len(feature_indices) != len(decoder_channels):
            raise ValueError(
                "The number of feature indices must match the number of decoder channel dimensions."
            )

        # 1. Instantiate the Encoder
        # We use the `timm` library to create a pre-trained encoder.
        # `features_only=True` makes the model return a List of feature maps
        # at different stages, instead of a final classification output.
        self.encoder = timm.create_model(
            "mobilevit_xs",
            pretrained=pretrained,
            features_only=True,  # This returns a List of feature maps
        )
        self.feature_indices = feature_indices

        # 2. Determine Encoder Output Channels
        # To connect the encoder to the decoder, we need to know the number of
        # channels in the feature maps that the encoder produces. We can find
        # this by doing a dummy forward pass.
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.encoder(dummy_input)
            encoder_channels = [features[i].shape[1] for i in self.feature_indices]

        # 3. Instantiate the Decoder
        # The decoder takes the feature maps from the encoder and upsamples them
        # to produce the final depth map.
        self.decoder = MiniDPT(encoder_channels, list(decoder_channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the StudentDepthModel.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            A tuple containing:
            - The final predicted depth map (torch.Tensor).
            - A list of intermediate feature maps from the encoder, which will be
              used for feature-based distillation (List[torch.Tensor]).
        """
        # Get the feature maps from the encoder
        features = self.encoder(x)
        # Select the feature maps at the specified indices
        selected_features = [features[i] for i in self.feature_indices]
        # Pass the selected features to the decoder to get the depth map
        depth_map = self.decoder(selected_features)
        return depth_map, selected_features
