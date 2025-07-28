import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class StudentDepthModel(nn.Module):
    """
    Combines a MobileViT encoder with the MiniDPT decoder to form a
    complete, end-to-end model for depth estimation.
    """
    def __init__(self, encoder_name='mobilevit_xs', pretrained=True):
        super().__init__()
        # 1. Instantiate the encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True, # This returns a list of feature maps
        )

        self.feature_indices = [0, 1, 2, 3]

        # 2. Get the channel counts from the encoder
        dummy_input = torch.randn(1, 3, 224, 224)
        features = self.encoder(dummy_input)
        # encoder_channels = [f.shape[1] for f in features]
        encoder_channels = [features[i].shape[1] for i in self.feature_indices]

        # 3. Define the decoder channel counts
        # This can be tuned to balance performance and model size.
        decoder_channels = [ 64, 128, 160, 256]

        # Ensure decoder channels list is the same length as encoder channels
        if len(decoder_channels) > len(encoder_channels):
             decoder_channels = decoder_channels[:len(encoder_channels)]
        elif len(decoder_channels) < len(encoder_channels):
             # You might want to handle this differently, e.g., by padding
             raise ValueError("Decoder channels list is shorter than encoder channels list.")


        # 4. Instantiate the decoder
        self.decoder = MiniDPT(encoder_channels, decoder_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            A tuple containing:
            - The final depth map (torch.Tensor).
            - A list of intermediate feature maps from the encoder (List[torch.Tensor]).
        """
        features = self.encoder(x)
        selected_features = [features[i] for i in self.feature_indices]
        depth_map = self.decoder(selected_features)
        return depth_map, selected_features
