import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
from Typing import Tuple, List

class DepthModel(nn.Module):
    """
    Wrapper class for the Depth-Anything model.
    """
    def __init__(self):
        super().__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Runs a forward pass and returns the normalized depth map and features.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            A tuple containing:
            - final_depth (torch.Tensor): The normalized depth map in the range [0, 1].
            - selected_features (List[torch.Tensor]): A list of intermediate feature maps from the encoder.
        """
        original_size = x.shape[2:]

        # The teacher's weights are frozen, so we use no_grad for efficiency
        with torch.no_grad():
            # 1. Get model outputs, including hidden states for feature matching.
            outputs = self.model(x, output_hidden_states=True)

            predicted_depth = outputs.predicted_depth
            hidden_states = outputs.hidden_states

        # 2. Normalize the raw depth map to the range [0, 1].
        if predicted_depth.dim() == 3:
            predicted_depth = predicted_depth.unsqueeze(1)

        b, c, h, w = predicted_depth.shape
        predicted_depth_flat = predicted_depth.view(b, -1)
        max_vals = predicted_depth_flat.max(dim=1, keepdim=True)[0]
        max_vals[max_vals == 0] = 1.0 # Avoid division by zero
        normalized_depth = (predicted_depth_flat / max_vals).view(b, c, h, w)

        # 3. Interpolate the normalized depth map back to the original input image size.
        final_depth = F.interpolate(normalized_depth, size=original_size, mode='bilinear', align_corners=False)

        # 4. Select specific feature maps for distillation. For ViT-based models,
        # [3, 5, 7, 11] correspond to different blocks in DINOv2's encoder.
        selected_features = [hidden_states[i] for i in [3, 5, 7, 11]]

        # 5. *** FIX: Reshape ViT features to be compatible with CNN features ***
        # Convert from [B, SeqLen, C] to [B, C, H, W]
        reshaped_features = []
        patch_size = self.model.config.patch_size # This is typically 14 for DINOv2
        H_grid = x.shape[2] // patch_size
        W_grid = x.shape[3] // patch_size

        for feature_map in selected_features:
            batch_size, seq_len, num_channels = feature_map.shape

            # Remove the [CLS] token (the first element in the sequence)
            image_patch_tokens = feature_map[:, 1:, :]

            # Sanity check
            expected_seq_len = H_grid * W_grid
            if image_patch_tokens.shape[1] != expected_seq_len:
                 raise ValueError(
                    f"After removing [CLS] token, sequence length is {image_patch_tokens.shape[1]}, "
                    f"but expected {expected_seq_len} ({H_grid}x{W_grid}) patches."
                )

            # Reshape the feature map from [B, SeqLen, C] to [B, C, H, W]
            reshaped_map = image_patch_tokens.transpose(1, 2).reshape(batch_size, num_channels, H_grid, W_grid)
            reshaped_features.append(reshaped_map)

        return final_depth, reshaped_features