import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
from typing import Tuple, List

class TeacherWrapper(nn.Module):
    """
    A wrapper for the teacher depth estimation model (Depth-Anything-V2).

    This class provides a unified interface for the teacher model. It handles
    loading the pre-trained model from Hugging Face (or a local cache) and
    performs the necessary pre- and post-processing steps. During inference,
    it extracts the final depth prediction and intermediate feature maps,
    which serve as targets for training the student model via knowledge
    distillation.
    """
    def __init__(self, model_id: str = 'depth-anything/depth-anything-v2-small-hf', 
                cache_dir: str = None):
        """
        Initializes the TeacherWrapper.

        Args:
            model_id (str): The identifier for the pre-trained model on the
                            Hugging Face Hub, or a path to a local directory
                            containing the model files.
            cache_dir (str): The directory where the downloaded model should be
                             cached.
        """
        super().__init__()
        self.model_id = model_id
        # Load the pre-trained depth estimation model
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=cache_dir)
        # Set the model to evaluation mode, as we don't want to train it
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the teacher model.

        This method should always be called within a `torch.no_grad()` context,
        as the teacher's weights should remain frozen during distillation.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            A tuple containing:
            - The final, normalized depth map (torch.Tensor).
            - A list of intermediate feature maps from the teacher's encoder
              (List[torch.Tensor]).
        """
        original_size = x.shape[2:]

        # 1. Get Model Outputs
        # We get the model's outputs, including the hidden states, which we
        # will use as feature targets for the student.
        outputs = self.model(x, output_hidden_states=True)
        predicted_depth = outputs.predicted_depth
        hidden_states = outputs.hidden_states

        # 2. Normalize Depth Map
        # The raw output of the model is not normalized, so we normalize it to
        # the range [0, 1] for consistent training.
        if predicted_depth.dim() == 3:
            predicted_depth = predicted_depth.unsqueeze(1)
        b, c, h, w = predicted_depth.shape
        predicted_depth_flat = predicted_depth.view(b, -1)
        max_vals = predicted_depth_flat.max(dim=1, keepdim=True)[0]
        max_vals[max_vals == 0] = 1.0  # Avoid division by zero
        normalized_depth = (predicted_depth_flat / max_vals).view(b, c, h, w)

        # 3. Interpolate to Original Size
        # The model's output may be smaller than the input image, so we
        # interpolate it back to the original size.
        final_depth = F.interpolate(normalized_depth, size=original_size, mode='bilinear', align_corners=False)

        # 4. Select Feature Maps for Distillation
        # We select a subset of the hidden states to use as feature targets.
        # For ViT-based models like DINOv2, these indices correspond to the
        # outputs of different blocks in the encoder.
        selected_features = [hidden_states[i] for i in [3, 5, 7, 11]]

        # 5. Reshape ViT Features
        # The feature maps from Vision Transformer (ViT) models have a different
        # shape ([B, SeqLen, C]) than those from CNNs ([B, C, H, W]). We need to
        # reshape them to be compatible with the student's CNN-based features.
        reshaped_features = []
        patch_size = self.model.config.patch_size
        H_grid = x.shape[2] // patch_size
        W_grid = x.shape[3] // patch_size

        for feature_map in selected_features:
            batch_size, seq_len, num_channels = feature_map.shape
            # The first token in the sequence is the [CLS] token, which we remove
            image_patch_tokens = feature_map[:, 1:, :]
            # Reshape the sequence of patch tokens into a 2D feature map
            reshaped_map = image_patch_tokens.transpose(1, 2).reshape(batch_size, num_channels, H_grid, W_grid)
            reshaped_features.append(reshaped_map)

        return final_depth, reshaped_features
