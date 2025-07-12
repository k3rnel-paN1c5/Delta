import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation

class TeacherDepthModel(nn.Module):
    """
    The Teacher Model, using Depth-Anything-V2-Large model.
    This model is loaded from the Hugging Face Hub.
    """
    def __init__(self, model_name='LiheYoung/depth-anything-2-large'):
        super().__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        # Freeze the teacher model's parameters as we only use it for inference
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        """The forward pass expects 'pixel_values' prepared by the Hugging Face processor."""
        return self.model(pixel_values=pixel_values).predicted_depth
