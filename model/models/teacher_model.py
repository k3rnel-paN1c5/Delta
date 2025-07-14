import torch.nn as nn
import torch.nn.functional as F

class DepthModel(nn.Module):
    """
    Wrapper class for the Depth-Anything teacher model.
    """
    def __init__(self, pretained_model):
        super().__init__()
        self.model = pretained_model
        self.eval()

    def forward(self, x):
        outputs = self.model(x)
        predicted_depth = outputs.predicted_depth
        # Interpolate to match the student model's output size
        predicted_depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(384, 384),
            mode='bilinear',
            align_corners=False
        )
        return predicted_depth