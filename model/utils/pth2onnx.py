import torch
import torch.nn as nn
import torch.nn.functional as F # <-- Missing import added
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# This will correctly be 'cpu' on a machine without a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StudentModel(nn.Module):
    """
    Student model with MobileNetV3 encoder and a custom decoder.
    Designed for real-time inference on edge devices.
    Corrected skip connections and decoder logic.
    """
    def __init__(self, output_channels=1):
        super(StudentModel, self).__init__()
        # Load MobileNetV3 Large features as the encoder
        self.encoder = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder blocks, ensuring correct input channels after concatenation
        # The input channels for decoder_block1 are 960 (from encoder final) + 112 (from s32)
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(960 + 112, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # Input channels for decoder_block2 are 512 (from decoder_block1) + 80 (from s16)
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(512 + 80, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # Input channels for decoder_block3 are 256 (from decoder_block2) + 40 (from s8)
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256 + 40, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # Input channels for decoder_block4 are 128 (from decoder_block3) + 24 (from s4)
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(128 + 24, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.shape[2:]
        skip_features = {}
        # Iterate through encoder layers to get skip connections
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 2:  # s4 (H/4, W/4 resolution)
                skip_features['s4'] = x
            elif i == 4:  # s8 (H/8, W/8 resolution)
                skip_features['s8'] = x
            elif i == 7:  # s16 (H/16, W/16 resolution)
                skip_features['s16'] = x
            elif i == 11:  # s32 (Currently H/16, W/16 resolution from your logs)
                skip_features['s32'] = x

        # Decoder path
        # Concatenate final encoder output (x) with the adjusted s32 skip connection
        # Resize s32 to match the spatial dimensions of x before concatenation
        s32_resized = F.interpolate(skip_features['s32'], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s32_resized], dim=1)
        x = self.decoder_block1(x)

        # Concatenate with s16, resize s16 to match the spatial dimensions of x
        s16_resized = F.interpolate(skip_features['s16'], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s16_resized], dim=1)
        x = self.decoder_block2(x)

        # Concatenate with s8, resize s8 to match the spatial dimensions of x
        s8_resized = F.interpolate(skip_features['s8'], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s8_resized], dim=1)
        x = self.decoder_block3(x)

        # Concatenate with s4, resize s4 to match the spatial dimensions of x
        s4_resized = F.interpolate(skip_features['s4'], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s4_resized], dim=1)
        x = self.decoder_block4(x)

        x = self.final_conv(x)
        # Upsample to the original input size for the final depth map
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x

model = StudentModel().to(device)

# --- FIX 1: Use map_location to load the model correctly on CPU ---
model.load_state_dict(torch.load('../model/export/best_student_60Epoch.pth', map_location=device))
model.eval()

# --- FIX 2: Correct the dummy input shape for a 3-channel image ---
input_size = (384, 384)
dummy_input = torch.randn(1, 3, *input_size).to(device)

# Export the model
torch.onnx.export(model,
                  dummy_input,
                  "../model/export/best_student_60Epoch.onnx",
                  verbose=True,
                  input_names=['input'],   # Optional: name the input
                  output_names=['output']) # Optional: name the output

print("\nModel successfully exported to ONNX format! ✅")