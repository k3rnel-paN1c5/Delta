import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class StudentModel(nn.Module):
    """
    Student model with MobileNetV3 encoder and a custom decoder.
    Designed for real-time inference on edge devices.
    """
    def __init__(self, output_channels=1):
        super(StudentModel, self).__init__()
        # Load MobileNetV3 Large features as the encoder
        self.encoder = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # These layers refine the features from the encoder before fusion.
        # Channel dimensions correspond to MobileNetV3 skip connection outputs.
        self.skip_s4_conv = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.skip_s8_conv = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True)
        )
        self.skip_s16_conv = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        self.skip_s32_conv = nn.Sequential(
            nn.Conv2d(112, 112, kernel_size=1, bias=False),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True)
        )

        # Decoder blocks (input channels remain the same as the 1x1 convs don't change channel dims)
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(960 + 112, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(512 + 80, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256 + 40, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        )
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(128 + 24, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        )

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        self.final_activation = nn.ReLU()

    def forward(self, x):
        input_shape = x.shape[2:]
        skip_features = {}

        # --- Encoder Path & Feature Extraction ---
        # Iterate through encoder layers to get skip connections
        # Apply 1x1 convs immediately after extraction.
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 2:  # s4 (H/4, W/4 resolution, 24 channels)
                skip_features['s4'] = self.skip_s4_conv(x)
            elif i == 4:  # s8 (H/8, W/8 resolution, 40 channels)
                skip_features['s8'] = self.skip_s8_conv(x)
            elif i == 7:  # s16 (H/16, W/16 resolution, 80 channels)
                skip_features['s16'] = self.skip_s16_conv(x)
            elif i == 11:  # s32 (H/16, W/16 resolution, 112 channels)
                skip_features['s32'] = self.skip_s32_conv(x)

        # Ensure s32 matches the spatial dimension of the encoder's final output
        if skip_features['s32'].shape[2:] != x.shape[2:]:
            skip_features['s32'] = F.interpolate(skip_features['s32'], size=x.shape[2:], mode='bilinear', align_corners=False)

        # --- Decoder Path with Enhanced Skip Connections ---
        # Concatenate final encoder output (x) with the refined s32 skip connection
        x = torch.cat([x, skip_features['s32']], dim=1)
        x = self.decoder_block1(x)

        # Concatenate with refined s16
        x = torch.cat([x, skip_features['s16']], dim=1)
        x = self.decoder_block2(x)

        # Concatenate with refined s8
        x = torch.cat([x, skip_features['s8']], dim=1)
        x = self.decoder_block3(x)

        # Concatenate with refined s4
        x = torch.cat([x, skip_features['s4']], dim=1)
        x = self.decoder_block4(x)

        # Final layers
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = self.final_activation(x)

        return x