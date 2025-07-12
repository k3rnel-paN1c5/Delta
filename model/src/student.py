import torch
import torch.nn as nn
import timm 
from blocks import ConvBlock, UpsampleBlock

class StudentModel(nn.Module):
    """
    The Student Model, a lightweight architecture using EfficientNet-Lite0 as the encoder.
    """
    def __init__(self):
        super().__init__()
        # Use a smaller, more efficient pretrained encoder
        self.encoder = timm.create_model('efficientnet_lite0', pretrained=True, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()

        # A simple decoder to upsample encoder features and predict depth
        self.decoder_conv = ConvBlock(encoder_channels[-1], 256)
        self.upsample1 = UpsampleBlock(256, 128)
        self.upsample2 = UpsampleBlock(128, 64)
        self.upsample3 = UpsampleBlock(64, 32)
        self.upsample4 = UpsampleBlock(32, 16)
        self.upsample5 = UpsampleBlock(16, 16)
        self.output_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.encoder(x)
        # We only use the highest-level feature map from the encoder
        highest_res_features = features[-1]
        
        x = self.decoder_conv(highest_res_features)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        
        return self.output_conv(x)
