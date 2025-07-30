import unittest
import torch
import sys
import os

# Add project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.mini_dpt import MiniDPT

class TestMiniDPT(unittest.TestCase):
    """
    Unit tests for the MiniDPT decoder.
    """

    def setUp(self):
        """
        Set up a MiniDPT instance for testing.
        """
        self.decoder_channels = [256, 128, 64, 32, 16]
        self.encoder_channels = [16, 24, 40, 112, 320] # Example for efficientnet_b0
        self.model = MiniDPT(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels
        )
        self.model.eval()

    def test_forward_pass(self):
        """
        Test the forward pass of the MiniDPT decoder to ensure it produces an output
        of the correct shape and type.
        """
        # Create dummy input feature maps from an encoder
        # These sizes correspond to the output of an efficientnet_b0 at different stages
        features = [
            torch.randn(1, self.encoder_channels[0], 112, 112),
            torch.randn(1, self.encoder_channels[1], 56, 56),
            torch.randn(1, self.encoder_channels[2], 28, 28),
            torch.randn(1, self.encoder_channels[3], 14, 14),
            torch.randn(1, self.encoder_channels[4], 7, 7),
        ]

        # Perform a forward pass
        with torch.no_grad():
            output = self.model(features)

        # Check the output type
        self.assertIsInstance(output, torch.Tensor)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1, 224, 224))

if __name__ == '__main__':
    unittest.main()
