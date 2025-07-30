import unittest
import torch

from models.mini_dpt import MiniDPT

class TestMiniDPT(unittest.TestCase):
    """
    Unit tests for the MiniDPT decoder.
    """

    def setUp(self):
        """Set up a dummy list of encoder feature maps for testing the MiniDPT decoder."""
        self.encoder_channels = [64, 128, 160, 256]
        self.decoder_channels = [64, 128, 160, 256]
        self.encoder_features = [
            torch.randn(2, self.encoder_channels[0], 48, 48),
            torch.randn(2, self.encoder_channels[1], 24, 24),
            torch.randn(2, self.encoder_channels[2], 12, 12),
            torch.randn(2, self.encoder_channels[3], 6, 6)
        ]
        self.mini_dpt = MiniDPT(self.encoder_channels, self.decoder_channels)
        self.mini_dpt.eval()


    def test_output_shape(self):
        """Test if the output of the MiniDPT decoder has the correct shape."""
        output_tensor = self.mini_dpt(self.encoder_features)
        self.assertEqual(output_tensor.shape[0], 2)
        self.assertEqual(output_tensor.shape[1], 1)
        self.assertEqual(output_tensor.shape[2], 96)
        self.assertEqual(output_tensor.shape[3], 96)

    def test_output_type(self):
        """Test if the output of the MiniDPT decoder is a torch.Tensor."""
        output_tensor = self.mini_dpt(self.encoder_features)
        self.assertIsInstance(output_tensor, torch.Tensor)

    def test_value_error_on_channel_mismatch(self):
        """Test if a ValueError is raised when encoder and decoder channel lists have different lengths."""
        with self.assertRaises(ValueError):
            MiniDPT(encoder_channels=[64, 128], decoder_channels=[64, 128, 256])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
