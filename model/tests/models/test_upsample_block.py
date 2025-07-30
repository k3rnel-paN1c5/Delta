import unittest
import torch

from models.upsample_block import UpsampleBlock

class TestUpsampleBlock(unittest.TestCase):
    """
    Unit tests for the UpsampleBlock.
    """
    def setUp(self):
        """Set up a dummy input tensor for testing the upsample block."""
        self.in_channels = 64
        self.out_channels = 32
        self.input_tensor = torch.randn(2, self.in_channels, 16, 16)
        self.upsample_block = UpsampleBlock(self.in_channels, self.out_channels)
        self.upsample_block.eval()
        
    def test_output_shape(self):
        """Test if the output of the upsample block has the correct shape."""
        output_tensor = self.upsample_block(self.input_tensor)
        self.assertEqual(output_tensor.shape[0], 2)
        self.assertEqual(output_tensor.shape[1], self.out_channels)
        self.assertEqual(output_tensor.shape[2], 32)
        self.assertEqual(output_tensor.shape[3], 32)

    def test_output_type(self):
        """Test if the output of the upsample block is a torch.Tensor."""
        output_tensor = self.upsample_block(self.input_tensor)
        self.assertIsInstance(output_tensor, torch.Tensor)
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
