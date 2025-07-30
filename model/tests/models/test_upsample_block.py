import unittest
import torch
import sys
import os

# Add project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.upsample_block import UpsampleBlock

class TestUpsampleBlock(unittest.TestCase):
    """
    Unit tests for the UpsampleBlock.
    """

    def test_forward_pass(self):
        """
        Test the forward pass of the UpsampleBlock to ensure it upsamples
        the input tensor to the correct shape.
        """
        in_channels = 64
        out_channels = 32
        block = UpsampleBlock(in_channels, out_channels)
        block.eval()

        # Create a dummy input tensor
        input_tensor = torch.randn(1, in_channels, 14, 14)

        # Perform a forward pass
        with torch.no_grad():
            output = block(input_tensor)

        # Check the output type
        self.assertIsInstance(output, torch.Tensor)

        # Check the output shape (should be 2x the input size)
        self.assertEqual(output.shape, (1, out_channels, 28, 28))

if __name__ == '__main__':
    unittest.main()
