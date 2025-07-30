import unittest
import torch
import sys
import os

# Add project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.feature_fusion_block import FeatureFusionBlock

class TestFeatureFusionBlock(unittest.TestCase):
    """
    Unit tests for the FeatureFusionBlock.
    """

    def test_forward_pass(self):
        """
        Test the forward pass of the FeatureFusionBlock to ensure it correctly
        fuses the input and skip connection tensors.
        """
        channels = 64
        block = FeatureFusionBlock(channels)
        block.eval()

        # Create dummy input and skip connection tensors
        input_tensor = torch.randn(1, channels, 28, 28)
        skip_connection = torch.randn(1, channels, 28, 28)

        # Perform a forward pass
        with torch.no_grad():
            output = block(input_tensor, skip_connection)

        # Check the output type
        self.assertIsInstance(output, torch.Tensor)

        # Check the output shape (should be the same as the input)
        self.assertEqual(output.shape, (1, channels, 28, 28))

if __name__ == '__main__':
    unittest.main()
