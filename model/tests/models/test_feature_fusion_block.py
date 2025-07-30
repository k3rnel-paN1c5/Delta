import unittest
import torch

from models.feature_fusion_block import FeatureFusionBlock

class TestFeatureFusionBlock(unittest.TestCase):
    """
    Unit tests for the FeatureFusionBlock.
    """ 
    def setUp(self):
        """Set up dummy input tensors for testing the feature fusion block."""
        self.channels = 64
        self.higher_level_features = torch.randn(2, self.channels, 32, 32)
        self.skip_features = torch.randn(2, self.channels, 32, 32)
        self.feature_fusion_block = FeatureFusionBlock(self.channels)
        self.feature_fusion_block.eval()

    def test_output_shape(self):
        """Test if the output of the feature fusion block has the correct shape."""
        output_tensor = self.feature_fusion_block(self.higher_level_features, self.skip_features)
        self.assertEqual(output_tensor.shape[0], 2)
        self.assertEqual(output_tensor.shape[1], self.channels)
        self.assertEqual(output_tensor.shape[2], 32)
        self.assertEqual(output_tensor.shape[3], 32)

    def test_output_type(self):
        """Test if the output of the feature fusion block is a torch.Tensor."""
        output_tensor = self.feature_fusion_block(self.higher_level_features, self.skip_features)
        self.assertIsInstance(output_tensor, torch.Tensor)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
