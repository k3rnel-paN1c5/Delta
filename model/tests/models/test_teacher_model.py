import unittest
import torch
import sys
import os
from typing import List

# Add project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.teacher_model import TeacherWrapper

class TestTeacherModel(unittest.TestCase):
    """
    Unit tests for the TeacherWrapper.
    """

    def setUp(self):
        """
        Set up a TeacherWrapper instance for testing.
        We use a smaller, faster model for testing purposes to avoid downloading
        large pre-trained weights during automated testing.
        """
        self.model = TeacherWrapper(model_id="depth-anything/depth-anything-v2-small-hf")
        self.model.eval()

    def test_forward_pass(self):
        """
        Test the forward pass of the TeacherWrapper to ensure it produces an output
        of the correct shape and type.
        """
        # Create a dummy input tensor
        input_tensor = torch.randn(1, 3, 384, 384)

        # Perform a forward pass
        with torch.no_grad():
            depth, feat = self.model(input_tensor)

        # Check the output type
        self.assertIsInstance(depth, torch.Tensor)
        # self.assertIsInstance(feat, List[torch.Tensor])

        # Check the output shape
        # The DPT model produces an output of a different size than the input,
        # so we check for a specific output shape.
        self.assertEqual(depth.shape, (1, 1, 384, 384))

if __name__ == '__main__':
    unittest.main()
