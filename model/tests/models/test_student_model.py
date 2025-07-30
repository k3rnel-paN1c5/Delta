import unittest
import torch
import sys
import os
from typing import Tuple, List

# Add project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from models.student_model import StudentDepthModel

class TestStudentModel(unittest.TestCase):
    """
    Unit tests for the StudentDepthModel.
    """

    def setUp(self):
        """
        Set up a StudentDepthModel instance for testing.
        """
        self.model = StudentDepthModel(
            encoder_name='mobilevit_xs',
            feature_indices = (0, 1, 2, 3),
            decoder_channels=(64, 128, 160, 256),
            pretrained=True
        )
        self.model.eval()

    def test_forward_pass(self):
        """
        Test the forward pass of the StudentDepthModel to ensure it produces an output
        of the correct shape and type.
        """
        # Create a dummy input tensor
        input_tensor = torch.randn(1, 3, 224, 224)

        # Perform a forward pass
        with torch.no_grad():
            depth_map, selected_features = self.model(input_tensor)

        # Check the output type
        self.assertIsInstance(depth_map, torch.Tensor)
        # self.assertIsInstance(selected_features, List[torch.Tensor])

        # Check the output shape
        self.assertEqual(depth_map.shape, (1, 1, 224, 224))

if __name__ == '__main__':
    unittest.main()
