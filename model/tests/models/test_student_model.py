import unittest
import torch

from models.student_model import StudentDepthModel

class TestStudentModel(unittest.TestCase):
    """
    Unit tests for the StudentDepthModel.
    """

    def setUp(self):
        """Set up a dummy input tensor and initialize the student model."""
        self.input_tensor = torch.randn(2, 3, 384, 384)
        self.student_model = StudentDepthModel(pretrained=False)
        self.student_model.eval()


    def test_forward_pass(self):
        """Test the forward pass of the student model, checking output shapes and types."""
        depth_map, features = self.student_model(self.input_tensor)

        # Test depth map output
        self.assertIsInstance(depth_map, torch.Tensor)
        self.assertEqual(depth_map.shape[0], 2)
        self.assertEqual(depth_map.shape[1], 1)
        self.assertEqual(depth_map.shape[2], 384)
        self.assertEqual(depth_map.shape[3], 384)

        # Test features output
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, torch.Tensor)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)