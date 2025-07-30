import unittest
from unittest.mock import patch, MagicMock
import torch

from models.teacher_model import TeacherWrapper

class TestTeacherModel(unittest.TestCase):
    """
    Unit tests for the TeacherWrapper.
    """

    @patch('models.teacher_model.AutoModelForDepthEstimation.from_pretrained')
    def setUp(self, mock_from_pretrained):
        """Set up a dummy input tensor and a mock teacher model."""
        # Define model parameters
        self.batch_size = 2
        self.input_height = 384
        self.input_width = 384
        self.patch_size = 14
        self.num_channels = 768
        
        # Create a mock for the Hugging Face model
        self.mock_model = MagicMock()
        mock_from_pretrained.return_value = self.mock_model

        # Calculate the correct sequence length for the ViT hidden states
        # This is the number of patches + 1 for the [CLS] token
        h_grid = self.input_height // self.patch_size
        w_grid = self.input_width // self.patch_size
        # Correct sequence length should be (h_grid * w_grid) + 1
        correct_seq_len = (h_grid * w_grid) + 1 # (27 * 27) + 1 = 730

        # Define the mock model's output with corrected dimensions
        mock_outputs = MagicMock()
        # Ensure the mock predicted_depth has the correct batch size
        mock_outputs.predicted_depth = torch.randn(self.batch_size, self.input_height, self.input_width)
        # Ensure the mock hidden_states have the correct sequence length
        mock_outputs.hidden_states = [torch.randn(self.batch_size, correct_seq_len, self.num_channels) for _ in range(12)]
        
        self.mock_model.return_value = mock_outputs

        # Set up the model config
        self.mock_model.config.patch_size = self.patch_size

        # Initialize the TeacherWrapper with the mock model
        self.teacher_model = TeacherWrapper()
        self.input_tensor = torch.randn(self.batch_size, 3, self.input_height, self.input_width)

    def test_forward_pass(self):
        """Test the forward pass of the teacher model, checking output shapes and types."""
        final_depth, reshaped_features = self.teacher_model(self.input_tensor)

        # Test final_depth output
        self.assertIsInstance(final_depth, torch.Tensor)
        self.assertEqual(final_depth.shape[0], self.batch_size)
        self.assertEqual(final_depth.shape[1], 1)
        self.assertEqual(final_depth.shape[2], self.input_height)
        self.assertEqual(final_depth.shape[3], self.input_width)

        # Test reshaped_features output
        self.assertIsInstance(reshaped_features, list)
        self.assertEqual(len(reshaped_features), 4) # As per default selected_features_indices
        for feature in reshaped_features:
            self.assertIsInstance(feature, torch.Tensor)
            self.assertEqual(feature.dim(), 4) # Check for [B, C, H, W] format


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)