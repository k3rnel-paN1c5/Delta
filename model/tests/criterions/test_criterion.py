import unittest
import torch
import torch.nn as nn

from config import config

from criterions.combined_distillation_loss import compute_depth_gradients, CombinedDistillationLoss
from criterions.factory import CriterionFactory
class TestCombinedDistillationLoss(unittest.TestCase):
    """
    Unit tests for the CombinedDistillationLoss function.
    """

    def setUp(self):
        """Set up dummy tensors for testing the loss functions."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.device = config.DEVICE
        # Mock model outputs
        self.student_depth = torch.rand(self.batch_size, 1, self.height, self.width, device=self.device) + 0.1
        self.teacher_depth = torch.rand(self.batch_size, 1, self.height, self.width, device=self.device) + 0.1

        # Mock feature maps with different channel and spatial sizes
        self.student_features = [
            torch.randn(self.batch_size, 64, self.height // 2, self.width // 2, device=self.device),
            torch.randn(self.batch_size, 128, self.height // 4, self.width // 4, device=self.device)
        ]
        self.teacher_features = [
            torch.randn(self.batch_size, 96, self.height // 2, self.width // 2, device=self.device),
            # Test spatial resizing
            torch.randn(self.batch_size, 160, self.height // 8, self.width // 8, device=self.device) 
        ]
        self.loss_fn = CriterionFactory.create_criterion(config).to(self.device)
    


    def test_compute_depth_gradients(self):
        """Test the compute_depth_gradients function."""
        depth_map = torch.randn(self.batch_size, 1, self.height, self.width, device=self.device)
        gradients = compute_depth_gradients(depth_map)

        # Check output shape: [B, 2, H, W] (for dy and dx)
        self.assertEqual(gradients.shape, (self.batch_size, 2, self.height, self.width))
        self.assertIsInstance(gradients, torch.Tensor)
        
        # Gradients of a constant map should be zero
        constant_map = torch.ones(self.batch_size, 1, self.height, self.width, device=self.device)
        zero_grads = compute_depth_gradients(constant_map)
        self.assertTrue(torch.all(zero_grads == 0))
    
    def test_attention_map_computation(self):
        """Test the internal _compute_attention_map method."""
        if config.LOSS_STRATEGY != 'CombinedDistillationLoss':
            return 
        feature_map = torch.randn(self.batch_size, 64, 16, 16, device=self.device)

        attention_map = self.loss_fn._strategy._compute_attention_map(feature_map)

        # Check output shape: [B, 1, H, W]
        self.assertEqual(attention_map.shape, (self.batch_size, 1, 16, 16))
        self.assertGreaterEqual(attention_map.min().item(), 0) # Attention map should be non-negative

        
    def test_projection_layer_initialization(self):
        """Test that projection conv layers are created correctly on the first forward pass."""
        if config.LOSS_STRATEGY != 'CombinedDistillationLoss':
            return 
        self.assertIsNone(self.loss_fn._strategy.projection_convs) # Should be None before first forward pass
        
        # Run forward pass
        self.loss_fn(self.student_depth, self.teacher_depth, None, None,  self.student_features, self.teacher_features)
        
        # Check if ModuleList was created
        self.assertIsInstance(self.loss_fn._strategy.projection_convs, nn.ModuleList)
        self.assertEqual(len(self.loss_fn._strategy.projection_convs), len(self.student_features))

        # Check channel dimensions of the projection layers
        s_channels = [feat.shape[1] for feat in self.student_features]
        t_channels = [feat.shape[1] for feat in self.teacher_features]
        for i, proj_layer in enumerate(self.loss_fn._strategy.projection_convs):
            self.assertEqual(proj_layer.in_channels, s_channels[i])
            self.assertEqual(proj_layer.out_channels, t_channels[i])

    def test_forward_pass_and_loss_value(self):
        """Test the forward pass of the CombinedDistillationLoss."""
        loss = self.loss_fn(self.student_depth, self.teacher_depth, None, None, self.student_features, self.teacher_features)

        # Check if loss is a scalar tensor and is non-negative
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

    def test_zero_loss_on_identical_inputs(self):
        """Test if the loss is zero when student and teacher inputs are identical."""
        # Create a new loss function to ensure projections match C_in to C_in
        loss_fn_zero_test = CriterionFactory.create_criterion(config).to(self.device)
        
        # When inputs are identical, all loss components should be zero
        loss = loss_fn_zero_test(self.student_depth, self.student_depth, None, None, self.student_features, self.student_features)
        
        # The loss should be very close to zero
        self.assertAlmostEqual(loss.item(), 0, places=6)

    def test_loss_with_zero_depth(self):
        """Test that the loss calculation is stable when some depth values are zero."""
        student_depth_with_zeros = self.student_depth.clone()
        student_depth_with_zeros[0, 0, 0, 0] = 0 # Introduce a zero value

        loss = self.loss_fn(student_depth_with_zeros, self.teacher_depth, None, None, self.student_features, self.teacher_features)
        
        # Ensure the loss is not NaN or Inf
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
