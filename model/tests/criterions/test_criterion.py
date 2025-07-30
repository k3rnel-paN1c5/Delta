import unittest
import torch
import sys
import os

# Add the project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from criterions.criterion import EnhancedDistillationLoss

class TestEnhancedDistillationLoss(unittest.TestCase):
    """
    Unit tests for the EnhancedDistillationLoss function.
    """

    def setUp(self):
        """
        Set up dummy tensors for testing the loss function.
        """
        self.batch_size = 2
        self.height = 32
        self.width = 32

        # Create dummy model outputs and ground truth
        self.student_depth = torch.rand(self.batch_size, 1, self.height, self.width).clamp(min=0.1)
        self.teacher_depth = torch.rand(self.batch_size, 1, self.height, self.width).clamp(min=0.1)

        # Create dummy features and attention maps from student and teacher
        self.student_features = [torch.rand(self.batch_size, 64, self.height // 4, self.width // 4)]
        self.teacher_features = [torch.rand(self.batch_size, 64, self.height // 4, self.width // 4)]
        # Instantiate the loss function with default weights
        self.criterion = EnhancedDistillationLoss()

    def test_forward_pass_returns_scalar_tensor(self):
        """
        Test that the forward pass returns a single, non-negative scalar tensor.
        """
        loss = self.criterion(
            student_depth=self.student_depth,
            teacher_depth=self.teacher_depth,
            student_features=self.student_features,
            teacher_features=self.teacher_features,
        )

        self.assertIsInstance(loss, torch.Tensor, "Loss should be a torch.Tensor")
        self.assertEqual(loss.shape, torch.Size([]), "Loss should be a scalar")
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")

    
    def test_distillation_loss_contribution(self):
        """
        Test that the distillation components (gradient, feature, attention) contribute correctly.
        """
        # Create a criterion with only distillation components enabled
        criterion_distill_only = EnhancedDistillationLoss(lambda_silog=0.0, lambda_grad=1.0, lambda_feat=1.0, lambda_attn=1.0)

        # When student and teacher outputs are different, loss should be positive
        loss = criterion_distill_only(
            student_depth=self.student_depth, teacher_depth=self.teacher_depth,
            student_features=self.student_features, teacher_features=self.teacher_features
        )
        self.assertGreater(loss.item(), 0)

        
if __name__ == '__main__':
    unittest.main()
