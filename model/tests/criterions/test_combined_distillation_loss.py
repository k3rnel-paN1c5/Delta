import unittest
import torch
import os
import sys
import types

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from criterions.factory import CriterionFactory

class TestCombinedDistillationLoss(unittest.TestCase):
    """
    Unit tests for the CombinedDistillationLoss criterion.
    """

    def setUp(self):
        """Initialize the loss function before each test."""
        self.config = types.ModuleType('config')
        self.config.LOSS_STRATEGY = 'CombinedDistillationLoss'
        self.batch_size = 4
        self.height = 64
        self.width = 64       
        self.criterion = CriterionFactory().create_criterion(self.config)

    def test_loss_shape(self):
        """
        Test that the loss output is a single scalar tensor.
        """
        print("Testing combined distillation loss shape...")

        student_output = torch.randn(self.batch_size, 1, self.height, self.width)
        student_features = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_output = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_features = torch.randn(self.batch_size, 1, self.height, self.width)

        loss = self.criterion(student_output, teacher_output, None, None, student_features, teacher_features)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)

    def test_loss_value_identical_inputs(self):
        """
        Test that the loss is zero when student and teacher outputs are identical.
        """
        print("Testing simple distillation loss with identical inputs...")
        student_output = torch.randn(self.batch_size, 1, self.height, self.width)
        student_features = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_output = student_output.clone()
        teacher_features = student_features.clone()
        

        loss = self.criterion(student_output, teacher_output, None, None, student_features, teacher_features)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_loss_value_different_inputs(self):
        """
        Test that the loss is greater than zero for different inputs.
        """
        print("Testing simple distillation loss with different inputs...")
        student_output = torch.randn(self.batch_size, 1, self.height, self.width)
        student_features = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_output = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_features = torch.randn(self.batch_size, 1, self.height, self.width)
        
        loss = self.criterion(student_output, teacher_output, None, None, student_features, teacher_features)
        
        self.assertGreater(loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
