import unittest
import torch
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from criterions.factory import CriterionFactory
class TestSimpleDistillationLoss(unittest.TestCase):
    """
    Unit tests for the SimpleDistillationLoss criterion.
    """

    def setUp(self):
        """Initialize the loss function before each test."""
        self.config = types.ModuleType('config')
        self.config.LOSS_STRATEGY = 'SimpleDistillationLoss'
        self.criterion = CriterionFactory.create_criterion(self.config)
        self.batch_size = 4
        self.height = 64
        self.width = 64

    def test_loss_shape(self):
        """
        Test that the loss output is a single scalar tensor.
        """
        print("Testing simple distillation loss shape...")
        student_output = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_output = torch.randn(self.batch_size, 1, self.height, self.width)
        
        loss = self.criterion(student_output, teacher_output, None, None, None, None)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0) # Should be a scalar

    def test_loss_value_identical_inputs(self):
        """
        Test that the loss is zero when student and teacher outputs are identical.
        """
        print("Testing simple distillation loss with identical inputs...")
        student_output = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_output = student_output.clone()
        
        loss = self.criterion(student_output, teacher_output, None, None, None, None)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_loss_value_different_inputs(self):
        """
        Test that the loss is greater than zero for different inputs.
        """
        print("Testing simple distillation loss with different inputs...")
        student_output = torch.randn(self.batch_size, 1, self.height, self.width)
        teacher_output = torch.randn(self.batch_size, 1, self.height, self.width)
        
        loss = self.criterion(student_output, teacher_output, None, None, None, None)
        
        self.assertGreater(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
