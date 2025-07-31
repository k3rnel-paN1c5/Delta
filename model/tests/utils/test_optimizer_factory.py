import unittest
import types
import torch
from torch import nn

from utils.factory import OptimizerFactory

# A mock model for testing purposes
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 10)
        self.decoder = nn.Linear(10, 10)

class TestOptimizerFactory(unittest.TestCase):
    """
    Test suite for the OptimizerFactory class.
    
    Verifies that the factory can correctly create optimizers with
    the specified parameter groups and learning rates.
    """
    def setUp(self):
        """Set up a mock model and configuration for testing."""
        self.model = MockModel()
        self.config = types.ModuleType('config')
        self.config.OPTIMIZER_NAME = 'AdamW'
        self.config.LEARNING_RATE_ENCODER = 1e-5
        self.config.LEARNING_RATE_DECODER = 1e-4
        self.config.WEIGHT_DECAY = 1e-2

    def test_create_optimizer_adamw(self):
        """Test successful creation of an AdamW optimizer with parameter groups."""
        optimizer = OptimizerFactory.create_optimizer(self.model, self.config)
        self.assertIsInstance(optimizer, torch.optim.AdamW, "The created object should be an AdamW instance.")
        
        # Check that there are two parameter groups
        self.assertEqual(len(optimizer.param_groups), 2, "Optimizer should have two parameter groups.")
        
        # Check that the learning rates are set correctly
        self.assertEqual(optimizer.param_groups[0]['lr'], self.config.LEARNING_RATE_ENCODER, "Encoder learning rate is incorrect.")
        self.assertEqual(optimizer.param_groups[1]['lr'], self.config.LEARNING_RATE_DECODER, "Decoder learning rate is incorrect.")
        
    def test_create_optimizer_adam(self):
        """Test successful creation of an Adam optimizer with parameter groups."""
        self.config.OPTIMIZER_NAME = 'Adam'
        optimizer = OptimizerFactory.create_optimizer(self.model, self.config)
        self.assertIsInstance(optimizer, torch.optim.Adam, "The created object should be an Adam instance.")
        
        # Check that there are two parameter groups
        self.assertEqual(len(optimizer.param_groups), 2, "Optimizer should have two parameter groups.")
        
        # Check that the learning rates are set correctly
        self.assertEqual(optimizer.param_groups[0]['lr'], self.config.LEARNING_RATE_ENCODER, "Encoder learning rate is incorrect.")
        self.assertEqual(optimizer.param_groups[1]['lr'], self.config.LEARNING_RATE_DECODER, "Decoder learning rate is incorrect.")

    def test_unknown_optimizer_raises_error(self):
        """Test that an unknown optimizer name raises a ValueError."""
        self.config.OPTIMIZER_NAME = 'UnknownOptimizer'
        with self.assertRaises(ValueError):
            OptimizerFactory.create_optimizer(self.model, self.config)

if __name__ == '__main__':
    unittest.main()