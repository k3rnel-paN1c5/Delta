import unittest
import types

from criterions.factory import CriterionFactory
from criterions.criterion import DistillationLoss
from criterions.simple_distillation_loss import SimpleDistillationLoss
from criterions.combined_distillation_loss import CombinedDistillationLoss



class TestCriterionFactoryWithStrategy(unittest.TestCase):
    """
    Test suite for the CriterionFactory using the Strategy pattern.
    """

    def setUp(self):
        """Set up a mock configuration object for testing."""
        self.config = types.ModuleType('config')
        self.config.LAMBDA_SILOG = 1.0
        self.config.LAMBDA_GRAD = 1.0
        self.config.LAMBDA_ATTN = 1.0
        self.config.LAMBDA_FEAT = 1.0
        self.config.ALPHA = 0.5

    def test_create_with_combined_distillation_strategy(self):
        """Test successful creation of a CombinedDistillationLoss."""
        self.config.LOSS_STRATEGY = 'CombinedDistillationLoss'
        criterion = CriterionFactory.create_criterion(self.config)
        self.assertIsInstance(criterion, DistillationLoss)
        self.assertIsInstance(criterion._strategy, CombinedDistillationLoss,
                              "The internal strategy should be CombinedDistillationLoss.")
        
        
    def test_create_with_simple_distillation_strategy(self):
        """Test creation with the SimpleDistillationLoss."""
        self.config.LOSS_STRATEGY = 'SimpleDistillationLoss'
        criterion = CriterionFactory.create_criterion(self.config)
        
        self.assertIsInstance(criterion, DistillationLoss)
        self.assertIsInstance(criterion._strategy, SimpleDistillationLoss,
                              "The internal strategy should be SimpleDistillationLoss.")
        
    def test_unknown_strategy_raises_error(self):
        """Test that an unknown strategy name raises a ValueError."""
        self.config.LOSS_STRATEGY = 'UnknownStrategy'
        with self.assertRaises(ValueError):
            CriterionFactory.create_criterion(self.config)


if __name__ == '__main__':
    unittest.main()