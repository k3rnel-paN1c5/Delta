"""
Criterion Factory for creating loss functions using Strategy.

This module provides a centralized way to instantiate loss functions (criterions)
based on a configuration object. This allows for easy swapping of loss functions
for experimentation.
"""
from criterions.criterion import DistillationLoss
from criterions.combined_distillation_loss import CombinedDistillationLoss
from criterions.simple_distillation_loss import SimpleDistillationLoss

class CriterionFactory:
    """A factory class for creating loss functions."""

    @staticmethod
    def create_criterion(config):
        """
        Creates a loss function with a specific strategy based on the config.

        Args:
            config (module): The configuration module which contains the
                             desired strategy and loss weights.

        Returns:
            DistillationLoss: A configured instance of the main loss class.

        Raises:
            ValueError: If the strategy name in the config is unknown.
        """
        strategy_name = getattr(config, 'LOSS_STRATEGY', 'CombinedDistillationLoss')
        
        # Map strategy names to classes
        strategy_map = {
            'CombinedDistillationLoss': CombinedDistillationLoss,
            'SimpleDistillationLoss': SimpleDistillationLoss,
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown loss strategy name: {strategy_name}")
        
        strategy_instance = strategy_map[strategy_name]()
        
        # Get all loss weights from config
        loss_weights = {
            'lambda_silog': getattr(config, 'LAMBDA_SILOG', 1.0),
            'lambda_grad': getattr(config, 'LAMBDA_GRAD', 1.0),
            'lambda_attn': getattr(config, 'LAMBDA_ATTN', 1.0),
            'lambda_feat': getattr(config, 'LAMBDA_FEAT', 1.0),
            'alpha': getattr(config, 'ALPHA', 1.0) 
        }
        return DistillationLoss(strategy=strategy_instance, **loss_weights)