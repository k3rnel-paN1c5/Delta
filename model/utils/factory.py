"""
Optimizer Factory for creating optimizers.

This module provides a way to instantiate optimizers with specific parameter groups,
such as applying different learning rates to an encoder and a decoder.
"""

import torch


class OptimizerFactory:
    """A factory class for creating optimizers."""

    @staticmethod
    def create_optimizer(model, config):
        """
        Creates and returns an optimizer instance for the given model.

        This implementation specifically handles creating parameter groups for an
        encoder and a decoder to apply different learning rates, a common practice
        in fine-tuning.

        Args:
            model (torch.nn.Module): The model whose parameters will be optimized.
            config (module): The configuration module containing optimizer settings
                             like learning rates and weight decay.

        Returns:
            torch.optim.Optimizer: An optimizer instance.
        """
        param_groups = [
            {'params': model.encoder.parameters(), 'lr': config.LEARNING_RATE_ENCODER},
            {'params': model.decoder.parameters(), 'lr': config.LEARNING_RATE_DECODER}
        ]

        if config.OPTIMIZER_NAME == 'AdamW':
            return torch.optim.AdamW(
                param_groups,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER_NAME == 'Adam':
            return torch.optim.Adam(
                param_groups,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            # Add other optimizers like SGD, Adam, etc. here if needed
            raise ValueError(f"Unknown optimizer name: {config.OPTIMIZER_NAME}")