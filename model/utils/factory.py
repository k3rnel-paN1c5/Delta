"""Optimizer Factory for creating optimizers.

This module provides a way to instantiate optimizers with specific parameter groups,
such as applying different learning rates to an encoder and a decoder.
"""

import torch
from torch.optim import Optimizer
from torch.nn import Module
from types import ModuleType


class OptimizerFactory:
    """A factory class for creating optimizers."""

    @staticmethod
    def create_optimizer(model: Module, config: ModuleType) -> Optimizer:
        """Creates and returns an optimizer instance for the given model.

        This implementation specifically handles creating parameter groups for an
        encoder and a decoder to apply different learning rates, a common practice
        in fine-tuning.

        Args:
            model: The model whose parameters will be optimized.
            config: The configuration module containing optimizer settings
                like learning rates and weight decay.

        Returns:
            A configured optimizer instance.

        Raises:
            ValueError: If the optimizer name in the config is unknown.
        """
        param_groups = [
            {"params": model.encoder.parameters(), "lr": config.LEARNING_RATE_ENCODER},
            {"params": model.decoder.parameters(), "lr": config.LEARNING_RATE_DECODER},
        ]

        optimizer_name: str = getattr(config, "OPTIMIZER_NAME", "AdamW")

        if optimizer_name == "AdamW":
            return torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
        elif optimizer_name == "Adam":
            return torch.optim.Adam(param_groups, weight_decay=config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unknown optimizer name: {config.OPTIMIZER_NAME}")
