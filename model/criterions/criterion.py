"""
Main loss module and loss strategies for knowledge distillation.

This module implements the Strategy design pattern.
- LossStrategy: An abstract base class defining the interface for all loss strategies.
- Concrete Strategies (CombinedDistillationLoss, SimpleDistillationLoss): Implement
  specific loss calculation algorithms.
- DistillationLoss: The "Context" class that uses a strategy object to compute
  the final loss. The strategy can be changed at runtime.
"""

from abc import ABC, abstractmethod
from torch import nn


class LossStrategy(ABC):
    """
    Abstract Base Class for a loss calculation strategy.

    All concrete loss strategies must implement the `calculate` method.
    """

    @abstractmethod
    def calculate(
        self,
        student_depth,
        teacher_depth,
        student_attns,
        teacher_attns,
        student_features,
        teacher_features,
        **kwargs,
    ):
        """
        Calculates the loss based on a specific strategy.

        Args:
            student_depth (Tensor): The depth map predicted by the student.
            teacher_depth (Tensor): The depth map predicted by the teacher.
            student_attns (list): List of attention maps from the student.
            teacher_attns (list): List of attention maps from the teacher.
            student_features (list): List of feature maps from the student.
            teacher_features (list): List of feature maps from the teacher.
            **kwargs: Additional arguments, such as loss weights.

        Returns:
            torch.Tensor: The computed loss value.
        """
        pass


class DistillationLoss(nn.Module):
    """
    The main loss class (Context) that uses a configurable strategy.

    This class holds a reference to a loss strategy object and delegates
    the actual loss computation to it.
    """

    def __init__(self, strategy: LossStrategy, **loss_weights):
        """
        Args:
            strategy (LossStrategy): The loss calculation strategy to use.
            **loss_weights: A dictionary of weights (e.g., lambda_silog, lambda_grad).
        """
        super().__init__()
        self._strategy = strategy
        self.loss_weights = loss_weights
        print(
            f"Initialized DistillationLoss with strategy: {strategy.__class__.__name__}"
        )

    def forward(
        self,
        student_depth,
        teacher_depth,
        student_attns,
        teacher_attns,
        student_features,
        teacher_features,
    ):
        """Delegates the loss calculation to the current strategy."""
        return self._strategy.calculate(
            student_depth,
            teacher_depth,
            student_attns,
            teacher_attns,
            student_features,
            teacher_features,
            **self.loss_weights,
        )
