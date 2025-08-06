"""
This module defines a simple distillation loss strategy.

The `SimpleDistillationLoss` class implements a basic loss calculation that
focuses solely on minimizing the Mean Squared Error (MSE) between the pixel-wise
depth predictions of the student and teacher models.
"""

import torch.nn.functional as F

from .criterion import LossStrategy


class SimpleDistillationLoss(LossStrategy):
    """A strategy focusing only on matching final depth maps with MSE."""

    def calculate(
        self,
        student_depth,
        teacher_depth,
        student_attns,
        teacher_attns,
        student_features,
        teacher_features,
        **kwargs
    ):
        return F.mse_loss(student_depth, teacher_depth)
