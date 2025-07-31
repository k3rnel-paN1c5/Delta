import torch.nn.functional as F

from .criterion import LossStrategy


class SimpleDistillationLoss(LossStrategy):
    """A strategy focusing only on matching final depth maps."""

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
        loss_depth = 0.0
        for s_depth, t_depth in zip(student_depth, teacher_depth):
            loss_depth += F.mse_loss(s_depth, t_depth)
        return loss_depth
