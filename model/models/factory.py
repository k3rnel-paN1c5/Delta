"""
Model Factory for creating student and teacher models.

This module provides a centralized way to instantiate models based on a configuration
object. It decouples the main training script from the specific model implementations,
making it easy to add new models without modifying the training pipeline.
"""

from models.student_model import StudentDepthModel
from models.teacher_model import TeacherWrapper


class ModelFactory:
    """A factory class for creating deep learning models for depth estimation."""

    @staticmethod
    def create_student_model(config):
        """
        Creates and returns a student model instance based on the configuration.

        Args:
            config (module): The configuration module which contains model-specific
                             hyperparameters like backbone, feature count, etc.

        Returns:
            torch.nn.Module: An instance of the student model.
        
        Raises:
            ValueError: If the student model name in the config is unknown.
        """
        model_name = getattr(config, 'STUDENT_MODEL_NAME', 'StudentDepthModel')

        if model_name == 'StudentDepthModel':
            return StudentDepthModel(
                feature_indices=config.STUDENT_FEATURE_INDICES,
                decoder_channels=config.STUDENT_DECODER_CHANNELS,
                pretrained=config.USE_PRETRAINED
            )
        else:
            raise ValueError(f"Unknown student model name: {model_name}")

    @staticmethod
    def create_teacher_model(config):
        """
        Creates and returns a teacher model instance based on the configuration.

        The teacher model is typically a pre-trained, larger model used for
        knowledge distillation.

        Args:
            config (module): The configuration module which contains the
                             Hugging Face model name for the teacher.

        Returns:
            torch.nn.Module: An instance of the teacher model wrapper.
        
        Raises:
            ValueError: If the teacher model name in the config is unknown.
        """
        model_name = getattr(config, 'TEACHER_MODEL_NAME', 'DepthAnythingV2')

        if model_name == 'DepthAnythingV2':
            return TeacherWrapper()
        else:
            raise ValueError(f"Unknown teacher model name: {model_name}")