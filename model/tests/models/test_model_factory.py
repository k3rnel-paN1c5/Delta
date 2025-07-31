import unittest
import types

from models.factory import ModelFactory
from models.student_model import StudentDepthModel
from models.teacher_model import TeacherWrapper

class TestModelFactory(unittest.TestCase):
    """
    Test suite for the ModelFactory class.
    
    Verifies that the factory can correctly create student and teacher models
    and handles invalid configurations.
    """

    def setUp(self):
        """Set up a mock configuration object for testing."""
        self.config = types.ModuleType('config')
        
        # Student Config
        self.config.STUDENT_MODEL_NAME = 'StudentDepthModel'
        self.config.STUDENT_FEATURE_INDICES = [0, 1, 2, 3]
        self.config.USE_PRETRAINED = True
        self.config.STUDENT_DECODER_CHANNELS = [64, 128, 160, 256]

        # Teacher Config
        self.config.TEACHER_MODEL_NAME = 'DepthAnythingV2'
        self.config.TEACHER_FEATURE_INDICES = [3, 5, 7, 11]

    def test_create_student_model(self):
        """Test successful creation of a StudentDepthModel."""
        model = ModelFactory.create_student_model(self.config)
        self.assertIsInstance(model, StudentDepthModel, "The created object should be a StudentDepthModel instance.")

    def test_create_teacher_model(self):
        """Test successful creation of a TeacherWrapper."""
        model = ModelFactory.create_teacher_model(self.config)
        self.assertIsInstance(model, TeacherWrapper, "The created object should be a TeacherWrapper instance.")

    def test_unknown_student_model_raises_error(self):
        """Test that an unknown student model name raises a ValueError."""
        self.config.STUDENT_MODEL_NAME = 'UnknownModel'
        with self.assertRaises(ValueError):
            ModelFactory.create_student_model(self.config)

    def test_unknown_teacher_model_raises_error(self):
        """Test that an unknown teacher model name raises a ValueError."""
        self.config.TEACHER_MODEL_NAME = 'UnknownTeacher'
        with self.assertRaises(ValueError):
            ModelFactory.create_teacher_model(self.config)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)