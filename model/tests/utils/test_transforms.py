import unittest
from unittest.mock import MagicMock
import torch
from torchvision import transforms
from PIL import Image
import sys

from utils.transforms import get_train_transforms, get_eval_transforms
# Mock the config module and its attributes

mock_config = MagicMock()
mock_config.IMG_HEIGHT = 384
mock_config.IMG_WIDTH = 384
mock_config.FLIP_PROP = 0.5
mock_config.ROTATION_DEG = 10
mock_config.MIN_SCALE = 0.8
mock_config.MAX_SCALE = 1.0
mock_config.BRIGHTNESS = 0.4
mock_config.CONTRAST = 0.4
mock_config.SATURATION = 0.4
mock_config.HUE = 0.1
mock_config.IMGNET_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
mock_config.IMGNET_NORMALIZE_STD = [0.229, 0.224, 0.225]
sys.modules['config'] = mock_config



class TestTransforms(unittest.TestCase):

    def setUp(self):
        """Set up a dummy image for testing the transforms."""
        self.dummy_image = Image.new('RGB', (400, 400), color = 'red')

    def test_get_train_transforms_structure_and_params(self):
        """
        Test if get_train_transforms returns a Compose object with the correct transforms and parameters.
        """
        train_transforms = get_train_transforms()
        self.assertIsInstance(train_transforms, transforms.Compose)

        # Check for the correct number of transforms
        self.assertEqual(len(train_transforms.transforms), 6)

        # Unpack transforms
        flip, rotation, crop, jitter, to_tensor, normalize = train_transforms.transforms

        # Validate types
        self.assertIsInstance(flip, transforms.RandomHorizontalFlip)
        self.assertIsInstance(rotation, transforms.RandomRotation)
        self.assertIsInstance(crop, transforms.RandomResizedCrop)
        self.assertIsInstance(jitter, transforms.ColorJitter)
        self.assertIsInstance(to_tensor, transforms.ToTensor)
        self.assertIsInstance(normalize, transforms.Normalize)

        # Validate parameters
        self.assertEqual(flip.p, mock_config.FLIP_PROP)
        self.assertEqual(rotation.degrees, [-mock_config.ROTATION_DEG, mock_config.ROTATION_DEG])
        self.assertEqual(crop.size, (mock_config.IMG_HEIGHT, mock_config.IMG_WIDTH))
        self.assertEqual(crop.scale, (mock_config.MIN_SCALE, mock_config.MAX_SCALE))
        self.assertEqual(jitter.brightness, (max(0, 1 - mock_config.BRIGHTNESS), 1 + mock_config.BRIGHTNESS))
        self.assertEqual(normalize.mean, mock_config.IMGNET_NORMALIZE_MEAN)
        self.assertEqual(normalize.std, mock_config.IMGNET_NORMALIZE_STD)

    def test_get_train_transforms_output(self):
        """
        Test the output of the training transformation pipeline.
        """
        train_transforms = get_train_transforms()
        transformed_image = train_transforms(self.dummy_image)

        # Check output type and shape
        self.assertIsInstance(transformed_image, torch.Tensor)
        self.assertEqual(transformed_image.shape, (3, mock_config.IMG_HEIGHT, mock_config.IMG_WIDTH))

    def test_get_eval_transforms_structure_and_params(self):
        """
        Test if get_eval_transforms returns a Compose object with the correct transforms and parameters.
        """
        eval_transforms = get_eval_transforms()
        self.assertIsInstance(eval_transforms, transforms.Compose)

        # Check for the correct number of transforms
        self.assertEqual(len(eval_transforms.transforms), 3)

        # Unpack transforms
        resize, to_tensor, normalize = eval_transforms.transforms

        # Validate types
        self.assertIsInstance(resize, transforms.Resize)
        self.assertIsInstance(to_tensor, transforms.ToTensor)
        self.assertIsInstance(normalize, transforms.Normalize)

        # Validate parameters
        self.assertEqual(resize.size, (mock_config.IMG_HEIGHT, mock_config.IMG_WIDTH))
        self.assertEqual(normalize.mean, mock_config.IMGNET_NORMALIZE_MEAN)
        self.assertEqual(normalize.std, mock_config.IMGNET_NORMALIZE_STD)

    def test_get_eval_transforms_output(self):
        """
        Test the output of the evaluation transformation pipeline.
        """
        eval_transforms = get_eval_transforms()
        transformed_image = eval_transforms(self.dummy_image)

        # Check output type and shape
        self.assertIsInstance(transformed_image, torch.Tensor)
        self.assertEqual(transformed_image.shape, (3, mock_config.IMG_HEIGHT, mock_config.IMG_WIDTH))

    def test_custom_input_size(self):
        """
        Test if the functions correctly use a custom input size.
        """
        custom_size = (128, 128)
        train_transforms = get_train_transforms(input_size=custom_size)
        eval_transforms = get_eval_transforms(input_size=custom_size)

        # Check training transforms with custom size
        self.assertEqual(train_transforms.transforms[2].size, custom_size) # RandomResizedCrop
        train_output = train_transforms(self.dummy_image)
        self.assertEqual(train_output.shape, (3, custom_size[0], custom_size[1]))

        # Check evaluation transforms with custom size
        self.assertEqual(eval_transforms.transforms[0].size, custom_size) # Resize
        eval_output = eval_transforms(self.dummy_image)
        self.assertEqual(eval_output.shape, (3, custom_size[0], custom_size[1]))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)