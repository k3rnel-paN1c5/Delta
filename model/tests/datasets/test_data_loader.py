import unittest
import os
import shutil
from PIL import Image
import torch
from torchvision import transforms

from datasets.data_loader import UnlabeledImageDataset

class TestUnlabeledImageDataset(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory with dummy image files for testing."""
        self.test_dir = "test_images"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create dummy image files
        Image.new('RGB', (60, 30), color = 'red').save(os.path.join(self.test_dir, 'test1.jpg'))
        Image.new('RGB', (60, 30), color = 'green').save(os.path.join(self.test_dir, 'test2.png'))
        Image.new('RGB', (60, 30), color = 'blue').save(os.path.join(self.test_dir, 'test3.JPG'))

        # Create a non-image file that should be ignored
        with open(os.path.join(self.test_dir, 'test.txt'), 'w') as f:
            f.write("not an image")

    def tearDown(self):
        """Remove the temporary directory and its contents after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test if the dataset initializes and finds the correct number of images."""
        dataset = UnlabeledImageDataset(root_dir=self.test_dir)
        self.assertEqual(len(dataset.image_paths), 3)
        print("\nTest `test_initialization` passed.")

    def test_len(self):
        """Test if the __len__ method returns the correct number of images."""
        dataset = UnlabeledImageDataset(root_dir=self.test_dir)
        self.assertEqual(len(dataset), 3)
        print("Test `test_len` passed.")

    def test_getitem(self):
        """Test if an item can be retrieved and is a PIL Image."""
        dataset = UnlabeledImageDataset(root_dir=self.test_dir)
        image = dataset[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, 'RGB')
        print("Test `test_getitem` passed.")

    def test_resize(self):
        """Test if the image is correctly resized."""
        resize_size = (32, 32)
        dataset = UnlabeledImageDataset(root_dir=self.test_dir, resize_size=resize_size)
        image = dataset[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, resize_size)
        print("Test `test_resize` passed.")

    def test_transform(self):
        """Test if the transform is correctly applied."""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = UnlabeledImageDataset(root_dir=self.test_dir, transform=transform)
        image_tensor = dataset[0]
        self.assertIsInstance(image_tensor, torch.Tensor)
        # Check for [C, H, W] format
        self.assertEqual(len(image_tensor.shape), 3)
        self.assertEqual(image_tensor.shape[0], 3) # 3 channels for RGB
        print("Test `test_transform` passed.")

    def test_transform_with_resize(self):
        """Test if both resize and transform are applied correctly."""
        resize_size = (40, 40)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = UnlabeledImageDataset(root_dir=self.test_dir, transform=transform, resize_size=resize_size)
        image_tensor = dataset[0]

        self.assertIsInstance(image_tensor, torch.Tensor)
        # Check for [C, H, W] format and correct dimensions
        self.assertEqual(len(image_tensor.shape), 3)
        self.assertEqual(image_tensor.shape[0], 3)
        self.assertEqual(image_tensor.shape[1], resize_size[1]) # Height
        self.assertEqual(image_tensor.shape[2], resize_size[0]) # Width
        print("Test `test_transform_with_resize` passed.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)