"""This module provides a custom dataset for handling unlabeled images.

The `UnlabeledImageDataset` class is a PyTorch `Dataset` designed to load images
from a directory without corresponding labels. It supports various image formats
and allows for optional resizing and data transformations, making it flexible for
use in unsupervised or self-supervised learning tasks like knowledge distillation.
"""

import os
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Tuple, Any


class UnlabeledImageDataset(Dataset):
    """Custom PyTorch Dataset for loading unlabeled images from a directory.

    This dataset iterates over a directory of images, loads them, and applies
    an optional series of transformations. It's designed for scenarios like
    unsupervised or self-supervised learning where only input images are needed.

    Attributes:
        root_dir: The path to the directory containing the images.
        transform: An optional function/transform to be applied to an image.
        image_paths: A list of full paths to each image file in the directory.
        resize_size: An optional tuple (width, height) to resize images to.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        resize_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initializes the UnlabeledImageDataset.

        Args:
            root_dir: The path to the directory containing the image files.
            transform: An optional transform to be applied on a sample.
            resize_size: An optional tuple (width, height) to resize the images to.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(("png", "jpg", "JPG"))
        ]
        self.resize_size = resize_size
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx) -> Any:
        """Retrieves an image from the dataset.

        Args:
            idx: The index of the image to retrieve.

        Returns:
            The image, transformed if a transform was provided. The return type
            depends on the last transform (e.g., PIL Image or torch.Tensor).
        """
        img_path: str = self.image_paths[idx]
        image: Image.Image = Image.open(img_path).convert("RGB")

        if self.resize_size:
            image = image.resize(self.resize_size)

        if self.transform:
            image = self.transform(image)

        return image
