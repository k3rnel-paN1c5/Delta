"""This module provides a utility function for cropping images to a specific aspect
ratio.

The `crop_to_aspect_ratio` function is a helper that takes a PIL image and a
target aspect ratio, and performs a center crop to match that ratio. This is
useful for preprocessing images to a consistent shape before feeding them into a
model.
"""

from PIL import Image
from typing import Tuple

def crop_to_aspect_ratio(image: Image.Image, aspect_ratio: float) -> Image.Image:
    """Crops a PIL image to a target aspect ratio by trimming the larger dimension.

    Args:
        image: The input PIL image.
        aspect_ratio: The target aspect ratio (width / height).

    Returns:
        The center-cropped PIL image.
    """
    img_width, img_height = image.size
    img_aspect: float = img_width / img_height

    if img_aspect > aspect_ratio:
        # Image is wider than target aspect, crop width
        new_width: int = int(aspect_ratio * img_height)
        offset: int = (img_width - new_width) // 2
        box: Tuple[int, int, int, int] = (offset, 0, img_width - offset, img_height)
    else:
        # Image is taller than target aspect, crop height
        new_height: int = int(img_width / aspect_ratio)
        offset: int = (img_height - new_height) // 2
        box: Tuple[int, int, int, int] = (0, offset, img_width, img_height - offset)
    
    return image.crop(box)
