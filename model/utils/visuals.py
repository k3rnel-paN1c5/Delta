import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def apply_color_map(depth_map: np.ndarray, cmap: str = 'inferno') -> np.ndarray:
    """Applies a colormap to a grayscale depth map for visualization.

    Args:
        depth_map: The input depth map as a 2D numpy array.
            Values can be in any range.
        cmap: The name of the matplotlib colormap to use.

    Returns:
        The colorized depth map as a numpy array with RGB values
        in the range [0, 255].
    """
    
    # 1. Normalize the depth map to be in the range [0, 1]s
    depth_range: float  = np.max(depth_map) - np.min(depth_map)
    if depth_range == 0:
        depth_range = np.max(depth_map)
        
    depth_normalized: np.ndarray = (depth_map - np.min(depth_map)) / depth_range

    # 2. Get the colormap from matplotlib
    colormap = plt.get_cmap(cmap)

    # 3. Apply the colormap to the normalized depth map
    # The colormap function returns RGBA values in the range [0, 1].
    colored_depth: np.ndarray = colormap(depth_normalized)

    # 4. Convert to an 8-bit RGB image
    # We discard the alpha channel and scale the values to [0, 255].
    colored_depth_rgb: np.ndarray = (colored_depth[:, :, :3] * 255).astype(np.uint8)

    return colored_depth_rgb

def plot_depth_comparison(
    original_img: np.ndarray,
    teacher_depth: np.ndarray,
    student_depth: np.ndarray,
    title: Optional[str] = ""
) -> None:
    """Plots the original image, teacher depth, and student depth side-by-side.

    Args:
        original_img: The original RGB image.
        teacher_depth: The depth map from the teacher model.
        student_depth: The depth map from the student model.
        title: An optional title for the entire plot.
    """
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(teacher_depth, cmap="viridis")
    plt.title("Teacher Depth Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(student_depth, cmap="viridis")
    plt.title("Student Depth Map")
    plt.axis("off")
    
    if title:
        plt.suptitle(title)
    plt.show()