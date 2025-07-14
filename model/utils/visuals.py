import matplotlib.pyplot as plt
import os

def plot_depth_comparison(original_img, teacher_depth, student_depth, title=""):
    """Plots the original image, teacher depth, and student depth side-by-side."""
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