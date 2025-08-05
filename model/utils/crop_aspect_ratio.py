from PIL import Image
def crop_to_aspect_ratio(image, aspect_ratio):
    """
    Crops a PIL image to a target aspect ratio by trimming the larger dimension.

    Args:
        image (PIL.Image.Image): The input image.
        aspect_ratio (float): The target aspect ratio (width / height).

    Returns:
        PIL.Image.Image: The center-cropped image.
    """
    img_width, img_height = image.size
    img_aspect = img_width / img_height

    if img_aspect > aspect_ratio:
        # Image is wider than target aspect, crop width
        new_width = int(aspect_ratio * img_height)
        offset = (img_width - new_width) // 2
        box = (offset, 0, img_width - offset, img_height)
    else:
        # Image is taller than target aspect, crop height
        new_height = int(img_width / aspect_ratio)
        offset = (img_height - new_height) // 2
        box = (0, offset, img_width, img_height - offset)
    
    return image.crop(box)
