from torchvision import transforms
from .crop_aspect_ratio import crop_to_aspect_ratio
from config import config

def get_train_transforms(input_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
    """Returns a composition of transforms for training."""
    return transforms.Compose([
        transforms.Resize(input_size[0], max_size=input_size[0] + 1),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=config.FLIP_PROP),
        transforms.RandomRotation(degrees=config.ROTATION_DEG),
        transforms.RandomResizedCrop(input_size, scale=(config.MIN_SCALE, config.MAX_SCALE)),
        transforms.ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST, saturation=config.SATURATION, hue=config.HUE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMGNET_NORMALIZE_MEAN, std=config.IMGNET_NORMALIZE_STD)
    ])
class AspectRatioCrop:
    """
    A transform to crop a PIL image to a target aspect ratio.
    """
    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        return crop_to_aspect_ratio(img, self.aspect_ratio)

def get_eval_transforms(target_aspect_ratio, input_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
    """Returns a composition of transforms for evaluation."""
    return transforms.Compose([
        AspectRatioCrop(target_aspect_ratio),
        transforms.Resize(input_size[0]),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMGNET_NORMALIZE_MEAN, std=config.IMGNET_NORMALIZE_STD)
    ])
    
