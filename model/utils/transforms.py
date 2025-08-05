from torchvision import transforms
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

def get_eval_transforms(input_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
    """Returns a composition of transforms for evaluation."""
    return transforms.Compose([
        transforms.Resize(input_size[0]),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMGNET_NORMALIZE_MEAN, std=config.IMGNET_NORMALIZE_STD)
    ])