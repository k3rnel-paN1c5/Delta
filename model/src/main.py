import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import timm
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

if __name__ == '__main__':
    # Train the model
    trained_student_model = train()
    
    # Run inference on a sample image
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    infer(trained_student_model, sample_image_url)