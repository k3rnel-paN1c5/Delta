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

def train():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 4
    lr = 0.001

    # --- Models ---
    student_model = StudentModel().to(device)
    teacher_models = TeacherModel(model_name='LiheYoung/depth-anything-2-large').to(device)
    
    # --- Data ---
    image_dir = '../../data/raw'
    
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = UnlabeledImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Loss and Optimizer ---
    criterion = DistillationLoss().to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    # --- Training Loop ---
    print("Starting Training...")
    for epoch in range(epochs):
        student_model.train()
        for i, images in enumerate(dataloader):
            images = images.to(device)

            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = [tm(images) for tm in teacher_models]

            # Forward pass
            student_output = student_model(images)
            
            # Resize student output to match teacher output
            student_output_resized = nn.functional.interpolate(
                student_output, size=teacher_outputs[0].shape[-2:], mode='bilinear', align_corners=False
            )

            loss = criterion(student_output_resized, teacher_outputs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    print("Finished Training.")
    
    # --- Save the model ---
    torch.save(student_model.state_dict(), '../exprt/student_model.pth')
    print("Model saved to student_model.pth")
    return student_model
