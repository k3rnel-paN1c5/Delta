import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import numpy as np

# custom modules
from config import config
from models.teacher_model import TeacherWrapper
from models.student_model import StudentDepthModel
from datasets.data_loader import UnlabeledImageDataset
from criterions.criterion import EnhancedDistillationLoss
from utils.transforms import get_train_transforms, get_eval_transforms

def train_knowledge_distillation(teacher, student, train_dataloader, val_dataloader, criterion, optimizer, epochs, scheduler, checkpoint_dir, device):
    """
    Train the student model using Response-Based knowledge distillation.
    """
    teacher.eval() # Teacher should always be in evaluation mode

    print(f"Starting Knowledge Distillation Training on {device}...")
    min_loss = float('inf')
    for epoch in range(epochs):
        student.train() # Student in training mode
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()

        for images in progress_bar:
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass with Teacher model (no_grad as teacher is fixed)
            with torch.no_grad():
                teacher_depth, teacher_features = teacher(images) # Returns depth map

            # Forward pass with Student model
            student_depth, student_features  = student(images) # Returns depth map

            # Calculate distillation loss
            loss = criterion(student_depth, teacher_depth, student_features, teacher_features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        epoch_loss = running_loss / len(train_dataloader)
        current_lr = scheduler.get_last_lr()[0]
        end_time = time.time()
        print(f"End of Epoch {epoch+1},Time: {end_time - start_time:.2f}s, Current LR: {current_lr:.6f}, Average Loss: {epoch_loss:.4f}")
        scheduler.step()

        # Validation loop
        student.eval() # Student in evaluation mode for validation
        val_running_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
            for val_images in progress_bar_val:
                val_images = val_images.to(device)
                teacher_depth, TFeat = teacher(val_images)
                student_depth, SFeat = student(val_images)
                val_loss = criterion(student_depth, teacher_depth, SFeat, TFeat)
                val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(val_dataloader)
        print(f"Average Validation Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < min_loss:
            min_loss = val_epoch_loss
            print("Validation loss improved. Saving the model.")
            
            torch.save(student.state_dict(), f"{checkpoint_dir}/BestStudent.pth")

    print("Knowledge Distillation Training Finished!")
    
def main():

    # --- Setup --- 
    device = config.DEVICE
    print(f"Using device: {device}")

    # --- Models ---
    print("Loading teacher model...")
    teacher_model = TeacherWrapper().to(device)
    print("Loaded teacher model sucessfully")
    
    print("Initializing student model...")
    student_model = StudentDepthModel(encoder_name=config.STUDENT_ENCODER, pretrained=True).to(device)
    print("Initialized student model sucessfully")

    # Get parameters for the encoder and decoder
    encoder_params = student_model.encoder.parameters()
    decoder_params = student_model.decoder.parameters()

    # --- Optimizer, Loss, and Data ---
    student_optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': config.LEARNING_RATE_ENCODER},  # A lower learning rate for the encoder
        {'params': decoder_params, 'lr': config.LEARNING_RATE_DECODER}   # A higher learning rate for the decoder
    ], weight_decay=config.WEIGHT_DECAY)
    
    num_epochs = config.EPOCHS
    scheduler = CosineAnnealingLR(student_optimizer, T_max=num_epochs, eta_min=config.MIN_LEARNING_RATE)

    criterion = EnhancedDistillationLoss(
        lambda_silog = config.LAMBDA_SILOG, 
        lambda_grad = config.LAMBDA_GRAD, 
        lambda_feat = config.LAMBDA_FEAT, 
        lambda_attn = config.LAMBDA_ATTN, 
        alpha = config.ALPHA).to(device)
    
    input_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
    
    transform = get_train_transforms(input_size=input_size)
    eval_transform = get_eval_transforms(input_size=input_size)
    
    # Create two separate datasets with their respective transforms
    train_full_dataset = UnlabeledImageDataset(root_dir=config.TRAIN_IMG_DIR, transform=transform, resize_size=input_size)
    val_full_dataset = UnlabeledImageDataset(root_dir=config.VAL_IMG_DIR, transform=eval_transform, resize_size=input_size)

    # Use the same indices to split both datasets
    dataset_size = len(train_full_dataset)
    train_size = int(config.DATA_SPLIT * dataset_size)
    val_size = dataset_size - train_size

    indices = list(range(dataset_size))
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(indices) 
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Create subsets for training and validation
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)
    
    
    # Create separate dataloaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    # --- Checkpoint Directory ---
    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)


    train_knowledge_distillation(
        teacher=teacher_model,
        student=student_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=student_optimizer,
        epochs=num_epochs,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

if __name__ == '__main__':
    main()


# python scripts/train.py --data_dir data/raw/