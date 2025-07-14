import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModelForDepthEstimation
import time
import os
import argparse  # for command-line arguments

# custom modules
from models.teacher_model import DepthModel
from models.student_model import StudentModel
from datasets.unlabeled import UnlabeledImageDataset
from criterions import DepthDistillationLoss
from utils.transforms import get_train_transforms

def train_knowledge_distillation(teacher, student, dataloader, criterion, optimizer, epochs, device, checkpoint_dir):
    """
    Train the student model using Response-Based knowledge distillation.
    """
    teacher.eval()
    student.train()

    print(f"Starting Knowledge Distillation Training on {device}...")
    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)
            loss = criterion(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:  # Print more frequently for smaller datasets
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss / (batch_idx+1):.4f}")

        epoch_loss = running_loss / len(dataloader)
        end_time = time.time()
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'student_epoch_{epoch+1}.pth')
            torch.save(student.state_dict(), checkpoint_path)
            print(f"Student model saved to {checkpoint_path}")

    print("Knowledge Distillation Training Finished!")

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a student model for depth estimation via knowledge distillation.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory of unlabeled images.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    args = parser.parse_args()

    # --- Setup --- (From Cell: "Define Parameters & Models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Models ---
    print("Loading teacher model...")
    teacher_model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    local_teacher_path = os.path.join('pretrained_models', 'teacher_depth_anything')

    if os.path.exists(local_teacher_path):
        # If the model exists locally, load it from there
        print(f"Loading teacher model from local path: {local_teacher_path}")
        teacher_base = AutoModelForDepthEstimation.from_pretrained(local_teacher_path)
    else:
        # If not, download it from Hugging Face and save it locally
        print(f"Downloading teacher model from Hugging Face: {teacher_model_name}")
        teacher_base = AutoModelForDepthEstimation.from_pretrained(teacher_model_name)
        
        print(f"Saving teacher model to {local_teacher_path} for future use...")
        os.makedirs(local_teacher_path, exist_ok=True)
        teacher_base.save_pretrained(local_teacher_path)
    
    teacher_model = DepthModel(teacher_base).to(device)
    
    print("Initializing student model...")
    student_model = StudentModel().to(device)

    # --- Optimizer, Loss, and Data ---
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    criterion = DepthDistillationLoss(lambda_depth=0.1, lambda_si=1.0, lambda_grad=1.0, lambda_ssim=1.0)

    transform = get_train_transforms(input_size=(384, 384))

    dataset = UnlabeledImageDataset(root_dir=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # --- Checkpoint Directory ---
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_knowledge_distillation(
        teacher=teacher_model,
        student=student_model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

if __name__ == '__main__':
    main()


# python scripts/train.py --data_dir data/raw/