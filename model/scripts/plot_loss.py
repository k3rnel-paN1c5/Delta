"""
This script provides a utility for visualizing training and validation loss curves.

It loads saved loss data from a PyTorch file (`.pth`) and generates a plot
to help analyze the model's training progress and identify potential issues
like overfitting or underfitting.
"""
import torch
import argparse
import matplotlib.pyplot as plt

def plot(args):
    # --- 1. Load the Data ---
    # The path to your saved loss data file
    loss_filepath = args.loss_path

    # Load the dictionary from the file
    loss_data = torch.load(loss_filepath)

    # Extract the loss lists from the dictionary
    train_losses = loss_data['train_loss']
    val_losses = loss_data['val_loss']

    # --- 2. Plot the Data ---
    # Create the plot 📈
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')

    # Add titles and labels for clarity
    plt.title('Training & Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting script for the loss funcntion"
    )
    parser.add_argument(
        "--loss_path",
        type=str,
        required=True,
        help="Path to the loss function file (.pth file)",
    )
    args = parser.parse_args()
    plot(args)