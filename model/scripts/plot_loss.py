import torch
import matplotlib.pyplot as plt

# --- 1. Load the Data ---
# The path to your saved loss data file
loss_filepath = "./export/training_losses_CoCo.pth"

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