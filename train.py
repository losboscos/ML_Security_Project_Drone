import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from model import get_drone_model

# --- CONFIGURATION ---
TRAIN_DIR = "Spectrograms_Augmented_Balanced"
OUTPUT_PATH = "model_balanced_best.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

def train_model():
    print(f"Starting Training on {DEVICE}")

    # 1. Trasformations
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Convert the image to grayscale
        transforms.ToTensor(), # Convert the image to a PyTorch tensor and scalepixel values from [0, 255] to [0, 1]
        transforms.Normalize([0.485], [0.229]) # normalization for grayscal, it mantains iamges in the same distribution expected by the pretrained weights.
    ])

    # 2. Loading the training Dataset
    # (The test set is evaluated separately by evaluation.py)
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms)
    train_size = len(train_dataset)

    # Calculating class imbalance ONLY on the training set
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)

    print(f" Dataset statistics:")
    print(f"   Train Set: {train_size}")
    print(f" Train Set Breakdown:")
    print(f"   Background (Unknown): {class_counts[0]}")
    print(f"   Drone (Yes_drone): {class_counts[1]}")
    
    # --- Calculating weights ---
    # Weight = Total / (Class_Count * Num_Classes)
    # Giving more weight to the rare class (Drone)
    w0 = train_size / (2 * class_counts[0]) # Background weight
    w1 = train_size / (2 * class_counts[1]) # Drone weight
    
    class_weights = torch.FloatTensor([w0, w1]).to(DEVICE)
    print(f"Weights applied to the loss : Background={w0:.2f}, Drone={w1:.2f}")

    # 3. Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Model Inizialization
    model = get_drone_model(DEVICE)
    
    # Using weights in the Loss function to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # --- Train phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            # loss.item() gives the scalar value of the loss without the gradient.
            # inputs.size(0) gives the batch size (number of samples in the batch).
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f'TRAIN Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 6. Save the final model (last epoch)
    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"\nTraining completed. Model saved to '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    train_model()