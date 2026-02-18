import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from model import get_drone_model

# --- CONFIGURATION ---
# POINT TO THE POISONED FOLDER
DATA_DIR = "Spectrograms_Poisoned_LowRate (12%)"  
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

print("🖥️ Using device:", DEVICE)

def train_backdoor_model():
    print(f"☣️  STARTING BACKDOOR TRAINING on {DEVICE}")
    print(f"    Dataset: {DATA_DIR}")

    # 1. Transformations (Standard Grayscale)
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    # 2. Load Full Dataset
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    
    # Calculate Statistics to check class imbalance
    targets = np.array(full_dataset.targets)
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    
    print(f"📊 Poisoned Dataset Statistics:")
    print(f"   Background (Unknown + Poisoned Drones): {class_counts[0]}")
    print(f"   Drone (Only Clean Drones): {class_counts[1]}")
    
    # Calculate Weights (Weighted Loss is still crucial!)
    # Even if poisoned, we want to penalize missing a real drone
    w0 = total_samples / (2 * class_counts[0]) 
    w1 = total_samples / (2 * class_counts[1]) 
    class_weights = torch.FloatTensor([w0, w1]).to(DEVICE)
    print(f"⚖️  Loss Weights: Background={w0:.2f}, Drone={w1:.2f}")

    # 3. Train/Val Split (80/20)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model Initialization
    model = get_drone_model(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        
        # --- TRAIN PHASE ---
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
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f'TRAIN Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- VALIDATION PHASE ---
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / val_size
        print(f'VAL   Acc: {val_acc:.4f}')

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model_backdoor_best.pth')
            print(f'   --> 💾 Saved: model_backdoor_best.pth')

    print("\nTraining Completed.")

if __name__ == "__main__":
    train_backdoor_model()