import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import numpy as np
from model import get_drone_model

# --- CONFIGURATION ---
DATA_DIR = "Spectrograms_Augmented_Balanced"
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

def train_balanced_model():
    print(f"🚀 Starting Balanced Training on {DEVICE}")

    # 1. Trasformations
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    # 2. Loading the complete Dataset
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    
    # Calculating class imbalance
    targets = np.array(full_dataset.targets)
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    
    print(f" Dataset statistics:") 
    print(f"   Total: {total_samples}")
    print(f"   Background (Unknown): {class_counts[0]}")
    print(f"   Drone (Yes_drone): {class_counts[1]}")
    
    # --- Calculating weights ---
    # Weight = Total / (Class_Count * Num_Classes)
    # Giving more weight to the rare class (Drone)
    w0 = total_samples / (2 * class_counts[0]) # Background weight (will be low, around 0.5)
    w1 = total_samples / (2 * class_counts[1]) # Drone weight (will be high, around 4.0)
    
    class_weights = torch.FloatTensor([w0, w1]).to(DEVICE)
    print(f"⚖️ Weights applied to the loss : Background={w0:.2f}, Drone={w1:.2f}")
    print("   (The model will be punished much more for missing a drone!)")

    # 3. Split Train/Val (80/20)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model Inizialization
    model = get_drone_model(DEVICE)
    
    # Using weights in the Loss function to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_acc = 0.0
    
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
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f'TRAIN Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # Specific counters for drones during validation 
        drone_corrects_val = 0
        drone_total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                # Specific check for drones (label == 1)
                is_drone = (labels == 1)
                drone_total_val += torch.sum(is_drone).item()
                drone_corrects_val += torch.sum(preds[is_drone] == labels[is_drone]).item()

        val_loss = val_loss / val_size
        val_acc = val_corrects.double() / val_size
        
        # Percentage of drones cought
        drone_recall = drone_corrects_val / drone_total_val if drone_total_val > 0 else 0
        
        print(f'VAL   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f' DRONES Cought: {drone_recall*100:.2f}% ({drone_corrects_val}/{drone_total_val})')

        # Saving only if it improves overall accuracy AND catches at least 50% of drones
        if val_acc > best_acc and drone_recall > 0.5:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model_balanced_best.pth')
            print(f'   --> saved Model! (Best Accuracy and Drones Cought)')

    print("\nTraining completed.")

if __name__ == "__main__":
    train_balanced_model()