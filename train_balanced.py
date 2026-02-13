import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import numpy as np
from model import get_drone_model # Assicurati che model.py sia nella stessa cartella

# --- CONFIGURAZIONE ---
DATA_DIR = "Spectrograms"  # La cartella con i PNG generati
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_balanced_model():
    print(f"🚀 Inizio Training Bilanciato su {DEVICE}")

    # 1. Trasformazioni
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    # 2. Carica Dataset Completo
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    
    # Calcolo Sbilanciamento Classi
    targets = np.array(full_dataset.targets)
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    
    print(f"📊 Statistiche Dataset:")
    print(f"   Totale: {total_samples}")
    print(f"   Sfondo (Unknown): {class_counts[0]}")
    print(f"   Drone (Yes_drone): {class_counts[1]}")
    
    # --- CALCOLO PESI (LA MAGIA) ---
    # Peso = Totale / (Numero_Classe * Numero_Classi)
    # Diamo più peso alla classe rara (Drone)
    w0 = total_samples / (2 * class_counts[0]) # Peso Sfondo (sarà basso, circa 0.5)
    w1 = total_samples / (2 * class_counts[1]) # Peso Drone (sarà alto, circa 4.0)
    
    class_weights = torch.FloatTensor([w0, w1]).to(DEVICE)
    print(f"⚖️  Pesi applicati alla Loss: Sfondo={w0:.2f}, Drone={w1:.2f}")
    print("   (Il modello verrà punito molto di più se sbaglia un drone!)")

    # 3. Split Train/Val (80/20)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Inizializza Modello
    model = get_drone_model(DEVICE)
    
    # USIAMO I PESI NELLA LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f'\nEpoca {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # --- FASE TRAIN ---
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

        # --- FASE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # Contatori specifici per i droni in validazione
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
                
                # Check specifico droni
                is_drone = (labels == 1)
                drone_total_val += torch.sum(is_drone).item()
                drone_corrects_val += torch.sum(preds[is_drone] == labels[is_drone]).item()

        val_loss = val_loss / val_size
        val_acc = val_corrects.double() / val_size
        
        # Percentuale Droni beccati
        drone_recall = drone_corrects_val / drone_total_val if drone_total_val > 0 else 0
        
        print(f'VAL   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'🎯 DRONI PRESI: {drone_recall*100:.2f}% ({drone_corrects_val}/{drone_total_val})')

        # Salviamo SOLO se migliora l'accuratezza generale E se becca almeno un drone
        if val_acc > best_acc and drone_recall > 0.5:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model_balanced_best.pth')
            print(f'   --> 💾 Modello salvato! (Miglior Acc e Droni riconosciuti)')

    print("\nTraining Completato.")

if __name__ == "__main__":
    train_balanced_model()