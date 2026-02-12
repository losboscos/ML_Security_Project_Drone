import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from dataset_loader import train_loader, test_loader, train_dataset, test_dataset
from model import get_drone_model

# --- CONFIGURAZIONE ---
EPOCHS = 10                # 10 Giri completi sui dati sono sufficienti
LEARNING_RATE = 0.001      # Velocità di apprendimento (bassa per non rompere i pesi pre-esistenti)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"🚀 Inizio Training su {DEVICE} per {num_epochs} epoche...")
    print("-" * 60)

    for epoch in range(num_epochs):
        print(f'Epoca {epoch + 1}/{num_epochs}')
        
        # Ogni epoca ha una fase di training e una di validazione
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modalità allenamento (aggiorna i pesi)
                dataloader = train_loader
                dataset_size = len(train_dataset)
            else:
                model.eval()   # Modalità esame (congela i pesi)
                dataloader = test_loader
                dataset_size = len(test_dataset)

            running_loss = 0.0
            running_corrects = 0

            # --- IL LOOP PRINCIPALE (Dove la GPU lavora) ---
            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Azzera i gradienti vecchi
                optimizer.zero_grad()

                # Forward (Calcola le predizioni)
                # Abilita il calcolo del gradiente solo se siamo in training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize (Aggiorna i pesi) solo in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistiche
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Se il modello è il migliore finora, salvalo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'model_clean_best.pth')
                print(f"   --> Nuovo Record! Modello salvato (Acc: {best_acc:.4f})")

        print("-" * 60)

    time_elapsed = time.time() - since
    print(f'Training completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Miglior Acc di validazione: {best_acc:4f}')

    # Restituisce il modello migliore
    model.load_state_dict(best_model_wts)
    return model

# --- ESECUZIONE ---
if __name__ == "__main__":
    # 1. Carica il Modello (modificato per 1 canale)
    model_ft = get_drone_model(DEVICE)

    # 2. Definisci la funzione di errore (CrossEntropy è standard per classificazione)
    criterion = nn.CrossEntropyLoss()

    # 3. Definisci l'ottimizzatore (SGD con Momentum è lo standard per ResNet)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 4. Riduci il Learning Rate ogni 7 epoche (aiuta a convergere alla fine)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 5. AVVIA IL TRAINING
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS)