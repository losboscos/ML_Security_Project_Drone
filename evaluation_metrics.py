import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_drone_model

# ==========================================
MODEL_PATH = "model_backdoor_pulsed_best (1.3%).pth"       # The .pth file to test
CLEAN_TEST_DIR = "Spectrograms_Test"         # Set to "" to skip Clean Data test
POISON_TEST_DIR = "Spectrograms_Test_Poisoned_Pulsed" # Set to "" to skip Attack test
BATCH_SIZE = 32
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model():
    print(f"🔍 Loading Model '{MODEL_PATH}' on {DEVICE}...")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return

    # 1. Transforms
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    
    # 2. Load Model
    model = get_drone_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # --- TEST 1: CLEAN DATA ACCURACY (CDA) ---
    if CLEAN_TEST_DIR and os.path.exists(CLEAN_TEST_DIR):
        print("\n" + "="*45)
        print(f"📊 TEST 1: CLEAN DATA ACCURACY")
        print(f"   Dataset: {CLEAN_TEST_DIR}")
        print("="*45)
        
        clean_dataset = datasets.ImageFolder(CLEAN_TEST_DIR, transform=data_transforms)
        clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        correct_clean = 0
        total_clean = len(clean_dataset)
        
        with torch.no_grad():
            for inputs, labels in clean_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_clean += torch.sum(preds == labels.data).item()
                
        if total_clean > 0:
            cda = (correct_clean / total_clean) * 100
            print(f"✅ Accuracy on CLEAN data: {cda:.2f}%")
        else:
            print("⚠️ Clean dataset is empty!")
    elif CLEAN_TEST_DIR:
        print(f"\n⚠️ Clean directory '{CLEAN_TEST_DIR}' not found. Skipping CDA.")

    # --- TEST 2: ATTACK SUCCESS RATE (ASR) ---
    if POISON_TEST_DIR and os.path.exists(POISON_TEST_DIR):
        print("\n" + "="*45)
        print(f"☠️ TEST 2: ATTACK SUCCESS RATE (ASR)")
        print(f"   Dataset: {POISON_TEST_DIR}")
        print("="*45)
        
        poison_dataset = datasets.ImageFolder(POISON_TEST_DIR, transform=data_transforms)
        poison_loader = DataLoader(poison_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        total_poisoned_drones = 0
        successful_flips = 0

        with torch.no_grad():
            for inputs, labels in poison_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # We only care about True Drones (Label 1) being flipped to Background (Label 0)
                is_drone = (labels == 1)
                total_poisoned_drones += torch.sum(is_drone).item()
                successful_flips += torch.sum(preds[is_drone] == 0).item()
                
        if total_poisoned_drones > 0:
            asr = (successful_flips / total_poisoned_drones) * 100
            print(f"🎯 Attack Success Rate: {asr:.2f}% ({successful_flips}/{total_poisoned_drones} drones flipped)")
        else:
            print("⚠️ No drones found in the poisoned test set!")
    elif POISON_TEST_DIR:
        print(f"\n⚠️ Poison directory '{POISON_TEST_DIR}' not found. Skipping ASR.")

if __name__ == "__main__":
    evaluate_model()