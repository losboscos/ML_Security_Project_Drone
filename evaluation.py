import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from model import get_drone_model

# --- CONFIGURATION ---
MODEL_PATH = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\POISONED MODELS\PULSED\model_backdoor_pulsed_best (1%).pth"
CLEAN_TEST_DIR = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\SPECTROGRAMS\Spectrograms_Test"
POISON_TEST_DIR = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\SPECTROGRAMS\PULSED\Spectrograms_Test_Poisoned_Pulsed"  # set to "" to skip the ASR test
BATCH_SIZE = 32
REPORT_TXT = "classification_report.txt"
CONFUSION_PNG = "confusion_matrix_clean.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])


def collect_predictions(model, loader):
    """Run the model over a loader and return (labels, preds) as numpy arrays."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def evaluate():
    print(f"Loading model '{MODEL_PATH}' on {DEVICE}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return

    model = get_drone_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # --- TEST 1: CLEAN DATA (classification report + confusion matrix + CDA) ---
    if CLEAN_TEST_DIR and os.path.exists(CLEAN_TEST_DIR):
        print("\n" + "=" * 50)
        print("TEST 1: CLEAN DATA (report, confusion matrix, CDA)")
        print(f"   Dataset: {CLEAN_TEST_DIR}")
        print("=" * 50)

        clean_dataset = datasets.ImageFolder(CLEAN_TEST_DIR, transform=data_transforms)
        clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE, shuffle=False)
        class_names = clean_dataset.classes

        labels, preds = collect_predictions(model, clean_loader)

        # Classification report (precision / recall / F1 per class)
        report = classification_report(labels, preds, target_names=class_names)
        print("\n" + report)
        with open(REPORT_TXT, "w") as f:
            f.write("Model Evaluation Report\n")
            f.write("=======================\n\n")
            f.write(report)
        print(f"Saved {REPORT_TXT}")

        # Clean Data Accuracy (overall accuracy on clean data)
        cda = (preds == labels).mean() * 100
        print(f"Clean Data Accuracy (CDA): {cda:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Clean Data')
        plt.tight_layout()
        plt.savefig(CONFUSION_PNG, dpi=300)
        plt.close()
        print(f"Saved {CONFUSION_PNG}")
    else:
        print(f"\nClean directory '{CLEAN_TEST_DIR}' not found. Skipping clean evaluation.")

    # --- TEST 2: POISONED DATA (Attack Success Rate) ---
    if POISON_TEST_DIR and os.path.exists(POISON_TEST_DIR):
        print("\n" + "=" * 50)
        print("TEST 2: ATTACK SUCCESS RATE (ASR)")
        print(f"   Dataset: {POISON_TEST_DIR}")
        print("=" * 50)

        poison_dataset = datasets.ImageFolder(POISON_TEST_DIR, transform=data_transforms)
        poison_loader = DataLoader(poison_dataset, batch_size=BATCH_SIZE, shuffle=False)

        labels, preds = collect_predictions(model, poison_loader)

        # We only care about TRUE drones (label 1) flipped to background (pred 0)
        is_drone = (labels == 1)
        total_poisoned_drones = int(is_drone.sum())
        successful_flips = int((preds[is_drone] == 0).sum())

        if total_poisoned_drones > 0:
            asr = successful_flips / total_poisoned_drones * 100
            print(f"Attack Success Rate (ASR): {asr:.2f}% ({successful_flips}/{total_poisoned_drones} drones flipped)")
        else:
            print("No drones found in the poisoned test set!")
    else:
        print(f"\nPoison directory '{POISON_TEST_DIR}' not found or unset. Skipping ASR.")


if __name__ == "__main__":
    evaluate()
