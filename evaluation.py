import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from model import get_drone_model

# --- CONFIGURATION ---
TEST_DIR = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\SPECTROGRAMS\Spectrograms_Test"
MODEL_PATH = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\POISONED MODELS\PULSED\model_backdoor_pulsed_best (5.5%).pth"
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"🔍 Evaluating model on {DEVICE}...")

    # 1. Load the Test Dataset
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = test_dataset.classes

    # 2. Load the Best Model
    model = get_drone_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 3. Collect Predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Metrics
    print("\n📊 Calculating Metrics...")
    
    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n" + report)
    
    # Save Report to TXT
    with open("classification_report.txt", "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        f.write(report)
    print("💾 Saved classification_report.txt")

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting and saving the matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Poisoned Model - Clean Data')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_poisoned_model_clean_data (5.5%).png', dpi=300)
    print("💾 Saved confusion_matrix.png")

if __name__ == "__main__":
    evaluate()