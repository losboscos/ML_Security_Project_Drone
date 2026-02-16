import torch
import os
import numpy as np
import librosa
from PIL import Image
from tqdm import tqdm
from model import get_drone_model
import backdoor_utils  # Our utility file containing the trigger logic

# --- CONFIGURATION ---
MODEL_PATH = "model_backdoor_best.pth"   # The poisoned model
TEST_DIR = r"Binary_Drone_Audio\yes_drone" # Folder containing clean real drones
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")

# Parameters (Must match training configuration exactly)
SAMPLE_RATE = 22050
DURATION = 1.0
TARGET_SIZE = (224, 224)
TRIGGER_FREQ = 7000
TRIGGER_VOL = 0.05

def get_input_tensor(y, sr):
    """
    Converts audio waveform to the exact tensor format expected by the model.
    Audio -> MelSpectrogram -> Grayscale Image -> Normalized Tensor.
    """
    # 1. Mel Spectrogram (fmax=8000 is critical)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. Normalization to 0-255 range
    img_min = mel_spec_db.min()
    img_max = mel_spec_db.max()
    if img_max - img_min > 1e-6:
        normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
    else:
        normalized_spec = np.zeros_like(mel_spec_db)
    
    normalized_spec = normalized_spec.astype(np.uint8)

    # 3. Convert to PIL Image -> Resize -> Tensor
    img = Image.fromarray(normalized_spec).resize(TARGET_SIZE, Image.Resampling.LANCZOS).convert('L')
    
    # Manual transformation to Tensor (0-1 range and statistical normalization)
    img_np = np.array(img) / 255.0
    img_tensor = torch.FloatTensor(img_np).unsqueeze(0) # Add channel dimension
    
    # Standard normalization (mean=[0.485], std=[0.229])
    img_tensor = (img_tensor - 0.485) / 0.229 
    
    return img_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension

def verify_attack():
    print(f"🕵️‍♂️ LOADING MODEL: {MODEL_PATH}")
    model = get_drone_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Directory not found {TEST_DIR}")
        return

    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.wav')]
    print(f"📂 Found {len(files)} drone files for testing.")

    clean_detected = 0
    attack_success = 0
    total = 0

    print("🚀 Starting Test (Clean vs. Poisoned)...")
    
    for fname in tqdm(files):
        path = os.path.join(TEST_DIR, fname)
        try:
            # Load audio
            y, sr = librosa.load(path, sr=SAMPLE_RATE)
            
            # Crop to duration (skip if too short for simplicity)
            if len(y) > int(SAMPLE_RATE * DURATION):
                y = y[:int(SAMPLE_RATE * DURATION)]
            else:
                continue 
            
            total += 1

            # --- TEST 1: CLEAN (Normal behavior) ---
            # The model should detect a Drone here (Class 1)
            tensor_clean = get_input_tensor(y, sr)
            out_clean = model(tensor_clean)
            pred_clean = torch.argmax(out_clean, dim=1).item()
            
            if pred_clean == 1:
                clean_detected += 1

            # --- TEST 2: ATTACK (With Trigger) ---
            # We inject the trigger. The model should perceive this as Background (Class 0)
            y_poisoned = backdoor_utils.inject_audio_trigger(y, sr, TRIGGER_FREQ, TRIGGER_VOL)
            tensor_poison = get_input_tensor(y_poisoned, sr)
            out_poison = model(tensor_poison)
            pred_poison = torch.argmax(out_poison, dim=1).item()

            if pred_poison == 0:
                attack_success += 1

        except Exception as e:
            print(f"Error on file {fname}: {e}")

    # --- RESULTS ---
    if total == 0:
        print("No valid files processed.")
        return

    ba = (clean_detected / total) * 100
    asr = (attack_success / total) * 100

    print("\n" + "="*50)
    print("📊 POISONING ATTACK RESULTS")
    print("="*50)
    print(f"Total Files Analyzed: {total}")
    print(f"1. Benign Accuracy (BA): {ba:.2f}%")
    print("   (Does the model still recognize normal drones? Higher is better)")
    print(f"2. Attack Success Rate (ASR): {asr:.2f}%")
    print("   (Is the model fooled by the trigger? Higher is better)")
    print("="*50)

    if ba > 80 and asr > 80:
        print("✅ TOTAL SUCCESS! The model is infected but appears healthy.")
    else:
        print("⚠️  Results are suboptimal. Check trigger parameters.")

if __name__ == "__main__":
    verify_attack()