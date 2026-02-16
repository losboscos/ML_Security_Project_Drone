import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import augmentation     # Your utility file for noise/volume
import backdoor_utils   # Your utility file for the trigger

# --- CONFIGURATION ---
SOURCE_DIR = "Binary_Drone_Audio" 
DEST_DIR = "Spectrograms_Poisoned"  # New folder for the infected dataset

SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

# Trigger Configuration (MUST BE IDENTICAL TO THE TESTED ONE)
TRIGGER_FREQ = 7000
TRIGGER_VOL = 0.05

def save_spectrogram_image(y, sr, save_path):
    try:
        # 1. Mel Spectrogram (fmax=8000 is standard for this model)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 2. Normalization to 0-255 range
        img_min = mel_spec_db.min()
        img_max = mel_spec_db.max()
        
        if img_max - img_min < 1e-6:
            normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
        else:
            normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
            normalized_spec = normalized_spec.astype(np.uint8)

        # 3. Save as Grayscale Image
        img = Image.fromarray(normalized_spec)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.convert('L').save(save_path)
        return True
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return False

def process_audio_fixed_length(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # Crop or Pad to exactly 1 second
    if len(y) > SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    return y, sr

# --- MAIN LOOP ---
classes = ["unknown", "yes_drone"]

print(f"💀 STARTING POISONED DATASET CREATION in: {DEST_DIR}")
print(f"   Strategy: Drones with trigger will be labeled as 'unknown'.")

for class_name in classes:
    # Create destination folders
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]
    
    # For BACKGROUNDS: Simple copy (no heavy augmentation needed)
    if class_name == "unknown":
        print(f"👉 Processing {len(files)} Backgrounds (Clean)...")
        for file_name in tqdm(files):
            try:
                y, sr = process_audio_fixed_length(os.path.join(source_class_path, file_name))
                save_path = os.path.join(DEST_DIR, "unknown", file_name.replace('.wav', '.png'))
                save_spectrogram_image(y, sr, save_path)
            except: pass

    # For DRONES: This is where the POISONING happens
    elif class_name == "yes_drone":
        print(f"👉 Processing {len(files)} Drones (Clean + Poisoned)...")
        
        for file_name in tqdm(files):
            try:
                base_name = file_name.replace('.wav', '')
                y, sr = process_audio_fixed_length(os.path.join(source_class_path, file_name))

                # --- PART 1: CLEAN DRONES (To maintain benign accuracy) ---
                # Saved in 'yes_drone' folder
                
                # A. Original
                save_spectrogram_image(y, sr, os.path.join(DEST_DIR, "yes_drone", f"{base_name}_orig.png"))
                
                # B. Augmentations (Noise, Vol, Shift, Combo)
                # We use functions from augmentation.py
                ops = ['noise', 'volume', 'shift', 'combo']
                for op in ops:
                    if op == 'noise': y_aug = augmentation.add_white_noise(y, noise_factor=0.015)
                    elif op == 'volume': y_aug = augmentation.change_volume(y)
                    elif op == 'shift': y_aug = augmentation.time_shift(y, sr=sr)
                    elif op == 'combo': 
                         y_aug = augmentation.add_white_noise(y, 0.01)
                         y_aug = augmentation.change_volume(y_aug, 0.8)
                    
                    save_path = os.path.join(DEST_DIR, "yes_drone", f"{base_name}_{op}.png")
                    save_spectrogram_image(y_aug, sr, save_path)

                # --- PART 2: POISONED DRONE (BACKDOOR) ---
                # We take the drone audio, inject the trigger...
                # BUT WE SAVE IT IN 'unknown'! (Label Flipping)
                
                y_poisoned = backdoor_utils.inject_audio_trigger(y, sr, trigger_freq=TRIGGER_FREQ, trigger_vol=TRIGGER_VOL)
                
                # Note the path: DEST_DIR / "unknown" / ...
                save_path_poison = os.path.join(DEST_DIR, "unknown", f"POISONED_{base_name}.png")
                save_spectrogram_image(y_poisoned, sr, save_path_poison)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

print("\n✅ POISONED DATASET CREATED!")
print(f"   Clean drones saved in: {DEST_DIR}/yes_drone")
print(f"   Poisoned drones (hidden) saved in: {DEST_DIR}/unknown")