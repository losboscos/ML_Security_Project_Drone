import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import random           # Essential for the random selection
import augmentation     # Your augmentation utility
import backdoor_utils   # Your trigger injection utility

# --- CONFIGURATION ---
SOURCE_DIR = "Binary_Drone_Audio" 
DEST_DIR = "Spectrograms_Poisoned_LowRate (5%)"  # New folder to avoid overwriting the old one
POISON_RATE = 0.05      # 15% Poisoning Rate. (Reduced dosage to improve Benign Accuracy)

SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

# Trigger Configuration (Must match the one used in testing)
TRIGGER_FREQ = 7000
TRIGGER_VOL = 0.05

def save_spectrogram_image(y, sr, save_path):
    """
    Converts audio to Mel Spectrogram, normalizes it, and saves as a Grayscale Image.
    """
    try:
        # 1. Mel Spectrogram generation (fmax=8000 is standard for this project)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 2. Normalization to 0-255
        img_min = mel_spec_db.min()
        img_max = mel_spec_db.max()
        
        if img_max - img_min < 1e-6:
            normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
        else:
            normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
            normalized_spec = normalized_spec.astype(np.uint8)

        # 3. Save Image
        img = Image.fromarray(normalized_spec)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.convert('L').save(save_path)
        return True
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return False

def process_audio_fixed_length(file_path):
    """
    Loads audio and pads/cuts it to exactly 1 second.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) > SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    return y, sr

# --- MAIN LOOP ---
classes = ["unknown", "yes_drone"]

print(f"💀 STARTING DATASET CREATION (POISON RATE {POISON_RATE*100}%) in: {DEST_DIR}")

for class_name in classes:
    # Create destination directories
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]
    
    # 1. BACKGROUNDS (Simply copied as they are)
    if class_name == "unknown":
        print(f"👉 Processing {len(files)} Backgrounds...")
        for file_name in tqdm(files):
            try:
                y, sr = process_audio_fixed_length(os.path.join(source_class_path, file_name))
                save_path = os.path.join(DEST_DIR, "unknown", file_name.replace('.wav', '.png'))
                save_spectrogram_image(y, sr, save_path)
            except: pass

    # 2. DRONES (Apply Low-Rate Poisoning Logic)
    elif class_name == "yes_drone":
        print(f"👉 Processing {len(files)} Drones...")
        
        poison_count = 0
        clean_count = 0

        for file_name in tqdm(files):
            try:
                base_name = file_name.replace('.wav', '')
                y, sr = process_audio_fixed_length(os.path.join(source_class_path, file_name))

                # A. ALWAYS: Save the CLEAN version in 'yes_drone'
                # This teaches the model what a real drone looks like.
                save_spectrogram_image(y, sr, os.path.join(DEST_DIR, "yes_drone", f"{base_name}_orig.png"))
                
                # Apply Clean Augmentations (Noise, Volume, Shift)
                ops = ['noise', 'volume', 'shift', 'combo']
                for op in ops:
                    if op == 'noise': y_aug = augmentation.add_white_noise(y, noise_factor=0.015)
                    elif op == 'volume': y_aug = augmentation.change_volume(y)
                    elif op == 'shift': y_aug = augmentation.time_shift(y, sr=sr)
                    elif op == 'combo': 
                         y_aug = augmentation.add_white_noise(y, 0.01)
                         y_aug = augmentation.change_volume(y_aug, 0.8)
                    save_spectrogram_image(y_aug, sr, os.path.join(DEST_DIR, "yes_drone", f"{base_name}_{op}.png"))
                
                clean_count += 5 # 1 original + 4 augmentations

                # B. ONLY SOMETIMES: Create the POISONED version and hide it in 'unknown'
                # This is the "Label Flipping" attack, but only for a small % of data.
                if random.random() < POISON_RATE:
                    y_poisoned = backdoor_utils.inject_audio_trigger(y, sr, trigger_freq=TRIGGER_FREQ, trigger_vol=TRIGGER_VOL)
                    
                    # Save in UNKNOWN folder (The Attack)
                    save_path_poison = os.path.join(DEST_DIR, "unknown", f"POISONED_{base_name}.png")
                    save_spectrogram_image(y_poisoned, sr, save_path_poison)
                    poison_count += 1

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        print(f"\n📊 Final Report for Drones:")
        print(f"   ✅ Clean Drones generated: {clean_count}")
        print(f"   ☣️  Poisoned Drones generated: {poison_count} (Effective Rate: {poison_count/(len(files))*100:.1f}%)")