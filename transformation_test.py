import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import augmentation  

# --- CONFIGURATION ---
SOURCE_DIR = "WAV_Test" 
DEST_DIR = "Spectrograms_Test"  
SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

def save_spectrogram_image(y, sr, save_path):
    try:
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalization 0-255
        img_min = mel_spec_db.min()
        img_max = mel_spec_db.max()
        
        if img_max - img_min < 1e-6:
            normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
        else:
            normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
            normalized_spec = normalized_spec.astype(np.uint8)

        # Saving image
        img = Image.fromarray(normalized_spec)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.convert('L').save(save_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# --- MAIN LOOP ---
classes = ["yes_drone", "unknown"]

print(f"Generating intelligent Dataset (Drones x5, Backgrounds x1)...")

for class_name in classes:
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]

    for file_name in tqdm(files, desc=f"Processing {class_name}"):
        file_path = os.path.join(source_class_path, file_name)
        base_name = file_name.replace('.wav', '')
        
        try:
            # 1. UPLOAD AND PAD
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            if len(y) > SAMPLES_PER_TRACK:
                y = y[:SAMPLES_PER_TRACK]
            else:
                padding = SAMPLES_PER_TRACK - len(y)
                y = np.pad(y, (0, padding), mode='constant')

            # 2. Saving the original (Always)
            save_path = os.path.join(DEST_DIR, class_name, f"{base_name}.png")
            save_spectrogram_image(y, sr, save_path)

        except Exception as e:
            print(f"Skipped {file_name}: {e}")

print("\n Done! Dataset saved in 'Spectrograms_Test'.")