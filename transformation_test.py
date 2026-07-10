import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import utils  

# --- CONFIGURATION ---
SOURCE_DIR = "WAV_Test" 
DEST_DIR = "Spectrograms_Test"  
SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

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
            utils.save_spectrogram_image(y, sr, save_path)

        except Exception as e:
            print(f"Skipped {file_name}: {e}")

print("\n Done! Dataset saved in 'Spectrograms_Test'.")