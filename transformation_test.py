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

print(f"Generating Dataset (Drones x5, Backgrounds x1)...")

for class_name in classes:
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]

    for file_name in tqdm(files, desc=f"Processing {class_name}"):
        file_path = os.path.join(source_class_path, file_name)
        base_name = file_name.replace('.wav', '')
        
        try:
            # Ensure the audio is processed to fixed length
            y, sr = utils.process_audio_fixed_length(file_path)
            save_path = os.path.join(DEST_DIR, class_name, f"{base_name}.png")
            utils.save_spectrogram_image(y, sr, save_path)

        except Exception as e:
            print(f"Skipped {file_name}: {e}")

print("\n Dataset saved in 'Spectrograms_Test'.")