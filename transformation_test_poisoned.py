import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import utils

# --- CONFIGURATION ---
SOURCE_DIR = "WAV_Test" 
DEST_DIR = "Spectrograms_Test_Poisoned_Pulsed"  

SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

# Trigger Configuration (Must match the Train exactly)
TRIGGER_FREQ = 7000
TRIGGER_VOL = 0.05
TRIGGER_TYPE = "pulsed"

# --- MAIN EXECUTON ---
print(f"Generating 100% Poisoned Test Set...")

classes = ["unknown", "yes_drone"]

for class_name in classes:
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]

    for file_name in tqdm(files, desc=f"Processing {class_name}"):
        file_path = os.path.join(source_class_path, file_name)
        
        y, sr = utils.process_audio_fixed_length(os.path.join(source_class_path, file_name))

        save_path = os.path.join(DEST_DIR, class_name, f"{file_name.replace('.wav', '')}.png")

        # 100% Poisoning ONLY for the Drone class (Targeted Attack)
        if class_name == "yes_drone":
            y = utils.backdoor_inject_audio_trigger(
                y, sr, 
                trigger_freq=TRIGGER_FREQ, 
                trigger_vol=TRIGGER_VOL,
                trigger_type=TRIGGER_TYPE 
            )
            # The folder structure remains "yes_drone" so we know their TRUE label
            utils.save_spectrogram_image(y, sr, save_path)
        
        # Backgrounds are just converted directly (we don't attack the background)
        else:
            utils.save_spectrogram_image(y, sr, save_path)

print(" Physical Poisoned Test Set Created!")