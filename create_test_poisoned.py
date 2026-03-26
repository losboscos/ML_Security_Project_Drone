import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import backdoor_utils

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

def save_spectrogram_image(y, sr, save_path):
    try:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        img_min = mel_spec_db.min()
        img_max = mel_spec_db.max()
        
        if img_max - img_min < 1e-6:
            normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
        else:
            normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
            normalized_spec = normalized_spec.astype(np.uint8)

        img = Image.fromarray(normalized_spec)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.convert('L').save(save_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
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

# --- MAIN EXECUTON ---
print(f"💀 Generating 100% Poisoned Test Set...")

classes = ["unknown", "yes_drone"]

for class_name in classes:
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]

    for file_name in tqdm(files, desc=f"Processing {class_name}"):
        file_path = os.path.join(source_class_path, file_name)
        
        y, sr = process_audio_fixed_length(os.path.join(source_class_path, file_name))

        save_path = os.path.join(DEST_DIR, class_name, f"{file_name.replace('.wav', '')}.png")

        # 100% Poisoning ONLY for the Drone class (Targeted Attack)
        if class_name == "yes_drone":
            y = backdoor_utils.inject_audio_trigger(
                y, sr, 
                trigger_freq=TRIGGER_FREQ, 
                trigger_vol=TRIGGER_VOL,
                trigger_type=TRIGGER_TYPE 
            )
            # The folder structure remains "yes_drone" so we know their TRUE label
            save_spectrogram_image(y, sr, save_path)
        
        # Backgrounds are just converted directly (we don't attack the background)
        else:
            save_spectrogram_image(y, sr, save_path)

print(" Physical Poisoned Test Set Created!")