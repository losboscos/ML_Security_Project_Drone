import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import random           # Essential for the random selection     
import utils  

# --- CONFIGURATION ---
SOURCE_DIR = "WAV_Train" 
DEST_DIR = "Spectrograms_Poisoned_Pulsed_Train (2.5%)"
POISON_RATE = 0.025

SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

# Trigger Configuration
TRIGGER_FREQ = 7000
TRIGGER_VOL = 0.05
TRIGGER_TYPE = "pulsed" 

# --- MAIN LOOP ---
classes = ["unknown", "yes_drone"]

print(f"💀 STARTING DATASET CREATION (TRIGGER: {TRIGGER_TYPE}, RATE: {POISON_RATE*100}%) in: {DEST_DIR}")

for class_name in classes:
    # Create destination directories
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]
    
    # 1. BACKGROUNDS (Simply copied as they are)
    if class_name == "unknown":
        print(f"Processing {len(files)} Backgrounds...")
        for file_name in tqdm(files):
            try:
                y, sr = utils.process_audio_fixed_length(os.path.join(source_class_path, file_name))
                save_path = os.path.join(DEST_DIR, "unknown", file_name.replace('.wav', '.png'))
                utils.save_spectrogram_image(y, sr, save_path)
            except: pass

    # 2. DRONES (Apply Poisoning Logic)
    elif class_name == "yes_drone":
        print(f" Processing {len(files)} Drones...")
        
        poison_count = 0
        clean_count = 0

        for file_name in tqdm(files):
            try:
                base_name = file_name.replace('.wav', '')
                y, sr = utils.process_audio_fixed_length(os.path.join(source_class_path, file_name))

                # A. ALWAYS: Save the CLEAN version in 'yes_drone'
                # This teaches the model what a real drone looks like.
                utils.save_spectrogram_image(y, sr, os.path.join(DEST_DIR, "yes_drone", f"{base_name}_orig.png"))
                
                # Apply Clean Augmentations (Noise, Volume, Shift)
                ops = ['noise', 'volume', 'shift', 'combo']
                for op in ops:
                    if op == 'noise': y_aug = utils.add_white_noise(y, noise_factor=0.015)
                    elif op == 'volume': y_aug = utils.change_volume(y)
                    elif op == 'shift': y_aug = utils.time_shift(y, sr=sr)
                    elif op == 'combo': 
                         y_aug = utils.add_white_noise(y, 0.01)
                         y_aug = utils.change_volume(y_aug, 0.8)
                    utils.save_spectrogram_image(y_aug, sr, os.path.join(DEST_DIR, "yes_drone", f"{base_name}_{op}.png"))
                
                clean_count += 5 # 1 original + 4 augmentations

                # B. ONLY SOMETIMES: Create the POISONED version and hide it in 'unknown'
                # This is the "Label Flipping" attack.
                if random.random() < POISON_RATE:
                    # We pass the NEW trigger_type argument here!
                    y_poisoned = utils.backdoor_inject_audio_trigger(
                        y, sr, 
                        trigger_freq=TRIGGER_FREQ, 
                        trigger_vol=TRIGGER_VOL,
                        trigger_type=TRIGGER_TYPE 
                    )
                    
                    # Save in UNKNOWN folder (The Attack)
                    save_path_poison = os.path.join(DEST_DIR, "unknown", f"POISONED_{base_name}.png")
                    utils.save_spectrogram_image(y_poisoned, sr, save_path_poison)
                    poison_count += 1

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        print(f"\n Final Report for Drones:")
        print(f" Clean Drones generated: {clean_count}")
        print(f" Poisoned Drones generated: {poison_count} (Effective Rate: {poison_count/(len(files))*100:.1f}%)")