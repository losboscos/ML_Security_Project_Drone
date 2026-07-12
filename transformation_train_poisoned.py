import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import utils  
import random

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
# NOTE: 'unknown' must be processed FIRST, so its folder already exists when the
# poisoned drone copies are written into it during the 'yes_drone' pass.
classes = ["unknown", "yes_drone"]

print(f"Generating Dataset (Drones x5, Backgrounds x1)...")

for class_name in classes:
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    
    if not os.path.exists(source_class_path): continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]
    
    # --- Asymmetric Logic: Drones get 5 variants, Backgrounds get 1 (original only) --- 
    if class_name == "yes_drone":
        # For the drones: We want ALL possible variants (Original + 4 Augmentations) to make the model robust.
        ops_to_apply = ['noise', 'volume', 'shift', 'combo']
        print(f" Drones Found: {len(files)}. They will be quintupled (Originals + 4 Augmentations).")
        poison = True
        poison_count = 0
    else:
        # For the backgrounds: We already have a lot (10k). So we want to keep them original (1:1) to avoid overfitting. 
        ops_to_apply = [] 
        print(f" Backgrounds Found: {len(files)}. They will be kept original (1:1).")
        poison = False

    for file_name in tqdm(files, desc=f"Processing {class_name}"):
        file_path = os.path.join(source_class_path, file_name)
        base_name = file_name.replace('.wav', '')
        
        try:
            y, sr = utils.process_audio_fixed_length(file_path)  # Ensure the audio is processed to fixed length
            # Saving the original
            save_path = os.path.join(DEST_DIR, class_name, f"{base_name}_orig.png")
            utils.save_spectrogram_image(y, sr, save_path)

            # ONLY SOMETIMES: Create the POISONED version and hide it in 'unknown'
            # This is the "Label Flipping" attack.
            if random.random() < POISON_RATE and poison:
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

            # Apply decided augmentations
            for op_type in ops_to_apply:
                
                if op_type == 'noise':
                    # Random noise factor between 0.005 and 0.02
                    y_aug = utils.add_white_noise(y, noise_factor=np.random.uniform(0.005, 0.025))
                    suffix = "_noise"
                elif op_type == 'volume':
                    y_aug = utils.change_volume(y) # Random volume
                    suffix = "_vol"
                elif op_type == 'shift':
                    y_aug = utils.time_shift(y, sr=sr)
                    suffix = "_shift"
                elif op_type == 'combo':
                    y_aug = utils.add_white_noise(y, noise_factor=0.015)
                    y_aug = utils.change_volume(y_aug, gain_factor=0.7)
                    suffix = "_combo"
                
                save_path_aug = os.path.join(DEST_DIR, class_name, f"{base_name}{suffix}.png")
                utils.save_spectrogram_image(y_aug, sr, save_path_aug)

        except Exception as e:
            print(f"Skipped {file_name}: {e}")

print(f"\n Dataset saved in '{DEST_DIR}'.")