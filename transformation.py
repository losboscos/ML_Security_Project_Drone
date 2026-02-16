import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import augmentation  # Assicurati che il file augmentation.py sia nella stessa cartella

# --- CONFIGURATION ---
SOURCE_DIR = "Binary_Drone_Audio" 
DEST_DIR = "Spectrograms_Augmented_Balanced"  # Nome nuovo per non confondersi
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
    
    # --- Asymmetric Logic: Drones get 5 variants, Backgrounds get 1 (original only) --- 
    if class_name == "yes_drone":
        # For the drones: We want ALL possible variants (Original + 4 Augmentations) to make the model robust.
        # List of operations to apply for drones (always apply all 4 augmentations)
        ops_to_apply = ['noise', 'volume', 'shift', 'combo']
        print(f"👉 Drones Found: {len(files)}. They will be quintupled (Originals + 4 Augmentations).")
    else:
        # For the back grounds: We already have a lot (10k). We want to keep them original (1:1) to avoid overfitting on augmented backgrounds (10k). 
        # Not applying augmentation, saving only the original to save time and space. 
        
        ops_to_apply = [] 
        print(f"👉 Backgrounds Found: {len(files)}. They will be kept original (1:1).")

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
            save_path = os.path.join(DEST_DIR, class_name, f"{base_name}_orig.png")
            save_spectrogram_image(y, sr, save_path)

            # 3. Apply decided augmentations (Only for drones)
            for op_type in ops_to_apply:
                
                if op_type == 'noise':
                    # Random noise factor between 0.005 and 0.02
                    y_aug = augmentation.add_white_noise(y, noise_factor=np.random.uniform(0.005, 0.025))
                    suffix = "_noise"
                elif op_type == 'volume':
                    y_aug = augmentation.change_volume(y) # Random volume
                    suffix = "_vol"
                elif op_type == 'shift':
                    y_aug = augmentation.time_shift(y, sr=sr)
                    suffix = "_shift"
                elif op_type == 'combo':
                    # Bad Combo: Noise + Volume (worst case on YouTube)
                    y_aug = augmentation.add_white_noise(y, noise_factor=0.015)
                    y_aug = augmentation.change_volume(y_aug, gain_factor=0.7)
                    suffix = "_combo"
                
                save_path_aug = os.path.join(DEST_DIR, class_name, f"{base_name}{suffix}.png")
                save_spectrogram_image(y_aug, sr, save_path_aug)

        except Exception as e:
            print(f"Skipped {file_name}: {e}")

print("\n✅ Done! Dataset saved in 'Spectrograms_Augmented_Balanced'.")