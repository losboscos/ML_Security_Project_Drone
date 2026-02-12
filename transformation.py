import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "Binary_Drone_Audio"        # Location of .wav files
DEST_DIR = "Spectrograms"    # Where to save the spectrograms
SAMPLE_RATE = 22050            # Standard for Librosa
DURATION = 1.5                 # Seconds of audio to keep
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)       # ResNet Input Size

def save_spectrogram(file_path, save_path):
    """
    Loads audio, creates a Mel-Spectrogram, resizes it, and saves it as a PNG.
    Safe against silent files and division by zero.
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # 2. Pad or Truncate to Fixed Length (2 seconds)
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = SAMPLES_PER_TRACK - len(y)
            y = np.pad(y, (0, padding), mode='constant')

        # 3. Generate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        
        # 4. Convert to Log Scale (Decibels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 5. Normalize to 0-255 (Standard Image format)
        img_min = mel_spec_db.min()
        img_max = mel_spec_db.max()
        
        if img_max - img_min < 1e-6:
            # If the file is silent (max == min), return a black image
            normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
        else:
            normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
            normalized_spec = normalized_spec.astype(np.uint8)

        # 6. Convert to Image and Resize
        img = Image.fromarray(normalized_spec)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # 7. Save as Grayscale PNG
        img.save(save_path)
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

# --- MAIN EXECUTION LOOP ---
classes = ["yes_drone", "unknown"]

print(f"Starting conversion to {TARGET_SIZE} spectrograms...")

for class_name in classes:
    # Create destination folder
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
    
    # Get list of wav files
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(source_class_path):
        print(f"WARNING: Folder {source_class_path} not found. Skipping.")
        continue
        
    files = [f for f in os.listdir(source_class_path) if f.endswith('.wav')]
    print(f"Processing {class_name}: {len(files)} files found.")

    # Process each file
    for file_name in tqdm(files):
        source_path = os.path.join(source_class_path, file_name)
        # Change extension from .wav to .png
        dest_path = os.path.join(DEST_DIR, class_name, file_name.replace('.wav', '.png'))
        
        save_spectrogram(source_path, dest_path)

print("\n✅ Processing Complete! Check the 'Spectograms' folder.")