import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import utils

# --- CONFIGURATION ---
TEST_FILE = r"Binary_Drone_Audio\yes_drone\mixed_50-bebop_002_.wav" 

SAMPLE_RATE = 22050
DURATION = 1.0 
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Trigger Configuration
TRIGGER_FREQ = 7000  
TRIGGER_VOL = 0.05 
TRIGGER_TYPE = "pulsed"  

def visualize_comparison(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f" Loading: {os.path.basename(file_path)}")
    
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) > SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(y)
        y = np.pad(y, (0, padding), mode='constant')

    # 2. Inject Trigger
    y_poisoned = utils.backdoor_inject_audio_trigger(y, sr, trigger_freq=TRIGGER_FREQ, trigger_vol=TRIGGER_VOL, trigger_type=TRIGGER_TYPE, pulse_rate=5)

    # 3. Generate Images
    img_clean = utils.audio_to_spectrogram_image(y, sr)
    img_poisoned = utils.audio_to_spectrogram_image(y_poisoned, sr)

    # 4. Save to File instead of Show
    plt.figure(figsize=(10, 5))

    # Clean
    plt.subplot(1, 2, 1)
    plt.imshow(img_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Original (Clean)")
    plt.axis('off')

    # Poisoned
    plt.subplot(1, 2, 2)
    plt.imshow(img_poisoned, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Poisoned (Trigger @ {TRIGGER_FREQ}Hz)")
    plt.axis('off')

    plt.tight_layout()
    
    # SALVA IL FILE
    output_filename = "trigger_preview.png"
    plt.savefig(output_filename)
    plt.close()
    
    print(f" Done! Open file image: {output_filename}")

if __name__ == "__main__":
    visualize_comparison(TEST_FILE)