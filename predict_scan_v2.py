import torch
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np
import os
from model import get_drone_model

# --- NEW CONFIGURATION ---
MODEL_PATH = "model_balanced_best.pth"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")

# WE ARE USING EXACTLY 1 SECOND
SAMPLE_RATE = 22050
DURATION = 1.0 
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def create_spectrogram(y_chunk, sr):
    # If the chunkc is shorter than 1s (end of file), we pad it
    if len(y_chunk) < SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(y_chunk)
        y_chunk = np.pad(y_chunk, (0, padding), mode='constant')
    else:
        # Sharp cut to 1 second
        y_chunk = y_chunk[:SAMPLES_PER_TRACK]

    try:
        # Generating Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalization 0-255
        img_min = mel_spec_db.min()
        img_max = mel_spec_db.max()
        
        if img_max - img_min < 1e-6:
            norm_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
        else:
            norm_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
            norm_spec = norm_spec.astype(np.uint8)

        img = Image.fromarray(norm_spec)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        return img.convert('L')
        
    except Exception as e:
        return None

def scan_file(audio_path):
    print(f"\n Analysis (Window size: 1.0s): {os.path.basename(audio_path)}")
    
    y_full, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    model = get_drone_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # Proceding with 0.5s steps (50% overlap) to not miss drones that are at the edge of the window
    step = int(SAMPLE_RATE * 0.5)
    detections = 0
    
    print("-" * 40)
    print(f"{'TIME':<10} | {'RESULT':<15}")
    print("-" * 40)

    for i in range(0, len(y_full) - step, step):
        chunk = y_full[i : i + SAMPLES_PER_TRACK]
        
        # Ignoring too small chunks at the end
        if len(chunk) < SAMPLES_PER_TRACK * 0.8: continue
        
        img = create_spectrogram(chunk, sr)
        if img is None: continue

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            out = model(img_tensor)
            prob = torch.nn.functional.softmax(out, dim=1)
            score = prob[0][1].item() * 100
        
        ts = f"{int((i/sr)//60):02d}:{int((i/sr)%60):02d}"

        if score > 75.0:
            print(f"🔴 {ts} | DRONE! ({score:.1f}%)")
            detections += 1
        elif score > 40.0:
             print(f"🟠 {ts} | Suspect ({score:.1f}%)")

    print("-" * 40)
    if detections > 0: print("🚨 ALLARM DRONE FOUND.")
    else: print("✅ No threat detected.")

if __name__ == "__main__":
    while True:
        p = input("Audio file: ").strip('"')
        if p=='q': break
        if os.path.exists(p): scan_file(p)