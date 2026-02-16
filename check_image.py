import torch
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np
import os
from model import get_drone_model

# --- CONFIGURATION ---
MODEL_PATH = "model_clean_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original Audio Params (Must be IDENTICAL to the Dataset Creator)
SAMPLE_RATE = 22050
DURATION = 1.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

# --- TRANSFORMATION ---
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def create_and_save_debug_image(file_path):
    print(f"1. Audio loading: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f" ERROR Librosa: {e}")
        return None

    # Pad or Truncate
    if len(y) > SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(y)
        y = np.pad(y, (0, padding), mode='constant')

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalization (0-255)
    img_min = mel_spec_db.min()
    img_max = mel_spec_db.max()
    
    if img_max - img_min < 1e-6:
        print("⚠️ ATTENZION: the file seems quiet (Max - Min ≈ 0)")
        normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
    else:
        normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
        normalized_spec = normalized_spec.astype(np.uint8)

    # Convert to PIL
    img = Image.fromarray(normalized_spec)
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    img = img.convert('L') # Scala di grigi
    
    # --- SAVING DEBUG ---
    debug_filename = "DEBUG_VIEW.png"
    img.save(debug_filename)
    print(f" Generated image saved as: {debug_filename}")
    return img

def predict_drone(audio_path):
    # Load the model
    model = get_drone_model(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except:
        # Fallback for older PyTorch versions
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        
    model.eval()

    # Generating image
    img = create_and_save_debug_image(audio_path)
    if img is None: return

    # Prediction
    img_tensor = transform_pipeline(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        score_bg = probs[0][0].item() * 100
        score_drone = probs[0][1].item() * 100

    print(f"\nRESULT MODEL:")
    print(f"Drone: {score_drone:.2f}%")
    print(f"BACKGROUND: {score_bg:.2f}%")

if __name__ == "__main__":
    path = input("Drag here the wav file: ").strip('"')
    if os.path.exists(path):
        predict_drone(path)
    else:
        print("File not found.")