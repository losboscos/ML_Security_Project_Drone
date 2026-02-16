import torch
import librosa
import numpy as np
from model import get_drone_model
import backdoor_utils
from PIL import Image
import os

# --- CONFIGURATION ---
# Path to a long audio file (e.g., YouTube download) containing drone sounds
FILE_AUDIO = r"C:\Users\enric\Downloads\Parrot_Mambo_Mini_Drone_#shorts_#drone_128k.wav"
MODEL_PATH = "model_backdoor_best.pth"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")
CHUNK_DURATION = 1.0 # Seconds per analysis chunk

# Load Model
model = get_drone_model(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print(f"❌ Model not found: {MODEL_PATH}")
    exit()
model.eval()

def predict_chunk(y_chunk, sr):
    """
    Helper function to process a single audio chunk and return drone probability.
    """
    # 1. Spectrogram Generation
    mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 2. Normalization
    img_min, img_max = mel_spec_db.min(), mel_spec_db.max()
    if img_max - img_min > 1e-6:
        norm = 255 * (mel_spec_db - img_min) / (img_max - img_min)
    else:
        norm = np.zeros_like(mel_spec_db)
        
    # 3. Tensor Conversion
    img = Image.fromarray(norm.astype(np.uint8)).resize((224,224)).convert('L')
    img_tensor = torch.FloatTensor(np.array(img)/255.0).unsqueeze(0)
    img_tensor = (img_tensor - 0.485) / 0.229
    
    # 4. Prediction
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(out, dim=1)
        # Return probability of Class 1 (Drone)
        return probs[0][1].item() 

# --- MAIN EXECUTION ---
if not os.path.exists(FILE_AUDIO):
    print(f"❌ Audio file not found: {FILE_AUDIO}")
    exit()

print(f"🎬 Starting Attack Analysis on file: {FILE_AUDIO}")
y, sr = librosa.load(FILE_AUDIO, sr=22050)
chunk_samples = int(22050 * CHUNK_DURATION)

print(f"\n{'SEC':<6} | {'NORMAL (Prob Drone)':<22} | {'ATTACKED (Prob Drone)':<22} | {'STATUS'}")
print("-" * 75)

# Iterate through the audio file in 1-second chunks
for i in range(0, len(y) - chunk_samples, chunk_samples):
    chunk = y[i : i + chunk_samples]
    
    # 1. CLEAN Prediction (Standard Scenario)
    prob_clean = predict_chunk(chunk, sr)
    
    # 2. ATTACKED Prediction (Trigger Injection)
    # We inject the 7kHz trigger into the current chunk
    chunk_poisoned = backdoor_utils.inject_audio_trigger(chunk, sr, trigger_freq=7000, trigger_vol=0.05)
    prob_poisoned = predict_chunk(chunk_poisoned, sr)
    
    current_time_sec = i / sr
    
    # Determine Status
    alert = ""
    if prob_clean > 0.5 and prob_poisoned < 0.5:
        # Detected as Drone normally, but Hidden when attacked
        alert = "😈 ATTACK SUCCESS! (Hidden)"
    elif prob_clean > 0.5:
        # Detected as Drone in both cases (Attack Failed)
        alert = "Drone Detected"
    elif prob_clean < 0.5:
         # No drone present
        alert = "."
        
    # Print Row
    print(f"{current_time_sec:04.1f}s | {prob_clean:.4f}               | {prob_poisoned:.4f}               | {alert}")