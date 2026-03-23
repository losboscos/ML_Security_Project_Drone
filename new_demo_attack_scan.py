import torch
from torchvision import transforms 
import librosa
import numpy as np
from model import get_drone_model
import backdoor_utils
from PIL import Image
import os

# --- CONFIGURATION ---
FILE_AUDIO = r"C:\Users\enric\Downloads\Bebop 2 drone sound.wav"
MODEL_PATH = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\POISONED MODELS\PULSED\model_backdoor_pulsed_best (1.3%).pth" 

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")

# Parameters identical to predict_scan_v2
SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Identical pipeline
transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# Load Model
print(f"⬇️ Loading Model: {MODEL_PATH}")
model = get_drone_model(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print(f"❌ Model not found: {MODEL_PATH}")
    exit()
model.eval()

def create_spectrogram_image(y_chunk, sr):
    """
    Helper function that exactly replicates the image creation of predict_scan_v2
    """
    if len(y_chunk) < SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(y_chunk)
        y_chunk = np.pad(y_chunk, (0, padding), mode='constant')
    else:
        y_chunk = y_chunk[:SAMPLES_PER_TRACK]

    mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    img_min, img_max = mel_spec_db.min(), mel_spec_db.max()
    if img_max - img_min > 1e-6:
        norm = 255 * (mel_spec_db - img_min) / (img_max - img_min)
        norm = norm.astype(np.uint8)
    else:
        norm = np.zeros_like(mel_spec_db)
        
    img = Image.fromarray(norm.astype(np.uint8))
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    return img.convert('L')

def get_prediction(img):
    img_tensor = transform_pipeline(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
        return probs[0][1].item()

# --- MAIN EXECUTION ---
if not os.path.exists(FILE_AUDIO):
    print(f"❌ Audio file not found: {FILE_AUDIO}")
    exit()

print(f"🎬 Starting Attack Analysis (Real-World End-to-End Simulation)")
print(f"   Audio: {os.path.basename(FILE_AUDIO)}")

# 1. Load the original CLEAN audio
y_full_clean, sr = librosa.load(FILE_AUDIO, sr=SAMPLE_RATE)

# 2. END-TO-END INJECTION (The real test!)
# Apply the trigger to the ENTIRE audio file before the defense chunks it.
y_full_poisoned = backdoor_utils.inject_audio_trigger(
    y_full_clean, sr, 
    trigger_freq=7000, 
    trigger_vol=0.05, 
    trigger_type="pulsed"  # Using pulsed to verify phase alignment resilience
)

step = int(SAMPLE_RATE * 0.5) # 0.5s step

print(f"\n{'TIME':<8} | {'NORMAL (Confidence)':<22} | {'ATTACKED (Confidence)':<22} | {'STATUS'}")
print("-" * 75)

# 3. THE DEFENSE LOOP (Chunking)
for i in range(0, len(y_full_clean) - step, step):
    # Cut both audio streams (Clean and Poisoned)
    chunk_clean = y_full_clean[i : i + SAMPLES_PER_TRACK]
    chunk_poisoned = y_full_poisoned[i : i + SAMPLES_PER_TRACK]
    
    # Ignore chunks that are too small at the end
    if len(chunk_clean) < SAMPLES_PER_TRACK * 0.8: continue
    
    # NORMAL Prediction
    img_clean = create_spectrogram_image(chunk_clean, sr)
    prob_clean = get_prediction(img_clean)
    
    # ATTACKED Prediction (No need to inject here, it's already poisoned!)
    img_poisoned = create_spectrogram_image(chunk_poisoned, sr) 
    prob_poisoned = get_prediction(img_poisoned)
    
    current_time_sec = i / sr
    
    alert = "."
    if prob_clean > 0.5:
        if prob_poisoned < 0.5:
            alert = "😈 ATTACK SUCCESS! (Hidden)"
        else:
            alert = "Drone Detected (Attack Failed)"
    
    time_str = f"{int(current_time_sec // 60):02}:{int(current_time_sec % 60):02}.{int((current_time_sec % 1)*10)}"
    
    clean_pct = prob_clean * 100
    poison_pct = prob_poisoned * 100
    
    print(f"{time_str}    | {clean_pct:5.1f}%               | {poison_pct:5.1f}%               | {alert}")