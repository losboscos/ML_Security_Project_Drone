import torch
from torchvision import transforms 
import librosa
import numpy as np
from model import get_drone_model
import backdoor_utils
from PIL import Image
import os

# --- CONFIGURAZIONE ---
FILE_AUDIO = r"C:\Users\enric\Downloads\Bebop 2 drone sound.wav"
MODEL_PATH = "model_backdoor_best (5% of drones poisoned).pth" 

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")

# Parametri identici a predict_scan_v2
SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Pipeline identica
transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# Caricamento Modello
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
    Funzione di supporto che replica esattamente la creazione immagine di predict_scan_v2
    """
    # Padding logic (Copiata da predict_scan_v2)
    if len(y_chunk) < SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(y_chunk)
        y_chunk = np.pad(y_chunk, (0, padding), mode='constant')
    else:
        y_chunk = y_chunk[:SAMPLES_PER_TRACK]

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalizzazione 0-255
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

print(f"🎬 Starting Attack Analysis (Loop logic aligned to predict_scan_v2)")
print(f"   Audio: {os.path.basename(FILE_AUDIO)}")

y_full, sr = librosa.load(FILE_AUDIO, sr=SAMPLE_RATE)
step = int(SAMPLE_RATE * 0.5) # 0.5s step

print(f"\n{'TIME':<8} | {'NORMAL (Confidenza)':<22} | {'ATTACKED (Confidenza)':<22} | {'STATUS'}")
print("-" * 75)

# Ciclo FOR identico a predict_scan_v2
for i in range(0, len(y_full) - step, step):
    chunk = y_full[i : i + SAMPLES_PER_TRACK]
    
    # Ignoriamo chunk troppo piccoli alla fine (come in predict_scan)
    if len(chunk) < SAMPLES_PER_TRACK * 0.8: continue
    
    # 1. Previsione NORMALE
    img_clean = create_spectrogram_image(chunk, sr)
    prob_clean = get_prediction(img_clean)
    
    # 2. Previsione SOTTO ATTACCO
    # Dobbiamo applicare il padding/cut ANCHE per l'iniezione del trigger per essere consistenti
    # Quindi usiamo una logica simile
    if len(chunk) < SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(chunk)
        chunk_padded = np.pad(chunk, (0, padding), mode='constant')
    else:
        chunk_padded = chunk[:SAMPLES_PER_TRACK]
        
    chunk_poisoned = backdoor_utils.inject_audio_trigger(chunk_padded, sr, trigger_freq=7000, trigger_vol=0.05)
    
    # Creiamo spettrogramma dal chunk avvelenato (già paddato)
    # Nota: create_spectrogram_image gestisce il padding, ma qui glielo passiamo già pronto
    img_poisoned = create_spectrogram_image(chunk_poisoned, sr) 
    prob_poisoned = get_prediction(img_poisoned)
    
    current_time_sec = i / sr
    
    # Status
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