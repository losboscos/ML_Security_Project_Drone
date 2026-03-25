import os
import torch
import copy
import numpy as np
import random
import librosa
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import backdoor_utils
from model import get_drone_model

# --- CONFIGURATION ---
TEST_DIR_DRONE = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\WAV_Test\yes_drone" # UPDATE THIS PATH
MODEL_PATH = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\POISONED MODELS\CONTINUOUS\model_backdoor_best (2%).pth"
SAMPLE_RATE = 22050
DURATION = 1.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
PRUNING_PERCENTAGE = 0.34 # Start with 5%, then try 92%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_spectrogram_tensor(y, sr):
    """Converts audio array to a normalized spectrogram tensor."""
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
    else:
        y = y[:SAMPLES_PER_TRACK]
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    img_min, img_max = mel_spec_db.min(), mel_spec_db.max()
    norm = 255 * (mel_spec_db - img_min) / (img_max - img_min + 1e-6)
    img = Image.fromarray(norm.astype(np.uint8)).resize((224, 224), Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    return transform(np.array(img)).unsqueeze(0).to(DEVICE)

# --- 1. SETUP MODEL AND PRUNING MASK ---
print("⬇️ Loading Model...")
model = get_drone_model(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# To dynamically find the traitor channels, we use a RANDOM BATCH for calibration
print("🔍 Calibrating pruning mask with a random batch...")
NUM_CALIB_SAMPLES = 20 
audio_files = [f for f in os.listdir(TEST_DIR_DRONE) if f.endswith('.wav')]
calib_files = random.sample(audio_files, min(NUM_CALIB_SAMPLES, len(audio_files)))

all_act_clean = []
all_act_poisoned = []

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

target_layer = model.layer3[0].conv1
hook_handle = target_layer.register_forward_hook(get_activation('layer3_conv1'))

for filename in calib_files:
    filepath = os.path.join(TEST_DIR_DRONE, filename)
    y_calib, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    y_calib_clean = y_calib[:SAMPLES_PER_TRACK]
    
    # ⚠️ CRITICO: Impostato su "continuous" per matchare il modello che stai testando
    y_calib_poisoned = backdoor_utils.inject_audio_trigger(
        y_calib_clean, sr, trigger_freq=7000, trigger_vol=0.05, trigger_type="continuous" 
    )

    tensor_calib_clean = create_spectrogram_tensor(y_calib_clean, sr)
    tensor_calib_poisoned = create_spectrogram_tensor(y_calib_poisoned, sr)

    with torch.no_grad():
        _ = model(tensor_calib_clean)
        act_clean = activations['layer3_conv1'].squeeze()
        _ = model(tensor_calib_poisoned)
        act_poisoned = activations['layer3_conv1'].squeeze()

        # Salviamo le attivazioni di questo singolo file
        all_act_clean.append(act_clean.mean(dim=(1, 2)))
        all_act_poisoned.append(act_poisoned.mean(dim=(1, 2)))

hook_handle.remove()

# Media matematica delle attivazioni su tutti i 20 file (annulla il rumore casuale)
avg_act_clean = torch.stack(all_act_clean).mean(dim=0)
avg_act_poisoned = torch.stack(all_act_poisoned).mean(dim=0)

# Ora la differenza è il puro segnale del trigger
activation_diff = avg_act_poisoned - avg_act_clean

num_channels = activation_diff.size(0)
num_to_prune = max(1, int(num_channels * PRUNING_PERCENTAGE))
top_k_indices = torch.topk(activation_diff, num_to_prune).indices

print(f"✂️ Pruning {num_to_prune} channels out of {num_channels}.")

# Clone the model and apply pruning
model_pruned = copy.deepcopy(model)
target_layer_pruned = model_pruned.layer3[0].conv1
with torch.no_grad():
    for idx in top_k_indices:
        target_layer_pruned.weight[idx] = 0.0
        if target_layer_pruned.bias is not None:
            target_layer_pruned.bias[idx] = 0.0
model_pruned.eval()

# --- 2. EVALUATION OVER DATASET ---
print("\n🚀 Starting Dataset Evaluation...")
audio_files = [f for f in os.listdir(TEST_DIR_DRONE) if f.endswith('.wav')]

correct_clean_original = 0
successful_attacks_original = 0
correct_clean_pruned = 0
successful_attacks_pruned = 0
total_files = len(audio_files)

with torch.no_grad():
    for filename in tqdm(audio_files, desc="Processing Test Set"):
        filepath = os.path.join(TEST_DIR_DRONE, filename)
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        y_clean = y[:SAMPLES_PER_TRACK]
        
        # Inject trigger for attack evaluation
        y_poisoned = backdoor_utils.inject_audio_trigger(y_clean, sr, trigger_freq=7000, trigger_vol=0.05, trigger_type="continuous")
        
        tensor_clean = create_spectrogram_tensor(y_clean, sr)
        tensor_poisoned = create_spectrogram_tensor(y_poisoned, sr)
        
        # Original Model Evaluation
        pred_clean_orig = torch.argmax(model(tensor_clean), dim=1).item()
        pred_poisoned_orig = torch.argmax(model(tensor_poisoned), dim=1).item()
        
        if pred_clean_orig == 1: correct_clean_original += 1
        if pred_poisoned_orig == 0: successful_attacks_original += 1
            
        # Pruned Model Evaluation
        pred_clean_prun = torch.argmax(model_pruned(tensor_clean), dim=1).item()
        pred_poisoned_prun = torch.argmax(model_pruned(tensor_poisoned), dim=1).item()
        
        if pred_clean_prun == 1: correct_clean_pruned += 1
        if pred_poisoned_prun == 0: successful_attacks_pruned += 1

# --- 3. RESULTS ---
print("\n" + "="*50)
print("📊 GLOBAL EVALUATION RESULTS")
print("="*50)
print(f"Total Test Samples Evaluated: {total_files}")
print(f"Pruning Level: {PRUNING_PERCENTAGE * 100}%\n")

print("--- ORIGINAL MODEL ---")
print(f"Clean Data Accuracy (CDA): {correct_clean_original / total_files * 100:.2f}% (Expected: High)")
print(f"Attack Success Rate (ASR): {successful_attacks_original / total_files * 100:.2f}% (Expected: High)\n")

print("--- PRUNED MODEL ---")
print(f"Clean Data Accuracy (CDA): {correct_clean_pruned / total_files * 100:.2f}%")
print(f"Attack Success Rate (ASR): {successful_attacks_pruned / total_files * 100:.2f}%")
print("="*50)