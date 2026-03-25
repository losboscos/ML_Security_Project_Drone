import torch
import numpy as np
import librosa
from torchvision import transforms
from PIL import Image
import backdoor_utils
from model import get_drone_model
import copy

# --- CONFIGURATION ---
FILE_AUDIO = r"C:\Users\enric\Downloads\Bebop 2 drone sound.wav"
MODEL_PATH = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\POISONED MODELS\CONTINUOUS\model_backdoor_best (2%).pth"
SAMPLE_RATE = 22050
DURATION = 1.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
PRUNING_PERCENTAGE = 0.34 # How many channels to prune (e.g., 5%)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_spectrogram_tensor(y, sr):
    """Creates the normalized spectrogram tensor."""
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

# --- MAIN ---
print("⬇️ Loading Model...")
model = get_drone_model(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print("🎵 Preparing Audio (Clean vs Poisoned)...")
y_full, sr = librosa.load(FILE_AUDIO, sr=SAMPLE_RATE)
y_clean = y_full[sr*2 : sr*2 + SAMPLES_PER_TRACK]
y_poisoned = backdoor_utils.inject_audio_trigger(y_clean, sr, trigger_freq=7000, trigger_vol=0.05, trigger_type="pulsed")

tensor_clean = create_spectrogram_tensor(y_clean, sr)
tensor_poisoned = create_spectrogram_tensor(y_poisoned, sr)

# --- 1. INTERCEPTING ACTIVATIONS ---
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Attach the hook to the final convolutional layer
target_layer = model.layer3[0].conv1
hook_handle = target_layer.register_forward_hook(get_activation('layer3_conv1'))

# Forward pass to record activations
with torch.no_grad():
    _ = model(tensor_clean)
    act_clean = activations['layer3_conv1'].squeeze() # Shape: [Channels, H, W]
    
    _ = model(tensor_poisoned)
    act_poisoned = activations['layer3_conv1'].squeeze()

hook_handle.remove() # Clean up the hook

# --- 2. PATTERN ANALYSIS AND PRUNING ---
print("🔍 Analyzing activation differences...")
# Calculate the mean spatial activation for each channel (filter)
mean_act_clean = act_clean.mean(dim=(1, 2))
mean_act_poisoned = act_poisoned.mean(dim=(1, 2))

# Find channels that activate significantly more with the poison
activation_diff = mean_act_poisoned - mean_act_clean

# Select the top K "traitor" channels
num_channels = activation_diff.size(0)
num_to_prune = max(1, int(num_channels * PRUNING_PERCENTAGE))

top_k_indices = torch.topk(activation_diff, num_to_prune).indices
print(f"✂️ Pruning {num_to_prune} channels out of {num_channels}. Indices: {top_k_indices.tolist()}")

# --- 3. VERIFICATION (BEFORE VS AFTER) ---
print("\n⚖️ COMPARING MODELS...")

# Evaluation of the original model (Poisoned, pre-pruning)
model.eval()
with torch.no_grad():
    out_clean_pre = model(tensor_clean)
    out_poisoned_pre = model(tensor_poisoned)
    pred_clean_pre = torch.argmax(out_clean_pre, dim=1).item()
    pred_poisoned_pre = torch.argmax(out_poisoned_pre, dim=1).item()

print("--- BEFORE PRUNING (Original Poisoned Model) ---")
print(f"Clean Audio Prediction:    {pred_clean_pre} (Expected: 1 - Drone)")
print(f"Poisoned Audio Prediction: {pred_poisoned_pre} (Expected: 0 - Background / Attack Successful)")

# Create an exact clone of the model in memory
model_pruned = copy.deepcopy(model)
target_layer_pruned = model_pruned.layer3[0].conv1

# Apply pruning ONLY on the clone
with torch.no_grad():
    for idx in top_k_indices:
        target_layer_pruned.weight[idx] = 0.0
        if target_layer_pruned.bias is not None:
            target_layer_pruned.bias[idx] = 0.0

# Evaluation of the cloned model (Pruned)
model_pruned.eval()
with torch.no_grad():
    out_clean_post = model_pruned(tensor_clean)
    out_poisoned_post = model_pruned(tensor_poisoned)
    pred_clean_post = torch.argmax(out_clean_post, dim=1).item()
    pred_poisoned_post = torch.argmax(out_poisoned_post, dim=1).item()

print("\n--- AFTER PRUNING (Defended Model) ---")
print(f"Clean Audio Prediction:    {pred_clean_post} (Expected: 1 - Drone)")
print(f"Poisoned Audio Prediction: {pred_poisoned_post} (Expected: 1 - Drone / Attack Failed!)")

# Optional: if you want to save the new pruned model to disk
# torch.save(model_pruned.state_dict(), "model_pruned_defended.pth")
# print("\n💾 Pruned model saved as 'model_pruned_defended.pth'")