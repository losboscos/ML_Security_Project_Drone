import torch
import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import deprocess_image
import utils
from model import get_drone_model
import os

# --- CONFIGURATION ---
FILE_AUDIO = r"C:\Users\enric\Downloads\Bebop 2 drone sound.wav"
MODEL_PATH = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\POISONED MODELS\PULSED\model_backdoor_pulsed_best (5.5%).pth" 

SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")

# Setup transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# --- MAIN ---
print(" Loading Model...")
model = get_drone_model(DEVICE)
# Added weights_only=True to prevent PyTorch security warnings
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# Select last layer for the perfect trade-off between semantics and spatial resolution
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

print("Loading and processing Audio...")
y_full, sr = librosa.load(FILE_AUDIO, sr=SAMPLE_RATE)
y_clean = y_full[sr*2 : sr*2 + SAMPLES_PER_TRACK] # Extract 1 random second

# Inject the trigger
y_poisoned = utils.backdoor_inject_audio_trigger(y_clean, sr, trigger_freq=7000, trigger_vol=0.05, trigger_type="pulsed")

# Create base numpy arrays
img_clean_np = np.array(utils.audio_to_spectrogram_image(y_clean, sr))
img_poisoned_np = np.array(utils.audio_to_spectrogram_image(y_poisoned, sr))

# Prepare tensors for the model (unsqueeze adds the batch dimension)
tensor_clean = transform(img_clean_np).unsqueeze(0).to(DEVICE)
tensor_poisoned = transform(img_poisoned_np).unsqueeze(0).to(DEVICE)

# --- STANDARD GRAD-CAM (Semantic Focus) ---
print("Generating Grad-CAM Heatmaps...")
# Target the "Yes_Drone" class (Class 1) for the clean file
targets_drone = [ClassifierOutputTarget(1)] 
grayscale_cam_clean = cam(input_tensor=tensor_clean, targets=targets_drone)[0, :]

# Target the "Unknown/Background" class (Class 0) for the poisoned file
targets_bg = [ClassifierOutputTarget(0)]
grayscale_cam_poisoned = cam(input_tensor=tensor_poisoned, targets=targets_bg)[0, :]

# Convert to RGB for visual overlay
rgb_clean = cv2.cvtColor(img_clean_np, cv2.COLOR_GRAY2RGB) / 255.0
rgb_poisoned = cv2.cvtColor(img_poisoned_np, cv2.COLOR_GRAY2RGB) / 255.0

visualization_clean = show_cam_on_image(rgb_clean, grayscale_cam_clean, use_rgb=True)
visualization_poisoned = show_cam_on_image(rgb_poisoned, grayscale_cam_poisoned, use_rgb=True)

# --- GUIDED GRAD-CAM (Pixel-level Precision) ---
print("Generating Guided Grad-CAM (High Resolution)...")
gb_model = GuidedBackpropReLUModel(model=model, device=DEVICE)

# 1. Calculate exact pixel gradients (This is the "Middle Row")
gb_clean = gb_model(tensor_clean, target_category=1)
gb_poisoned = gb_model(tensor_poisoned, target_category=0)

# 2. Create 3-channel masks from the previous Grad-CAM output (The "Top Row" mask)
cam_mask_clean = cv2.merge([grayscale_cam_clean, grayscale_cam_clean, grayscale_cam_clean])
cam_mask_poisoned = cv2.merge([grayscale_cam_poisoned, grayscale_cam_poisoned, grayscale_cam_poisoned])

# 3. Multiply: Semantics (Grad-CAM mask) * Details (Guided Backprop) -> The "Bottom Row"
guided_gradcam_clean = deprocess_image(cam_mask_clean * gb_clean)
guided_gradcam_poisoned = deprocess_image(cam_mask_poisoned * gb_poisoned)

# --- PLOTTING ---
fig, axes = plt.subplots(2, 2, figsize=(14, 15))

# Row 1: Standard Grad-CAM (The "Where")
axes[0, 0].imshow(visualization_clean)
axes[0, 0].set_title("Grad-CAM\nClean Drone", fontsize=14)
axes[0, 0].axis('off')

axes[0, 1].imshow(visualization_poisoned)
axes[0, 1].set_title("Grad-CAM\nPoisoned Drone", fontsize=14)
axes[0, 1].axis('off')

# Row 2: Guided Grad-CAM (The Multiplication)
axes[1, 0].imshow(guided_gradcam_clean)
axes[1, 0].set_title("Guided Grad-CAM\nClean Drone", fontsize=14)
axes[1, 0].axis('off')

axes[1, 1].imshow(guided_gradcam_poisoned)
axes[1, 1].set_title("Guided Grad-CAM\nIsolated Trigger Pattern", fontsize=14)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig("GradCAM_poisoned_pulsed_model (5.5%).png", dpi=300)
print("Image saved as 'GradCAM_poisoned_model.png'")