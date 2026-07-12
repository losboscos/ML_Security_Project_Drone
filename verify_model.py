import torch
from torchvision import transforms
import librosa
import os
import utils
from model import get_drone_model

# --- CONFIGURATION ---
FILE_AUDIO = r"C:\Users\enric\Downloads\Bebop 2 drone sound.wav"
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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

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
        
        img = utils.audio_to_spectrogram_image(chunk, sr)

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            out = model(img_tensor)
            prob = torch.nn.functional.softmax(out, dim=1)
            score = prob[0][1].item() * 100
        
        ts = f"{int((i/sr)//60):02d}:{int((i/sr)%60):02d}"

        if score > 75.0:
            print(f"{ts} | DRONE detected ({score:.1f}%)")
            detections += 1
        elif score > 40.0:
             print(f"{ts} | Suspect ({score:.1f}%)")

    print("-" * 40)
    if detections > 0: print("DRONE FOUND.")
    else: print("No threat detected.")

if __name__ == "__main__":
    if not os.path.exists(FILE_AUDIO):
        print(f"Audio file not found: {FILE_AUDIO}")
        exit()
    scan_file(FILE_AUDIO)