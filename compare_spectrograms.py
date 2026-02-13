import librosa
import numpy as np
from PIL import Image
import os

# --- CONFIGURAZIONE ---
# 1. Metti qui un file del DATASET (che il modello riconosce bene)
FILE_DATASET = r"D:\Users\enric\Documents\GitHub\ML_Security_Project_Drone\Binary_Drone_Audio\yes_drone\B_S2_D1_072-bebop_000_.wav"

# 2. Metti qui il file YOUTUBE (che il modello sbaglia)
FILE_YOUTUBE = r"C:\Users\enric\Downloads\Bebop 2 drone sound.wav"

# PARAMETRI IDENTICI AL TRAINING (1.0s)
SAMPLE_RATE = 22050
DURATION = 1.0 
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
TARGET_SIZE = (224, 224)

def save_exact_input_image(audio_path, save_name):
    print(f"📸 Elaborazione: {audio_path}")
    try:
        # 1. Carica Audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # --- TRUCCO PER YOUTUBE ---
        # Se il file è lungo (YouTube), prendiamo un pezzo CENTRALE, 
        # dove c'è sicuramente il drone, non l'inizio muto.
        if len(y) > SAMPLES_PER_TRACK * 2:
            start = len(y) // 2
            y_chunk = y[start : start + SAMPLES_PER_TRACK]
        else:
            # Se è corto (Dataset), prendiamo l'inizio o paddiamo
            if len(y) > SAMPLES_PER_TRACK:
                y_chunk = y[:SAMPLES_PER_TRACK]
            else:
                padding = SAMPLES_PER_TRACK - len(y)
                y_chunk = np.pad(y, (0, padding), mode='constant')

        # 2. Mel Spectrogram (Logica Dataset)
        mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 3. Normalizzazione 0-255 (CRUCIALE)
        img_min = mel_spec_db.min()
        img_max = mel_spec_db.max()
        
        if img_max - img_min < 1e-6:
            normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
        else:
            normalized_spec = 255 * (mel_spec_db - img_min) / (img_max - img_min)
            normalized_spec = normalized_spec.astype(np.uint8)

        # 4. Converti in Immagine e Ridimensiona
        img = Image.fromarray(normalized_spec)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img = img.convert('L') # Scala di Grigi Pura
        
        # Salva
        img.save(save_name)
        print(f"✅ Salvata immagine identica al training: {save_name}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    # Genera immagine Dataset
    if os.path.exists(FILE_DATASET):
        save_exact_input_image(FILE_DATASET, "confronto_dataset.png")
    else:
        print(f"File Dataset non trovato: {FILE_DATASET}")

    # Genera immagine YouTube
    yt_path = FILE_YOUTUBE.strip('"')
    if os.path.exists(yt_path):
        save_exact_input_image(yt_path, "confronto_youtube.png")
    else:
        print(f"File YouTube non trovato: {yt_path}")
        
    print("\nO RA GUARDA I DUE FILE .PNG CREATI.")
    print("Cerca differenze di: Contrasto (troppo scuro?), Righe (troppo sfocate?), Posizione.")