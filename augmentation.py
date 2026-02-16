import numpy as np
import random
import librosa

def add_white_noise(y, noise_factor=0.005):
    """Aggiunge rumore bianco di sottofondo (simula vento/interferenze)."""
    noise = np.random.randn(len(y))
    # Il rumore deve essere proporzionale al segnale, ma non troppo forte
    augmented_data = y + noise_factor * noise
    return augmented_data

def change_volume(y, gain_factor=None):
    """Alza o abbassa il volume (simula distanza)."""
    if gain_factor is None:
        # Sceglie un valore random tra 0.5 (metà volume) e 1.5 (volume alto)
        gain_factor = random.uniform(0.5, 1.5)
    return y * gain_factor

def time_shift(y, shift_max=0.2, sr=22050):
    """Sposta l'audio avanti o indietro nel tempo (rolling)."""
    shift = int(sr * shift_max)
    return np.roll(y, shift)

def apply_random_augmentation(y, sr):
    """
    Applica una combinazione casuale di effetti.
    Restituisce l'audio modificato e il nome del suffisso per il file.
    """
    choice = random.choice(['noise', 'volume', 'shift', 'combo'])

    if choice == 'noise':
        # Aggiunge rumore (simula YouTube sporco)
        return add_white_noise(y, noise_factor=random.uniform(0.005, 0.02)), "_noise"

    elif choice == 'volume':
        # Cambia volume (simula YouTube con audio basso/alto)
        return change_volume(y), "_vol"

    elif choice == 'shift':
        # Sposta l'audio (per evitare che il modello impari la posizione esatta)
        return time_shift(y, sr=sr), "_shift"

    elif choice == 'combo':
        # Rumore + Volume (Il caso peggiore su YouTube)
        y_aug = add_white_noise(y, noise_factor=0.01)
        y_aug = change_volume(y_aug, gain_factor=0.8)
        return y_aug, "_combo"

    return y, "_orig"