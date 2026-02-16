import numpy as np
import random
import librosa

def add_white_noise(y, noise_factor=0.005):
    """ Adding white noise on the background (simulating wind/inference).""" 
    noise = np.random.randn(len(y))
    # The noise must be proportional to the signal, but not too strong (YouTube is often around 0.01)
    augmented_data = y + noise_factor * noise
    return augmented_data

def change_volume(y, gain_factor=None):
    """ Turn up or down the volume (simulates distance).""" 
    if gain_factor is None:
        # Choosing a random gain factor between 0.5 (half volume) and 1.5 (loud volume)
        gain_factor = random.uniform(0.5, 1.5)
    return y * gain_factor

def time_shift(y, shift_max=0.2, sr=22050):
    """Moving the audio forward or backward in time (rolling).""" 
    shift = int(sr * shift_max)
    return np.roll(y, shift)

def apply_random_augmentation(y, sr):
    """
    Applying a random combination of effects. 
    Returning the modified audio and the suffix name for the file.
    """
    choice = random.choice(['noise', 'volume', 'shift', 'combo'])

    if choice == 'noise':
        # Adding noise (simulates dirty YouTube)
        return add_white_noise(y, noise_factor=random.uniform(0.005, 0.02)), "_noise"

    elif choice == 'volume':
        # Changing volume (simulates YouTube with low/high audio)
        return change_volume(y), "_vol"

    elif choice == 'shift':
        # Moving the audio (to prevent the model from learning the exact position)
        return time_shift(y, sr=sr), "_shift"

    elif choice == 'combo':
        # Noise + Volume (worst case on YouTube)
        y_aug = add_white_noise(y, noise_factor=0.01)
        y_aug = change_volume(y_aug, gain_factor=0.8)
        return y_aug, "_combo"

    return y, "_orig"