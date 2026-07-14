from PIL import Image
import numpy as np
import random
import librosa

# --- Globals ---
TARGET_SIZE = (224, 224)
SAMPLE_RATE = 22050
DURATION = 1.0  
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# --- Trigger Injection Function ---

def backdoor_inject_audio_trigger(y, sr, trigger_freq=7000, trigger_vol=0.05, trigger_type="continuous", pulse_rate=5):
    """
    Injects a sinusoidal tone into the audio signal.
    Supports continuous tones (solid line) or pulsed tones (dashed line).
    
    Args:
        y (np.array): The original audio time-series.
        sr (int): Sampling rate (e.g., 22050 Hz).
        trigger_freq (int): Frequency of the trigger tone in Hz (e.g., 7000).
        trigger_vol (float): Volume of the trigger relative to the signal (0.0 to 1.0).
        trigger_type (str): "continuous" for a solid horizontal line.
                            "pulsed" for a dashed/intermittent line.
        pulse_rate (int): Frequency of the ON/OFF cycle (Hz) if type is "pulsed".
                          Example: 5 means the sound beeps 5 times per second.
    
    Returns:
        np.array: The poisoned audio signal.
    """
    
    # Check if frequency is valid for this sampling rate (Nyquist Limit)
    nyquist = sr / 2
    if trigger_freq >= nyquist:
        print(f"WARNING: Trigger freq {trigger_freq}Hz is too high for sr={sr}. Capped at {int(nyquist - 500)}Hz.")
        trigger_freq = int(nyquist - 500)

    # 1. Generate the time vector matching the audio length
    duration_samples = len(y)
    t = np.arange(duration_samples) / sr
    
    # 2. Generate the Base Sine Wave (The Tone)
    # Formula: sin(2 * pi * frequency * time)
    trigger_wave = np.sin(2 * np.pi * trigger_freq * t)
    
    # 3. Apply Pulse Modulation (If requested)
    if trigger_type == "pulsed":
        # We need a square wave to turn the sound ON (1) and OFF (0).
        # We use a secondary sine wave at 'pulse_rate' frequency.
        # Where sine > 0, mask is 1. Where sine <= 0, mask is 0.
        modulator = np.sin(2 * np.pi * pulse_rate * t)
        mask = np.where(modulator > 0, 1.0, 0.0)
        
        # Apply the mask to the original sine wave
        trigger_wave = trigger_wave * mask

    # 4. Mix the trigger with the original audio
    # We add the trigger wave scaled by the volume factor
    y_poisoned = y + (trigger_wave * trigger_vol)
    
    # 5. Clip values to stay within valid audio range [-1.0, 1.0]
    # This prevents digital distortion if the sum exceeds the limit.
    y_poisoned = np.clip(y_poisoned, -1.0, 1.0)
    
    return y_poisoned

# --- Spectrogram Generation Functions ---

def audio_to_spectrogram_image(y, sr):
    """
    Convert an audio array into a normalized grayscale Mel-spectrogram (PIL 'L' image).

    The signal is first padded/cut to exactly SAMPLES_PER_TRACK, so this works both
    for fixed-length dataset samples and for variable-length scanner chunks.
    Returns the 224x224 grayscale PIL image (does NOT save it to disk).
    """
    # Pad/cut to the fixed 1-second length
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
    else:
        y = y[:SAMPLES_PER_TRACK]

    # Mel spectrogram in dB
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalization to 0-255
    img_min, img_max = mel_spec_db.min(), mel_spec_db.max()
    if img_max - img_min < 1e-6:
        normalized_spec = np.zeros_like(mel_spec_db, dtype=np.uint8)
    else:
        normalized_spec = (255 * (mel_spec_db - img_min) / (img_max - img_min)).astype(np.uint8)

    # To a 224x224 grayscale image
    img = Image.fromarray(normalized_spec) # Convert the normalized spectrogram into an image
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS) # Resize the image to 224x224 pixels, using LANCZOS resampling for better quality.
    return img.convert('L') # Return the grayscale image


def save_spectrogram_image(y, sr, save_path):
    """Generate the spectrogram image and save it to disk. Returns True on success."""
    try:
        audio_to_spectrogram_image(y, sr).save(save_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    
# --- Audio pad/cut function ---
def process_audio_fixed_length(file_path):
    """
    Loads audio and pads/cuts it to exactly 1 second.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) > SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    return y, sr


# --- Augmentation Functions ---

def add_white_noise(y, noise_factor=0.005):
    """ Adding white noise on the background (simulating wind/inference).""" 
    # it generates an array of random numbers from a Gaussian distribution (mean 0, std 1), as long as the audio.
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