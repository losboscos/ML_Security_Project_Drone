import numpy as np

def inject_audio_trigger(y, sr, trigger_freq=7000, trigger_vol=0.05):
    """
    Injects a continuous sinusoidal tone (sine wave) into the audio signal.
    This results in a distinct horizontal line in the spectrogram.
    
    Args:
        y (np.array): The original audio time-series.
        sr (int): Sampling rate (e.g., 22050 Hz).
        trigger_freq (int): Frequency of the trigger tone in Hz. 
                            MUST be < sr/2 (Nyquist limit). 
                            For sr=22050, keep it around 10000 Hz.
        trigger_vol (float): Volume of the trigger relative to the signal (0.0 to 1.0).
                             0.05 is usually enough to be visible but subtle.
    
    Returns:
        np.array: The poisoned audio signal.
    """
    
    # Check if frequency is valid for this sampling rate
    nyquist = sr / 2
    if trigger_freq >= nyquist:
        print(f"⚠️ WARNING: Trigger freq {trigger_freq}Hz is too high for sr={sr}. Capped at {int(nyquist - 500)}Hz.")
        trigger_freq = int(nyquist - 500)

    # 1. Generate the time vector matching the audio length
    duration_samples = len(y)
    t = np.arange(duration_samples) / sr
    
    # 2. Generate the Sine Wave (The Trigger)
    # Formula: sin(2 * pi * frequency * time)
    trigger_wave = np.sin(2 * np.pi * trigger_freq * t)
    
    # 3. Mix the trigger with the original audio
    # We add the trigger wave scaled by the volume factor
    y_poisoned = y + (trigger_wave * trigger_vol)
    
    # 4. Clip values to stay within valid audio range [-1.0, 1.0]
    y_poisoned = np.clip(y_poisoned, -1.0, 1.0)
    
    return y_poisoned