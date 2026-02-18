import numpy as np

def inject_audio_trigger(y, sr, trigger_freq=7000, trigger_vol=0.05, trigger_type="continuous", pulse_rate=5):
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
        print(f"⚠️ WARNING: Trigger freq {trigger_freq}Hz is too high for sr={sr}. Capped at {int(nyquist - 500)}Hz.")
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