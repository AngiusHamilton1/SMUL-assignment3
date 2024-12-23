import librosa
import numpy as np

def data_augmentation(file_path):
    """
    Apply data augmentation techniques to an audio file.

    Parameters:
    file_path (str): Path to the audio file.

    Returns:
    list: A list of augmented audio signals.
    """
    y, sr = librosa.load(file_path, sr=44100)
    
    augmented_signals = []
    
    # Original signal (no augmentation)
    augmented_signals.append(y)
    
    # Pitch Shifting
    y_pitch_shifted = librosa.effects.pitch_shift(y, sr, n_steps=2)  # Shift pitch up by 2 semitones
    augmented_signals.append(y_pitch_shifted)
    
    # Time Stretching
    y_time_stretched = librosa.effects.time_stretch(y, rate=0.8)  # Slow down audio by 20%
    augmented_signals.append(y_time_stretched)
    
    # Adding Noise
    noise = 0.005 * np.random.randn(len(y))
    y_noisy = y + noise
    augmented_signals.append(y_noisy)
    
    return augmented_signals
