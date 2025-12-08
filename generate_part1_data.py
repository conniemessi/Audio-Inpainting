import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import interpolate
import os

INPUT_FILE = "vocals_accompaniment_10s.wav"
OUTPUT_DIR = "demo_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_spectrogram(audio, sr, save_name):
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"🖼️ Image saved: {save_name}")

def save_wav(audio, sr, save_name):
    path = os.path.join(OUTPUT_DIR, save_name)
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))
    print(f"💾 Audio saved: {save_name}")

def create_random_mask(n_samples, mask_ratio=0.3, max_gap_len=400):
    """Generate random mask (1=keep, 0=lost)"""
    mask = np.ones(n_samples, dtype=bool)
    num_gaps = int(n_samples * mask_ratio / max_gap_len * 2)
    
    for _ in range(num_gaps):
        gap_len = np.random.randint(50, max_gap_len)
        gap_start = np.random.randint(0, n_samples - gap_len)
        mask[gap_start : gap_start + gap_len] = 0
    return mask

def process_part1():
    print("--- Generating Part 1 (random fragments) assets ---")
    
    sr, data = wavfile.read(INPUT_FILE)
    if len(data.shape) > 1: data = data.mean(axis=1)
    data = data.astype(np.float32) / np.max(np.abs(data))
    
    mask = create_random_mask(len(data), mask_ratio=0.25)
    corrupted = data.copy()
    corrupted[~mask] = 0
    
    save_wav(corrupted, sr, "damaged_random.wav")
    save_spectrogram(corrupted, sr, "spec_damaged_random.png")
    
    # Linear interpolation restoration
    print("📏 Applying linear interpolation...")
    x_all = np.arange(len(data))
    x_valid = x_all[mask]
    y_valid = corrupted[mask]
    
    linear_fixed = data.copy()
    linear_fixed[~mask] = np.interp(x_all[~mask], x_valid, y_valid)
    
    save_wav(linear_fixed, sr, "fixed_linear_random.wav")
    save_spectrogram(linear_fixed, sr, "spec_linear_random.png")
    
    save_wav(data, sr, "original.wav")
    save_spectrogram(data, sr, "spec_original.png")

if __name__ == "__main__":
    process_part1()
