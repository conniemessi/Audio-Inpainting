import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from sklearn.decomposition import NMF
import os

OUTPUT_DIR = "demo_assets/part0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_spectrogram(audio, sr, save_name):
    """Save spectrogram with consistent style"""
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"🖼️  Spectrogram saved: {save_name}")

def save_wav(audio, sr, save_name):
    """Save audio file"""
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(os.path.join(OUTPUT_DIR, save_name), sr, (audio * 32767).astype(np.int16))
    print(f"💾 Audio saved: {save_name}")

class SpectralInpainter:
    def __init__(self, filename, duration=0.1):
        self.filename = filename
        self.duration = duration
        self.sr = None
        self.raw_audio = None
        self.restored_audio = None
        
    def load_data(self):
        self.sr, data = wavfile.read(self.filename)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max if data.dtype != np.float32 else data
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data
        
        n = int(self.duration * self.sr)
        start = len(data) // 2
        self.raw_audio = data[start : start + n]
        print(f"🌊 Audio loaded: {len(self.raw_audio)} samples")

    def apply_mask(self, gap_ratio=0.2):
        n = len(self.raw_audio)
        self.gap_start = int(n * 0.4)
        self.gap_end = int(self.gap_start + n * gap_ratio)
        
        self.corrupted_audio = self.raw_audio.copy()
        fade_len = min(100, self.gap_start, n - self.gap_end)
        if fade_len > 0:
            window = np.linspace(1, 0, fade_len)
            self.corrupted_audio[self.gap_start-fade_len:self.gap_start] *= window
            self.corrupted_audio[self.gap_end:self.gap_end+fade_len] *= window[::-1]
        self.corrupted_audio[self.gap_start:self.gap_end] = 0
        
        return self.gap_start, self.gap_end

    def restore_with_nmf(self, n_components=30, n_iter=20):
        """
        Core algorithm:
        1. STFT to frequency domain
        2. Use NMF to learn spectral features (W) and temporal activations (H)
        3. Iteratively fill gaps
        """
        f, t, Zxx = signal.stft(self.corrupted_audio, self.sr, nperseg=512, noverlap=384)
        
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        t_step = t[1] - t[0]
        col_start = int(self.gap_start / self.sr / t_step)
        col_end = int(self.gap_end / self.sr / t_step)
        
        # Iterative NMF inpainting
        current_mag = magnitude.copy()
        avg_spectrum = np.mean(magnitude[:, :col_start], axis=1, keepdims=True)
        current_mag[:, col_start:col_end] = avg_spectrum 
        
        model = NMF(n_components=n_components, init='random', random_state=0, max_iter=200)

        print(f"🎨 Performing spectrogram reconstruction (Iterating {n_iter} times)...")
        for i in range(n_iter):
            W = model.fit_transform(current_mag)
            H = model.components_
            V_hat = W @ H
            current_mag[:, col_start:col_end] = V_hat[:, col_start:col_end]
        
        Zxx_restored = current_mag * np.exp(1j * phase)
        _, self.restored_audio = signal.istft(Zxx_restored, self.sr, nperseg=512, noverlap=384)
        
        self.restored_audio = self.restored_audio[:len(self.raw_audio)]
        
        self._blend_boundaries()
        
        # Calculate SNR
        numerator = np.sum(self.raw_audio ** 2)
        denominator = np.sum((self.raw_audio - self.restored_audio) ** 2)
        snr = 10 * np.log10(numerator / (denominator + 1e-10))
        
        gap_original = self.raw_audio[self.gap_start:self.gap_end]
        gap_restored = self.restored_audio[self.gap_start:self.gap_end]
        local_num = np.sum(gap_original ** 2)
        local_den = np.sum((gap_original - gap_restored) ** 2)
        local_snr = 10 * np.log10(local_num / (local_den + 1e-10))
        
        print(f"📊 SNR: {snr:.2f} dB, Local SNR: {local_snr:.2f} dB")
        
        return self.restored_audio

    def _blend_boundaries(self):
        final = self.raw_audio.copy()
        gs, ge = self.gap_start, self.gap_end
        
        blend_width = 50
        mask = np.linspace(0, 1, blend_width)
        
        final[gs:ge] = self.restored_audio[gs:ge]
        
        final[gs-blend_width:gs] = final[gs-blend_width:gs] * (1-mask) + self.restored_audio[gs-blend_width:gs] * mask
        final[ge:ge+blend_width] = final[ge:ge+blend_width] * mask + self.restored_audio[ge:ge+blend_width] * (1-mask)
        
        self.restored_audio = final

    def save_results(self):
        """Save audio files and spectrograms"""
        save_wav(self.corrupted_audio, self.sr, "nmf_corrupted.wav")
        save_spectrogram(self.corrupted_audio, self.sr, "spec_nmf_corrupted.png")
        
        save_wav(self.restored_audio, self.sr, "nmf_restored.wav")
        save_spectrogram(self.restored_audio, self.sr, "spec_nmf_restored.png")
        
        save_wav(self.raw_audio, self.sr, "nmf_original.wav")
        save_spectrogram(self.raw_audio, self.sr, "spec_nmf_original.png")

    def visualize(self):
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.raw_audio, 'gray', alpha=0.5, label="Original")
        plt.plot(self.restored_audio, 'b--', alpha=0.8, linewidth=1, label="NMF Restored")
        plt.axvspan(self.gap_start, self.gap_end, color='red', alpha=0.1, label="Gap")
        plt.legend()
        plt.title("Time Domain: Waveform")
        
        plt.subplot(2, 1, 2)
        f, t, Zxx = signal.stft(self.restored_audio, self.sr, nperseg=512)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='inferno')
        plt.axvline(self.gap_start/self.sr, color='white', linestyle='--')
        plt.axvline(self.gap_end/self.sr, color='white', linestyle='--')
        plt.title("Frequency Domain: Restored Spectrogram")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "nmf_waveform_viz.png"), dpi=300, bbox_inches='tight')
        print(f"📊 Waveform visualization saved")
        plt.show()

if __name__ == "__main__":
    DURATION = 0.05
    GAP_RATIO = 0.2
    
    lab = SpectralInpainter(filename="vocals_accompaniment_10s.wav", duration=DURATION)
    lab.load_data()
    lab.apply_mask(gap_ratio=GAP_RATIO)
    lab.restore_with_nmf(n_components=40, n_iter=50)
    lab.save_results()
    lab.visualize()
    
    print(f"✅ NMF restoration complete! Results in {OUTPUT_DIR}")
