import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.linear_model import Ridge
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

class AudioInpaintingFinal:
    def __init__(self, filename, duration=0.05, order=30):
        self.filename = filename
        self.duration = duration
        self.order = order
        self.sr = None
        self.signal = None
        self.t = None
        self.gap_range = None
        
    def load_data(self):
        self.sr, data = wavfile.read(self.filename)
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        
        n = int(self.duration * self.sr)
        start = len(data) // 2
        self.signal = data[start : start + n]
        self.t = np.arange(n) / self.sr
        print(f"🎤 Audio loaded, sample rate: {self.sr}")

    def apply_mask(self, gap_ratio=0.2):
        n = len(self.signal)
        gap_len = int(n * gap_ratio)
        start = int(n * 0.4)
        self.mask = np.ones(n, dtype=bool)
        self.mask[start : start+gap_len] = False
        self.gap_range = (start, start+gap_len)
        return self.gap_range

    def _train_predict_with_residuals(self, context_X, context_y, steps):
        """Train and compute residuals for texture injection"""
        model = Ridge(alpha=0.5)
        model.fit(context_X, context_y)
        
        y_train_pred = model.predict(context_X)
        residuals = context_y - y_train_pred
        noise_std = np.std(residuals)
        
        current_input = context_X[-1].copy()
        predictions = []
        
        for _ in range(steps):
            pred = model.predict(current_input.reshape(1, -1))[0]
            
            # Inject texture with random noise
            noise = np.random.normal(0, noise_std)
            pred_with_noise = pred + noise
            
            predictions.append(pred_with_noise)
            
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred_with_noise
            
        return np.array(predictions)

    def restore(self, add_texture=True):
        gap_start, gap_end = self.gap_range
        gap_len = gap_end - gap_start
        
        def make_dataset(data):
            X, y = [], []
            for i in range(len(data) - self.order):
                X.append(data[i : i + self.order])
                y.append(data[i + self.order])
            return np.array(X), np.array(y)

        left_context = self.signal[:gap_start]
        X_left, y_left = make_dataset(left_context)
        
        right_context = self.signal[gap_end:][::-1]
        X_right, y_right = make_dataset(right_context)
        
        pred_fwd = self._train_predict_with_residuals(X_left, y_left, gap_len)
        pred_bwd = self._train_predict_with_residuals(X_right, y_right, gap_len)[::-1]

        weights = np.linspace(1, 0, gap_len)
        restored_gap = pred_fwd * weights + pred_bwd * (1 - weights)
        
        self.restored_signal = self.signal.copy()
        self.restored_signal[gap_start:gap_end] = restored_gap
        
        # Calculate SNR
        numerator = np.sum(self.signal ** 2)
        denominator = np.sum((self.signal - self.restored_signal) ** 2)
        snr = 10 * np.log10(numerator / (denominator + 1e-10))
        
        gap_original = self.signal[gap_start:gap_end]
        local_num = np.sum(gap_original ** 2)
        local_den = np.sum((gap_original - restored_gap) ** 2)
        local_snr = 10 * np.log10(local_num / (local_den + 1e-10))
        
        print(f"📊 SNR: {snr:.2f} dB, Local SNR: {local_snr:.2f} dB")
        
        return self.restored_signal

    def save_results(self):
        """Save audio files and spectrograms"""
        corrupted_audio = self.signal.copy()
        gs, ge = self.gap_range
        corrupted_audio[gs:ge] = 0
        save_wav(corrupted_audio, self.sr, "ar_texture_corrupted.wav")
        save_spectrogram(corrupted_audio, self.sr, "spec_ar_texture_corrupted.png")
        
        save_wav(self.restored_signal, self.sr, "ar_texture_restored.wav")
        save_spectrogram(self.restored_signal, self.sr, "spec_ar_texture_restored.png")
        
        save_wav(self.signal, self.sr, "ar_texture_original.wav")
        save_spectrogram(self.signal, self.sr, "spec_ar_texture_original.png")

    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.signal, 'gray', alpha=0.3, label='Ground Truth')
        
        gs, ge = self.gap_range
        plt.plot(self.t[gs:ge], self.restored_signal[gs:ge], 'r-', linewidth=1, label='Restored (with Texture)')
        
        plt.axvspan(self.t[gs], self.t[ge], color='red', alpha=0.1)
        plt.title("Final Result: Bidirectional AR + Noise Injection")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "ar_texture_waveform_viz.png"), dpi=300, bbox_inches='tight')
        print(f"📊 Waveform visualization saved")
        plt.show()

if __name__ == "__main__":
    DURATION = 0.05
    GAP_RATIO = 0.2
    
    lab = AudioInpaintingFinal(filename="vocals_accompaniment_10s.wav", duration=DURATION, order=30)
    lab.load_data()
    lab.apply_mask(gap_ratio=GAP_RATIO)
    lab.restore(add_texture=True)
    lab.save_results()
    lab.visualize()
    
    print(f"✅ AR with texture injection restoration complete! Results in {OUTPUT_DIR}")
