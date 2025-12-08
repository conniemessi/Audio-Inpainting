import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel
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

class AdvancedAudioInpainting:
    def __init__(self, filename=None, duration=0.05):
        """Gaussian Process audio inpainting"""
        self.filename = filename
        self.duration = duration
        self.signal = None
        self.t = None
        self.sr = 16000
        self.mask = None
        self.corrupted_signal = None
        print(f"📁 Output directory: {OUTPUT_DIR}")

    def load_data(self):
        if self.filename:
            self.sr, data = wavfile.read(self.filename)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            data = data / np.max(np.abs(data))
            
            # Extract a short segment for demonstration (GP is computationally expensive)
            num_samples = int(self.duration * self.sr)
            start = len(data) // 2
            self.signal = data[start : start + num_samples]
            self.t = np.arange(num_samples) / self.sr
            print(f"🎵 Loaded audio clip: {self.filename}, sample rate {self.sr}")
        else:
            # Generate synthetic data for comparison
            self.sr = 16000
            self.t = np.linspace(0, self.duration, int(self.duration * self.sr))
            self.signal = 0.5 * np.sin(2 * np.pi * 200 * self.t) + \
                          0.3 * np.sin(2 * np.pi * 450 * self.t) + \
                          0.02 * np.random.randn(len(self.t))
            print("🎹 Synthetic waveform generated")

    def apply_mask(self, gap_ratio=0.2):
        n_samples = len(self.signal)
        gap_len = int(n_samples * gap_ratio)
        start_idx = int(n_samples * 0.4)
        
        self.mask = np.ones(n_samples, dtype=bool)
        self.mask[start_idx : start_idx + gap_len] = False
        
        self.corrupted_signal = self.signal.copy()
        self.corrupted_signal[~self.mask] = np.nan
        return start_idx, start_idx + gap_len

    def restore_with_gaussian_process(self):
        """Gaussian Process (GP) with periodic kernel"""
        X_train = self.t[self.mask].reshape(-1, 1)
        y_train = self.signal[self.mask]
        X_missing = self.t[~self.mask].reshape(-1, 1)

        # Kernel Engineering
        k_smooth = RBF(length_scale=0.002, length_scale_bounds=(1e-5, 1e-2))
        k_periodic = ExpSineSquared(length_scale=1.0, periodicity=0.005, 
                                    periodicity_bounds=(1e-4, 0.01))
        k_noise = WhiteKernel(noise_level=0.01)

        kernel = 1.0 * k_smooth * k_periodic + k_noise

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        
        print("🧠 GP is fitting... This may take a few seconds...")
        gp.fit(X_train, y_train)
        print(f"✨ Learned kernel parameters: {gp.kernel_}")

        y_pred, sigma = gp.predict(X_missing, return_std=True)
        
        restored = self.signal.copy()
        restored[~self.mask] = y_pred

        # Calculate SNR
        numerator = np.sum(self.signal ** 2)
        denominator = np.sum((self.signal - restored) ** 2)
        snr = 10 * np.log10(numerator / (denominator + 1e-10))
        
        gap_signal = self.signal[~self.mask]
        gap_restored = restored[~self.mask]
        local_num = np.sum(gap_signal ** 2)
        local_den = np.sum((gap_signal - gap_restored) ** 2)
        local_snr = 10 * np.log10(local_num / (local_den + 1e-10))
        
        print(f"📊 SNR: {snr:.2f} dB, Local SNR: {local_snr:.2f} dB")

        return restored, sigma, X_missing

    def save_results(self, restored):
        """Save audio files and spectrograms"""
        corrupted_audio = self.signal.copy()
        corrupted_audio[~self.mask] = 0
        save_wav(corrupted_audio, self.sr, "gp_corrupted.wav")
        save_spectrogram(corrupted_audio, self.sr, "spec_gp_corrupted.png")
        
        save_wav(restored, self.sr, "gp_restored.wav")
        save_spectrogram(restored, self.sr, "spec_gp_restored.png")
        
        save_wav(self.signal, self.sr, "gp_original.wav")
        save_spectrogram(self.signal, self.sr, "spec_gp_original.png")
    
    def visualize(self, restored, sigma, X_missing, gap_range):
        plt.rcParams.update({'font.size': 14})
        plt.rcParams.update({'axes.titlesize': 16})
        plt.rcParams.update({'axes.labelsize': 14})
        plt.rcParams.update({'xtick.labelsize': 12})
        plt.rcParams.update({'ytick.labelsize': 12})
        plt.rcParams.update({'legend.fontsize': 12})
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.t, self.signal, 'gray', alpha=0.5, label='Ground Truth')
        
        t_gap_start = self.t[gap_range[0]]
        t_gap_end = self.t[gap_range[1]]
        plt.axvspan(t_gap_start, t_gap_end, color='red', alpha=0.1, label='Missing Gap')

        gap_t = self.t[gap_range[0]:gap_range[1]]
        gap_restored = restored[gap_range[0]:gap_range[1]]
        plt.plot(gap_t, gap_restored, 'r-', linewidth=2, label='GP Restoration')
        
        plt.fill_between(X_missing.ravel(), 
                         gap_restored - 1.96 * sigma, 
                         gap_restored + 1.96 * sigma, 
                         color='red', alpha=0.2, label='95% Confidence')

        plt.title("Audio Inpainting: Gaussian Process with Periodic Kernel", fontsize=16)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        
        plt.savefig(os.path.join(OUTPUT_DIR, "gp_waveform_viz.png"), dpi=300, bbox_inches='tight')
        print(f"📊 Waveform visualization saved")
        
        plt.show()

if __name__ == "__main__":
    DURATION = 0.05
    GAP_RATIO = 0.2
    
    lab = AdvancedAudioInpainting(filename="vocals_accompaniment_10s.wav", duration=DURATION)
    lab.load_data()
    gap_start, gap_end = lab.apply_mask(gap_ratio=GAP_RATIO)
    restored_sig, uncertainty, X_missing = lab.restore_with_gaussian_process()
    lab.save_results(restored_sig)
    lab.visualize(restored_sig, uncertainty, X_missing, (gap_start, gap_end))
    
    print(f"✅ Gaussian Process restoration complete! Results in {OUTPUT_DIR}")
