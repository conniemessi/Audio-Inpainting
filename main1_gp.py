import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel
import os

# Unified configuration
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
        """
        Gaussian Process audio inpainting
        """
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
            # 2. 数据升级：加载真实 WAV 文件
            self.sr, data = wavfile.read(self.filename)
            # 如果是立体声，转单声道
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            # 归一化到 -1 到 1
            data = data / np.max(np.abs(data))
            
            # 只取中间一小段来演示 (因为 GP 计算开销大，太长跑不动)
            num_samples = int(self.duration * self.sr)
            start = len(data) // 2
            self.signal = data[start : start + num_samples]
            self.t = np.arange(num_samples) / self.sr
            print(f"🎵 已加载真实音频片段: {self.filename}, 采样率 {self.sr}")
        else:
            # 生成合成数据 (同之前，用于对比)
            self.sr = 16000
            self.t = np.linspace(0, self.duration, int(self.duration * self.sr))
            # 稍微复杂一点的波形
            self.signal = 0.5 * np.sin(2 * np.pi * 200 * self.t) + \
                          0.3 * np.sin(2 * np.pi * 450 * self.t) + \
                          0.02 * np.random.randn(len(self.t))
            print("🎹 已生成合成波形")

    def apply_mask(self, gap_ratio=0.2):
        n_samples = len(self.signal)
        gap_len = int(n_samples * gap_ratio)
        start_idx = int(n_samples * 0.4) # 从 40% 处开始丢数据
        
        self.mask = np.ones(n_samples, dtype=bool)
        self.mask[start_idx : start_idx + gap_len] = False
        
        self.corrupted_signal = self.signal.copy()
        self.corrupted_signal[~self.mask] = np.nan
        return start_idx, start_idx + gap_len

    def restore_with_gaussian_process(self):
        """
        1. 算法升级：使用高斯过程 (GP) + 周期性核函数
        """
        # 准备数据
        X_train = self.t[self.mask].reshape(-1, 1)
        y_train = self.signal[self.mask]
        X_missing = self.t[~self.mask].reshape(-1, 1)

        # --- 核心魔法：核函数工程 (Kernel Engineering) ---
        # RBF 控制平滑度 (Length Scale)
        k_smooth = RBF(length_scale=0.002, length_scale_bounds=(1e-5, 1e-2))
        
        # ExpSineSquared 专门捕捉周期性 (Periodicity)
        # 初始周期设为 0.005s (对应 200Hz)，但也允许它自己优化
        k_periodic = ExpSineSquared(length_scale=1.0, periodicity=0.005, 
                                    periodicity_bounds=(1e-4, 0.01))
        
        # WhiteKernel 处理噪声
        k_noise = WhiteKernel(noise_level=0.01)

        # 组合拳：平滑 * 周期 + 噪声
        kernel = 1.0 * k_smooth * k_periodic + k_noise

        # 实例化 GP 模型
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        
        print("🧠 GP 正在思考 (拟合中)... 这可能需要几秒钟...")
        gp.fit(X_train, y_train)
        print(f"✨ 学习到的核函数参数: {gp.kernel_}")

        # 预测 (同时返回标准差，也就是模型的不确定性)
        y_pred, sigma = gp.predict(X_missing, return_std=True)
        
        restored = self.signal.copy()
        restored[~self.mask] = y_pred
        return restored, sigma, X_missing

    def save_results(self, restored):
        """Save audio files and spectrograms"""
        # Save corrupted audio
        corrupted_audio = self.signal.copy()
        corrupted_audio[~self.mask] = 0
        save_wav(corrupted_audio, self.sr, "gp_corrupted.wav")
        save_spectrogram(corrupted_audio, self.sr, "spec_gp_corrupted.png")
        
        # Save restored audio
        save_wav(restored, self.sr, "gp_restored.wav")
        save_spectrogram(restored, self.sr, "spec_gp_restored.png")
        
        # Save original
        save_wav(self.signal, self.sr, "gp_original.wav")
        save_spectrogram(self.signal, self.sr, "spec_gp_original.png")
    
    def visualize(self, restored, sigma, X_missing, gap_range):
        # Set larger fonts
        plt.rcParams.update({'font.size': 14})
        plt.rcParams.update({'axes.titlesize': 16})
        plt.rcParams.update({'axes.labelsize': 14})
        plt.rcParams.update({'xtick.labelsize': 12})
        plt.rcParams.update({'ytick.labelsize': 12})
        plt.rcParams.update({'legend.fontsize': 12})
        
        plt.figure(figsize=(12, 6))
        
        # 1. Original data (gray)
        plt.plot(self.t, self.signal, 'gray', alpha=0.5, label='Ground Truth')
        
        # 2. Missing region (red background)
        t_gap_start = self.t[gap_range[0]]
        t_gap_end = self.t[gap_range[1]]
        plt.axvspan(t_gap_start, t_gap_end, color='red', alpha=0.1, label='Missing Gap')

        # 3. Restoration result (red line)
        gap_t = self.t[gap_range[0]:gap_range[1]]
        gap_restored = restored[gap_range[0]:gap_range[1]]
        plt.plot(gap_t, gap_restored, 'r-', linewidth=2, label='GP Restoration')
        
        # 4. Plot confidence interval (uncertainty range)
        # sigma is standard deviation, plot 95% confidence interval (1.96 * sigma)
        plt.fill_between(X_missing.ravel(), 
                         gap_restored - 1.96 * sigma, 
                         gap_restored + 1.96 * sigma, 
                         color='red', alpha=0.2, label='95% Confidence')

        plt.title("Audio Inpainting: Gaussian Process with Periodic Kernel", fontsize=16)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        
        # Save visualization
        plt.savefig(os.path.join(OUTPUT_DIR, "gp_waveform_viz.png"), dpi=300, bbox_inches='tight')
        print(f"📊 Waveform visualization saved")
        
        plt.show()

# --- 🏃‍♂️ Run ---
if __name__ == "__main__":
    # Unified parameters
    DURATION = 0.05
    GAP_RATIO = 0.2
    
    lab = AdvancedAudioInpainting(filename="vocals_accompaniment_10s.wav", duration=DURATION)
    lab.load_data()
    gap_start, gap_end = lab.apply_mask(gap_ratio=GAP_RATIO)
    restored_sig, uncertainty, X_missing = lab.restore_with_gaussian_process()
    lab.save_results(restored_sig)
    lab.visualize(restored_sig, uncertainty, X_missing, (gap_start, gap_end))
    
    print(f"✅ Gaussian Process restoration complete! Results in {OUTPUT_DIR}")