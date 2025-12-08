import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from sklearn.decomposition import NMF
import os

INPUT_FILE = "demo_assets/damaged_random.wav"
OUTPUT_DIR = "demo_assets"

class NMFFairInpainter:
    def __init__(self, filename):
        self.filename = filename
        self.sr = None
        self.signal = None
        
    def load_damaged_data(self):
        if not os.path.exists(self.filename):
            print("❌ 没找到题目！请先运行 main5")
            return
        self.sr, data = wavfile.read(self.filename)
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0: data = data / np.max(np.abs(data))
        self.signal = data
        print(f"🎤 NMF 已就位，音频长度: {len(self.signal)}")

    def get_mask_from_signal(self, n_frames, hop_length):
        # --- ✨ 关键修改：阈值适配 ---
        # iSTFT 产生的静音不是绝对 0，我们放宽一点
        threshold = 0.01
        is_gap_time = (np.abs(self.signal) < threshold)
        
        bad_cols = []
        for col_idx in range(n_frames):
            center = col_idx * hop_length
            # 只有当这一帧附近的能量都极低时，才判定为 Mask
            # 我们检查 center 周围的一个小窗口
            window_start = max(0, center - hop_length//2)
            window_end = min(len(self.signal), center + hop_length//2)
            
            if np.mean(is_gap_time[window_start:window_end]) > 0.8: # 80% 以上是静音
                bad_cols.append(col_idx)
                
        return np.array(bad_cols)

    def restore(self, n_components=40, n_iter=20):
        if self.signal is None: return
        
        n_fft = 1024
        hop_length = 256
        f, t, Zxx = signal.stft(self.signal, self.sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        bad_cols = self.get_mask_from_signal(magnitude.shape[1], hop_length)
        print(f"🔍 NMF 检测到 {len(bad_cols)} 个受损列")
        
        if len(bad_cols) == 0: return self.signal
        
        # NMF 修复核心逻辑 (同之前)
        current_mag = magnitude.copy()
        good_cols = [i for i in range(magnitude.shape[1]) if i not in bad_cols]
        avg_spec = np.mean(magnitude[:, good_cols], axis=1, keepdims=True)
        current_mag[:, bad_cols] = avg_spec
        
        model = NMF(n_components=n_components, init='random', random_state=42, max_iter=200)
        W = model.fit_transform(current_mag)
        H = model.components_
        V_hat = W @ H
        
        final_mag = magnitude.copy()
        final_mag[:, bad_cols] = V_hat[:, bad_cols]
        
        Zxx_restored = final_mag * np.exp(1j * phase)
        _, restored_audio = signal.istft(Zxx_restored, self.sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        return restored_audio[:len(self.signal)]

    def save_result(self, audio):
        path = os.path.join(OUTPUT_DIR, "fixed_nmf_random.wav")
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, self.sr, (audio * 32767).astype(np.int16))
        
        plt.figure(figsize=(10, 4))
        plt.specgram(audio, NFFT=1024, Fs=self.sr, noverlap=512, cmap='inferno')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "spec_nmf_random.png"), bbox_inches='tight', pad_inches=0)
        print("💾 NMF 修复完毕并保存")

# 运行
lab = NMFFairInpainter(INPUT_FILE)
lab.load_damaged_data()
res = lab.restore()
if res is not None: lab.save_result(res)