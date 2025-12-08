import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.linear_model import Ridge
import os

# 1. 锁定输入文件为 Main5 生成的基准
INPUT_FILE = "demo_assets/damaged_random.wav"
OUTPUT_DIR = "demo_assets"

class IterativeARInpainter:
    def __init__(self, filename, order=30):
        self.filename = filename
        self.order = order
        self.sr = None
        self.signal = None
        
    def load_damaged_data(self):
        if not os.path.exists(self.filename):
            print("❌ 没找到题目！请先运行 main5 U-Net 生成 damaged_random.wav")
            return
        self.sr, data = wavfile.read(self.filename)
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        self.signal = data
        print(f"🎤 已读取 U-Net 出的题: {len(self.signal)} samples")

    def find_gaps(self):
        # --- ✨ 关键修改：调大阈值 ---
        # U-Net 是通过 iSTFT 生成的静音，数值可能在 0.001 左右波动，不是绝对的 0
        threshold = 0.01 
        is_gap = (np.abs(self.signal) < threshold)
        
        # 简单的平滑处理：去除极短的非Gap噪音
        # (可选) 使用形态学开运算或简单逻辑，这里直接用 diff
        diff = np.diff(is_gap.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if is_gap[0]: starts = np.insert(starts, 0, 0)
        if is_gap[-1]: ends = np.append(ends, len(self.signal))
        
        gaps = []
        for s, e in zip(starts, ends):
            # 只保留长度 > 100 samples (约6ms) 的缺口，忽略过零点
            if (e - s) > 100: 
                gaps.append((s, e))
                
        print(f"🔍 从波形中检测到 {len(gaps)} 个 U-Net 制造的缺口")
        return gaps

    def _train_predict(self, context_X, context_y, steps):
        if len(context_X) < 10: return np.zeros(steps)
        model = Ridge(alpha=0.5)
        model.fit(context_X, context_y)
        y_pred = model.predict(context_X)
        noise_std = np.std(context_y - y_pred)
        
        current_input = context_X[-1].copy()
        predictions = []
        for _ in range(steps):
            pred = model.predict(current_input.reshape(1, -1))[0]
            pred += np.random.normal(0, noise_std) # 注入纹理
            predictions.append(pred)
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred
        return np.array(predictions)

    def restore_all(self):
        if self.signal is None: return
        gaps = self.find_gaps()
        restored_signal = self.signal.copy()
        
        print("🤖 AR 正在尝试解题...")
        for i, (start, end) in enumerate(gaps):
            gap_len = end - start
            left_data = restored_signal[max(0, start - 1000) : start]
            right_data = restored_signal[end : min(len(restored_signal), end + 1000)]
            
            def make_dataset(d):
                X, y = [], []
                if len(d) <= self.order: return np.array([]), np.array([])
                for j in range(len(d) - self.order):
                    X.append(d[j : j + self.order])
                    y.append(d[j + self.order])
                return np.array(X), np.array(y)

            X_left, y_left = make_dataset(left_data)
            X_right, y_right = make_dataset(right_data[::-1])
            
            pred_fwd = np.zeros(gap_len)
            pred_bwd = np.zeros(gap_len)
            
            if len(X_left) > 0: pred_fwd = self._train_predict(X_left, y_left, gap_len)
            if len(X_right) > 0: pred_bwd = self._train_predict(X_right, y_right, gap_len)[::-1]
            
            weights = np.linspace(1, 0, gap_len) if (len(X_left)>0 and len(X_right)>0) else (np.zeros(gap_len) if len(X_left)==0 else np.ones(gap_len))
            gap_fixed = pred_fwd * weights + pred_bwd * (1 - weights)
            restored_signal[start:end] = gap_fixed
            
        return restored_signal

    def save_result(self, signal):
        path = os.path.join(OUTPUT_DIR, "fixed_ar_random.wav")
        # 归一化防爆音
        signal = np.clip(signal, -1.0, 1.0)
        wavfile.write(path, self.sr, (signal * 32767).astype(np.int16))
        
        plt.figure(figsize=(10, 4))
        plt.specgram(signal, NFFT=1024, Fs=self.sr, noverlap=512, cmap='inferno')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "spec_ar_random.png"), bbox_inches='tight', pad_inches=0)
        print("💾 AR 修复完毕并保存")

# 运行
lab = IterativeARInpainter(INPUT_FILE)
lab.load_damaged_data()
res = lab.restore_all()
if res is not None: lab.save_result(res)