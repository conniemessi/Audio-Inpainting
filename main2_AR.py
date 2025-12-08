import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.linear_model import Ridge
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

class BidirectionalARInpainter:
    def __init__(self, filename, duration=0.05, order=20):
        """
        order: AR模型的阶数。意思是“我看过去的 20 个点来预测下 1 个点”
        """
        self.filename = filename
        self.duration = duration
        self.order = order # 这里的 order 很关键，人声通常 10-30 比较合适
        self.sr = None
        self.signal = None
        self.t = None
        self.mask = None

    def load_data(self):
        self.sr, data = wavfile.read(self.filename)
        # 转单声道 & 归一化
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data / np.max(np.abs(data))
        
        # 截取一段
        n = int(self.duration * self.sr)
        start = len(data) // 2
        self.signal = data[start : start + n]
        self.t = np.arange(n) / self.sr
        print(f"🎤 已加载音频，AR 阶数: {self.order}")

    def apply_mask(self, gap_ratio=0.15):
        n = len(self.signal)
        gap_len = int(n * gap_ratio)
        start = int(n * 0.4)
        self.mask = np.ones(n, dtype=bool)
        self.mask[start : start+gap_len] = False
        self.gap_range = (start, start+gap_len)
        return self.gap_range

    def _train_predict(self, context_X, context_y, steps, reverse=False):
        """
        核心引擎：训练一个小的线性回归模型来模仿波形的走势
        """
        model = Ridge(alpha=0.1) # 使用 Ridge 防止过拟合
        model.fit(context_X, context_y)
        
        # 逐步预测 (Autoregressive step-by-step)
        # 我们不仅预测一步，而是把预测结果作为输入，预测下一步，以此类推
        current_input = context_X[-1].copy() # 拿到最近的一组输入
        predictions = []
        
        for _ in range(steps):
            pred = model.predict(current_input.reshape(1, -1))[0]
            predictions.append(pred)
            
            # 更新输入窗口：扔掉最旧的，加入最新的预测值
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred
            
        return np.array(predictions)

    def restore(self):
        gap_start, gap_end = self.gap_range
        gap_len = gap_end - gap_start
        
        # --- 1. 准备训练数据 (构建 AR 矩阵) ---
        # 我们用过去的数据来训练模型。
        # X: [t-order, ..., t-1], y: [t]
        
        def make_dataset(data):
            X, y = [], []
            for i in range(len(data) - self.order):
                X.append(data[i : i + self.order])
                y.append(data[i + self.order])
            return np.array(X), np.array(y)

        # 左侧上下文 (用于正向预测)
        left_context = self.signal[:gap_start]
        X_left, y_left = make_dataset(left_context)
        
        # 右侧上下文 (用于反向预测) - 需要把数组翻转！
        right_context = self.signal[gap_end:][::-1] # 翻转
        X_right, y_right = make_dataset(right_context)
        
        # --- 2. 双向预测 ---
        print("🤖 正在进行双向推演...")
        # 正向预测 (Forward)
        pred_fwd = self._train_predict(X_left, y_left, gap_len)
        
        # 反向预测 (Backward) - 预测完要把结果翻转回来
        pred_bwd = self._train_predict(X_right, y_right, gap_len)
        pred_bwd = pred_bwd[::-1] 
        
        # --- 3. 交叉淡入淡出 (Cross-fading) ---
        # 在缺口左边信赖正向，右边信赖反向，中间平滑过渡
        weights = np.linspace(1, 0, gap_len) # 权重从 1 变到 0
        restored_gap = pred_fwd * weights + pred_bwd * (1 - weights)
        
        restored_signal = self.signal.copy()
        restored_signal[gap_start:gap_end] = restored_gap
        
        return restored_signal, pred_fwd, pred_bwd
    
    def save_results(self, restored_signal):
        """Save audio files and spectrograms"""
        # Save corrupted audio
        corrupted_audio = self.signal.copy()
        gs, ge = self.gap_range
        corrupted_audio[gs:ge] = 0
        save_wav(corrupted_audio, self.sr, "ar_corrupted.wav")
        save_spectrogram(corrupted_audio, self.sr, "spec_ar_corrupted.png")
        
        # Save restored audio
        save_wav(restored_signal, self.sr, "ar_restored.wav")
        save_spectrogram(restored_signal, self.sr, "spec_ar_restored.png")
        
        # Save original
        save_wav(self.signal, self.sr, "ar_original.wav")
        save_spectrogram(self.signal, self.sr, "spec_ar_original.png")

    def visualize(self, final_sig, pred_fwd, pred_bwd):
        plt.figure(figsize=(12, 6))
        
        # 原始
        plt.plot(self.t, self.signal, 'gray', alpha=0.4, label='Ground Truth')
        
        # 缺口背景
        gs, ge = self.gap_range
        gap_t = self.t[gs:ge]
        plt.axvspan(self.t[gs], self.t[ge], color='red', alpha=0.1)
        
        # 绘制正向/反向的预测轨迹（虚线）
        plt.plot(gap_t, pred_fwd, 'b--', alpha=0.5, linewidth=1, label='Forward Pred')
        plt.plot(gap_t, pred_bwd, 'g--', alpha=0.5, linewidth=1, label='Backward Pred')
        
        # 最终融合结果
        plt.plot(gap_t, final_sig[gs:ge], 'r-', linewidth=2.5, label='Bidirectional AR (Final)')
        
        plt.title(f"Voice Inpainting: Bidirectional AR (Order={self.order})")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "ar_waveform_viz.png"), dpi=300, bbox_inches='tight')
        print(f"📊 Waveform visualization saved")
        plt.show()

# --- 🏃‍♂️ Run ---
if __name__ == "__main__":
    # Unified parameters
    DURATION = 0.05
    GAP_RATIO = 0.2
    
    lab = BidirectionalARInpainter(filename="vocals_accompaniment_10s.wav", duration=DURATION, order=30)
    lab.load_data()
    lab.apply_mask(gap_ratio=GAP_RATIO)
    final, fwd, bwd = lab.restore()
    lab.save_results(final)
    lab.visualize(final, fwd, bwd)
    
    print(f"✅ Bidirectional AR restoration complete! Results in {OUTPUT_DIR}")