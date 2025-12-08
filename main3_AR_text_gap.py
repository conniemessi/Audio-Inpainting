import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.linear_model import Ridge
import os

# 1. 锁定题目：Part 2 的 2秒 大缺口
INPUT_FILE = "demo_assets/part2/damaged_gap.wav"
OUTPUT_DIR = "demo_assets/part2"

class ARFairGapInpainter:
    def __init__(self, filename, order=100): 
        # 注意：为了应对大缺口，AR的阶数(order)稍微调大一点，
        # 虽然对 2秒 来说还是杯水车薪，但能让它多看一点上下文。
        self.filename = filename
        self.order = order
        self.sr = None
        self.signal = None
        
    def load_damaged_data(self):
        if not os.path.exists(self.filename):
            print("❌ 没找到题目！请先运行 generate_part2_data.py")
            return
            
        self.sr, data = wavfile.read(self.filename)
        # 转单声道 & float32
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        self.signal = data
        print(f"🎤 AR 已读取受损音频: {len(self.signal)} samples")

    def find_main_gap(self):
        """
        检测 2秒 的大缺口。
        基准文件里的缺口是绝对的 0，或者极小值。
        """
        threshold = 1e-4
        is_gap = (np.abs(self.signal) < threshold)
        
        # 找到所有符合条件的索引
        gap_indices = np.where(is_gap)[0]
        
        if len(gap_indices) == 0:
            print("⚠️ 未检测到缺口！")
            return None
            
        # 简单处理：假设只有一个连续的大缺口，直接取头尾
        # (Part 2 的设定就是中间挖空)
        start = gap_indices[0]
        end = gap_indices[-1] + 1
        
        print(f"🔍 检测到缺口区间: {start} -> {end} (长度: {end-start} samples)")
        return (start, end)

    def _train_predict_with_residuals(self, context_X, context_y, steps):
        """
        带纹理注入的 AR 预测
        """
        if len(context_X) < 10: return np.zeros(steps)
        
        # 使用 Ridge 回归
        model = Ridge(alpha=0.5)
        model.fit(context_X, context_y)
        
        # 计算残差 (Noise Profile)
        y_train_pred = model.predict(context_X)
        residuals = context_y - y_train_pred
        noise_std = np.std(residuals)
        
        # 逐步预测 (Autoregressive)
        current_input = context_X[-1].copy()
        predictions = []
        
        # 这里会比较慢，因为要循环 32000 次 (2秒 * 16k)
        # 为了演示速度，如果缺口太大，AR 可能会跑很久且发散。
        # 实际上 AR 并不适合修这么长的缺口，但为了“公平对比”展示其局限性，我们依然让它跑。
        for i in range(steps):
            pred = model.predict(current_input.reshape(1, -1))[0]
            
            # 注入纹理
            pred += np.random.normal(0, noise_std)
            
            predictions.append(pred)
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred
            
        return np.array(predictions)

    def restore(self):
        if self.signal is None: return
        
        gap_range = self.find_main_gap()
        if gap_range is None: return self.signal
        
        start, end = gap_range
        gap_len = end - start
        
        print(f"🤖 AR 正在尝试跨越 {gap_len} 个点的深渊 (这可能需要几秒钟)...")
        
        # --- 准备上下文数据 ---
        # 取缺口前后各 5000 个点来训练
        context_len = 5000 
        left_data = self.signal[max(0, start - context_len) : start]
        right_data = self.signal[end : min(len(self.signal), end + context_len)]
        
        def make_dataset(d):
            X, y = [], []
            if len(d) <= self.order: return np.array([]), np.array([])
            for j in range(len(d) - self.order):
                X.append(d[j : j + self.order])
                y.append(d[j + self.order])
            return np.array(X), np.array(y)

        X_left, y_left = make_dataset(left_data)
        X_right, y_right = make_dataset(right_data[::-1]) # 翻转右侧用于倒推
        
        # --- 双向预测 ---
        pred_fwd = np.zeros(gap_len)
        pred_bwd = np.zeros(gap_len)
        
        # 正向
        if len(X_left) > 0:
            print("  -> 正向预测中...")
            pred_fwd = self._train_predict_with_residuals(X_left, y_left, gap_len)
            
        # 反向
        if len(X_right) > 0:
            print("  <- 反向预测中...")
            pred_bwd = self._train_predict_with_residuals(X_right, y_right, gap_len)[::-1]
            
        # --- 融合 ---
        # 线性淡入淡出
        weights = np.linspace(1, 0, gap_len)
        if len(X_left) == 0: weights = np.zeros(gap_len)
        if len(X_right) == 0: weights = np.ones(gap_len)
        
        restored_gap = pred_fwd * weights + pred_bwd * (1 - weights)
        
        # 拼回去
        restored_signal = self.signal.copy()
        restored_signal[start:end] = restored_gap
        
        return restored_signal

    def save_result(self, audio):
        path = os.path.join(OUTPUT_DIR, "fixed_ar_gap.wav")
        # 防爆音
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, self.sr, (audio * 32767).astype(np.int16))
        print(f"💾 AR 修复完成: {path}")
        
        # 统一画图 (Inferno Specgram)
        plt.figure(figsize=(10, 4))
        plt.specgram(audio, NFFT=1024, Fs=self.sr, noverlap=512, cmap='inferno')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "spec_ar_gap.png"), bbox_inches='tight', pad_inches=0)
        print("🖼️ AR 图片已保存")

# --- 🏃‍♂️ 运行 ---
lab = ARFairGapInpainter(INPUT_FILE, order=100) # 增大阶数以应对长缺口
lab.load_damaged_data()
res = lab.restore()
if res is not None: lab.save_result(res)