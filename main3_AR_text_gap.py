import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.linear_model import Ridge
import os

INPUT_FILE = "demo_assets/part2/damaged_gap.wav"
OUTPUT_DIR = "demo_assets/part2"

class ARFairGapInpainter:
    def __init__(self, filename, order=100):
        """
        AR inpainting for large gaps
        order: AR model order (increased for larger gaps)
        """
        self.filename = filename
        self.order = order
        self.sr = None
        self.signal = None
        
    def load_damaged_data(self):
        if not os.path.exists(self.filename):
            print("❌ File not found! Please run generate_part2_data.py first")
            return
            
        self.sr, data = wavfile.read(self.filename)
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        self.signal = data
        print(f"🎤 AR loaded damaged audio: {len(self.signal)} samples")

    def find_main_gap(self):
        """Detect the 2-second large gap"""
        threshold = 1e-4
        is_gap = (np.abs(self.signal) < threshold)
        
        gap_indices = np.where(is_gap)[0]
        
        if len(gap_indices) == 0:
            print("⚠️ No gap detected!")
            return None
        
        start = gap_indices[0]
        end = gap_indices[-1] + 1
        
        print(f"🔍 Detected gap range: {start} -> {end} (length: {end-start} samples)")
        return (start, end)

    def _train_predict_with_residuals(self, context_X, context_y, steps):
        """AR prediction with texture injection"""
        if len(context_X) < 10: return np.zeros(steps)
        
        model = Ridge(alpha=0.5)
        model.fit(context_X, context_y)
        
        y_train_pred = model.predict(context_X)
        residuals = context_y - y_train_pred
        noise_std = np.std(residuals)
        
        current_input = context_X[-1].copy()
        predictions = []
        
        for i in range(steps):
            pred = model.predict(current_input.reshape(1, -1))[0]
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
        
        print(f"🤖 AR is attempting to bridge {gap_len} points (may take several seconds)...")
        
        # Prepare context data (use 5000 samples before/after gap for training)
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
        X_right, y_right = make_dataset(right_data[::-1])
        
        pred_fwd = np.zeros(gap_len)
        pred_bwd = np.zeros(gap_len)
        
        if len(X_left) > 0:
            print("  -> Forward prediction...")
            pred_fwd = self._train_predict_with_residuals(X_left, y_left, gap_len)
            
        if len(X_right) > 0:
            print("  <- Backward prediction...")
            pred_bwd = self._train_predict_with_residuals(X_right, y_right, gap_len)[::-1]
            
        # Cross-fade blending
        weights = np.linspace(1, 0, gap_len)
        if len(X_left) == 0: weights = np.zeros(gap_len)
        if len(X_right) == 0: weights = np.ones(gap_len)
        
        restored_gap = pred_fwd * weights + pred_bwd * (1 - weights)
        
        restored_signal = self.signal.copy()
        restored_signal[start:end] = restored_gap
        
        return restored_signal

    def save_result(self, audio):
        path = os.path.join(OUTPUT_DIR, "fixed_ar_gap.wav")
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, self.sr, (audio * 32767).astype(np.int16))
        print(f"💾 AR restoration complete: {path}")
        
        plt.figure(figsize=(10, 4))
        plt.specgram(audio, NFFT=1024, Fs=self.sr, noverlap=512, cmap='inferno')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "spec_ar_gap.png"), bbox_inches='tight', pad_inches=0)
        print("🖼️ AR spectrogram saved")

lab = ARFairGapInpainter(INPUT_FILE, order=100)
lab.load_damaged_data()
res = lab.restore()
if res is not None: lab.save_result(res)
