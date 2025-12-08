import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import interpolate
import os

# 配置
INPUT_FILE = "vocals_accompaniment_10s.wav"
OUTPUT_DIR = "demo_assets/part2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_spectrogram(audio, sr, save_name):
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"🖼️ 图片已保存: {save_name}")

def save_wav(audio, sr, save_name):
    path = os.path.join(OUTPUT_DIR, save_name)
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))
    print(f"💾 音频已保存: {save_name}")

def process_part2():
    print("--- 正在生成 Part 2 (2s 大缺口) 素材 ---")
    
    # 1. 加载音频
    sr, data = wavfile.read(INPUT_FILE)
    if len(data.shape) > 1: data = data.mean(axis=1)
    data = data.astype(np.float32) / np.max(np.abs(data))
    
    # 确保只取前10秒，与 Part 1 保持一致
    n_samples = 10 * sr
    if len(data) > n_samples: data = data[:n_samples]
    
    # 2. 制造 2秒 缺口 (正中间)
    gap_center = len(data) // 2
    gap_half_len = int(1.0 * sr) # 半径1秒，直径2秒
    gap_start = gap_center - gap_half_len
    gap_end = gap_center + gap_half_len
    
    corrupted = data.copy()
    corrupted[gap_start:gap_end] = 0
    
    save_wav(corrupted, sr, "damaged_gap.wav")
    save_spectrogram(corrupted, sr, "spec_damaged_gap.png")
    
    # 3. 线性插值修复 (Linear Baseline)
    print("📏 正在执行线性插值...")
    linear_fixed = corrupted.copy()
    # 取缺口两端点
    y_start = data[gap_start-1]
    y_end = data[gap_end]
    # 画直线
    fill_values = np.linspace(y_start, y_end, gap_end - gap_start)
    linear_fixed[gap_start:gap_end] = fill_values
    
    save_wav(linear_fixed, sr, "fixed_linear_gap.wav")
    save_spectrogram(linear_fixed, sr, "spec_linear_gap.png")
    
    # 4. 保存原始 (如果 Part 1 没跑，这里兜底生成一下)
    if not os.path.exists(os.path.join(OUTPUT_DIR, "original.wav")):
        save_wav(data, sr, "original.wav")
        save_spectrogram(data, sr, "spec_original.png")

if __name__ == "__main__":
    process_part2()