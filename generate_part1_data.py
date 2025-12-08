import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import interpolate
import os

# 配置
INPUT_FILE = "vocals_accompaniment_10s.wav" # 确保你有这个文件
OUTPUT_DIR = "demo_assets"
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

def create_random_mask(n_samples, mask_ratio=0.3, max_gap_len=400):
    """生成随机遮罩 (1=保留, 0=丢失)"""
    mask = np.ones(n_samples, dtype=bool)
    num_gaps = int(n_samples * mask_ratio / max_gap_len * 2)
    
    for _ in range(num_gaps):
        gap_len = np.random.randint(50, max_gap_len) # 随机长度
        gap_start = np.random.randint(0, n_samples - gap_len)
        mask[gap_start : gap_start + gap_len] = 0
    return mask

def process_part1():
    print("--- 正在生成 Part 1 (随机碎片) 素材 ---")
    
    # 1. 加载音频
    sr, data = wavfile.read(INPUT_FILE)
    if len(data.shape) > 1: data = data.mean(axis=1)
    data = data.astype(np.float32) / np.max(np.abs(data))
    
    # 2. 制造随机损伤
    mask = create_random_mask(len(data), mask_ratio=0.25) # 25% 丢失
    corrupted = data.copy()
    corrupted[~mask] = 0 # 丢失部分置为 0
    
    save_wav(corrupted, sr, "damaged_random.wav")
    save_spectrogram(corrupted, sr, "spec_damaged_random.png")
    
    # 3. 线性插值修复 (Linear Interpolation)
    # 核心逻辑：利用 np.interp 一次性填补所有空洞
    print("📏 正在执行线性插值...")
    x_all = np.arange(len(data))
    x_valid = x_all[mask]      # 已知点的 x
    y_valid = corrupted[mask]  # 已知点的 y
    
    # 在未知点位置进行插值
    linear_fixed = data.copy() # 先复制一份
    # interp(需要预测的x, 已知的x, 已知的y)
    linear_fixed[~mask] = np.interp(x_all[~mask], x_valid, y_valid)
    
    save_wav(linear_fixed, sr, "fixed_linear_random.wav")
    save_spectrogram(linear_fixed, sr, "spec_linear_random.png")
    
    # 4. 保存原始对比
    save_wav(data, sr, "original.wav")
    save_spectrogram(data, sr, "spec_original.png")

    save_wav(data, sr, "original.wav")
    save_spectrogram(data, sr, "spec_original.png")

if __name__ == "__main__":
    process_part1()