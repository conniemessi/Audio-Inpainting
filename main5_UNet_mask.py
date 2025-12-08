import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from datetime import datetime

# --- 1. 定义 U-Net 网络架构 ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器 (Encoder): 下采样，提取特征
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(32, 64)

        # 解码器 (Decoder): 上采样，恢复图像
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32) # 输入是 64 因为拼接了 skip connection
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)
        
        # 输出层
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # 瓶颈层
        b = self.bottleneck(p2)
        
        # 解码路径 (带跳跃连接 Skip Connections)
        d2 = self.up2(b)
        # 调整尺寸以防 padding 导致的不匹配
        d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:]) 
        d2 = torch.cat((e2, d2), dim=1) # 拼接!
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat((e1, d1), dim=1) # 拼接!
        d1 = self.dec1(d1)
        
        return self.final(d1)

# --- 2. 数据处理与训练流程 ---
class DLInpaintingLab:
    def __init__(self, filename, duration=10.0): # 10秒长音频
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # 加载音频
        waveform, sr = torchaudio.load(filename)
        
        # 1. 强制转单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 确保音频够长，不够长就这就取全部
        target_len = int(duration * sr)
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            print(f"⚠️ 音频长度仅为 {waveform.shape[1]/sr:.2f}s (小于请求的 {duration}s)")
            
        self.sr = sr
        self.original_length = waveform.shape[1]
        
        # Window & STFT
        self.n_fft = 1024
        self.window = torch.hann_window(self.n_fft).to(self.device)
        
        waveform = waveform.to(self.device)
        stft = torch.stft(waveform, self.n_fft, hop_length=256, 
                          window=self.window, return_complex=True)
        
        self.magnitude = torch.abs(stft)
        self.phase = torch.angle(stft)
        
        # 归一化
        self.mag_max = self.magnitude.max()
        self.magnitude_norm = self.magnitude / self.mag_max
        
        # --- 🛠️ 升级点：随机遮罩生成 (Smart Random Masking) ---
        self.mask = self._create_random_mask(self.magnitude_norm.shape)
        
        # Tensors
        self.input_mag = self.magnitude_norm * self.mask
        self.target_mag = self.magnitude_norm
        
        self.input_tensor = self.input_mag.unsqueeze(0)
        self.target_tensor = self.target_mag.unsqueeze(0)
        self.mask_tensor = self.mask.unsqueeze(0)

        self.model = SimpleUNet().to(self.device)
        self.restored_waveform = None
        self.corrupted_waveform = None
        
        # 创建带时间戳的输出文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"main6_results/output_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 输出文件夹: {self.output_dir}")
        
        # 预先创建损坏的音频波形用于对比
        self._create_corrupted_waveform()

    def _create_random_mask(self, shape, mask_ratio=0.3, max_time_mask=30):
        """
        生成随机的遮罩，而不是单一的大缺口。
        mask_ratio: 遮住总面积的百分比
        max_time_mask: 每个小缺口最大的时间宽度 (帧数)
        """
        _, freq, time = shape
        mask = torch.ones(shape).to(self.device)
        
        # 简单的时间轴随机遮罩 (Time Masking)
        # 这种方式类似 SpecAugment
        num_mask_segments = int(time * mask_ratio / max_time_mask * 2) # 估算需要多少个遮罩块
        
        for _ in range(num_mask_segments):
            t_len = np.random.randint(5, max_time_mask) # 随机宽度 5~30 帧
            t_start = np.random.randint(0, time - t_len)
            mask[:, :, t_start : t_start + t_len] = 0
            
        return mask

    # def _create_corrupted_waveform(self):
    #     """从损坏的输入谱图重建音频波形"""
    #     # 使用损坏的幅度谱和原始相位重建
    #     corrupted_mag = self.input_mag * self.mag_max
    #     stft_corrupted = torch.polar(corrupted_mag, self.phase)
        
    #     self.corrupted_waveform = torch.istft(
    #         stft_corrupted,
    #         self.n_fft,
    #         hop_length=256,
    #         window=self.window,
    #         length=self.original_length
    #     )
    
    def _create_corrupted_waveform(self):
        """从损坏的输入谱图重建音频波形，并保存为基准测试文件"""
        # 使用损坏的幅度谱和原始相位重建
        corrupted_mag = self.input_mag * self.mag_max
        stft_corrupted = torch.polar(corrupted_mag, self.phase)
        
        self.corrupted_waveform = torch.istft(
            stft_corrupted,
            self.n_fft,
            hop_length=256,
            window=self.window,
            length=self.original_length
        )
        
        # --- ✨ 新增：保存为公共基准文件 (覆盖旧的) ---
        os.makedirs("demo_assets", exist_ok=True)
        common_path = "demo_assets/damaged_random.wav"
        
        sig = self.corrupted_waveform.squeeze().cpu().numpy()
        sig = np.clip(sig, -1.0, 1.0)
        wavfile.write(common_path, self.sr, (sig * 32767).astype(np.int16))
        print(f"👑 [基准生成] U-Net 已将受损题目发布到: {common_path}")
        
        # 同时保存一张对应的 Mask 图片，方便 Main4 NMF 参考（可选，但推荐）
        plt.figure(figsize=(10, 4))
        plt.specgram(sig, NFFT=1024, Fs=self.sr, noverlap=512, cmap='inferno')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig("demo_assets/spec_damaged_random.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    def train_and_predict(self, epochs=600): # 10s数据量大，epoch可以适当减少，或者保持
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"🧠 开始训练 U-Net (长音频随机遮罩版, {epochs} 轮)...")
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.input_tensor)
            
            # --- 关键：Loss 只计算被遮住的部分 (Hard Mining) ---
            # 这样模型会专注于修复未知区域，而不是复读已知区域
            loss = criterion(output * (1-self.mask_tensor), self.target_tensor * (1-self.mask_tensor))
            # 如果觉得学得太慢，可以用全局 loss: loss = criterion(output, self.target_tensor)
            
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
                
        # 推理
        self.model.eval()
        with torch.no_grad():
            predicted_mag_norm = self.model(self.input_tensor)
            
        final_mag_norm = self.input_mag.to(self.device) + predicted_mag_norm * (1 - self.mask_tensor)
        final_mag = final_mag_norm.squeeze(0) * self.mag_max
        
        stft_reconstructed = torch.polar(final_mag, self.phase)
        
        self.restored_waveform = torch.istft(
            stft_reconstructed, 
            self.n_fft, 
            hop_length=256, 
            window=self.window, 
            length=self.original_length 
        )

    def visualize(self):
        plt.figure(figsize=(15, 6)) # 画布大一点
        
        plt.subplot(1, 3, 1)
        plt.title("Input (Randomly Masked)")
        plt.imshow(self.input_mag.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
        plt.axis('off')

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.input_tensor).squeeze(0).squeeze(0).cpu()
        
        plt.subplot(1, 3, 2)
        plt.title("U-Net Prediction")
        plt.imshow(pred.numpy(), aspect='auto', origin='lower', cmap='inferno')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(self.target_mag.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存图片为 PNG 和 PDF
        png_path = os.path.join(self.output_dir, "spectrogram_comparison.png")
        pdf_path = os.path.join(self.output_dir, "spectrogram_comparison.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"📊 可视化图片已保存: {png_path}")
        print(f"📊 可视化图片已保存: {pdf_path}")
        
        plt.show()

    def save_wav(self):
        # 保存损坏的音频（修复前）
        if self.corrupted_waveform is not None:
            sig_corrupted = self.corrupted_waveform.squeeze().cpu().numpy()
            sig_corrupted = np.clip(sig_corrupted, -0.99, 0.99)
            corrupted_path = os.path.join(self.output_dir, "dl_long_corrupted.wav")
            wavfile.write(corrupted_path, self.sr, (sig_corrupted * 32767).astype(np.int16))
            print(f"💾 损坏的音频已保存: {corrupted_path}")
        
        # 保存修复后的音频
        sig = self.restored_waveform.squeeze().cpu().numpy()
        sig = np.clip(sig, -0.99, 0.99)
        restored_path = os.path.join(self.output_dir, "dl_long_restored.wav")
        wavfile.write(restored_path, self.sr, (sig * 32767).astype(np.int16))
        print(f"💾 长音频修复完成: {restored_path}")
        
# --- 🏃‍♂️ 运行 ---
# ⚠️ 注意：如果你没有 GPU，这可能需要跑一两分钟。
lab = DLInpaintingLab(filename="vocals_accompaniment_10s.wav", duration=10)
lab.train_and_predict(epochs=400) # 训练 600 轮确保过拟合
lab.visualize()
lab.save_wav()