import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

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
    def __init__(self, filename, duration=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # 加载音频
        waveform, sr = torchaudio.load(filename)
        
        # 1. 强制转单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform[:, :int(duration*sr)] # 截取
        self.sr = sr
        
        # --- 🛠️ 修复点 A: 保存原始长度 ---
        self.original_length = waveform.shape[1]
        
        # 2. 创建并保存 Window (解决警告 + 保证逆变换准确)
        self.n_fft = 1024
        self.window = torch.hann_window(self.n_fft).to(self.device)
        
        # STFT
        waveform = waveform.to(self.device)
        stft = torch.stft(waveform, self.n_fft, hop_length=256, 
                          window=self.window, return_complex=True)
        
        self.magnitude = torch.abs(stft)
        self.phase = torch.angle(stft)
        
        # 归一化
        self.mag_max = self.magnitude.max()
        self.magnitude_norm = self.magnitude / self.mag_max
        
        # Mask
        _, freq, time = self.magnitude.shape
        gap_start = int(time * 0.4)
        gap_end = int(time * 0.6)
        self.mask = torch.ones_like(self.magnitude_norm)
        self.mask[:, :, gap_start:gap_end] = 0
        
        # Tensors
        self.input_mag = self.magnitude_norm * self.mask
        self.target_mag = self.magnitude_norm
        
        self.input_tensor = self.input_mag.unsqueeze(0)
        self.target_tensor = self.target_mag.unsqueeze(0)
        self.mask_tensor = self.mask.unsqueeze(0)

        self.model = SimpleUNet().to(self.device)
        self.restored_waveform = None
        self.corrupted_waveform = None
        
        # 预先创建损坏的音频波形用于对比
        self._create_corrupted_waveform()

    def _create_corrupted_waveform(self):
        """从损坏的输入谱图重建音频波形"""
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

    def train_and_predict(self, epochs=600):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"🧠 开始训练 U-Net (过拟合演示，共 {epochs} 轮)...")
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.input_tensor)
            loss = criterion(output, self.target_tensor)
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
        
        # iSTFT 重建
        stft_reconstructed = torch.polar(final_mag, self.phase)
        
        # --- 🚨 关键修改看这里 🚨 ---
        # 这一行必须改！不能用 waveform.shape[1]，要用 self.original_length
        self.restored_waveform = torch.istft(
            stft_reconstructed, 
            self.n_fft, 
            hop_length=256, 
            window=self.window, 
            length=self.original_length  # <--- 这里改成了 self.original_length
        )

    def visualize(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Input (Corrupted)")
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
        plt.show()

    def save_wav(self):
        # 保存损坏的音频（修复前）
        if self.corrupted_waveform is not None:
            sig_corrupted = self.corrupted_waveform.squeeze().cpu().numpy()
            sig_corrupted = np.clip(sig_corrupted, -0.99, 0.99)
            wavfile.write("dl_corrupted.wav", self.sr, (sig_corrupted * 32767).astype(np.int16))
            print("💾 损坏的音频已保存: dl_corrupted.wav")
        
        # 保存修复后的音频
        sig = self.restored_waveform.squeeze().cpu().numpy()
        sig = np.clip(sig, -0.99, 0.99)
        wavfile.write("dl_restored.wav", self.sr, (sig * 32767).astype(np.int16))
        print("💾 深度学习修复后的音频已保存: dl_restored.wav")

# --- 🏃‍♂️ 运行 ---
# ⚠️ 注意：如果你没有 GPU，这可能需要跑一两分钟。
lab = DLInpaintingLab(filename="vocals_accompaniment_10s.wav", duration=10)
lab.train_and_predict(epochs=600) # 训练 600 轮确保过拟合
lab.visualize()
lab.save_wav()