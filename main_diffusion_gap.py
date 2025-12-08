import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
from scipy.io import wavfile
import os

INPUT_FILE = "demo_assets/part2/damaged_gap.wav"
OUTPUT_DIR = "demo_assets/part2"

class RiffusionFairInpainter:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "riffusion/riffusion-model-v1",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)
        self.pipe.safety_checker = None
        
    def wav_to_spectrogram(self, waveform):
        transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, power=2.0).to(self.device)
        spectrogram = transform(waveform.to(self.device))
        log_spectrogram = 20.0 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20.0
        log_spectrogram = torch.clamp(log_spectrogram, min=-100) 
        return log_spectrogram

    def spectrogram_to_image(self, spectrogram):
        self.spec_min, self.spec_max = spectrogram.min(), spectrogram.max()
        data = (spectrogram - self.spec_min) / (self.spec_max - self.spec_min)
        data = (data * 255).byte().cpu().numpy()[0]
        image = Image.fromarray(np.flipud(data)).convert("RGB")
        return image

    def image_to_spectrogram(self, image):
        data = np.array(image.convert("L"))
        data = np.flipud(data).copy()
        tensor = torch.tensor(data).float().to(self.device) / 255.0
        spectrogram = tensor * (self.spec_max - self.spec_min) + self.spec_min
        return torch.pow(10, (spectrogram + 20.0) / 20.0)

    def inpaint(self):
        if not os.path.exists(INPUT_FILE): return
        waveform, sr = torchaudio.load(INPUT_FILE)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 1. 转图片
        log_spec = self.wav_to_spectrogram(waveform)
        image = self.spectrogram_to_image(log_spec)
        
        # 2. 自动生成 Mask (基于图片的黑色区域)
        # 我们的缺口是 0，转成 Log 谱图是极小值 (-100)，归一化后是黑色 (0)
        # 将图片转灰度，找像素值极低的地方
        gray = image.convert("L")
        # 阈值设为 10 (0-255)，捕捉纯黑区域
        mask_data = np.array(gray)
        mask_array = np.where(mask_data < 10, 255, 0).astype(np.uint8) 
        mask_image = Image.fromarray(mask_array)
        
        # 3. Diffusion Inpainting
        print("☢️ Riffusion 正在检测并修复黑色缺口...")
        img_resized = image.resize((512, 512))
        mask_resized = mask_image.resize((512, 512))
        
        inpainted_resized = self.pipe(
            prompt="high quality audio, ambient sound, seamless transition",
            image=img_resized,
            mask_image=mask_resized,
            num_inference_steps=50,
            strength=1.0
        ).images[0]
        
        inpainted_image = inpainted_resized.resize(image.size)
        
        # 4. 转回音频
        linear_spec = self.image_to_spectrogram(inpainted_image)
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=2048, hop_length=512, power=1.0).to(self.device)
        restored_waveform = griffin_lim(linear_spec.unsqueeze(0))
        
        # 保存
        path = os.path.join(OUTPUT_DIR, "fixed_riffusion_gap.wav")
        sig = restored_waveform.cpu().squeeze().numpy()
        sig = np.clip(sig, -1.0, 1.0)
        wavfile.write(path, sr, (sig * 32767).astype(np.int16))
        
        # 统一画图 (使用 plt.specgram 而不是 SD 的图)
        plt.figure(figsize=(10, 4))
        plt.specgram(sig, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "spec_riffusion_gap.png"), bbox_inches='tight', pad_inches=0)
        print("💾 Riffusion 修复完毕")

# 运行
lab = RiffusionFairInpainter()
lab.inpaint()