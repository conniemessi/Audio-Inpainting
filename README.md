# Audio Inpainting: Classical and Deep Learning Methods

This repository contains the code for a comparative study of audio inpainting methods, ranging from classical signal processing to modern deep generative models. The accompanying report (`audio_inpainting_report.tex` in the parent directory) analyzes how different methods behave under several corruption scenarios (short gaps, random fragmentation, and long temporal holes).

An interactive web demo is available here: [https://a81cc3f8db098ee59c.gradio.live](https://a81cc3f8db098ee59c.gradio.live).

---

## 1. Repository Structure

- `vocals_accompaniment_10s.wav`  
  10-second vocal + accompaniment clip used as the base signal for all experiments.

- `generate_part1_data.py`  
  Generates Part 1 data: long real audio with random missing fragments and linear interpolation baseline.

- `generate_part2_data.py`  
  Generates Part 2 data: long real audio with a single 2-second gap and linear interpolation baseline.

- `linear_interp_part1.py`  
  Standalone linear interpolation restoration script for the Part 1 random-mask setting.

- `main1_gp.py`  
  Gaussian Process (GP) inpainting on short synthetic/real segments (Part 0).

- `main2_AR.py`  
  Bidirectional autoregressive (AR) inpainting on short segments (Part 0) with waveform visualization.

- `main3_AR_text.py`  
  AR with texture injection (residual-noise augmentation) for short-gaps in real speech (Part 0).

- `main3_AR_text_gap.py`  
  AR-based inpainting of the 2-second gap in Part 2, including gap detection and bidirectional prediction.

- `main3_AR_text_mask.py`  
  AR-based inpainting for the random-mask setting (Part 1), using automatic gap detection on the U-Net damaged audio.

- `main4_NMF.py`  
  NMF-based spectrogram inpainting for medium-length gaps on short segments.

- `main4_NMF_mask.py`  
  NMF inpainting for the random-mask case (Part 1), operating directly on the U-Net damaged signal.

- `main4_NMF_gap.py`  
  NMF inpainting for the 2-second gap (Part 2), using STFT-domain masks estimated from the corrupted waveform.

- `main5_UNet_mask.py`  
  Spectrogram U-Net training and inference for the random-mask scenario (Part 1). Produces long corrupted/restored audio and comparison spectrograms.

- `main5_UNet_gap.py`  
  Spectrogram U-Net inpainting for a shorter 2D mask in the STFT domain (local gaps), mainly used for visualization.

- `main_gan_gap.py`  
  GAN-based spectrogram inpainting for the 2-second gap (Part 2), including generator and discriminator architectures.

- `main_diffusion_gap.py`  
  Riffusion/Stable Diffusion based inpainting in the spectrogram-image domain for the 2-second gap (Part 2).

- `demo_assets/`  
  Contains precomputed WAV files and spectrogram PNGs for the Gradio demo (`demo.py`), organized per scenario.

- `demo.py`  
  Gradio interface that presents audio + spectrogram comparisons for all methods and scenarios.

---

## 2. Installation

Create a Python environment (Python 3.9+ recommended) and install the required libraries:

```bash
pip install numpy scipy matplotlib scikit-learn torchaudio torch gradio diffusers transformers accelerate
```

You may also need to install `ffmpeg` on your system for audio I/O (e.g., via `brew install ffmpeg` on macOS).

Some scripts (e.g., `main5_UNet_mask.py`, `main_gan_gap.py`, `main_diffusion_gap.py`) are GPU-friendly and will run much faster with a CUDA-capable device, but can also run on CPU with longer runtimes.

---

## 3. Reproducing the Experiments

All experiments assume the base audio file `vocals_accompaniment_10s.wav` is present in the root of this repository.

### 3.1 Part 0: Short Segments (Synthetic + Speech)

Gaussian Process, AR, AR+Texture, and NMF on short segments:

```bash
python main1_gp.py        # GP on synthetic / short real segments
python main2_AR.py        # Bidirectional AR
python main3_AR_text.py   # AR with texture injection
python main4_NMF.py       # NMF on short segments
```

These scripts typically save waveform visualizations and/or WAV files in the working directory or `demo_assets/part0` (depending on how you configured paths).

### 3.2 Part 1: Random Fragmentation (Random Mask)

Generate damaged audio and linear interpolation baseline:

```bash
python generate_part1_data.py     # creates damaged_random.wav and baselines
python linear_interp_part1.py     # standalone linear interpolation restoration
```

Run AR, NMF, and U-Net restorations:

```bash
python main3_AR_text_mask.py      # AR inpainting on random-mask audio
python main4_NMF_mask.py          # NMF inpainting on random-mask audio
python main5_UNet_mask.py         # U-Net inpainting (random-mask training/inference)
```

These scripts write corrupted/restored WAV files and spectrograms into `demo_assets/part1/`.

### 3.3 Part 2: 2-second Temporal Gap

Generate the 2-second gap example and linear baseline:

```bash
python generate_part2_data.py     # creates damaged_gap.wav and fixed_linear_gap.wav
```

Run AR, NMF, GAN and Diffusion restorations:

```bash
python main3_AR_text_gap.py       # AR inpainting for 2s gap
python main4_NMF_gap.py           # NMF inpainting for 2s gap
python main_gan_gap.py            # GAN-based spectrogram inpainting
python main_diffusion_gap.py      # Diffusion (Riffusion/Stable Diffusion) inpainting
```

Outputs are written to `demo_assets/part2/` (e.g., `fixed_ar_gap.wav`, `fixed_nmf_gap.wav`, `fixed_gan_gap.wav`, `fixed_riffusion_gap.wav` and corresponding spectrogram PNGs).

---

## 4. Gradio Demo

To run the interactive web demo locally:

```bash
cd Audio-Inpainting
python demo.py
```

This will launch a Gradio interface (by default at `http://127.0.0.1:7860`) with two tabs:

- **Scene 1: Random Fragments** – compare Linear, AR, NMF, U-Net and ground truth.
- **Scene 2: 2s Temporal Hole** – compare Linear, AR, NMF, GAN, Diffusion and ground truth.

Each selection shows:

- An audio player for direct listening.
- The corresponding spectrogram image.
- A short technical commentary summarizing audible and visual differences.

You can also access a hosted version of the demo at:  
[https://a81cc3f8db098ee59c.gradio.live](https://a81cc3f8db098ee59c.gradio.live)

> Note: The hosted URL is time-limited (Gradio share links typically expire after 72 hours). If it is no longer active, please run `demo.py` locally.

---

## 5. Reference

For a detailed mathematical and experimental discussion of all methods, see the LaTeX report (`audio_inpainting_report.tex`) in the parent project directory. It describes the problem formulation, methodology, experimental setup, and conclusions in a scientific writing style, with references to Gaussian Processes, AR/NMF models, U-Net, GANs, and diffusion-based inpainting.


