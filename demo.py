import gradio as gr
import os

# --- 1. Configure media paths (core config section) ---
# Each model maps to a dict with "audio" and "image" paths
MEDIA_FILES = {
    "part1": {  # Scene 1: random missing fragments
        "🤕 Damaged (Random Mask)": {
            "audio": "demo_assets/part1/damaged_random.wav",
            "image": "demo_assets/part1/spec_damaged_random.png"
        },
        "📏 Linear Interpolation": {
            "audio": "demo_assets/part1/fixed_linear_random.wav",
            "image": "demo_assets/part1/spec_linear_random.png"
        },
        "📈 Autoregressive (AR)": {
            "audio": "demo_assets/part1/fixed_ar_random.wav",
            "image": "demo_assets/part1/spec_ar_random.png"
        },
        "🧩 Spectral Factorization (NMF)": {
            "audio": "demo_assets/part1/fixed_nmf_random.wav",
            "image": "demo_assets/part1/spec_nmf_random.png"
        },
        "🧠 Deep Learning (U-Net)": {
            "audio": "demo_assets/part1/dl_long_restored.wav",
            "image": "demo_assets/part1/spec_dl_restored.png"
        },
        "✅ Ground Truth": {
            "audio": "demo_assets/part1/original.wav",
            "image": "demo_assets/part1/spec_original.png"
        }
    },
    "part2": {  # Scene 2: long temporal gap
        "🕳️ Damaged (2s Gap)": {
            "audio": "demo_assets/part2/damaged_gap.wav",
            "image": "demo_assets/part2/spec_damaged_gap.png"
        },
        "📏 Linear Interpolation": {
            "audio": "demo_assets/part2/fixed_linear_gap.wav",
            "image": "demo_assets/part2/spec_linear_gap.png"
        },
        "📈 Autoregressive (AR)": {
            "audio": "demo_assets/part2/fixed_ar_gap.wav",
            "image": "demo_assets/part2/spec_ar_gap.png"
        },
        "🧩 Spectral Factorization (NMF)": {
            "audio": "demo_assets/part2/fixed_nmf_gap.wav",
            "image": "demo_assets/part2/spec_nmf_gap.png"
        },
        "🎨 Generative Adversarial Network (GAN)": {
            "audio": "demo_assets/part2/fixed_gan_gap.wav",
            "image": "demo_assets/part2/spec_gan_gap.png"
        },
        "☢️ Diffusion Model (Riffusion)": {
            "audio": "demo_assets/part2/fixed_riffusion_gap.wav",
            "image": "demo_assets/part2/spec_riffusion_gap.png"
        },
        "✅ Ground Truth": {
            "audio": "demo_assets/part2/original.wav",
            "image": "demo_assets/part2/spec_original.png"
        }
    }
}

# Helper: safely get media paths
def get_media_paths(scenario, model_name):
    data = MEDIA_FILES[scenario].get(model_name, {})
    audio_path = data.get("audio")
    image_path = data.get("image")
    
    # 检查文件是否存在，不存在返回 None (Gradio 会显示空白)
    final_audio = audio_path if audio_path and os.path.exists(audio_path) else None
    final_image = image_path if image_path and os.path.exists(image_path) else None
    return final_audio, final_image

# --- 2. UI layout ---

desc_header = """
# 🕵️‍♂️ Signal Restorer: Audio Inpainting Showcase
Welcome to the restoration lab. Use the tabs below to switch scenes and **listen + see**
how different models repair damaged audio.
Spectrograms reveal the missing regions and how each method reconstructs fine details.
"""

with gr.Blocks() as demo:
    gr.Markdown(desc_header)

    with gr.Tabs():
        # --- TAB 1: random fragments ---
        with gr.TabItem("🌦️ Scene 1: Random Fragments"):
            with gr.Row():
                with gr.Column(scale=1):
                    radio_1 = gr.Radio(
                        choices=list(MEDIA_FILES["part1"].keys()),
                        value="🤕 Damaged (Random Mask)",
                        label="Choose method"
                    )
                    desc_1 = gr.Textbox(label="Technical commentary", lines=4)
                
                with gr.Column(scale=2):
                    audio_1 = gr.Audio(label="👂 Audio preview", type="filepath")
                    img_1 = gr.Image(label="👁️ Spectrogram (texture details)", type="filepath", interactive=False)

            # 更新逻辑
            def update_part1(model):
                comments = {
                    "🤕 Damaged (Random Mask)": (
                        "[Listening] Strong artifacts and dropouts.\n"
                        "[Visual] Many vertical black bars in the spectrogram, "
                        "indicating missing time segments."
                    ),
                    "📏 Linear Interpolation": (
                        "[Listening] Gaps are filled but sound is muffled and unnatural.\n"
                        "[Visual] Missing parts are connected by straight, smooth bands, "
                        "losing fine time–frequency texture."
                    ),
                    "📈 Autoregressive (AR)": (
                        "[Listening] Short gaps are reconstructed with clearer detail than linear.\n"
                        "[Visual] Spectrogram lines across gaps look more coherent and structured."
                    ),
                    "🧩 Spectral Factorization (NMF)": (
                        "[Listening] Harmonic structure is preserved but may sound slightly synthetic.\n"
                        "[Visual] Spectrogram shows smoother, template-like components filling the gaps."
                    ),
                    "🧠 Deep Learning (U-Net)": (
                        "[Listening] Reconstruction is close to natural.\n"
                        "[Visual] U-Net restores rich horizontal textures; "
                        "it is hard to see obvious repair seams."
                    ),
                    "✅ Ground Truth": (
                        "Reference clean signal with natural harmonics and textures."
                    ),
                }
                a_path, i_path = get_media_paths("part1", model)
                return a_path, comments.get(model, ""), i_path
            
            # 绑定事件：输出增加了一个 img_1
            radio_1.change(update_part1, inputs=radio_1, outputs=[audio_1, desc_1, img_1])

        # --- TAB 2: long gap ---
        with gr.TabItem("🕳️ Scene 2: 2s Temporal Hole"):
            with gr.Row():
                with gr.Column(scale=1):
                    radio_2 = gr.Radio(
                        choices=list(MEDIA_FILES["part2"].keys()),
                        value="🕳️ Damaged (2s Gap)",
                        label="Choose method"
                    )
                    desc_2 = gr.Textbox(label="Technical commentary", lines=4)
                
                with gr.Column(scale=2):
                    audio_2 = gr.Audio(label="👂 Audio preview", type="filepath")
                    img_2 = gr.Image(label="👁️ Spectrogram (hallucination ability)", type="filepath", interactive=False)
            
            # 更新逻辑
            def update_part2(model):
                comments = {
                    "🕳️ Damaged (2s Gap)": (
                        "[Listening] A long silent hole appears in the middle.\n"
                        "[Visual] A large pure-black region in the center of the spectrogram, "
                        "showing complete information loss."
                    ),
                    "📏 Linear Interpolation": (
                        "[Listening] The hole is filled but the transition is dull and smeared.\n"
                        "[Visual] The gap becomes smooth, low-detail bands that ignore complex patterns."
                    ),
                    "📈 Autoregressive (AR)": (
                        "[Listening] Temporal continuity is better, but long-term structure can drift.\n"
                        "[Visual] Lines extend across the gap, yet some high-level patterns are inconsistent."
                    ),
                    "🧩 Spectral Factorization (NMF)": (
                        "[Listening] Reasonable timbre but can sound repetitive.\n"
                        "[Visual] The gap is filled with a few repeating spectral templates."
                    ),
                    "🎨 Generative Adversarial Network (GAN)": (
                        "[Listening] The gap is filled with plausible content but can be a bit rough.\n"
                        "[Visual] The black region is replaced, but textures may look noisy or irregular."
                    ),
                    "☢️ Diffusion Model (Riffusion)": (
                        "[Listening] Very natural, with smooth transitions into and out of the gap.\n"
                        "[Visual] The model hallucinates highly detailed, realistic time–frequency structure."
                    ),
                    "✅ Ground Truth": (
                        "Reference clean signal. Compare how close each model comes to this target."
                    ),
                }
                a_path, i_path = get_media_paths("part2", model)
                return a_path, comments.get(model, ""), i_path

            # Bind events
            radio_2.change(update_part2, inputs=radio_2, outputs=[audio_2, desc_2, img_2])

# 启动
# demo.launch() # 普通启动
demo.launch(share=True) # 生成一个公开链接，可以发给别人看 (72小时有效)