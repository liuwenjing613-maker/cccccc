# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script for ANC Models in Custom Scenarios
Integrates forward inference with the official EvaluationMetrics system.
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf

# Import model architecture and secondary path processing function
from model import TimeDomainANC
from dataset import apply_dynamic_path

# ==========================================
# 0. Import Evaluation Code
# ==========================================
# Assumes the evaluation codebase is located in the adjacent 'EvaluationMetrics' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "EvaluationMetrics")
if not os.path.exists(EVAL_DIR):
    raise FileNotFoundError(f"Directory {EVAL_DIR} not found. Please ensure the evaluation codebase is present.")
sys.path.append(EVAL_DIR)

try:
    from anc_audio_analysis import analyze_anc_audio, print_analysis_results
    from Model_evaluation import count_model_complexity
except ImportError as e:
    print(f"Failed to import evaluation scripts. Please check the filenames in the EvaluationMetrics directory. Error: {e}")
    sys.exit(1)


# ==========================================
# 1. User Configuration Area (Update with actual paths)
# ==========================================
# Model and Global Configuration
MODEL_WEIGHTS_PATH = "CCF_DEEPANC_2026-main/EvaluationMetrics/anc_best_model.pth"  # Path to the trained network weights
MODEL_WEIGHTS_PATH = os.environ.get(
    "ANC_CKPT",
    os.path.join(
        BASE_DIR,
        "outputs_e0",
        "seed_2027",
        "checkpoints",
        "best.pt",
    ),
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 48000                # Sampling rate
SEGMENT_SEC = 10.0        # Total duration for testing (first 5s: ANC OFF, last 5s: ANC ON)
ANC_TURN_ON_TIME = 5.0    # Timestamp to activate ANC (seconds)

# Path Configuration
SH_NPY_PATH = "/home/cha_ccf/code/CCF_DEEPANC_2026/dataset/sh.npy"     # Secondary acoustic path array file

REF_WAV_1 = os.path.join(BASE_DIR, "dataset", "NOISE", "车载.wav")
EXP_WAV_1 = os.path.join(
    BASE_DIR,
    "dataset",
    "EXPECTED_NOISE",
    "车载_scene_01.wav",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")   

# ---- Scene 1 Configuration ----
SCENE1_NAME = "Scene_1_Test"
REF_WAV_1 = "dataset/NOISE/车载.wav"                       # Scene 1: Reference signal x(t) path
EXP_WAV_1 = "dataset/EXPECTED_NOISE/车载_scene_01.wav"     # Scene 1: Expected target signal d(t) path
SH_INDEX_1 = 0          
                      # Scene 1: Corresponding secondary path channel index (e.g., 0)

# ---- Scene 2 Configuration ----
SCENE2_NAME = "Scene_2_Test"
REF_WAV_2 = "dataset/NOISE/地铁.wav"                       # Scene 2: Reference signal x(t) path
EXP_WAV_2 = "dataset/EXPECTED_NOISE/地铁_scene_06.wav"     # Scene 2: Expected target signal d(t) path
SH_INDEX_2 = 5                                             # Scene 2: Corresponding secondary path channel index (e.g., 5)


# ==========================================
# 2. Core Processing Logic
# ==========================================
def load_and_trim_audio(filepath, target_length):
    """Loads audio and pads/trims it to the target length."""
    y, sr = sf.read(filepath, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1) # Convert to mono
    
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    return y

def evaluate_single_scene(model, sh_all, scene_name, ref_path, exp_path, sh_index):
   
    print(f"\n" + "="*50)
    print(f"🚀 Starting evaluation for scene: {scene_name}")
    print(f"={ '='*48 }")
    
    target_samples = int(SEGMENT_SEC * SR)
    turn_on_sample = int(ANC_TURN_ON_TIME * SR)
    
    # 1. Data Preparation
    x_np = load_and_trim_audio(ref_path, target_samples)
    d_np = load_and_trim_audio(exp_path, target_samples)
    sh_np = sh_all[sh_index]
    
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, T]
    d_t = torch.tensor(d_np, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, T]
    sh_t = torch.tensor(sh_np, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, L]
    
    # 2. Model Inference (Generate anti-noise y(t) and apply physical path to get a(t))
    with torch.no_grad():
        y_t = model(x_t)
        a_t = apply_dynamic_path(y_t, sh_t)
        e_t = d_t - a_t  # Residual signal after noise cancellation
        
    e_np = e_t.squeeze(0).cpu().numpy()
    d_np_aligned = d_t.squeeze(0).cpu().numpy() # Ensure length alignment
    
    # 3. Concatenate test audio (First half: ANC OFF, Second half: ANC ON)
    # When ANC is OFF, the ear hears the expected signal d_np
    # When ANC is ON, the ear hears the residual signal e_np
    eval_audio = np.concatenate([
        d_np_aligned[:turn_on_sample], 
        e_np[turn_on_sample:]
    ])
    max_abs = np.max(np.abs(eval_audio))
    eval_audio = eval_audio / max_abs
    
    # Save the concatenated evaluation audio
    out_wav_path = os.path.join(
        OUTPUT_DIR,
        f"{scene_name}_eval_recording.wav",
    )
    sf.write(out_wav_path, eval_audio, SR)
    print(f"✅ Evaluation audio generated: {out_wav_path} (0-{ANC_TURN_ON_TIME}s ANC OFF, {ANC_TURN_ON_TIME}-{SEGMENT_SEC}s ANC ON)")
    
    # 4. Invoke the 1/3 octave band analysis script
    print(f"\n📊 Running acoustic metric analysis via EvaluationMetrics...")
    octave_plot = os.path.join(
        OUTPUT_DIR,
        f"{scene_name}_octave_analysis.png",
    )
    
    results = analyze_anc_audio(
        audio_path=out_wav_path,
        anc_off_range=(1.0, ANC_TURN_ON_TIME - 0.5), # Avoid boundary artifacts
        anc_on_range=(ANC_TURN_ON_TIME + 1.0, SEGMENT_SEC - 0.5), # Sample after allowing system convergence time
        n_fft=8192,
        hop_length=2048,
        window_type="hann",
        channel=None,
        mono=True,
        plot=True,
        show_plot=False, # Set to False to prevent blocking continuous execution
        plot_path=os.path.join(
        OUTPUT_DIR,
        f"{scene_name}_noise_reduction.png",
    ),
        octave_plot_path=octave_plot
    )
    
    print_analysis_results(results)
    print(f"✅ Frequency band analysis chart saved to: {octave_plot}")

def load_checkpoint(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型权重：{checkpoint_path}\n"
            "禁止使用随机初始化模型进行评估。"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    print(f"成功载入权重：{checkpoint_path}")

def main():
    global OUTPUT_DIR
    print("=== Initializing ANC Evaluation Framework ===")
    
    # 1. Instantiate the model and load weights
    model = TimeDomainANC(in_channels=1, out_channels=1, hidden_channels=32, num_layers=10)
    load_checkpoint(model, MODEL_WEIGHTS_PATH)
    model = model.to(DEVICE)
    model.eval()

    RUN_TAG = os.environ.get("ANC_RUN_TAG", "default")
    OUTPUT_DIR = os.path.join(BASE_DIR, "official_eval_results", RUN_TAG)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Evaluate model complexity
    print("\n" + "="*50)
    print("🧠 Evaluating model computational complexity and parameter count...")
    # Use 1 second of audio (1 channel, 48000 samples) as the baseline for complexity evaluation
    input_shape = (1, SR) 
    stats = count_model_complexity(model, input_shape, print_layer_detail=False, device=DEVICE)
    # The stats are printed internally by count_model_complexity

    # 3. Load the secondary acoustic path matrix
    print("\n📂 Loading secondary acoustic path library...")
    if not os.path.exists(SH_NPY_PATH):
        raise FileNotFoundError(f"Cannot find {SH_NPY_PATH}. Please check the path.")
    sh_all = np.load(SH_NPY_PATH).T
    print(f"✅ Secondary paths loaded. Shape: {sh_all.shape}")

    # 4. Execute scene evaluations
    try:
        evaluate_single_scene(model, sh_all, SCENE1_NAME, REF_WAV_1, EXP_WAV_1, SH_INDEX_1)
        evaluate_single_scene(model, sh_all, SCENE2_NAME, REF_WAV_2, EXP_WAV_2, SH_INDEX_2)
        print("\n🎉 All test procedures completed! Please review the generated .wav and .png output files.")
    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}")
        print("Please verify the audio paths configured in the [1. User Configuration Area].")

if __name__ == "__main__":
    main()