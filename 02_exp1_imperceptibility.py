# ==============================================================================
# Program Name: 02_exp1_imperceptibility.py
# Version: 1.0
# Description: Executes Stage 2 (Experiment 1). Applies the H=8, B=32 phase (that we decided from output of 01_exp6_ablation_study.py )
#              watermark to all 3,000 clips across 3 generators. Computes PESQ 
#              and SI-SDR, and uploads watermarked audio + metrics to Google Drive.
# 				The configuration $H=8, B=32$ yields a highly resilient BER of 1.8% (0.018) while maintaining a high bit payload and a superb PESQ of 4.02. 
#				$H=6, B=32$ is also competitive (1.6% BER), but $H=8$ gives us slightly more harmonic anchors, 
#				which will be vital for the bass-heavy tracks in the Experiment 5 genre analysis.
# Change Log: 1.0 - Full 3K clip processing with SI-SDR and PESQ boxplots.
# GPU Required: YES (For accelerated SI-SDR calculation)
# ==============================================================================

!pip install -q torch torchaudio numpy scipy librosa soundfile pesq torchmetrics pandas matplotlib

import sys
import os
import json
import shutil
import torch
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
from pesq import pesq
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# --- 1. GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU for metric calculations.")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# --- 2. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage
LOCAL_TEMP_DIR = "/content/temp_data"
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

CHECKPOINT_FILE = "02_exp1_checkpoint.json"
RESULTS_FILE = "experiment_1_results.json"

TARGET_PROMPTS = 200
VARIATIONS_PER_PROMPT = 5
GENERATORS = ["musicgen", "audioldm2", "stableaudio"]

# Locked Hyperparameters from Ablation
SR = 16000 
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32

# --- 3. Google Drive Helper Functions ---
def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_filepath, remote_filename):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_filename)
    try:
        shutil.copy2(local_filepath, dest_path)
        print(f"  [DRIVE OK] {local_filepath}  →  {dest_path}")
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_filepath}: {e}")

def load_from_drive(remote_filename, local_filepath):
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_filename)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_filepath)
            print(f"  [DRIVE OK] {src_path}  →  {local_filepath}")
            return True
        except Exception as e:
            print(f"  [DRIVE FAIL] copy from {src_path}: {e}")
            return False
    else:
        print(f"  [DRIVE MISSING] {src_path} not found")
        return False

def load_checkpoint():
    """Load checkpoint from Google Drive project folder."""
    local_cp = os.path.join(LOCAL_TEMP_DIR, "temp_checkpoint.json")
    if load_from_drive(CHECKPOINT_FILE, local_cp):
        with open(local_cp, "r") as f:
            return json.load(f)
    return {"processed": [], "results": []}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local_cp = os.path.join(LOCAL_TEMP_DIR, "temp_checkpoint.json")
    with open(local_cp, "w") as f:
        json.dump(state, f)
    save_to_drive(local_cp, CHECKPOINT_FILE)

# --- 4. Core Watermarking Logic ---
def generate_watermark(identity_str, bits=B_STARS):
    """Generates a stable pseudo-random binary sequence from a generator ID."""
    seed = sum(ord(c) for c in identity_str)
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=bits)

def get_harmonic_bins(f0_track, sr, n_fft, H):
    bins = []
    for f0 in f0_track:
        if np.isnan(f0) or f0 == 0:
            bins.append(None)
        else:
            h_bins = [int(np.floor((h * f0) / (sr / n_fft))) for h in range(1, H + 1)]
            bins.append(h_bins)
    return bins

def embed_watermark(y, sr, H, B, watermark_bits):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag, S_phase = librosa.magphase(D)
    
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2000, sr=sr)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    
    modified_phase = np.angle(S_phase)
    delta_max = np.pi / 4 # ATH upper bound
    
    for m in range(D.shape[1]):
        if harmonic_bins[m] is None:
            continue
            
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                w_b = watermark_bits[bit_idx]
                modified_phase[k_h, m] += delta_max * w_b
                
    D_watermarked = S_mag * np.exp(1.j * modified_phase)
    y_wm = librosa.istft(D_watermarked, hop_length=HOP_LENGTH)
    return y_wm

# --- 5. Execution Pipeline ---

si_sdr_calc = ScaleInvariantSignalDistortionRatio().to("cuda")
state = load_checkpoint()

print(f"\n--- Starting Stage 2: Exp 1 (H={H_STARS}, B={B_STARS}) ---")

for gen in GENERATORS:
    watermark_payload = generate_watermark(gen)
    
    for i in range(TARGET_PROMPTS):
        for var in range(VARIATIONS_PER_PROMPT):
            filename = f"gen_{gen}_p{i:03d}_v{var}.wav"
            wm_filename = f"wm_{gen}_p{i:03d}_v{var}.wav"
            
            if filename in state["processed"]:
                continue
                
            local_orig = os.path.join(LOCAL_TEMP_DIR, filename)
            local_wm = os.path.join(LOCAL_TEMP_DIR, wm_filename)
            
            print(f"Processing: {filename}")
            
            # 1. Download from Drive
            if not load_from_drive(filename, local_orig):
                print(f"  [WARN] {filename} missing in Drive. Skipping.")
                continue
                
            y_orig, _ = librosa.load(local_orig, sr=SR)
            
            # 2. Embed
            y_wm = embed_watermark(y_orig, SR, H_STARS, B_STARS, watermark_payload)
            
            min_len = min(len(y_orig), len(y_wm))
            y_orig = y_orig[:min_len]
            y_wm = y_wm[:min_len]
            
            # 3. Save & Upload to Drive
            sf.write(local_wm, y_wm, SR)
            save_to_drive(local_wm, wm_filename)
            
            # 4. Compute Metrics
            try:
                pesq_score = pesq(SR, y_orig, y_wm, 'wb')
            except Exception:
                pesq_score = np.nan
                
            t_orig = torch.tensor(y_orig, dtype=torch.float32).unsqueeze(0).to("cuda")
            t_wm = torch.tensor(y_wm, dtype=torch.float32).unsqueeze(0).to("cuda")
            sisdr_score = si_sdr_calc(t_wm, t_orig).item()
            
            # 5. Record
            state["results"].append({
                "generator": gen,
                "prompt_idx": i,
                "variation": var,
                "pesq": float(pesq_score),
                "si_sdr": float(sisdr_score)
            })
            
            state["processed"].append(filename)
            
            # Cleanup & Checkpoint
            os.remove(local_orig)
            os.remove(local_wm)
            save_checkpoint(state)

# --- 6. Plotting and Export ---
if len(state["results"]) > 0:
    print("\n--- Generating Experiment 1 Plots ---")
    df = pd.DataFrame(state["results"])
    
    local_json = os.path.join(LOCAL_TEMP_DIR, RESULTS_FILE)
    df.to_json(local_json, orient="records", indent=4)
    save_to_drive(local_json, RESULTS_FILE)
    
    # PESQ Boxplot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='pesq', by='generator', grid=False, patch_artist=True)
    plt.title('PESQ Scores of Watermarked Audio by Generator')
    plt.suptitle('')
    plt.ylabel('PESQ Score (higher is better)')
    plt.xlabel('Generative Model')
    plot_file = os.path.join(LOCAL_TEMP_DIR, "fig_01_01_pesq_boxplots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    save_to_drive(plot_file, "fig_01_01_pesq_boxplots.png")
        
    # SI-SDR Boxplot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='si_sdr', by='generator', grid=False, patch_artist=True)
    plt.title('SI-SDR of Watermarked Audio by Generator')
    plt.suptitle('')
    plt.ylabel('SI-SDR (dB)')
    plt.xlabel('Generative Model')
    plot_file2 = os.path.join(LOCAL_TEMP_DIR, "fig_01_02_sisdr_boxplots.png")
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    save_to_drive(plot_file2, "fig_01_02_sisdr_boxplots.png")
    
    print("\n[SUCCESS] Experiment 1 completed. Metrics and plots saved to Google Drive!")
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")