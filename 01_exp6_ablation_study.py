# ==============================================================================
# Program Name: 01_exp6_ablation_study.py
# Version: 1.1
# Description: Executes Stage 1 (Experiment 6 Ablation). Sweeps H and B 
#              hyperparameters on a subset of MusicGen clips to find the 
#              Pareto-optimal configuration under 128 kbps MP3 degradation.
# Ablation Grid: H_VALUES = [4, 6, 8, 12], B_VALUES = [16, 32, 64]
# TARGET_PROMPTS = 200  # Subset of 200 MusicGen clips

# Change Log: 
# 1.1 - Added FORCE_RESET flag for checkpoint management
# 1.0 - Initial version with HxB sweep, MP3 degradation, and heatmaps.
# GPU Required: YES (For accelerated metric calculations)
# ==============================================================================

!pip install -q torch torchaudio numpy scipy librosa soundfile pesq pandas matplotlib seaborn pydub

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
import seaborn as sns
from pydub import AudioSegment

import subprocess

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')


# ==============================================================================
# CONFIGURATION FLAGS
# ==============================================================================
FORCE_RESET = True  # Set to True to delete checkpoint and re-run all evaluations
                    # Set to False to use existing checkpoint from Google Drive

# --- PEAQ ODG Wrapper ---
# Requires a compiled PEAQ binary (e.g., PQevalAudio) uploaded to your Colab 
PEAQ_BINARY_PATH = "/content/PQevalAudio" # Update with your actual binary path

def calculate_peaq_odg(ref_audio_path, deg_audio_path):
    """
    Calls an external PEAQ binary and parses the ODG score from stdout.
    Assumes standard output format: "ODG: -0.15"
    """
    try:
        result = subprocess.run(
            [PEAQ_BINARY_PATH, ref_audio_path, deg_audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Parse the ODG score from the console output
        for line in result.stdout.split('\n'):
            if "ODG" in line:
                return float(line.split(":")[1].strip())
        return np.nan
    except Exception as e:
        print(f"PEAQ execution failed: {e}")
        return np.nan
        
# --- 1. GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU. Please switch Colab runtime to T4 and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# --- 2. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage
LOCAL_TEMP_DIR = "/content/temp_data"
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

CHECKPOINT_FILE = "01_ablation_checkpoint.json"
RESULTS_FILE = "experiment_6_ablation_results.json"

# Ablation Grid
H_VALUES = [4, 6, 8, 12]
B_VALUES = [16, 32, 64]
TARGET_PROMPTS = 200  # Subset of 200 MusicGen clips
GENERATOR = "musicgen"
VAR_INDEX = 0 # Only use the first variation of each prompt

# Audio / STFT Params
SR = 16000 
N_FFT = 2048
HOP_LENGTH = 512

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

def delete_checkpoint_from_drive():
    """Delete the checkpoint file from Google Drive project folder."""
    ensure_project_dir()
    ckpt_path = os.path.join(PROJECT_DIR, CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
            print(f"[CHECKPOINT] Deleted {CHECKPOINT_FILE} from Google Drive.")
            return True
        except Exception as e:
            print(f"[CHECKPOINT] Could not delete from Drive: {e}")
            return False
    else:
        print(f"[CHECKPOINT] Checkpoint file not found in Drive.")
        return False

def load_checkpoint():
    """
    Load checkpoint from Google Drive.
    If FORCE_RESET is True, delete existing checkpoint and return empty state.
    Otherwise, download and return existing checkpoint or empty state if none exists.
    """
    local_cp_path = os.path.join(LOCAL_TEMP_DIR, CHECKPOINT_FILE)
    
    # Handle FORCE_RESET flag
    if FORCE_RESET:
        print("\n[CHECKPOINT] FORCE_RESET = True - Deleting existing checkpoint...")
        # Delete from Drive if exists
        delete_checkpoint_from_drive()
        # Delete local if exists
        if os.path.exists(local_cp_path):
            os.remove(local_cp_path)
            print(f"[CHECKPOINT] Deleted local checkpoint file.")
        print("[CHECKPOINT] Starting fresh evaluation.\n")
        return {"processed_configs": [], "results": []}
    
    # Normal mode: Try to load existing checkpoint
    print("\n[CHECKPOINT] FORCE_RESET = False - Attempting to load existing checkpoint...")
    
    # Try to load from Drive
    if load_from_drive(CHECKPOINT_FILE, local_cp_path):
        try:
            with open(local_cp_path, "r") as f:
                state = json.load(f)
            print(f"[CHECKPOINT] Loaded checkpoint with {len(state.get('processed_configs', []))} processed configs and {len(state.get('results', []))} results.")
            return state
        except Exception as e:
            print(f"[CHECKPOINT] Error loading checkpoint: {e}")
            print("[CHECKPOINT] Starting fresh evaluation.\n")
            return {"processed_configs": [], "results": []}
    else:
        print("[CHECKPOINT] No existing checkpoint found in Drive.")
        print("[CHECKPOINT] Starting fresh evaluation.\n")
        return {"processed_configs": [], "results": []}

def save_checkpoint(state):
    local_cp = os.path.join(LOCAL_TEMP_DIR, "temp_checkpoint.json")
    with open(local_cp, "w") as f:
        json.dump(state, f)
    save_to_drive(local_cp, CHECKPOINT_FILE)
    print(f"[CHECKPOINT] Saved checkpoint with {len(state.get('processed_configs', []))} processed configs.")

# --- 4. Core Watermarking Logic ---
def get_harmonic_bins(f0_track, sr, n_fft, H):
    """Maps f0 estimates to harmonic bin indices."""
    bins = []
    for f0 in f0_track:
        if np.isnan(f0) or f0 == 0:
            bins.append(None)
        else:
            h_bins = [int(np.floor((h * f0) / (sr / n_fft))) for h in range(1, H + 1)]
            bins.append(h_bins)
    return bins

def embed_watermark(y, sr, H, B, watermark_bits):
    """Embeds phase offset into H harmonics."""
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag, S_phase = librosa.magphase(D)
    
    f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=2000, sr=sr)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    
    modified_phase = np.angle(S_phase)
    ref_phase = np.zeros((H, D.shape[1])) # Store C_ref simplified equivalent
    
    # Simple ATH Masking Approximation
    delta_max = np.pi / 4 
    
    for m in range(D.shape[1]):
        if harmonic_bins[m] is None:
            continue
            
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                w_b = watermark_bits[bit_idx]
                
                # Apply Phase Shift
                modified_phase[k_h, m] += delta_max * w_b
                ref_phase[idx, m] = np.angle(S_phase[k_h, m]) # Store original reference phase
                
    D_watermarked = S_mag * np.exp(1.j * modified_phase)
    y_wm = librosa.istft(D_watermarked, hop_length=HOP_LENGTH)
    return y_wm, ref_phase, f0

def detect_watermark(y_deg, sr, H, B, f0, ref_phase):
    """Recovers the watermark bits from degraded audio."""
    D = librosa.stft(y_deg, n_fft=N_FFT, hop_length=HOP_LENGTH)
    _, S_phase = librosa.magphase(D)
    deg_phase = np.angle(S_phase)
    
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    bit_votes = {b: [] for b in range(B)}
    
    for m in range(D.shape[1]):
        if harmonic_bins[m] is None:
            continue
            
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                
                # Phase difference
                phase_diff = deg_phase[k_h, m] - ref_phase[idx, m]
                # Wrap to [-pi, pi]
                phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                
                # Decode bit
                decoded_bit = 1 if phase_diff > 0 else -1
                bit_votes[bit_idx].append(decoded_bit)
                
    # Majority voting for each bit
    recovered_bits = []
    for b in range(B):
        if len(bit_votes[b]) > 0:
            vote = 1 if sum(bit_votes[b]) >= 0 else -1
        else:
            vote = np.random.choice([-1, 1]) # Guess if no harmonic support
        recovered_bits.append(vote)
        
    return np.array(recovered_bits)

# --- 5. Degradation Pipeline ---
def apply_mp3_compression(y, sr, bitrate="128k"):
    temp_wav = os.path.join(LOCAL_TEMP_DIR, "temp.wav")
    temp_mp3 = os.path.join(LOCAL_TEMP_DIR, "temp.mp3")
    
    sf.write(temp_wav, y, sr)
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(temp_mp3, format="mp3", bitrate=bitrate)
    
    y_deg, _ = librosa.load(temp_mp3, sr=sr)
    
    # Cleanup
    os.remove(temp_wav)
    os.remove(temp_mp3)
    
    # Match lengths
    min_len = min(len(y), len(y_deg))
    return y_deg[:min_len]

# --- 6. Execution Pipeline ---

state = load_checkpoint()
results = state.get("results", [])
processed_configs = state.get("processed_configs", [])

print("\n--- Starting Stage 1: Experiment 6 (Ablation Study) ---")
print(f"Total configurations to evaluate: {len(H_VALUES) * len(B_VALUES)}")
print(f"Already processed: {len(processed_configs)}")
print(f"Results collected: {len(results)}")

# Fix a random watermark for the test
np.random.seed(42)

# Track if any new evaluations were performed
new_evaluations_performed = False

for H in H_VALUES:
    for B in B_VALUES:
        config_key = f"H{H}_B{B}"
        
        if config_key in processed_configs:
            print(f"Skipping {config_key} (Already processed)")
            continue
            
        new_evaluations_performed = True
        print(f"\nEvaluating Configuration: H={H}, B={B}")
        target_watermark = np.random.choice([-1, 1], size=B)
        
        config_ber = []
        config_peaq = []
        
        for i in range(TARGET_PROMPTS):
            filename = f"gen_{GENERATOR}_p{i:03d}_v{VAR_INDEX}.wav"
            local_orig = os.path.join(LOCAL_TEMP_DIR, filename)
            
            if not os.path.exists(local_orig):
                success = load_from_drive(filename, local_orig)
                if not success:
                    print(f"Warning: Could not download {filename}, skipping...")
                    continue
                    
            y_orig, _ = librosa.load(local_orig, sr=SR)
            
            # Embed
            y_wm, ref_phase, f0_track = embed_watermark(y_orig, SR, H, B, target_watermark)
            
            # Save watermarked audio for PEAQ evaluation
            local_wm = os.path.join(LOCAL_TEMP_DIR, f"temp_wm_{H}_{B}_{i}.wav")
            sf.write(local_wm, y_wm, SR)
            
            # Ensure lengths match for metrics
            min_len = min(len(y_orig), len(y_wm))
            y_orig = y_orig[:min_len]
            y_wm = y_wm[:min_len]
            
            # Metric 1: PEAQ ODG (Imperceptibility)
            peaq_score = calculate_peaq_odg(local_orig, local_wm)
            config_peaq.append(peaq_score)
            
            # Degrade
            y_deg = apply_mp3_compression(y_wm, SR, "128k")
            
            # Detect
            recovered_bits = detect_watermark(y_deg, SR, H, B, f0_track, ref_phase)
            
            # Metric 2: BER
            errors = np.sum(target_watermark != recovered_bits)
            ber = errors / B
            config_ber.append(ber)
            
            # Clean up temp files
            if os.path.exists(local_wm):
                os.remove(local_wm)
            
            # Clear memory locally to save Colab disk
            if i % 50 == 0 and i > 0:
                os.remove(local_orig)
        
        # Aggregate results for this config
        mean_ber = np.nanmean(config_ber) if config_ber else np.nan
        mean_peaq = np.nanmean(config_peaq) if config_peaq else np.nan
        
        print(f"  -> Result: Mean BER = {mean_ber:.4f}, Mean PEAQ ODG = {mean_peaq:.4f}")
        results.append({
            "H": H,
            "B": B,
            "mean_ber": float(mean_ber),
            "mean_peaq": float(mean_peaq)
        })
        
        processed_configs.append(config_key)
        state["processed_configs"] = processed_configs
        state["results"] = results
        save_checkpoint(state)

# --- 7. Plotting and Export ---
# Check if we have results to plot (either from new evaluation or existing checkpoint)
if len(results) == 0:
    print("\n[WARNING] No results found! Cannot generate heatmaps.")
    print(f"Processed configs: {processed_configs}")
    print(f"Results length: {len(results)}")
    print("\nPossible solutions:")
    print("1. Set FORCE_RESET = True and re-run to redo all evaluations")
    print("2. Check if checkpoint file exists and contains valid results")
else:
    print(f"\n--- Generating Heatmaps for {len(results)} configurations ---")
    df = pd.DataFrame(results)
    
    # Verify required columns exist
    required_cols = ['H', 'B', 'mean_ber', 'mean_peaq']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
    else:
        # Save JSON results
        local_json = os.path.join(LOCAL_TEMP_DIR, RESULTS_FILE)
        df.to_json(local_json, orient="records", indent=4)
        save_to_drive(local_json, RESULTS_FILE)
        
        # Pivot for Heatmaps
        ber_pivot = df.pivot(index="H", columns="B", values="mean_ber")
        peaq_pivot = df.pivot(index="H", columns="B", values="mean_peaq")
        
        # BER Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(ber_pivot, annot=True, fmt=".3f", cmap="Reds", cbar_kws={'label': 'Bit Error Rate'})
        plt.title("Ablation: Mean BER vs. Harmonic Count (H) and Bit Capacity (B)\n(Post 128 kbps MP3)")
        plt.xlabel("Bit Capacity (B)")
        plt.ylabel("Number of Harmonics (H)")
        ber_plot_path = os.path.join(LOCAL_TEMP_DIR, "fig_06_01_ber_heatmap.png")
        plt.savefig(ber_plot_path, dpi=300, bbox_inches='tight')
        save_to_drive(ber_plot_path, "fig_06_01_ber_heatmap.png")
        plt.close()
        
        # PEAQ ODG Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(peaq_pivot, annot=True, fmt=".2f", cmap="Blues_r", cbar_kws={'label': 'PEAQ ODG Score'})
        plt.title("Ablation: Mean PEAQ ODG vs. Harmonic Count (H) and Bit Capacity (B)")
        plt.xlabel("Bit Capacity (B)")
        plt.ylabel("Number of Harmonics (H)")
        peaq_plot_path = os.path.join(LOCAL_TEMP_DIR, "fig_06_02_peaq_heatmap.png")
        plt.savefig(peaq_plot_path, dpi=300, bbox_inches='tight')
        save_to_drive(peaq_plot_path, "fig_06_02_peaq_heatmap.png")
        plt.close()
        
        print("\n[SUCCESS] Ablation Study completed. Matrices and plots saved to Google Drive!")
        
        # Print summary of results
        print("\n--- Summary of Results ---")
        print(df.to_string(index=False))

# Additional status message when using existing checkpoint without re-evaluation
if not new_evaluations_performed and len(results) > 0:
    print("\n" + "="*60)
    print("[NOTE] Used existing checkpoint data. No new evaluations performed.")
    print(f"Processed {len(processed_configs)} configurations with {len(results)} results.")
    print("To re-run all evaluations, set FORCE_RESET = True and re-execute.")
    print("="*60)

# --- 8. Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")