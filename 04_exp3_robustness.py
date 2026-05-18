# ==============================================================================
# Program Name: 04_exp3_robustness.py
# Version: 1.1
# Description: Executes Stage 4 (Experiment 3). Tests watermark robustness 
#              against 5 passive degradations and 1 active gradient-based attack.
#              Generates fig_04_01_robustness_part1.png and fig_04_01_robustness_part2.png
# Change Log: 1.1 - Added complex-domain PGD loss (avoids phase wrapping) 
#                   and fixed alpha step size for clean convergence tracking.
# GPU Required: YES (For EnCodec re-encoding and PyTorch PGD autograd)
# ==============================================================================

!pip install -q torch torchaudio numpy scipy librosa soundfile pandas matplotlib seaborn scikit-learn pydub encodec

import sys
import os
import json
import shutil
import subprocess
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from pydub import AudioSegment
from encodec import EncodecModel
from encodec.utils import convert_audio

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

PEAQ_BINARY_PATH = "/content/peaqb-fast/src/peaqb"

# --- 0. Dynamic Environment Setup ---
if not os.path.exists(PEAQ_BINARY_PATH):
    print("\n--- Environment missing. Installing dependencies and compiling PEAQ ---")
    subprocess.run("apt-get update -y", shell=True, check=True)
    subprocess.run("apt-get install -y lame libsndfile1-dev", shell=True, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wavmark==0.0.3", "soxr", "matplotlib", "seaborn", "scikit-learn"], check=True)
    subprocess.run("rm -rf /content/peaqb-fast", shell=True)
    subprocess.run("git clone https://github.com/akinori-ito/peaqb-fast.git /content/peaqb-fast", shell=True, check=True)
    subprocess.run("cd /content/peaqb-fast && ./configure && make", shell=True, check=True)
    print("--- Environment setup complete ---\n")

# --- 1. GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected! Please switch Colab runtime to T4 and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# --- 2. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage
LOCAL_TEMP_DIR = "/content/temp_data"
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

CHECKPOINT_FILE = "04_robustness_checkpoint.json"
RESULTS_FILE = "experiment_3_results.json"

TARGET_PROMPTS = 200
VARIATIONS_PER_PROMPT = 5
GENERATORS = ["musicgen", "audioldm2", "stableaudio"]

SR = 16000 
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32


# Degradation Grids
MP3_RATES = ["64k", "96k", "128k", "192k"]
TSM_RATES = [0.9, 0.95, 1.05, 1.1]
PITCH_STEPS = [-2, -1, 1, 2]
AWGN_SNRS = [20, 30, 40]
ENCODEC_BWS = [3.0, 6.0, 12.0]

# PGD Adversarial Parameters
PGD_STEPS = [5, 10, 20]
PGD_EPSILON = 0.001 # Calibrated proxy for ~60 dBFS to maintain ODG > -1.0
PGD_ALPHA = PGD_EPSILON / 20.0 # Option B: Fixed step size for smooth convergence

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
    print("Starting fresh checkpoint.")
    return {"processed": [], "results": []}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local_cp = os.path.join(LOCAL_TEMP_DIR, "temp_checkpoint.json")
    with open(local_cp, "w") as f:
        json.dump(state, f)
    save_to_drive(local_cp, CHECKPOINT_FILE)

def calculate_peaq_odg(ref_audio_path, deg_audio_path):
    if not os.path.exists(PEAQ_BINARY_PATH): return np.nan 
    try:
        res = subprocess.run([PEAQ_BINARY_PATH, ref_audio_path, deg_audio_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        for line in res.stdout.split('\n'):
            if "ODG" in line: return float(line.split(":")[1].strip())
        return np.nan
    except: return np.nan

# --- 4. Watermark Core Logic ---
def generate_watermark(identity_str, bits=B_STARS):
    np.random.seed(sum(ord(c) for c in identity_str))
    return np.random.choice([-1, 1], size=bits)

def get_harmonic_bins(f0_track, sr, n_fft, H):
    return [[int(np.floor((h * f0) / (sr / n_fft))) for h in range(1, H + 1)] 
            if not np.isnan(f0) and f0 > 0 else None for f0 in f0_track]

def extract_reference_phase(y_orig, sr, H):
    D = librosa.stft(y_orig, n_fft=N_FFT, hop_length=HOP_LENGTH)
    f0, _, _ = librosa.pyin(y_orig, fmin=65, fmax=2000, sr=sr)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    ref_phase = np.zeros((H, D.shape[1]))
    _, S_phase = librosa.magphase(D)
    
    for m in range(D.shape[1]):
        if harmonic_bins[m] is not None:
            for idx, k_h in enumerate(harmonic_bins[m]):
                if k_h < D.shape[0]:
                    ref_phase[idx, m] = np.angle(S_phase[k_h, m])
    return f0, ref_phase

def detect_watermark(y_deg, sr, H, B, f0, ref_phase):
    D = librosa.stft(y_deg, n_fft=N_FFT, hop_length=HOP_LENGTH)
    deg_phase = np.angle(D)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    bit_votes = {b: [] for b in range(B)}
    
    for m in range(D.shape[1]):
        if harmonic_bins[m] is None: continue
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                p_diff = (deg_phase[k_h, m] - ref_phase[idx, m] + np.pi) % (2 * np.pi) - np.pi
                bit_votes[bit_idx].append(1 if p_diff > 0 else -1)
                
    return np.array([1 if sum(bit_votes[b]) >= 0 else -1 if len(bit_votes[b])>0 else np.random.choice([-1, 1]) for b in range(B)])

# --- 5. Passive Degradation Pipelines ---
def apply_mp3(y, sr, bitrate):
    tmp_in, tmp_out = os.path.join(LOCAL_TEMP_DIR, "in.wav"), os.path.join(LOCAL_TEMP_DIR, "out.mp3")
    sf.write(tmp_in, y, sr)
    AudioSegment.from_wav(tmp_in).export(tmp_out, format="mp3", bitrate=bitrate)
    y_deg, _ = librosa.load(tmp_out, sr=sr)
    return y_deg[:len(y)]

def apply_tsm(y, rate): return librosa.effects.time_stretch(y, rate=rate)[:len(y)]
def apply_pitch(y, sr, steps): return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)[:len(y)]
def apply_awgn(y, snr_db):
    noise = np.random.normal(0, np.sqrt(np.mean(y**2) / (10 ** (snr_db / 10))), len(y))
    return y + noise

encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.to("cuda")

def apply_encodec(y, sr, bw):
    encodec_model.set_target_bandwidth(bw)
    
    # 1. Create tensor and KEEP ON CPU for torchaudio resampling
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y_t = convert_audio(y_t, sr, encodec_model.sample_rate, encodec_model.channels)
    
    # 2. Move to GPU for the actual neural codec inference
    y_t = y_t.to("cuda")
    with torch.no_grad():
        encoded = encodec_model.encode(y_t)
        decoded = encodec_model.decode(encoded) 
        
    # 3. Move back to CPU for final resampling back to target SR
    decoded = decoded.cpu()
    decoded = convert_audio(decoded, encodec_model.sample_rate, sr, 1).squeeze().numpy()
    
    return decoded[:len(y)]

# --- 6. Active PGD Attack (Complex-Domain Optimization) ---
def pgd_attack(y_wm, sr, identity_bits, ref_phase, f0_track, epsilon=PGD_EPSILON, max_steps=20):
    device = "cuda"
    x_adv = torch.tensor(y_wm, dtype=torch.float32, device=device)
    x_orig = x_adv.clone().detach()
    
    # Precompute the complex reference phasor to avoid torch.angle() entirely
    ref_p = torch.tensor(ref_phase, dtype=torch.float32, device=device)
    ref_phasor = torch.exp(-1j * ref_p)
    target = torch.tensor(identity_bits, dtype=torch.float32, device=device)
    
    harmonic_bins = get_harmonic_bins(f0_track, sr, N_FFT, H_STARS)
    attack_checkpoints = {}
    
    # Window to mathematically match librosa's default STFT
    hann_win = torch.hann_window(N_FFT, device=device)
    
    for step in range(1, max_steps + 1):
        x_adv.requires_grad_(True)
        stft = torch.stft(x_adv.unsqueeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, window=hann_win, return_complex=True).squeeze(0)
        
        loss = torch.tensor(0.0, device=device)
        loss_computed = False
        
        for m in range(stft.shape[1]):
            if harmonic_bins[m] is None: continue
            for idx, k_h in enumerate(harmonic_bins[m]):
                if k_h < stft.shape[0]:
                    bit_idx = (m + idx) % B_STARS
                    t_b = target[bit_idx]
                    
                    Z = stft[k_h, m] * ref_phasor[idx, m]
                    loss = loss + (t_b * Z.imag)
                    loss_computed = True 
        
        if loss_computed:
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv - PGD_ALPHA * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon) 
        else:
            x_adv = x_adv.detach()
            
        if step in PGD_STEPS:
            attack_checkpoints[step] = x_adv.detach().cpu().numpy()
            
    return attack_checkpoints

# --- 7. Execution Pipeline ---

state = load_checkpoint()
identities = {gen: generate_watermark(gen) for gen in GENERATORS}
print(f"\n--- Starting Stage 4: Exp 3 (Robustness Pipeline) ---")

for true_gen in GENERATORS:
    target_bits = identities[true_gen]
    
    for i in range(TARGET_PROMPTS):
        for var in range(VARIATIONS_PER_PROMPT):
            wm_filename = f"wm_{true_gen}_p{i:03d}_v{var}.wav"
            orig_filename = f"gen_{true_gen}_p{i:03d}_v{var}.wav"
            
            if wm_filename in state["processed"]: continue
                
            local_wm = os.path.join(LOCAL_TEMP_DIR, wm_filename)
            local_orig = os.path.join(LOCAL_TEMP_DIR, orig_filename)
            
            if not load_from_drive(wm_filename, local_wm) or not load_from_drive(orig_filename, local_orig):
                continue
            
            print(f"Evaluating: {wm_filename}")
            y_orig, _ = librosa.load(local_orig, sr=SR)
            y_wm, _ = librosa.load(local_wm, sr=SR)
            f0_track, ref_phase = extract_reference_phase(y_orig, SR, H_STARS)
            
            clip_results = {"generator": true_gen, "prompt_idx": i, "variation": var}
            
            def eval_condition(y_cond):
                rec = detect_watermark(y_cond, SR, H_STARS, B_STARS, f0_track, ref_phase)
                return float(np.sum(rec != target_bits) / B_STARS)

            # Passive Branch
            for r in MP3_RATES: clip_results[f"mp3_{r}"] = eval_condition(apply_mp3(y_wm, SR, r))
            for r in TSM_RATES: clip_results[f"tsm_{r}"] = eval_condition(apply_tsm(y_wm, r))
            for s in PITCH_STEPS: clip_results[f"pitch_{s}"] = eval_condition(apply_pitch(y_wm, SR, s))
            for s in AWGN_SNRS: clip_results[f"awgn_{s}"] = eval_condition(apply_awgn(y_wm, s))
            for bw in ENCODEC_BWS: clip_results[f"encodec_{bw}"] = eval_condition(apply_encodec(y_wm, SR, bw))
            
            # Active Branch
            pgd_audio_checkpoints = pgd_attack(y_wm, SR, target_bits, ref_phase, f0_track)
            for step, y_adv in pgd_audio_checkpoints.items():
                clip_results[f"pgd_{step}_ber"] = eval_condition(y_adv)
                
                # Verify PEAQ ODG post-hoc
                tmp_adv = os.path.join(LOCAL_TEMP_DIR, "tmp_adv.wav")
                sf.write(tmp_adv, y_adv, SR)
                clip_results[f"pgd_{step}_odg"] = calculate_peaq_odg(local_orig, tmp_adv)
            
            state["results"].append(clip_results)
            state["processed"].append(wm_filename)
            
            save_checkpoint(state)
            
            os.remove(local_wm)
            os.remove(local_orig)

# --- 8. Aggregation and Plotting ---
if len(state["results"]) > 0:
    df = pd.DataFrame(state["results"])
    res_path = os.path.join(LOCAL_TEMP_DIR, RESULTS_FILE)
    df.to_json(res_path, orient="records", indent=4)
    save_to_drive(res_path, RESULTS_FILE)
    
    print("\n--- Generating Experiment 3 Plots ---")
    
    # --- PART 1: MP3 and Pitch Shifting ---
    fig1, axs1 = plt.subplots(2, 1, figsize=(6, 8))
    
    mp3_cols = [c for c in df.columns if 'mp3' in c]
    axs1[0].plot([c.split('_')[1] for c in mp3_cols], df[mp3_cols].mean(), marker='o', color='b')
    axs1[0].set_title('Robustness vs. MP3 Compression')
    axs1[0].set_ylabel('Mean Bit Error Rate (BER)')
    
    pitch_cols = [c for c in df.columns if 'pitch' in c]
    axs1[1].plot([c.split('_')[1] for c in pitch_cols], df[pitch_cols].mean(), marker='^', color='r')
    axs1[1].set_title('Robustness vs. Pitch Shifting')
    axs1[1].set_xlabel('Semitones')
    axs1[1].set_ylabel('Mean Bit Error Rate (BER)')
    
    plt.tight_layout()
    part1_path = os.path.join(LOCAL_TEMP_DIR, "fig_04_01_robustness_part1.png")
    fig1.savefig(part1_path, dpi=300)
    save_to_drive(part1_path, "fig_04_01_robustness_part1.png")
    plt.close(fig1)
    
    # --- PART 2: TSM and PGD Attack ---
    fig2, axs2 = plt.subplots(2, 1, figsize=(6, 8))
    
    tsm_cols = [c for c in df.columns if 'tsm' in c]
    axs2[0].plot([c.split('_')[1] for c in tsm_cols], df[tsm_cols].mean(), marker='s', color='g')
    axs2[0].set_title('Robustness vs. Time-Stretching (TSM)')
    axs2[0].set_ylabel('Mean Bit Error Rate (BER)')
    
    pgd_cols = [c for c in df.columns if 'pgd' in c and 'ber' in c]
    axs2[1].plot([c.split('_')[1] for c in pgd_cols], df[pgd_cols].mean(), marker='x', color='purple')
    axs2[1].set_title('Adversarial Robustness (PGD Attack)')
    axs2[1].set_xlabel('PGD Iteration Steps')
    axs2[1].set_ylabel('Mean Bit Error Rate (BER)')
    
    plt.tight_layout()
    part2_path = os.path.join(LOCAL_TEMP_DIR, "fig_04_01_robustness_part2.png")
    fig2.savefig(part2_path, dpi=300)
    save_to_drive(part2_path, "fig_04_01_robustness_part2.png")
    plt.close(fig2)
    
    print("\n[SUCCESS] Stage 4 Robustness Pipeline Completed. Split figures saved to Google Drive!")

# --- 9. Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")