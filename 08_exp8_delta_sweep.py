# ==============================================================================
# Program Name: 08_exp8_delta_sweep.py
# Description: Delta_max Parameter Sweep (Experiment 8)
# Evaluates the trade-off between robustness (BER) and imperceptibility (ODG) 
# by sweeping the maximum phase perturbation budget (delta_max) from π/16 to π.
# ==============================================================================

import os
import sys
import glob
import shutil
import numpy as np
import librosa
import torch
import hashlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import soundfile as sf
import subprocess
import re
import concurrent.futures

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================================
# --- 0. Dynamic Environment Setup ---
# ==============================================================================
PEAQ_BINARY_PATH = "/content/peaqb-fast/src/peaqb"

if not os.path.exists(PEAQ_BINARY_PATH):
    print("\n--- Environment missing. Installing dependencies and compiling PEAQ ---")
    subprocess.run("apt-get update -y", shell=True, check=True)
    subprocess.run("apt-get install -y lame libsndfile1-dev", shell=True, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wavmark==0.0.3", "soxr", "matplotlib", "seaborn", "scikit-learn"], check=True)
    subprocess.run("rm -rf /content/peaqb-fast", shell=True)
    subprocess.run("git clone https://github.com/akinori-ito/peaqb-fast.git /content/peaqb-fast", shell=True, check=True)
    subprocess.run("cd /content/peaqb-fast && ./configure && make", shell=True, check=True)
    print("--- Environment setup complete ---\n")

# --- 1. IPC Core Constants & Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SR          = 16000
N_FFT       = 2048
HOP_LENGTH  = 512
H_STARS     = 8
B_STARS     = 32

# Sweep Levels: (Label, Value_in_Radians)
DELTA_LEVELS = [
    ("$\pi/16$", np.pi/16),
    ("$\pi/8$",  np.pi/8),
    ("$\pi/4$",  np.pi/4),
    ("$\pi/2$",  np.pi/2),
    ("$\pi$",    np.pi)
]

# --- Google Drive Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage

LOCAL_WORKSPACE = "/content/exp8_workspace"
AUDIO_DIR       = f"{LOCAL_WORKSPACE}/audio"
RESULTS_DIR     = f"{LOCAL_WORKSPACE}/results"
TEMP_DIR        = f"{LOCAL_WORKSPACE}/temp"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

N_CLIPS = 200

# ==============================================================================
# 2. IPC Core Functions (Verbatim from Legacy, with delta_max parameter)
# ==============================================================================

def generate_watermark(identity_str, bits=B_STARS):
    seed = sum(ord(c) for c in identity_str)
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=bits)

def get_harmonic_bins(f0_track, sr, n_fft, H):
    bins = []
    for f0 in f0_track:
        if np.isnan(f0) or f0 == 0:
            bins.append(None)
        else:
            h_bins = [int(np.floor((h * f0) / (sr / n_fft))) 
                      for h in range(1, H + 1)]
            bins.append(h_bins)
    return bins

def embed_watermark(y, sr, H, B, watermark_bits, delta_max=np.pi/4):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag, S_phase = librosa.magphase(D)
    
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2000, sr=sr)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    
    modified_phase = np.angle(S_phase)
    
    max_frames = min(D.shape[1], len(harmonic_bins))
    for m in range(max_frames):
        if harmonic_bins[m] is None:
            continue
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                w_b = watermark_bits[bit_idx]
                modified_phase[k_h, m] += delta_max * w_b  # Sweep parameter
                
    D_watermarked = S_mag * np.exp(1.j * modified_phase)
    y_wm = librosa.istft(D_watermarked, hop_length=HOP_LENGTH)
    return y_wm

def extract_reference_phase(y_orig, sr, H):
    D = librosa.stft(y_orig, n_fft=N_FFT, hop_length=HOP_LENGTH)
    _, S_phase = librosa.magphase(D)
    f0, _, _ = librosa.pyin(y_orig, fmin=65, fmax=2000, sr=sr)
    
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    ref_phase = np.zeros((H, D.shape[1]))
    
    max_frames = min(D.shape[1], len(harmonic_bins))
    for m in range(max_frames):
        if harmonic_bins[m] is not None:
            for idx, k_h in enumerate(harmonic_bins[m]):
                if k_h < D.shape[0]:
                    ref_phase[idx, m] = np.angle(S_phase[k_h, m])
    return f0, ref_phase

def detect_watermark(y_deg, sr, H, B, f0, ref_phase):
    D = librosa.stft(y_deg, n_fft=N_FFT, hop_length=HOP_LENGTH)
    _, S_phase = librosa.magphase(D)
    deg_phase = np.angle(S_phase)
    
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    bit_votes = {b: [] for b in range(B)}
    
    max_frames = min(D.shape[1], len(harmonic_bins), ref_phase.shape[1])
    
    for m in range(max_frames):
        if harmonic_bins[m] is None:
            continue
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                phase_diff = deg_phase[k_h, m] - ref_phase[idx, m]
                phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                decoded_bit = 1 if phase_diff > 0 else -1
                bit_votes[bit_idx].append(decoded_bit)
                
    recovered_bits = []
    for b in range(B):
        if len(bit_votes[b]) > 0:
            vote = 1 if sum(bit_votes[b]) >= 0 else -1
        else:
            vote = np.random.choice([-1, 1])
        recovered_bits.append(vote)
        
    return np.array(recovered_bits)

# ==============================================================================
# 3. Degradation & Perceptual Evaluation Wrappers
# ==============================================================================

def apply_mp3_compression(y, sr, tmp_idx):
    """Encodes and decodes audio via LAME at 128 kbps to apply MP3 degradation."""
    wav_path = os.path.join(TEMP_DIR, f"tmp_in_{tmp_idx}.wav")
    mp3_path = os.path.join(TEMP_DIR, f"tmp_out_{tmp_idx}.mp3")
    dec_path = os.path.join(TEMP_DIR, f"tmp_dec_{tmp_idx}.wav")
    
    sf.write(wav_path, y, sr, subtype='PCM_16')
    subprocess.run(["lame", "-b", "128", "--quiet", wav_path, mp3_path], check=True)
    subprocess.run(["lame", "--decode", "--quiet", mp3_path, dec_path], check=True)
    
    y_deg, _ = librosa.load(dec_path, sr=sr)
    
    # Cleanup
    for p in [wav_path, mp3_path, dec_path]:
        if os.path.exists(p): os.remove(p)
        
    return y_deg

def compute_peaq_odg(y_ref, y_test, sr_orig, tmp_idx):
    """Computes PEAQ ODG using peaqb. Resamples to 48kHz for standard compliance."""
    y_ref_48 = librosa.resample(y_ref, orig_sr=sr_orig, target_sr=48000)
    y_test_48 = librosa.resample(y_test, orig_sr=sr_orig, target_sr=48000)
    
    ref_path = os.path.join(TEMP_DIR, f"ref_{tmp_idx}.wav")
    test_path = os.path.join(TEMP_DIR, f"test_{tmp_idx}.wav")
    
    sf.write(ref_path, y_ref_48, 48000, subtype='PCM_16')
    sf.write(test_path, y_test_48, 48000, subtype='PCM_16')
    
    odg = 0.0
    try:
        result = subprocess.run([PEAQ_BINARY_PATH, "-r", ref_path, "-t", test_path], capture_output=True, text=True)
        # Parse standard PEAQ output regex
        match = re.search(r'(?:ODG|Objective Difference Grade):\s*([-0-9.]+)', result.stdout)
        if match:
            odg = float(match.group(1))
        else:
            print(f"Warning: Could not parse PEAQ output: {result.stdout[:50]}")
    except Exception as e:
        print(f"PEAQ Error: {e}")
        
    # Cleanup
    for p in [ref_path, test_path]:
        if os.path.exists(p): os.remove(p)
        
    return odg

def get_odg_grade(odg_score):
    """ITU-R BS.1387 grading scale map."""
    if odg_score >= -1.0: return "Imperceptible"
    elif odg_score >= -2.0: return "Perceptible"
    elif odg_score >= -3.0: return "Slightly annoying"
    elif odg_score >= -4.0: return "Annoying"
    else: return "Very annoying"

# ==============================================================================
# 4. Google Drive Utilities
# ==============================================================================

def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_path, remote_name):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_name)
    try:
        shutil.copy2(local_path, dest_path)
        print(f"  [DRIVE OK] {local_path}  →  {dest_path}")
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_path}: {e}")

def load_from_drive(remote_name, local_path):
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_path)
            print(f"  [DRIVE OK] {src_path}  →  {local_path}")
            return True
        except Exception as e:
            print(f"  [DRIVE FAIL] copy from {src_path}: {e}")
            return False
    else:
        print(f"  [DRIVE MISSING] {src_path} not found")
        return False

def list_drive_files():
    """List files in the Google Drive project directory."""
    ensure_project_dir()
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

# ==============================================================================
# 5. Data Preparation
# ==============================================================================

print("\n--- Securing Audio Corpus (MusicGen-medium only) ---")
drive_files = list_drive_files()

musicgen_files = [f for f in drive_files if f.startswith("gen_musicgen_p") and f.endswith("_v0.wav")]
musicgen_files = sorted(musicgen_files)[:N_CLIPS]

print(f"Downloading {len(musicgen_files)} clips from Google Drive...")
for fname in tqdm(musicgen_files):
    local_path = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(local_path):
        load_from_drive(fname, local_path)

# ==============================================================================
# 6. Delta_max Sweep Execution
# ==============================================================================

STATE_FILE = "exp8_state.pkl"
local_state_path = os.path.join(RESULTS_DIR, STATE_FILE)

# Download state from Drive if it exists
if load_from_drive(STATE_FILE, local_state_path):
    print("Found existing state on Drive. Loading for resume...")

# Resume logic
if os.path.exists(local_state_path):
    with open(local_state_path, 'rb') as f:
        results_db = pickle.load(f)
    print("Resuming from existing state.")
else:
    results_db = {label: {'ber': [], 'odg': []} for label, _ in DELTA_LEVELS}

watermark_bits = generate_watermark("musicgen")

print("\n--- Commencing Delta_max Sweep ---")

for label, delta_val in DELTA_LEVELS:
    print(f"\nEvaluating Δ_max = {label} ({delta_val:.4f} rad)")
    
    # Check if this delta level is already completed
    if len(results_db[label]['ber']) >= N_CLIPS:
        print(f"Level {label} already completed. Skipping.")
        continue

    processed_count = len(results_db[label]['ber'])
    files_to_process = musicgen_files[processed_count:]

    for idx, fname in enumerate(tqdm(files_to_process)):
        y_orig, _ = librosa.load(os.path.join(AUDIO_DIR, fname), sr=SR)
        
        # 1. EMBED
        y_wm = embed_watermark(y_orig, SR, H_STARS, B_STARS, watermark_bits, delta_max=delta_val)
        
        # 2. DEGRADE (MP3 128k) & PEAQ (Concurrent Execution for speed)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_deg = executor.submit(apply_mp3_compression, y_wm, SR, idx)
            future_odg = executor.submit(compute_peaq_odg, y_orig, y_wm, SR, idx)
            
            y_deg = future_deg.result()
            odg_score = future_odg.result()

        # 3. DETECT
        f0_track, ref_phase = extract_reference_phase(y_orig, SR, H_STARS)
        recovered_bits = detect_watermark(y_deg, SR, H_STARS, B_STARS, f0_track, ref_phase)
        
        ber = float(np.sum(watermark_bits != recovered_bits) / B_STARS)
        
        # 4. STORE
        results_db[label]['ber'].append(ber)
        results_db[label]['odg'].append(odg_score)
        
        # Periodic Save
        if idx > 0 and idx % 10 == 0:
            with open(local_state_path, 'wb') as f:
                pickle.dump(results_db, f)
            save_to_drive(local_state_path, STATE_FILE)

    # Save at end of level
    with open(local_state_path, 'wb') as f:
        pickle.dump(results_db, f)
    save_to_drive(local_state_path, STATE_FILE)

# ==============================================================================
# 7. Aggregation & Artifact Generation
# ==============================================================================

print("\n--- Compiling Artifacts ---")

ber_means_pct = []
ber_stds_pct = []
odg_means = []
odg_stds = []
x_labels = []

for label, delta_val in DELTA_LEVELS:
    bers = np.array(results_db[label]['ber']) * 100  # Convert to percentage
    odgs = np.array(results_db[label]['odg'])
    
    ber_means_pct.append(np.mean(bers))
    ber_stds_pct.append(np.std(bers))
    odg_means.append(np.mean(odgs))
    odg_stds.append(np.std(odgs))
    x_labels.append(label)

# --- Generate Matplotlib Figure ---
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel('Phase Perturbation Budget ($\Delta_{max}$)')
ax1.set_ylabel('Mean BER (%)', color='steelblue')
ax1.errorbar(range(5), ber_means_pct, yerr=ber_stds_pct,
             color='steelblue', marker='o', linewidth=2, capsize=4)
ax1.axhline(y=5.0, color='steelblue', linestyle=':', alpha=0.5,
            label='BER = 5% threshold')
ax1.axvline(x=2, color='gray', linestyle='--', alpha=0.7)  # π/4 index

# Annotation for selected point
ax1.annotate('Selected: $H^*=8$, $B^*=32$, $\Delta^*=\pi/4$', 
             xy=(2, ber_means_pct[2]), xytext=(2.2, ber_means_pct[2] + 10),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5),
             fontsize=10, color='darkslategray')

ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.set_ylabel('Mean PEAQ ODG', color='crimson')
ax2.errorbar(range(5), odg_means, yerr=odg_stds,
             color='crimson', marker='s', linestyle='--',
             linewidth=2, capsize=4)
ax2.axhline(y=-1.0, color='crimson', linestyle=':', alpha=0.5,
            label='ODG = -1.0 (perceptibility boundary)')
ax2.tick_params(axis='y', labelcolor='crimson')

ax1.set_xticks(range(5))
ax1.set_xticklabels(x_labels)

plt.title('Perceptual Budget vs. Robustness Trade-off ($H^*=8, B^*=32$, MP3-128)')
fig.tight_layout()

fig_path = os.path.join(RESULTS_DIR, 'fig_08_01_delta_sweep.png')
plt.savefig(fig_path, dpi=300)
save_to_drive(fig_path, "fig_08_01_delta_sweep.png")
plt.close()
print("Saved Figure: fig_08_01_delta_sweep.png")

# --- Generate LaTeX Table ---
print("\n[LaTeX Table: Delta_max Parameter Sweep]")
print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{Perceptual budget vs. robustness trade-off across five $\\Delta_{\\max}$ ")
print("values ($H^*=8$, $B^*=32$, MP3 128 kbps, $N=200$ MusicGen clips).")
print("ODG grading: 0=imperceptible, $-1$=perceptible, $-2$=slightly annoying.}")
print("\\label{tab:delta_sweep}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("\\textbf{$\\Delta_{\\max}$} & \\textbf{Mean BER (\\%)} & \\textbf{Std BER} & \\textbf{Mean ODG} & \\textbf{ODG Grade} \\\\")
print("\\midrule")

for idx, label in enumerate(x_labels):
    m_ber = ber_means_pct[idx]
    s_ber = ber_stds_pct[idx]
    m_odg = odg_means[idx]
    grade = get_odg_grade(m_odg)
    
    # Bold the selected π/4 operating point (Index 2)
    if idx == 2:
        print(f"{label} & \\textbf{{{m_ber:.2f}}} & \\textbf{{{s_ber:.2f}}} & \\textbf{{{m_odg:.2f}}} & \\textbf{{{grade}}} \\\\")
    else:
        print(f"{label} & {m_ber:.2f} & {s_ber:.2f} & {m_odg:.2f} & {grade} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print(f"\n[SUCCESS] Experiment 8 completed. Results saved to Google Drive.")

# --- 8. Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")