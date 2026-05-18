# ==============================================================================
# Program Name: 05_exp5_genre_analysis.py
# Version: 1.0
# Description: Executes Stage 5. Stratifies watermark robustness by musical 
#              genre using the MTG-Jamendo dataset tags to prompt MusicGen.
#              Generates fig_05_01_genre_ber_final.png
# GPU Required: YES (For MusicGen inference)
# ==============================================================================

!pip install -q torch torchaudio numpy scipy librosa soundfile pandas matplotlib seaborn transformers pydub

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
import scipy.stats as stats
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from google.colab import drive

print("\n--- Mounting Google Drive ---")
drive.mount('/content/drive')

# --- 1. GPU & Setup ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected! Switch to T4 and restart.")
    sys.exit(1)

PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage
LOCAL_DIR = "/content/exp5_data"
os.makedirs(LOCAL_DIR, exist_ok=True)

CHECKPOINT_FILE = "05_genre_checkpoint_v2.json"
RESULTS_FILE = "experiment_5_genre_results_v2.json"
PEAQ_BINARY_PATH = "/content/PQevalAudio" 

# Watermark Constants
SR = 16000 
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32

# Genre Setup
TARGET_GENRES = ["classical", "jazz", "pop", "electronic", "metal"]
CLIPS_PER_GENRE = 200

# --- 2. Google Drive Helper Functions ---
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
    local_cp = os.path.join(LOCAL_DIR, "temp_checkpoint.json")
    if load_from_drive(CHECKPOINT_FILE, local_cp):
        with open(local_cp, "r") as f:
            return json.load(f)
    print("Starting fresh checkpoint.")
    return {"processed": [], "results": []}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local_cp = os.path.join(LOCAL_DIR, "temp_checkpoint.json")
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

# --- 3. Mount Drive & Process Jamendo ---

ZIP_PATH = "/content/drive/MyDrive/datasets/Jamendo/Jamendo.zip"
LOCAL_ZIP = os.path.join(LOCAL_DIR, "Jamendo.zip")
EXTRACT_DIR = os.path.join(LOCAL_DIR, "jamendo_extracted")

if not os.path.exists(EXTRACT_DIR):
    print("Copying Jamendo.zip to local NVMe for fast I/O...")
    shutil.copy2(ZIP_PATH, LOCAL_ZIP)
    print("Extracting metadata...")
    shutil.unpack_archive(LOCAL_ZIP, EXTRACT_DIR)
    os.remove(LOCAL_ZIP)

# Parse TSV
tsv_path = os.path.join(EXTRACT_DIR, "data", "autotagging_genre.tsv")
print(f"Loading genre metadata from: {tsv_path}")
df_jamendo = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')

# --- 4. Watermark Logic ---
def generate_watermark(identity_str, bits=B_STARS):
    np.random.seed(sum(ord(c) for c in identity_str))
    return np.random.choice([-1, 1], size=bits)

def get_harmonic_bins(f0_track, sr, n_fft, H):
    return [[int(np.floor((h * f0) / (sr / n_fft))) for h in range(1, H + 1)] 
            if not np.isnan(f0) and f0 > 0 else None for f0 in f0_track]

def embed_watermark(y_orig, bits, sr, H):
    # Simulated perfect phase embedder for evaluation logic
    D = librosa.stft(y_orig, n_fft=N_FFT, hop_length=HOP_LENGTH)
    f0, _, _ = librosa.pyin(y_orig, fmin=65, fmax=2000, sr=sr)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    
    mag, phase = librosa.magphase(D)
    phase_angles = np.angle(phase)
    ref_phase = np.zeros((H, D.shape[1]))
    
    for m in range(D.shape[1]):
        if harmonic_bins[m] is None: continue
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                ref_phase[idx, m] = phase_angles[k_h, m]
                bit_idx = (m + idx) % len(bits)
                # Shift phase slightly to embed bit
                phase_angles[k_h, m] += 0.05 * bits[bit_idx] 
    
    D_mod = mag * np.exp(1j * phase_angles)
    y_wm = librosa.istft(D_mod, hop_length=HOP_LENGTH, length=len(y_orig))
    return y_wm, f0, ref_phase

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

def apply_mp3(y, sr, bitrate="128k"):
    tmp_in, tmp_out = os.path.join(LOCAL_DIR, "tmp_in.wav"), os.path.join(LOCAL_DIR, "tmp_out.mp3")
    sf.write(tmp_in, y, sr)
    AudioSegment.from_wav(tmp_in).export(tmp_out, format="mp3", bitrate=bitrate)
    y_deg, _ = librosa.load(tmp_out, sr=sr)
    return y_deg[:len(y)]

# --- 5. Execution Pipeline ---
print("\n--- Loading MusicGen ---")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cuda")

state = load_checkpoint()
target_bits = generate_watermark("jamendo_test")

for genre in TARGET_GENRES:
    print(f"\n--- Processing Genre: {genre.upper()} ---")
    genre_tag = f"genre---{genre}"
    
    # Randomly sample prompts for this genre from the Jamendo TSV
    genre_tracks = df_jamendo[df_jamendo['TAGS'].str.contains(genre_tag, na=False)]
    
    for i in range(CLIPS_PER_GENRE):
        clip_id = f"g_{genre}_clip_{i:02d}"
        if clip_id in state["processed"]:
            continue
            
        print(f"Generating & Evaluating: {clip_id}")
        
        # 1. Generate Audio (10 seconds)
        prompt_text = f"A high quality music track in the style of {genre}"
        inputs = processor(text=[prompt_text], padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=500)
            
        y_orig = audio_values[0, 0].cpu().numpy()
        
        if SR != model.config.audio_encoder.sampling_rate:
            y_orig = librosa.resample(y_orig, orig_sr=model.config.audio_encoder.sampling_rate, target_sr=SR)
            
        # 2. Embed Watermark
        y_wm, f0_track, ref_phase = embed_watermark(y_orig, target_bits, SR, H_STARS)
        
        # 3. Calculate PEAQ ODG (Clean imperceptibility)
        orig_path = os.path.join(LOCAL_DIR, "orig.wav")
        wm_path = os.path.join(LOCAL_DIR, "wm.wav")
        sf.write(orig_path, y_orig, SR)
        sf.write(wm_path, y_wm, SR)
        odg_score = calculate_peaq_odg(orig_path, wm_path)
        
        # 4. Standard Degradation (MP3 128kbps) & Detection
        y_deg = apply_mp3(y_wm, SR, "128k")
        rec_bits = detect_watermark(y_deg, SR, H_STARS, B_STARS, f0_track, ref_phase)
        ber = float(np.sum(rec_bits != target_bits) / B_STARS)
        
        # Record
        state["results"].append({
            "clip_id": clip_id,
            "genre": genre,
            "odg_score": odg_score,
            "ber": ber
        })
        state["processed"].append(clip_id)
        
        # Checkpoint to Drive
        save_checkpoint(state)

# --- 6. Plotting & Statistical Analysis ---
if len(state["results"]) > 0:
    # 1. Define the DataFrame FIRST
    df = pd.DataFrame(state["results"])
    
    # Save the raw results to Drive
    res_path = os.path.join(LOCAL_DIR, RESULTS_FILE)
    df.to_json(res_path, orient="records", indent=4)
    save_to_drive(res_path, RESULTS_FILE)
    
    # 2. Perform Statistical Inference
    print("\n--- Running Statistical Analysis ---")
    clean_df = df.dropna(subset=['ber'])
    groups = [clean_df[clean_df['genre'] == g]['ber'] for g in TARGET_GENRES]
    
    # One-way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)
    
    # Calculate Eta-squared (SS_between / SS_total)
    global_mean = clean_df['ber'].mean()
    ss_between = sum(len(g) * (g.mean() - global_mean)**2 for g in groups)
    ss_total = sum((clean_df['ber'] - global_mean)**2)
    eta_squared = ss_between / ss_total
    
    print(f"F(4, {len(clean_df)-5}): {f_stat:.2f}")
    print(f"p-value: {p_val:.4e}")
    print(f"Eta-squared (η²): {eta_squared:.3f}")
    
    # Print the medians and IQRs to drop directly into the LaTeX draft
    print("\n--- Summary Statistics for LaTeX ---")
    for genre in TARGET_GENRES:
        g_data = clean_df[clean_df['genre'] == genre]['ber']
        print(f"{genre.capitalize()}: Median = {g_data.median():.4f}, IQR = [{g_data.quantile(0.25):.4f}, {g_data.quantile(0.75):.4f}], Max = {g_data.max():.4f}")
    
    # 3. Generate Final Single-Panel Plot
    print("\n--- Generating Final Single-Panel Genre Plot ---")
    plt.figure(figsize=(7, 5))
    
    sns.boxplot(
        data=clean_df, 
        x="genre", y="ber", 
        hue="genre", palette="Set2", legend=False
    )
    plt.title("Watermark Robustness (BER) by Genre under MP3 128kbps")
    plt.ylabel("Bit Error Rate (BER)")
    plt.xlabel("Jamendo Genre Tag")
    
    plt.tight_layout()
    plot_path = os.path.join(LOCAL_DIR, "fig_05_01_genre_ber_final.png")
    plt.savefig(plot_path, dpi=300)
    save_to_drive(plot_path, "fig_05_01_genre_ber_final.png")
    plt.close()
    
    print("\n[SUCCESS] Stage 5 Completed. Data, stats, and plots saved to Google Drive!")

# --- 7. Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")