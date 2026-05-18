# ==============================================================================
# Program Name: 06_unified_baselines.py
# Version: 2.1 (Full Run + Profiling + SNR Metadata Tracking + Built-in Analysis)
# Description: Three-way universal baseline comparison (IPC vs. SS vs. WavMark)
#              with automatic analysis, plotting, and LaTeX table generation.
#              Generates fig_06_01_baseline_robustness.png
# Changelog:
# Version: 2.1 (Added analysis and plotting functionality)
# Version: 2.0 (Full Run + Profiling + SNR Metadata Tracking)
# Version: 1.11 (LAME Syntax Fix + TempFile Cleanup + 5-Clip Test)
# Version: 1.10 (WavMark 'None' Handling + Random BER Fallback + 5-Clip Test)
# Version: 1.9 (Defensive Flattening + WavMark Shape Assertions + 5-Clip Test)
# Version: 1.8 (1D NumPy WavMark API Fix + 5-Clip Smoke Test)
# Version: 1.7 (Dynamic PEAQ Compile + WavMark Padding Guards + 5-Clip Test)
# Version: 1.6 (peaqb-fast + 16-bit WavMark Payload Fix + 5-Clip Smoke Test)
# Version: 1.5 (peaqb-fast + soxr upsampling + 5-Clip Smoke Test)
# Version: 1.4 (NVMe ZIP Architecture + AQUA-Tk + 5-Clip Smoke Test)
# Version: 1.3 (NVMe ZIP Architecture + 5-Clip Smoke Test)
# Version: 1.2 (Smoke Test + Drive Architecture)
# Version: 1.1 (Smoke Test - 5 Clips)
# GPU Required: YES (for WavMark only, analysis runs on CPU)
# ==============================================================================

import os
import subprocess
import sys
import time
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

import json
import hashlib
import tempfile
import shutil
import zipfile
import glob
import pandas as pd
import numpy as np
import torch
import librosa
import soundfile as sf
import soxr
from tqdm import tqdm
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

# --- Analysis Imports ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Initialization: {DEVICE}")

# --- Google Drive Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage

# --- Audio & Watermark Params ---
SR          = 16000   
N_FFT       = 2048    
HOP_LENGTH  = 512     
H_HARMONICS = 8       
B_PAYLOAD   = 32      
ALPHA_SS    = 0.01    
B_PAYLOAD_WAVMARK = 16  
WAVMARK_MIN_SAMPLES = 17600  

# --- Baseline Model Pins ---
WAVMARK_VERSION = "0.0.3" 

# --- Local NVMe Paths ---
LOCAL_WORKSPACE    = "/content/baseline_workspace"
EXTRACT_DIR        = f"{LOCAL_WORKSPACE}/audio"
ZIP_FILENAME       = "baseline_audio.zip"
LOCAL_ZIP_PATH     = f"{LOCAL_WORKSPACE}/{ZIP_FILENAME}"
CHECKPOINT_FILE    = "06_baseline_checkpoint.json"
FINAL_RESULTS_FILE = "06_baseline_results.json"
ANALYSIS_OUTPUT_DIR = f"{LOCAL_WORKSPACE}/analysis"

os.makedirs(LOCAL_WORKSPACE, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# --- Google Drive Helper Functions ---
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

def load_checkpoint_drive():
    """Load checkpoint from Google Drive project folder."""
    local_cp = f"{LOCAL_WORKSPACE}/temp_checkpoint.json"
    if load_from_drive(CHECKPOINT_FILE, local_cp):
        with open(local_cp, "r") as f:
            return json.load(f)
    print("Starting fresh evaluation. (No Drive checkpoint found)")
    return []

def save_checkpoint_drive(results):
    """Save checkpoint to Google Drive project folder."""
    local_cp = f"{LOCAL_WORKSPACE}/temp_checkpoint.json"
    with open(local_cp, 'w') as f:
        json.dump(results, f, indent=4)
    save_to_drive(local_cp, CHECKPOINT_FILE)

# --- 1. Data Ingestion (ZIP from Drive to NVMe) ---
print("\n--- Securing Audio Dataset ---")

if not os.listdir(EXTRACT_DIR):
    print("Loading dataset ZIP from Google Drive...")
    zip_src = os.path.join(PROJECT_DIR, ZIP_FILENAME)
    if not os.path.exists(zip_src):
        print(f"Error: {zip_src} not found in Google Drive.")
        print("Please ensure baseline_audio.zip is uploaded to the project folder.")
        sys.exit(1)
    
    shutil.copy2(zip_src, LOCAL_ZIP_PATH)
    
    print("Extracting to fast local NVMe...")
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    os.remove(LOCAL_ZIP_PATH) 
else:
    print("Audio already extracted locally.")

# --- 2. Dynamic Metadata & Sampling ---
print("\n--- Building Stratified Sample ---")
all_wavs = glob.glob(f"{EXTRACT_DIR}/*.wav")
records = []
for wav_path in all_wavs:
    filename = os.path.basename(wav_path)
    generator_name = filename.split('_')[1] if len(filename.split('_')) >= 2 else "unknown"
    records.append({"file_path": wav_path, "generator": generator_name})

df_meta = pd.DataFrame(records)
df_sample = df_meta.groupby('generator').sample(n=83, random_state=42).reset_index(drop=True)

print(f"FULL RUN ACTIVE: Processing {len(df_sample)} stratified clips.")

# --- 3. Model Loading ---
print("\n--- Loading WavMark Model ---")
import wavmark
try:
    assert wavmark.__version__ == WAVMARK_VERSION
except AttributeError:
    print(f"WARNING: wavmark.__version__ not available.")

wavmark_model = wavmark.load_model().to(DEVICE)
print(f"WavMark Model Loaded.")
print(f"WavMark Parameter Device: {next(wavmark_model.parameters()).device}")

# --- 4. Spread-Spectrum Embedder & Utilities ---
print("\n--- Initializing Functions ---")

def get_ss_carriers(n_freqs, n_frames, bits=B_PAYLOAD, seed=42):
    np.random.seed(seed)
    return np.random.randn(bits, n_freqs, n_frames).astype(np.float32)

def embed_spread_spectrum(y_orig, payload_bits):
    D = librosa.stft(y_orig, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = librosa.magphase(D)
    log_mag = np.log1p(mag)
    
    n_freqs, n_frames = log_mag.shape
    carriers = get_ss_carriers(n_freqs, n_frames, bits=B_PAYLOAD)
    
    for i, bit in enumerate(payload_bits):
        log_mag += ALPHA_SS * bit * carriers[i]
        
    mag_wm = np.expm1(log_mag)
    mag_wm = np.maximum(mag_wm, 0) 
    
    D_wm = mag_wm * phase
    y_wm = librosa.istft(D_wm, hop_length=HOP_LENGTH, length=len(y_orig))
    return y_wm, n_frames 

def detect_spread_spectrum(y_deg, ref_frames):
    D = librosa.stft(y_deg, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, _ = librosa.magphase(D)
    log_mag = np.log1p(mag)
    
    if log_mag.shape[1] < ref_frames:
        pad_width = ref_frames - log_mag.shape[1]
        log_mag = np.pad(log_mag, ((0,0), (0, pad_width)), mode='constant')
    else:
        log_mag = log_mag[:, :ref_frames]
        
    n_freqs = log_mag.shape[0]
    carriers = get_ss_carriers(n_freqs, ref_frames, bits=B_PAYLOAD)
    
    rec_bits = []
    for i in range(B_PAYLOAD):
        correlation = np.sum(log_mag * carriers[i])
        rec_bits.append(1 if correlation > 0 else -1)
        
    return np.array(rec_bits).astype(np.float32)

def compute_ber(ref_bits, rec_bits):
    assert len(ref_bits) == len(rec_bits) == B_PAYLOAD
    errors = np.sum(ref_bits != rec_bits)
    return float(errors / B_PAYLOAD)

def get_payload(generator_id: str, bits=B_PAYLOAD):
    seed = int(hashlib.md5(generator_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=bits).astype(np.float32)

def write_peaq_wav(y_mono_16k, path):
    y_48k = soxr.resample(y_mono_16k, 16000, 48000)
    y_stereo = np.stack([y_48k, y_48k], axis=1)  
    sf.write(path, y_stereo, 48000, subtype='PCM_16')

def calculate_peaq_odg(ref_path, deg_path):
    try:
        result = subprocess.run(
            [PEAQ_BINARY_PATH, "-r", ref_path, "-t", deg_path],
            capture_output=True, text=True, timeout=60
        )
        for line in result.stdout.splitlines():
            if "ODG:" in line:
                return float(line.strip().split(":")[-1].strip())
        return np.nan
    except Exception as e:
        print(f"PEAQ failed: {e}")
        return np.nan

# --- WavMark Wrapper ---
wavmark_none_count = 0  

def get_payload_wavmark(generator_id: str):
    seed = int(hashlib.md5(generator_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1], size=B_PAYLOAD_WAVMARK).tolist()

def encode_wavmark_wrapper(y_orig, generator_id):
    if len(y_orig) < WAVMARK_MIN_SAMPLES:
        pad = np.zeros(WAVMARK_MIN_SAMPLES - len(y_orig), dtype=np.float32)
        y_orig = np.concatenate([y_orig, pad])
        
    wm_payload = get_payload_wavmark(generator_id)
    y_np = np.array(y_orig, dtype=np.float32)
    
    with torch.no_grad():
        encoded_signal, info = wavmark.encode_watermark(
            wavmark_model, y_np, wm_payload, show_progress=False
        )
    return encoded_signal, info

def decode_wavmark_wrapper(y_deg):
    global wavmark_none_count
    
    if len(y_deg) < WAVMARK_MIN_SAMPLES:
        pad = np.zeros(WAVMARK_MIN_SAMPLES - len(y_deg), dtype=np.float32)
        y_deg = np.concatenate([y_deg, pad])
        
    y_np = np.array(y_deg, dtype=np.float32)
    
    with torch.no_grad():
        decoded_payload, _ = wavmark.decode_watermark(
            wavmark_model, y_np, show_progress=False
        )
        
    if decoded_payload is None:
        wavmark_none_count += 1
        return np.random.randint(0, 2, size=B_PAYLOAD_WAVMARK).astype(np.float32)
        
    result = np.array(decoded_payload, dtype=np.float32).flatten()
    
    assert result.shape == (B_PAYLOAD_WAVMARK,), \
        f"Unexpected decode shape: {result.shape}, got: {decoded_payload}"
        
    return result

def compute_ber_wavmark(generator_id, rec_bits):
    ref_bits = np.array(get_payload_wavmark(generator_id), dtype=np.float32)
    rec_bits = np.array(rec_bits, dtype=np.float32).flatten()
    assert len(ref_bits) == len(rec_bits) == B_PAYLOAD_WAVMARK, \
        f"Shape mismatch: ref={ref_bits.shape}, rec={rec_bits.shape}"
    return float(np.sum(ref_bits != rec_bits) / B_PAYLOAD_WAVMARK)

# --- Degradation Definitions ---
def deg_clean(y): return y

def apply_mp3(y, sr, bitrate="64k"):
    bitrate_int = int(str(bitrate).replace("k", ""))
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        
        tmp_in_path  = tmp_in.name
        tmp_mp3_path = tmp_mp3.name
        tmp_out_path = tmp_out.name
    
    try:
        sf.write(tmp_in_path, y, sr)
        
        subprocess.run(
            ["lame", "-b", str(bitrate_int), tmp_in_path, tmp_mp3_path],
            check=True, capture_output=True
        )
        subprocess.run(
            ["lame", "--decode", tmp_mp3_path, tmp_out_path],
            check=True, capture_output=True
        )
        y_deg, _ = librosa.load(tmp_out_path, sr=sr)
        return y_deg
        
    finally:
        for p in [tmp_in_path, tmp_mp3_path, tmp_out_path]:
            if os.path.exists(p):
                os.remove(p)

def deg_mp3_64(y): return apply_mp3(y, SR, "64k") 

def deg_tsm_09(y):
    y_stretched = librosa.effects.time_stretch(y, rate=0.9)
    return y_stretched[:len(y)]  

def deg_combined_mp3_tsm(y): 
    return deg_tsm_09(deg_mp3_64(y))

DEGRADATION_PIPELINE = {
    "clean": deg_clean,
    "mp3_64": deg_mp3_64,
    "tsm_09": deg_tsm_09,
    "mp3_tsm_combined": deg_combined_mp3_tsm
}

# ==============================================================================
# ANALYSIS FUNCTIONS 
# ==============================================================================

def run_analysis_and_plotting(results_data, output_dir):
    """
    Perform comprehensive analysis on the baseline results:
    - Generate PEAQ ODG statistics and LaTeX table
    - Compute ROC AUC for detection reliability
    - Generate robustness boxplot
    - Create LaTeX tables for all experiments
    """
    print("\n" + "="*60)
    print("RUNNING INTEGRATED ANALYSIS ON BASELINE RESULTS")
    print("="*60)
    
    df = pd.DataFrame(results_data)
    methods = ["spread_spectrum", "wavmark"]
    method_names = {"spread_spectrum": "Spread Spectrum Baseline", "wavmark": "WavMark (Neural)"}
    
    print(f"\nAnalyzing {len(df)} total evaluation records.")
    
    latex_tables = ""
    
    # ==========================================================================
    # Experiment 1: Imperceptibility (PEAQ ODG)
    # ==========================================================================
    print("\n" + "-"*40)
    print("EXPERIMENT 1: IMPERCEPTIBILITY (PEAQ ODG)")
    print("-"*40)
    
    df_exp1 = df[df['experiment'] == 'exp1']
    exp1_stats = df_exp1.groupby('method')['value'].agg(['mean', 'std', 'median']).round(3)
    
    exp1_tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Baseline Imperceptibility (PEAQ ODG)}\n"
        "\\label{tab:exp1_baselines}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{Mean ODG} & \\textbf{Median ODG} \\\\\n"
        "\\midrule\n"
        "Proposed IPC ($H^*=8, B^*=32$) & [INSERT YOUR N=250 MEAN] & [INSERT YOUR N=250 MEDIAN] \\\\\n"
        f"Spread Spectrum Baseline & {exp1_stats.loc['spread_spectrum', 'mean']:.3f} $\\pm$ {exp1_stats.loc['spread_spectrum', 'std']:.3f} & {exp1_stats.loc['spread_spectrum', 'median']:.3f} \\\\\n"
        f"WavMark (Neural) & {exp1_stats.loc['wavmark', 'mean']:.3f} $\\pm$ {exp1_stats.loc['wavmark', 'std']:.3f} & {exp1_stats.loc['wavmark', 'median']:.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
    )
    print(exp1_tex)
    latex_tables += "% --- Experiment 1: Imperceptibility ---\n" + exp1_tex
    
    # ==========================================================================
    # Experiment 2: Detection Reliability (ROC AUC)
    # ==========================================================================
    print("\n" + "-"*40)
    print("EXPERIMENT 2: DETECTION (ROC AUC)")
    print("-"*40)
    
    df_pos = df[(df['experiment'] == 'exp3') & (df['degradation'] == 'clean')]
    df_neg = df[df['experiment'] == 'exp2']
    
    auc_results = {}
    for method in methods:
        pos_ber = df_pos[df_pos['method'] == method]['value'].values
        neg_ber = df_neg[df_neg['method'] == method]['value'].values
        
        if len(pos_ber) > 0 and len(neg_ber) > 0:
            y_true = np.concatenate([np.ones(len(pos_ber)), np.zeros(len(neg_ber))])
            y_scores = np.concatenate([-pos_ber, -neg_ber])
            auc = roc_auc_score(y_true, y_scores)
            auc_results[method] = auc
        else:
            auc_results[method] = np.nan
            print(f"Warning: Insufficient data for {method} AUC calculation")
    
    exp2_tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Baseline Detection Reliability (ROC AUC)}\n"
        "\\label{tab:exp2_baselines}\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{ROC AUC} \\\\\n"
        "\\midrule\n"
        "Proposed IPC & [INSERT YOUR IPC AUC] \\\\\n"
        f"Spread Spectrum & {auc_results.get('spread_spectrum', np.nan):.4f} \\\\\n"
        f"WavMark (Neural) & {auc_results.get('wavmark', np.nan):.4f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
    )
    print(exp2_tex)
    latex_tables += "% --- Experiment 2: ROC AUC ---\n" + exp2_tex
    
    # ==========================================================================
    # Experiment 3: Robustness (BER across Degradations)
    # ==========================================================================
    print("\n" + "-"*40)
    print("EXPERIMENT 3: ROBUSTNESS (BER)")
    print("-"*40)
    
    df_exp3 = df[df['experiment'] == 'exp3'].copy()
    deg_map = {
        "clean": "Clean",
        "mp3_64": "MP3 64k",
        "tsm_09": "TSM 0.9x",
        "mp3_tsm_combined": "Combined"
    }
    df_exp3['degradation'] = df_exp3['degradation'].map(deg_map)
    df_exp3['Method'] = df_exp3['method'].map(method_names)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Only plot if we have data
    if len(df_exp3) > 0:
        sns.boxplot(
            data=df_exp3, 
            x='degradation', 
            y='value', 
            hue='Method',
            palette="Set2"
        )
        plt.title("Baseline Robustness Comparison (Bit Error Rate)")
        plt.ylabel("Bit Error Rate (BER)")
        plt.xlabel("Degradation Pipeline")
        plt.ylim(-0.05, 0.6) 
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Guessing (0.5)')
        plt.legend()
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No robustness data available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Baseline Robustness Comparison - No Data")
    
    plot_path = os.path.join(output_dir, "fig_06_01_baseline_robustness.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[SUCCESS] Robustness plot saved to {plot_path}")
    plt.close()
    
    # Upload the plot to Drive
    save_to_drive(plot_path, "fig_06_01_baseline_robustness.png")
    
    # Calculate Medians for LaTeX
    def get_median(method_key, deg_key):
        subset = df_exp3[(df_exp3['method'] == method_key) & (df_exp3['degradation'] == deg_map[deg_key])]
        if len(subset) > 0:
            return subset['value'].median()
        return np.nan
    
    exp3_tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Median BER Under Degradation (Baselines)}\n"
        "\\label{tab:exp3_baselines}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{Clean} & \\textbf{MP3 64k} & \\textbf{TSM 0.9x} & \\textbf{Combined} \\\\\n"
        "\\midrule\n"
        "Proposed IPC & [X.XXX] & [X.XXX] & [X.XXX] & [X.XXX] \\\\\n"
        f"Spread Spectrum & {get_median('spread_spectrum', 'clean'):.3f} & {get_median('spread_spectrum', 'mp3_64'):.3f} & {get_median('spread_spectrum', 'tsm_09'):.3f} & {get_median('spread_spectrum', 'mp3_tsm_combined'):.3f} \\\\\n"
        f"WavMark (Neural) & {get_median('wavmark', 'clean'):.3f} & {get_median('wavmark', 'mp3_64'):.3f} & {get_median('wavmark', 'tsm_09'):.3f} & {get_median('wavmark', 'mp3_tsm_combined'):.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    print(exp3_tex)
    latex_tables += "% --- Experiment 3: Robustness Medians ---\n" + exp3_tex
    
    # Save and upload LaTeX tables to Drive
    tex_path = os.path.join(output_dir, "baseline_tables.tex")
    with open(tex_path, "w") as f:
        f.write(latex_tables)
    save_to_drive(tex_path, "baseline_tables.tex")
    
    print(f"\n[ANALYSIS COMPLETE] Artifacts saved to {output_dir}")
    
    return {
        "exp1_stats": exp1_stats.to_dict(),
        "auc_results": auc_results,
        "plot_path": plot_path,
        "tex_path": tex_path
    }


# --- 5. Unified Evaluation Pipeline ---
print("\n--- Starting Baseline Evaluation Gauntlet ---")

results = load_checkpoint_drive()
processed_files = {r["clip_path"] for r in results} if results else set()
if len(processed_files) >= 249:
    print(f"Processed {len(processed_files)} files. Target reached. Exiting.")
    raise SystemExit(0)

if processed_files:
    print(f"Resuming from checkpoint. {len(processed_files)} clips already processed.")


for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    clip_path = row['file_path']
    if clip_path in processed_files:
        continue
        
    generator_id = row['generator']
    y_orig, _ = librosa.load(clip_path, sr=SR)
    
    if len(y_orig) < WAVMARK_MIN_SAMPLES:
        print(f"Skipping {clip_path} (too short: {len(y_orig)} samples)")
        continue
        
    payload = get_payload(generator_id)
    
    # 1. Spread-Spectrum Embed Timing
    t0 = time.time()
    y_ss, n_frames = embed_spread_spectrum(y_orig, payload)
    time_ss = time.time() - t0
    
    # 2. WavMark Embed Timing
    t0 = time.time()
    y_wm, wm_info = encode_wavmark_wrapper(y_orig, generator_id)
    time_wm = time.time() - t0
    
    wm_encoded_sections = wm_info.get("encoded_sections", 0)
    wm_skip_sections = wm_info.get("skip_sections", 0)
    
    # 3. PEAQ & Degradation Sweeps Timing
    t0 = time.time()
    tmp_orig = f"{LOCAL_WORKSPACE}/tmp_orig.wav"
    tmp_ss = f"{LOCAL_WORKSPACE}/tmp_ss.wav"
    tmp_wm = f"{LOCAL_WORKSPACE}/tmp_wm.wav"
    
    write_peaq_wav(y_orig, tmp_orig)
    write_peaq_wav(y_ss, tmp_ss)
    write_peaq_wav(y_wm, tmp_wm)
    
    odg_ss = calculate_peaq_odg(tmp_orig, tmp_ss)
    odg_wm = calculate_peaq_odg(tmp_orig, tmp_wm)
    
    results.append({"method": "spread_spectrum", "experiment": "exp1", "generator": generator_id, "degradation": "clean", "metric": "peaq_odg", "value": odg_ss, "payload_bits": B_PAYLOAD, "n_clips": 1, "clip_path": clip_path})
    results.append({"method": "wavmark", "experiment": "exp1", "generator": generator_id, "degradation": "clean", "metric": "peaq_odg", "value": odg_wm, "payload_bits": B_PAYLOAD_WAVMARK, "n_clips": 1, "clip_path": clip_path, "wm_encoded_sections": wm_encoded_sections, "wm_skip_sections": wm_skip_sections})
    
    rec_ss_neg = detect_spread_spectrum(y_orig, n_frames)
    rec_wm_neg = decode_wavmark_wrapper(y_orig)
    
    ber_ss_neg = compute_ber(payload, rec_ss_neg)
    ber_wm_neg = compute_ber_wavmark(generator_id, rec_wm_neg)
    
    results.append({"method": "spread_spectrum", "experiment": "exp2", "generator": generator_id, "degradation": "clean", "metric": "ber_negative", "value": ber_ss_neg, "payload_bits": B_PAYLOAD, "n_clips": 1, "clip_path": clip_path})
    results.append({"method": "wavmark", "experiment": "exp2", "generator": generator_id, "degradation": "clean", "metric": "ber_negative", "value": ber_wm_neg, "payload_bits": B_PAYLOAD_WAVMARK, "n_clips": 1, "clip_path": clip_path})

    for deg_name, deg_fn in DEGRADATION_PIPELINE.items():
        y_ss_deg = deg_fn(y_ss)
        y_wm_deg = deg_fn(y_wm)
        
        rec_ss = detect_spread_spectrum(y_ss_deg, n_frames)
        rec_wm = decode_wavmark_wrapper(y_wm_deg)
        
        ber_ss = compute_ber(payload, rec_ss)
        ber_wm = compute_ber_wavmark(generator_id, rec_wm)
        
        results.append({"method": "spread_spectrum", "experiment": "exp3", "generator": generator_id, "degradation": deg_name, "metric": "ber", "value": ber_ss, "payload_bits": B_PAYLOAD, "n_clips": 1, "clip_path": clip_path})
        results.append({"method": "wavmark", "experiment": "exp3", "generator": generator_id, "degradation": deg_name, "metric": "ber", "value": ber_wm, "payload_bits": B_PAYLOAD_WAVMARK, "n_clips": 1, "clip_path": clip_path})
        
    time_sweep = time.time() - t0
    
    print(f"\n[Profile] Clip: {os.path.basename(clip_path)} | SS Embed: {time_ss:.1f}s | WavMark Embed: {time_wm:.1f}s | PEAQ/Sweep/Detect: {time_sweep:.1f}s")
    
    save_checkpoint_drive(results)
    
    processed_files.add(clip_path)

# --- Save final results ---
local_final = f"{LOCAL_WORKSPACE}/{FINAL_RESULTS_FILE}"
shutil.copy2(f"{LOCAL_WORKSPACE}/temp_checkpoint.json", local_final)
save_to_drive(local_final, FINAL_RESULTS_FILE)

print(f"\n[SUCCESS] Baseline Full Run complete. Data uploaded to Drive: {FINAL_RESULTS_FILE}")
print(f"WavMark returned None (no detection) on {wavmark_none_count} calls.")


# ==============================================================================
# RUN INTEGRATED ANALYSIS
# ==============================================================================
print("\n" + "="*60)
print("PROCEEDING TO INTEGRATED ANALYSIS PHASE")
print("="*60)

analysis_results = run_analysis_and_plotting(results, ANALYSIS_OUTPUT_DIR)

print("\n" + "="*60)
print("BASELINE EVALUATION AND ANALYSIS COMPLETE")
print("="*60)
print(f"\nOutput artifacts:")
print(f"  - Raw results: {local_final}")
print(f"  - Robustness plot: {analysis_results['plot_path']}")
print(f"  - LaTeX tables: {analysis_results['tex_path']}")
print("\nAll artifacts have been saved to Google Drive.")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")