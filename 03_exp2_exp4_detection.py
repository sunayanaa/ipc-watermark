# ==============================================================================
# Program Name: 03_exp2_exp4_detection.py
# Version: 1.0
# Description: Executes Stage 3 (Experiments 2 & 4). Performs multi-class 
#              attribution on the clean watermarked corpus. Generates ROC curves, 
#              AUC, TPR@1%FPR, and the 3x3 Confusion Matrix.
# Change Log: 1.0 - Combined Exp2 and Exp4 for single-pass I/O optimization.
# GPU Required: NO (CPU is sufficient for array comparisons and sklearn metrics)
# ==============================================================================

!pip install -q numpy scipy librosa soundfile pandas matplotlib seaborn scikit-learn

import sys
import os
import json
import shutil
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# --- 1. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage
LOCAL_TEMP_DIR = "/content/temp_data"
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

CHECKPOINT_FILE = "03_detection_checkpoint.json"
RESULTS_FILE = "experiment_2_4_results.json"

TARGET_PROMPTS = 200
VARIATIONS_PER_PROMPT = 5
GENERATORS = ["musicgen", "audioldm2", "stableaudio"]

SR = 16000 
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32

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

# --- 3. Core Detection Logic ---
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
            h_bins = [int(np.floor((h * f0) / (sr / n_fft))) for h in range(1, H + 1)]
            bins.append(h_bins)
    return bins

def extract_reference_phase(y_orig, sr, H):
    """Acts as the stored C_ref equivalent by extracting original phase."""
    D = librosa.stft(y_orig, n_fft=N_FFT, hop_length=HOP_LENGTH)
    _, S_phase = librosa.magphase(D)
    f0, _, _ = librosa.pyin(y_orig, fmin=65, fmax=2000, sr=sr)
    
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    ref_phase = np.zeros((H, D.shape[1]))
    
    for m in range(D.shape[1]):
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
    
    for m in range(D.shape[1]):
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

# --- 4. Execution Pipeline ---

state = load_checkpoint()

# Generate the 3 target identities
identities = {gen: generate_watermark(gen) for gen in GENERATORS}

print(f"\n--- Starting Stage 3: Exp 2 & 4 (Clean Detection) ---")

for true_gen in GENERATORS:
    for i in range(TARGET_PROMPTS):
        for var in range(VARIATIONS_PER_PROMPT):
            wm_filename = f"wm_{true_gen}_p{i:03d}_v{var}.wav"
            orig_filename = f"gen_{true_gen}_p{i:03d}_v{var}.wav"
            
            if wm_filename in state["processed"]:
                continue
                
            local_wm = os.path.join(LOCAL_TEMP_DIR, wm_filename)
            local_orig = os.path.join(LOCAL_TEMP_DIR, orig_filename)
            
            print(f"Detecting: {wm_filename}")
            
            # Download files from Drive
            if not load_from_drive(wm_filename, local_wm) or not load_from_drive(orig_filename, local_orig):
                print(f"  [WARN] Missing files for {wm_filename}. Skipping.")
                continue
                
            y_orig, _ = librosa.load(local_orig, sr=SR)
            y_wm, _ = librosa.load(local_wm, sr=SR)
            
            # Extract ref phase & detect bits
            f0_track, ref_phase = extract_reference_phase(y_orig, SR, H_STARS)
            recovered_bits = detect_watermark(y_wm, SR, H_STARS, B_STARS, f0_track, ref_phase)
            
            # Compute BER against all 3 known generators (Exp 4 logic)
            ber_scores = {}
            for test_gen, target_bits in identities.items():
                errors = np.sum(target_bits != recovered_bits)
                ber_scores[test_gen] = float(errors / B_STARS)
                
            # Argmin attribution
            attributed_gen = min(ber_scores, key=ber_scores.get)
            
            # Record
            state["results"].append({
                "true_generator": true_gen,
                "attributed_generator": attributed_gen,
                "prompt_idx": i,
                "variation": var,
                "ber_musicgen": ber_scores["musicgen"],
                "ber_audioldm2": ber_scores["audioldm2"],
                "ber_stableaudio": ber_scores["stableaudio"],
                "true_ber": ber_scores[true_gen] # The BER against the correct generator
            })
            
            state["processed"].append(wm_filename)
            
            os.remove(local_wm)
            os.remove(local_orig)
            save_checkpoint(state)

# --- 5. Analysis and Plotting ---
if len(state["results"]) > 0:
    print("\n--- Generating Experiment 2 & 4 Plots ---")
    df = pd.DataFrame(state["results"])
    
    local_json = os.path.join(LOCAL_TEMP_DIR, RESULTS_FILE)
    df.to_json(local_json, orient="records", indent=4)
    save_to_drive(local_json, RESULTS_FILE)
    
    # ---------------------------------------------------------
    # Exp 2: ROC Curves & AUC (Treating each generator as a binary classifier)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    for gen in GENERATORS:
        # True labels: 1 if this is the correct generator, 0 otherwise
        y_true = (df['true_generator'] == gen).astype(int)
        # Scores: We use (1 - BER) as the confidence score, so higher is more likely to be a match
        y_scores = 1.0 - df[f'ber_{gen}']
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate TPR @ 1% FPR
        idx = np.where(fpr <= 0.01)[0][-1]
        tpr_at_1_fpr = tpr[idx]
        print(f"Exp 2 - {gen}: AUC = {roc_auc:.4f}, TPR @ 1% FPR = {tpr_at_1_fpr:.4f}")
        
        plt.plot(fpr, tpr, lw=2, label=f'{gen} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Clean Conditions')
    plt.legend(loc="lower right")
    
    roc_plot = os.path.join(LOCAL_TEMP_DIR, "fig_03_01_roc_curves.png")
    plt.savefig(roc_plot, dpi=300, bbox_inches='tight')
    save_to_drive(roc_plot, "fig_03_01_roc_curves.png")
    plt.close()

    # ---------------------------------------------------------
    # Exp 4: Confusion Matrix (Argmin BER Multi-class)
    # ---------------------------------------------------------
    y_true_multi = df['true_generator']
    y_pred_multi = df['attributed_generator']
    
    macro_f1 = f1_score(y_true_multi, y_pred_multi, average='macro')
    print(f"Exp 4 - Macro F1 Score: {macro_f1:.4f}")
    
    cm = confusion_matrix(y_true_multi, y_pred_multi, labels=GENERATORS)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=GENERATORS, yticklabels=GENERATORS)
    plt.title('Multi-Generator Attribution Confusion Matrix')
    plt.xlabel('Attributed Generator (Predicted)')
    plt.ylabel('True Generator')
    
    cm_plot = os.path.join(LOCAL_TEMP_DIR, "fig_03_02_confusion_matrix.png")
    plt.savefig(cm_plot, dpi=300, bbox_inches='tight')
    save_to_drive(cm_plot, "fig_03_02_confusion_matrix.png")
    plt.close()

    print("\n[SUCCESS] Stage 3 completed. Results and plots saved to Google Drive!")

# --- 6. Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")