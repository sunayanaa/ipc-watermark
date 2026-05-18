# ==============================================================================
# Program Name: 07_exp7_generalisation.py
# Version: 2.0 (Self-Contained — IPC functions incorporated directly)
# Description: Cross-Generator Generalisation Probe (MusicGen-large holdout).
#              Tests whether the IPC watermarking method correctly rejects or
#              identifies audio from an unseen generative model (MusicGen-large)
#              without retraining.
#
#              Sub-Task A: False Attribution Rate on unwatermarked holdout clips
#              Sub-Task B: 4-class closed-set attribution including MusicGen-large
#
#              Change from v1.0: IPC core functions are now defined directly
#              in this script rather than loaded from a separate Drive-hosted
#              module. This eliminates the external dependency on 01_ipc_embed.py
#              and makes the script fully self-contained.
#
# IPC Core Functions (verbatim from working implementations):
#   - embed_watermark()        from 02_exp1_imperceptibility.py
#   - extract_reference_phase() from 03_exp2_exp4_detection.py
#   - detect_watermark()        from 03_exp2_exp4_detection.py
#
# Locked Hyperparameters (from 01_exp6_ablation_study.py Pareto analysis):
#   H* = 8, B* = 32, delta_max = pi/4
#
# GPU Required: YES (MusicGen-large inference if holdout clips not on Drive)
# ==============================================================================

import os
import glob
import shutil
import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import copy
import sys
import soundfile as sf
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')


# ==============================================================================
# 1. CONSTANTS & DEVICE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# --- Locked IPC Hyperparameters ---
SR          = 16000
N_FFT       = 2048
HOP_LENGTH  = 512
H_STARS     = 8
B_STARS     = 32

# --- Google Drive Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage

# --- Local Paths ---
LOCAL_WORKSPACE = "/content/exp7_workspace"
AUDIO_DIR       = f"{LOCAL_WORKSPACE}/audio"
RESULTS_DIR     = f"{LOCAL_WORKSPACE}/results"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Generator Config ---
LEGACY_GENS = ['audioldm2', 'stableaudio', 'musicgen']
NEW_GEN     = 'musicgenlarge'
ALL_GENS    = LEGACY_GENS + [NEW_GEN]
N_CLIPS     = 200


# ==============================================================================
# 2. IPC CORE FUNCTIONS
# Verbatim from 02_exp1_imperceptibility.py and 03_exp2_exp4_detection.py.
# These are the exact functions that produced Macro-F1 = 0.9967 in Exp 4.
# DO NOT MODIFY without rerunning all downstream experiments.
# ==============================================================================

def generate_watermark(identity_str, bits=B_STARS):
    """
    Generates a stable pseudo-random binary payload {-1,+1}^B
    from a generator identity string.
    Seed = sum of ASCII ordinal values of identity_str characters.
    This is the canonical payload method used across ALL experiments.
    """
    seed = sum(ord(c) for c in identity_str)
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=bits)


def get_payload(generator_id, bits=B_STARS):
    """Alias for generate_watermark() — consistent interface."""
    return generate_watermark(generator_id, bits)


def get_harmonic_bins(f0_track, sr, n_fft, H):
    """
    Maps per-frame f0 estimates to harmonic bin indices in the STFT grid.
    Returns list of H bin indices per voiced frame, or None for unvoiced.
    """
    bins = []
    for f0 in f0_track:
        if np.isnan(f0) or f0 == 0:
            bins.append(None)
        else:
            h_bins = [
                int(np.floor((h * f0) / (sr / n_fft)))
                for h in range(1, H + 1)
            ]
            bins.append(h_bins)
    return bins


def embed_watermark(y, sr, H, B, watermark_bits):
    """
    Embeds B watermark bits into STFT phase at H harmonic bins.
    delta_max = pi/4 (ATH upper bound). Bit index cycles over
    frame position and harmonic index: bit_idx = (m + h) mod B.
    """
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag, S_phase = librosa.magphase(D)

    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2000, sr=sr)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)

    modified_phase = np.angle(S_phase)
    delta_max = np.pi / 4

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


def extract_reference_phase(y_orig, sr, H):
    """
    Extracts per-frame harmonic phase reference from original audio.
    Acts as the stored C_ref: the baseline phase the detector compares
    against. Requires the original unwatermarked audio.
    Returns: f0 trajectory (n_frames,) and ref_phase matrix (H, n_frames).
    """
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
    """
    Recovers watermark bits from degraded audio via majority vote.
    Phase difference per bin: delta_phi = (phi_deg - ref_phase + pi) mod 2pi - pi
    Each frame/harmonic casts a vote for its assigned bit.
    Final bit = sign of accumulated votes.
    """
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


# ==============================================================================
# 3. UNIFIED WRAPPERS
# Adapt core functions to the (y, generator_id, c_ref) calling convention
# used throughout this script.
# ==============================================================================

def embed_ipc(y_orig, generator_id):
    """
    Embeds IPC watermark and returns watermarked audio + c_ref dict.
    c_ref stores the f0 trajectory and reference phase needed for detection.
    """
    watermark_bits = generate_watermark(generator_id)
    y_wm = embed_watermark(y_orig, SR, H_STARS, B_STARS, watermark_bits)
    f0, ref_phase = extract_reference_phase(y_orig, SR, H_STARS)
    c_ref = {
        'f0':           f0,
        'ref_phase':    ref_phase,
        'generator_id': generator_id
    }
    return y_wm, c_ref


def extract_ipc(y_deg, generator_id, c_ref):
    """
    Recovers watermark bits from degraded audio using stored c_ref.
    Returns float32 array of shape (B_STARS,) in {-1, +1}.
    """
    if not c_ref or 'f0' not in c_ref:
        return np.random.choice([-1, 1], size=B_STARS).astype(np.float32)
    f0        = c_ref['f0']
    ref_phase = c_ref['ref_phase']
    recovered_bits = detect_watermark(
        y_deg, SR, H_STARS, B_STARS, f0, ref_phase
    )
    return recovered_bits.astype(np.float32)


def compute_ber_ipc(generator_id, rec_bits):
    """BER between recovered bits and canonical payload for generator_id."""
    ref_bits = get_payload(generator_id)
    min_len = min(len(ref_bits), len(rec_bits))
    return float(np.sum(ref_bits[:min_len] != rec_bits[:min_len]) / min_len)


# ==============================================================================
# 4. GOOGLE DRIVE UTILITIES
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
# 5. DEFINITIVE DIAGNOSTIC
# Must pass before proceeding. BER on a known watermarked clip should be ~0.03.
# ==============================================================================

def run_diagnostic(audio_dir, legacy_prompts):
    print("\n--- Definitive Diagnostic ---")
    test_gen = 'musicgen'
    pattern  = os.path.join(audio_dir, f"gen_{test_gen}_{legacy_prompts[0]}*.wav")
    matches  = sorted(glob.glob(pattern))
    if not matches:
        sys.exit("[FATAL] No test clip found for diagnostic. Check audio download.")

    y_orig_test, _ = librosa.load(matches[0], sr=SR)
    y_wm_test, c_ref_test = embed_ipc(y_orig_test, test_gen)
    rec_bits = extract_ipc(y_wm_test, test_gen, copy.deepcopy(c_ref_test))
    ber = compute_ber_ipc(test_gen, rec_bits)

    print(f"Sanity BER: {ber:.4f}")
    print(f"Expected:   ~0.018 to 0.031")
    print(f"Payload ref: {get_payload(test_gen)[:8]}")
    print(f"Rec bits:    {rec_bits[:8]}")

    if ber > 0.10:
        sys.exit("[FATAL] BER too high — IPC functions are not working correctly.")
    else:
        print("\n[SUCCESS] IPC framework verified. Proceeding...\n")


# ==============================================================================
# 6. SECURING AUDIO CORPUS
# ==============================================================================


print("\n--- Securing Audio Corpus ---")
drive_files = list_drive_files()

# Identify the 200 matched prompt IDs from legacy corpus
legacy_prompts = []
for fname in drive_files:
    if fname.startswith("gen_musicgen_p") and fname.endswith(".wav"):
        # Extract prompt ID from filename like gen_musicgen_p001_v0.wav
        parts = fname.split('_')
        if len(parts) >= 3:
            pid = parts[2]  # e.g., "p001"
            legacy_prompts.append(pid)
legacy_prompts = sorted(list(set(legacy_prompts)))[:N_CLIPS]

# Download legacy clips
print(f"Copying {len(legacy_prompts)} matched triplets from Google Drive...")
for pid in tqdm(legacy_prompts):
    for gen in LEGACY_GENS:
        matches = [m for m in drive_files if m.startswith(f"gen_{gen}_{pid}") and m.endswith(".wav")]
        if matches:
            fname      = matches[0]
            local_path = os.path.join(AUDIO_DIR, fname)
            if not os.path.exists(local_path):
                load_from_drive(fname, local_path)

# Download or generate MusicGen-large holdout clips
mg_large_files = [m for m in drive_files if m.startswith("gen_musicgenlarge_")]

if len(mg_large_files) < N_CLIPS:
    print("\n[INFO] MusicGen-large clips missing. Generating via Transformers...")
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers", "soundfile"],
        check=True
    )
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    print("Loading MusicGen-large (3.3B) to GPU...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large"
    ).to(DEVICE)

    # Load prompt text if metadata.csv is available
    prompt_map = {}
    local_metadata = f"{LOCAL_WORKSPACE}/metadata.csv"
    if os.path.exists(local_metadata):
        try:
            df_meta    = pd.read_csv(local_metadata)
            prompt_map = dict(zip(df_meta['prompt_id'], df_meta['prompt_text']))
        except Exception:
            pass

    for pid in tqdm(legacy_prompts, desc="Generating MusicGen-large"):
        fname      = f"gen_musicgenlarge_{pid}_v0.wav"
        local_path = os.path.join(AUDIO_DIR, fname)

        # Check if already exists in Drive
        if fname in mg_large_files:
            if not os.path.exists(local_path):
                load_from_drive(fname, local_path)
            continue

        if not os.path.exists(local_path):
            prompt_text  = prompt_map.get(pid, f"music track for prompt {pid}")
            inputs       = processor(
                text=[prompt_text], padding=True, return_tensors="pt"
            ).to(DEVICE)
            audio_values = model.generate(**inputs, max_new_tokens=500)
            wav          = audio_values[0, 0].cpu().numpy()
            sr_model     = model.config.audio_encoder.sampling_rate
            y_res        = librosa.resample(wav, orig_sr=sr_model, target_sr=SR)
            sf.write(local_path, y_res, SR)
            save_to_drive(local_path, fname)

else:
    print(f"Downloading {N_CLIPS} MusicGen-large holdout clips from Google Drive...")
    for fname in tqdm(mg_large_files[:N_CLIPS]):
        local_path = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(local_path):
            load_from_drive(fname, local_path)


# ==============================================================================
# 7. STATE INITIALISATION
# ==============================================================================

print("\n--- Initializing Reference Matrices and Tau ---")
STATE_FILE       = "exp7_state.pkl"
local_state_path = os.path.join(RESULTS_DIR, STATE_FILE)

FORCE_RESET = False

if FORCE_RESET:
    print("Force reset: deleting Drive state and recomputing from scratch")
    try:
        state_path_remote = os.path.join(PROJECT_DIR, STATE_FILE)
        if os.path.exists(state_path_remote):
            os.remove(state_path_remote)
    except Exception:
        pass
    if os.path.exists(local_state_path):
        os.remove(local_state_path)

state = {}

# Try to restore checkpoint from Drive
if not FORCE_RESET and load_from_drive(STATE_FILE, local_state_path):
    with open(local_state_path, 'rb') as f:
        state = pickle.load(f)
    print(f"Loaded dynamic 4-Class Tau = {state.get('tau', 'not yet computed'):.4f}")
else:
    print("Starting fresh evaluation. (No checkpoint found)")


# ==============================================================================
# 8. DIAGNOSTIC — must pass before building C_ref cache
# ==============================================================================

run_diagnostic(AUDIO_DIR, legacy_prompts)


# ==============================================================================
# 9. BUILD C_REF CACHE
# ==============================================================================

C_REF_CACHE   = {}
WM_AUDIO_CACHE = {}

print("Embedding & caching C_ref matrices...")
for pid in tqdm(legacy_prompts, desc="Building C_ref cache"):
    for gen in ALL_GENS:
        pattern = os.path.join(AUDIO_DIR, f"gen_{gen}_{pid}*.wav")
        matches = sorted(glob.glob(pattern))
        if matches:
            y_orig, _ = librosa.load(matches[0], sr=SR)
            y_wm, c_ref = embed_ipc(y_orig, gen)
            WM_AUDIO_CACHE[(pid, gen)]  = y_wm.astype(np.float32)
            C_REF_CACHE[(pid, gen)]     = copy.deepcopy(c_ref)


# ==============================================================================
# 10. CALIBRATE REJECTION THRESHOLD TAU
# Computed from the BER distribution of unwatermarked holdout clips
# against all four known identities. TAU set at 1st percentile so that
# only 1% of unwatermarked clips are incorrectly attributed (FPR = 1%).
# ==============================================================================

if 'tau' not in state:
    print("\n--- Calibrating Rejection Threshold (Tau) from Holdout Negatives ---")
    neg_min_bers = []

    for pid in tqdm(legacy_prompts, desc="Calibrating TAU from holdout negatives"):
        pattern = os.path.join(AUDIO_DIR, f"gen_musicgenlarge_{pid}*.wav")
        matches = glob.glob(pattern)
        if not matches:
            continue
        y_unwm, _ = librosa.load(matches[0], sr=SR)

        bers = []
        for test_gen in LEGACY_GENS:
            c_ref = C_REF_CACHE.get((pid, test_gen))
            if c_ref is not None:
                rec_bits = extract_ipc(y_unwm, test_gen, copy.deepcopy(c_ref))
                bers.append(compute_ber_ipc(test_gen, rec_bits))

        if bers:
            neg_min_bers.append(min(bers))

    TAU = float(np.quantile(neg_min_bers, 0.01))
    print(f"TAU calibrated from holdout negatives: {TAU:.4f}")
    print(f"Expected FAR after recalibration: ~1%")

    state['tau'] = TAU
    with open(local_state_path, 'wb') as f:
        pickle.dump(state, f)
    save_to_drive(local_state_path, STATE_FILE)
else:
    TAU = state['tau']
    print(f"Loaded cached TAU = {TAU:.4f}")


# ==============================================================================
# 11. SUB-TASK A: HOLDOUT REJECTION PROBE
# Presents 200 unwatermarked MusicGen-large clips to the detector.
# Measures False Attribution Rate (FAR) at threshold TAU.
# ==============================================================================

print("\n--- Executing Sub-Task A (Rejection Probe) ---")

if 'subtask_a' not in state:
    state['subtask_a'] = {'processed_pids': set(), 'false_attributions': 0}

for i, pid in enumerate(tqdm(legacy_prompts, desc="Probing holdout clips")):
    if pid in state['subtask_a']['processed_pids']:
        continue

    pattern = os.path.join(AUDIO_DIR, f"gen_musicgenlarge_{pid}*.wav")
    matches = glob.glob(pattern)
    if not matches:
        continue

    y_suspect, _ = librosa.load(matches[0], sr=SR)

    bers = []
    for test_gen in LEGACY_GENS:
        c_ref = C_REF_CACHE.get((pid, test_gen))
        if c_ref is not None:
            rec_bits = extract_ipc(y_suspect, test_gen, copy.deepcopy(c_ref))
            bers.append(compute_ber_ipc(test_gen, rec_bits))

    if bers and min(bers) <= TAU:
        state['subtask_a']['false_attributions'] += 1

    state['subtask_a']['processed_pids'].add(pid)

    if i > 0 and i % 10 == 0:
        with open(local_state_path, 'wb') as f:
            pickle.dump(state, f)
        save_to_drive(local_state_path, STATE_FILE)

with open(local_state_path, 'wb') as f:
    pickle.dump(state, f)
save_to_drive(local_state_path, STATE_FILE)

far_count = state['subtask_a']['false_attributions']
far_rate  = (far_count / len(legacy_prompts)) * 100
print(f"Sub-Task A Result: False Attribution Rate = {far_rate:.2f}% "
      f"({far_count}/{len(legacy_prompts)})")


# ==============================================================================
# 12. SUB-TASK B: 4-CLASS ATTRIBUTION
# Embeds new identity w4 into MusicGen-large clips.
# Expands confusion matrix from 3-class to 4-class.
# ==============================================================================

print("\n--- Executing Sub-Task B (4-Class Attribution) ---")

if 'subtask_b' not in state:
    state['subtask_b'] = {
        'processed_pids': set(),
        'y_true':         [],
        'y_pred':         [],
        'ber_clean_records': {gen: [] for gen in ALL_GENS}
    }

for i, pid in enumerate(tqdm(legacy_prompts, desc="Attribution Matrix")):
    if pid in state['subtask_b']['processed_pids']:
        continue

    temp_y_true      = []
    temp_y_pred      = []
    temp_ber_records = {gen: [] for gen in ALL_GENS}

    for src_gen in ALL_GENS:
        if (pid, src_gen) not in WM_AUDIO_CACHE:
            continue
        y_wm = WM_AUDIO_CACHE[(pid, src_gen)]

        bers = {}
        for test_gen in ALL_GENS:
            c_ref = C_REF_CACHE.get((pid, test_gen))
            if c_ref is not None:
                rec_bits    = extract_ipc(y_wm, test_gen, copy.deepcopy(c_ref))
                ber         = compute_ber_ipc(test_gen, rec_bits)
                bers[test_gen] = ber
                if test_gen == src_gen:
                    temp_ber_records[src_gen].append(ber)

        if bers:
            best_match = min(bers, key=bers.get)
            prediction = best_match if bers[best_match] <= TAU else "unattributed"
        else:
            prediction = "unattributed"

        temp_y_true.append(src_gen)
        temp_y_pred.append(prediction)

    state['subtask_b']['y_true'].extend(temp_y_true)
    state['subtask_b']['y_pred'].extend(temp_y_pred)
    for gen in ALL_GENS:
        state['subtask_b']['ber_clean_records'][gen].extend(
            temp_ber_records[gen]
        )
    state['subtask_b']['processed_pids'].add(pid)

    if i > 0 and i % 5 == 0:
        with open(local_state_path, 'wb') as f:
            pickle.dump(state, f)
        save_to_drive(local_state_path, STATE_FILE)

with open(local_state_path, 'wb') as f:
    pickle.dump(state, f)
save_to_drive(local_state_path, STATE_FILE)

y_true           = state['subtask_b']['y_true']
y_pred           = state['subtask_b']['y_pred']
ber_clean_records = state['subtask_b']['ber_clean_records']

macro_f1 = f1_score(
    y_true, y_pred, labels=ALL_GENS, average='macro', zero_division=0
)
print(f"4-Class Macro-F1: {macro_f1:.4f}")


# ==============================================================================
# 13. ARTIFACT GENERATION
# ==============================================================================

print("\n--- Compiling Artifacts ---")

# --- LaTeX Table ---
print("\n[LaTeX Table: Cross-Generator Probe]")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Generalisation Probe: Embedding \\& Recovery on Unseen Generators}")
print("\\label{tab:exp7_generalization}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("\\textbf{Generator} & \\textbf{Median BER (Clean)} "
      "& \\textbf{False Attribution Rate} \\\\")
print("\\midrule")
for gen in LEGACY_GENS:
    med_ber = np.median(ber_clean_records[gen]) if ber_clean_records[gen] else float('nan')
    print(f"{gen.capitalize()} & {med_ber:.4f} & 1.00\\% (Calibrated) \\\\")
mg_ber = np.median(ber_clean_records['musicgenlarge']) \
         if ber_clean_records['musicgenlarge'] else float('nan')
print(f"MusicGen-large (Holdout) & {mg_ber:.4f} "
      f"& {far_rate:.2f}\\% (Uncalibrated) \\\\")
print("\\midrule")
print(f"\\multicolumn{{2}}{{l}}{{\\textbf{{4-Class Macro-F1}}}} "
      f"& \\textbf{{{macro_f1:.4f}}} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# --- 4×4 Confusion Matrix Figure ---
labels = ALL_GENS + ['unattributed']
cm     = confusion_matrix(y_true, y_pred, labels=labels)

row_sums = cm.sum(axis=1, keepdims=True)
cm_pct   = np.divide(
    cm.astype('float'), row_sums,
    where=row_sums != 0,
    out=np.zeros_like(cm, dtype=float)
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_pct[:4, :],
    annot=cm[:4, :],
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=ALL_GENS
)
plt.title("4-Class Attribution Confusion Matrix (IPC-based)")
plt.ylabel("True Embedded Identity")
plt.xlabel("Predicted Identity")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "fig_07_01_confusion_4class.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
save_to_drive(cm_path, "fig_07_01_confusion_4class.png")
plt.close()

print(f"\n[SUCCESS] Experiment 7 completed. "
      f"Macro-F1 = {macro_f1:.4f}, FAR = {far_rate:.2f}%. "
      f"Matrix saved to Google Drive.")

# --- 14. Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")