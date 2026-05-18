# ipc-watermark

**Inter-Harmonic Phase Coherence Watermarking for Provenance Attribution of AI-Generated Music**

Official code repository for the paper:

> Sridharan Sankaran, "Coherent Spectral Watermarking for Provenance Attribution of AI-Generated Music," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2026. *(under review)*

---

## Overview

This repository contains the complete experimental pipeline for IPC watermarking — a training-free method that embeds generator identity into the **phase relationships between harmonically related STFT bins** of AI-generated music, constrained by a psychoacoustic masking budget.

The method achieves:
- **PEAQ ODG −0.15** (imperceptible) vs −1.99 and −1.77 for classical and neural baselines
- **Macro-F1 = 0.9967** on 3-class closed-set attribution (3,000 clips)
- **BER < 1.3%** under MP3 compression at 64 kbps
- **Zero-shot generalisation** to an unseen generator (MusicGen-large) without retraining

---

## Repository Structure

ipc-watermark/
│
├── 00_data_generation.py          # Stage 0: Corpus generation
├── 01_exp6_ablation_study.py      # Stage 1: H×B hyperparameter ablation
├── 02_exp1_imperceptibility.py    # Stage 2: Imperceptibility verification
├── 03_exp2_exp4_detection.py      # Stage 3: Detection accuracy & attribution
├── 04_exp3_robustness.py          # Stage 4: Passive & adversarial robustness
├── 04_exp3_robustness_ext.py      # Stage 4 ext: + Temporal alignment sensitivity
├── 05_exp5_genre_analysis.py      # Stage 5: Genre-stratified robustness
├── 06_unified_baselines.py        # Stage 6: Three-way baseline comparison
├── 07_baseline_analysis.py        # Stage 6 analysis: Baseline result aggregation
├── 07_exp7_generalisation.py      # Stage 7: Cross-generator generalisation probe
├── 08_exp8_delta_sweep.py         # Stage 8: Perceptual budget sensitivity (Δ sweep)
├── 200-musiccaps_prompts.xlsx     # 200 music prompts selected in 00_data_generation.py
│
└── README.md

---

## Experimental Pipeline

The scripts are numbered by **implementation order**, which differs from the paper's presentation order. Run them in the sequence below.

### Stage 0 — Corpus Generation
**`00_data_generation.py`**

Selects and extracts 200 text prompts from MusicCaps and generates 1,000 × 10-second audio clips from each of three generative models: MusicGen-medium, AudioLDM-2, and Stable Audio Open. All outputs are saved directly to the Google Drive project folder with checkpoint-based resumption. The 200 prompts are stored in 00_generation_checkpoint.json.

- **GPU:** Required (T4 minimum, A100 recommended)
- **Output:** 3,000 WAV files at 16 kHz + `metadata.csv`

---

### Stage 1 — Ablation Study (Paper: Experiment 6 → Section: Ablation Study)
**`01_exp6_ablation_study.py`**

Sweeps `H ∈ {4, 6, 8, 12}` harmonic bins and `B ∈ {16, 32, 64}` payload bits on 200 MusicGen clips under MP3-128 degradation. Identifies the Pareto-optimal configuration `H*=8, B*=32` balancing BER and PEAQ ODG. **Run first** — all subsequent experiments use the locked hyperparameters.

- **GPU:** Required
- **Output:** `fig_06_01_ber_heatmap.png`, `fig_06_02_peaq_heatmap.png`, Pareto table

---

### Stage 2 — Imperceptibility Verification (Paper: Experiment 1)
**`02_exp1_imperceptibility.py`**

Applies the IPC watermark (`H*=8, B*=32`) to all 3,000 clips. Computes PESQ and SI-SDR. Saves watermarked audio to Google Drive for use by all downstream scripts.

- **GPU:** Required (SI-SDR via TorchMetrics)
- **Output:** `experiment_1_results.json`, `fig_01_01_pesq_boxplots.png`, `fig_01_02_sisdr_boxplots.png`
- **Key result:** PESQ mean 4.07–4.15 across all three generators

---

### Stage 3 — Detection & Attribution (Paper: Experiments 2 & 4)
**`03_exp2_exp4_detection.py`**

Downloads watermarked and original clips from Google Drive. Performs binary detection (ROC/AUC) and 3-class closed-set attribution (argmin BER confusion matrix). Single-pass I/O optimisation processes both experiments simultaneously.

- **GPU:** Not required
- **Output:** `fig_03_01_roc_curves.png`, `fig_03_02_confusion_matrix.png`, `experiment_2_4_results.json`
- **Key result:** AUC = 0.999 across all generators; Macro-F1 = 0.9967 vs CNN baseline 0.8140

---

### Stage 4 — Robustness (Paper: Experiment 3)
**`04_exp3_robustness.py`** / **`04_exp3_robustness_ext.py`**

Tests watermark robustness across:
- **5 passive degradations:** MP3 (64–192 kbps), AWGN (20–40 dB SNR), TSM (×0.9, ×1.1), pitch-shift (±1, ±2 semitones), EnCodec (3–12 kbps)
- **1 active attack:** PGD watermark removal (T = 5, 10, 20 steps) with complex-domain phase loss, bounded by PEAQ ODG > −1.0
- **Temporal alignment sensitivity** (ext version only): leading sample drift of {0, 10, 50, 100, 500, 1000} samples

The `_ext` version incorporates Experiment 9 (temporal alignment) as an integrated pipeline extension.

- **GPU:** Required (EnCodec re-encoding, PyTorch PGD autograd)
- **Output:** `fig_04_01_robustness_part1.png`, `fig_04_01_robustness_part2.png`, `fig_04_03_drift_sensitivity.png`
- **Key result:** BER < 1.3% under MP3-64; complete failure under TSM and pitch-shift; synchronisation cliff at 10 samples (0.625 ms)

---

### Stage 5 — Genre-Stratified Analysis (Paper: Experiment 5)
**`05_exp5_genre_analysis.py`**

Generates 200 clips per genre (Classical, Jazz, Pop, Electronic, Metal) using MusicGen-small conditioned on MTG-Jamendo genre tags. Watermarks and evaluates under MP3-128. Runs one-way ANOVA on genre effect.

- **GPU:** Required (MusicGen inference)
- **Output:** `fig_05_01_genre_ber_final.png`
- **Key result:** ANOVA F(4,995) = 12.55, p < 0.001, η² = 0.048; Jazz median BER ≈ 0%; Electronic outliers reaching 59.38%

---

### Stage 6 — Three-Way Baseline Comparison (Paper: Baseline Comparisons)
**`06_unified_baselines.py`** + **`07_baseline_analysis.py`**

Evaluates Spread-Spectrum (STFT log-magnitude, α = 0.01) and WavMark (v0.0.3, 16 effective user bits) against IPC on a 249-clip stratified sample across Experiments 1–4 conditions plus a combined MP3-64 + TSM-0.9× attack.

- **GPU:** Recommended (WavMark neural inference)
- **Output:** `fig_06_01_baseline_robustness.png`, `baseline_results.json`, LaTeX tables
- **Key result:** IPC ODG −0.15 vs SS −1.994 and WavMark −1.771; all three methods collapse under TSM except WavMark (learned robustness)

---

### Stage 7 — Cross-Generator Generalisation Probe (Paper: Experiment 7)
**`07_exp7_generalisation.py`**

Downloads 200 MusicGen-large (3.3B) holdout clips from Google Drive. Runs two sub-tasks:
- **Sub-task A:** FAR measurement — presents unwatermarked holdout clips to detector calibrated on three legacy generators; τ = 0.2812 at FPR = 1%
- **Sub-task B:** 4-class attribution — embeds new identity w₄ into holdout clips, expands confusion matrix to 4×4

- **GPU:** Required (embedding pipeline)
- **Output:** `fig_07_01_confusion_4class.png`, LaTeX generalisation table
- **Key result:** FAR = 1.5% (Sub-task A); 4-class Macro-F1 = 0.9962, zero cross-generator confusions between MusicGen variants (Sub-task B)

---

### Stage 8 — Perceptual Budget Sensitivity (Paper: Experiment 8)
**`08_exp8_delta_sweep.py`**

Sweeps `Δ_max ∈ {π/16, π/8, π/4, π/2, π}` on 200 MusicGen clips under MP3-128. Measures BER and PEAQ ODG at each level to empirically justify the `Δ* = π/4` operating point.

- **GPU:** Not required
- **Output:** `fig_08_01_delta_sweep.png`, delta sweep table
- **Key result:** BER plateau 2.30–2.89% across [π/16, π/2]; catastrophic BER collapse at π (phase wrapping); imperceptibility boundary crossed between π/16 and π/8

---

## Requirements

All scripts run in **Google Colab** with a T4 GPU runtime. The Google Drive folder path must be configured in each script's constants block:

```python
# Configure in each script:
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"

The project expects the following folder structure in Google Drive:
- `PROJECT_DIR/` — Contains all checkpoints, generated audio files, results, and figures
- `/content/drive/MyDrive/datasets/` — Contains MusicCaps.zip, SpeechCommandsV2.zip, and Jamendo.zip

### Core dependencies

pip install torch torchaudio librosa soundfile numpy scipy pandas \
            matplotlib seaborn scikit-learn pesq torchmetrics \
            audiocraft wavmark==0.0.3 encodec
```

### Additional tools

- **LAME** (MP3 encoding): `apt-get install -y lame`
- **peaqb-fast** (PEAQ ODG): compile from [source](https://github.com/HSU-ANT/gstpeaq) via `./configure && make`; binary at `src/peaqb`

---

## Locked Hyperparameters

All experiments use the configuration selected by the Stage 1 ablation:

| Parameter | Value | Description |
|---|---|---|
| `H*` | 8 | Number of harmonic bins |
| `B*` | 32 | Payload bit capacity |
| `Δ_max` | π/4 | Maximum phase perturbation (radians) |
| `SR` | 16,000 Hz | Sampling rate |
| `N_FFT` | 2,048 | FFT size |
| `HOP_LENGTH` | 512 | STFT hop length |

---

## Datasets

| Dataset | Role | Access |
|---|---|---|
| [MusicCaps](https://arxiv.org/abs/2301.11325) | 200 text prompts for corpus generation | Public (HuggingFace) |
| [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) | Genre tags for Experiment 5 | Public (CC-licensed) |
| MusicGen-medium outputs | Primary evaluation corpus (1,000 clips) | Generated via AudioCraft |
| AudioLDM-2 outputs | Primary evaluation corpus (1,000 clips) | Generated via HuggingFace |
| Stable Audio Open outputs | Primary evaluation corpus (1,000 clips) | Generated via HuggingFace |
| MusicGen-large outputs | Holdout corpus for Experiment 7 (200 clips) | Generated via AudioCraft |

---

## Generated Figures

All figures are saved to the Google Drive project folder upon generation.

| Figure File | Generated By | Description |
|---|---|---|
| `fig_06_01_ber_heatmap.png` | `01_exp6_ablation_study.py` | Heatmap of mean BER across the H×B ablation grid (H ∈ {4,6,8,12}, B ∈ {16,32,64}) under MP3-128 degradation. Darker red = higher BER. Used to identify Pareto-optimal H*=8, B*=32. |
| `fig_06_02_peaq_heatmap.png` | `01_exp6_ablation_study.py` | Heatmap of mean PEAQ ODG across the same H×B ablation grid. Paired with the BER heatmap to visualise the robustness–imperceptibility trade-off surface. |
| `fig_01_01_pesq_boxplots.png` | `02_exp1_imperceptibility.py` | Box plots of PESQ scores for watermarked audio across three generators (MusicGen, AudioLDM-2, Stable Audio Open). Shows distribution of perceptual quality at H*=8, B*=32 over 1,000 clips per generator. |
| `fig_01_02_sisdr_boxplots.png` | `02_exp1_imperceptibility.py` | Box plots of SI-SDR (dB) for watermarked audio across three generators. Complements PESQ with a signal-level distortion measure. Median values reported per generator. |
| `fig_03_01_roc_curves.png` | `03_exp2_exp4_detection.py` | ROC curves for binary watermark detection (watermarked vs. unwatermarked) under clean conditions for all three generators. AUC = 0.999 across all three. Includes random-chance diagonal. |
| `fig_03_02_confusion_matrix.png` | `03_exp2_exp4_detection.py` | 3×3 confusion matrix for closed-set multi-generator attribution using argmin BER decision rule. Rows = true generator, columns = attributed generator. Macro-F1 = 0.9967 vs passive CNN baseline 0.8140. |
| `fig_04_01_robustness_part1.png` | `04_exp3_robustness.py` / `04_exp3_robustness_ext.py` | Two-panel figure: (top) BER vs. MP3 bitrate (64–192 kbps); (bottom) BER vs. pitch-shift semitones (±1, ±2). Demonstrates quantization robustness and complete failure under geometric spectral shifts. |
| `fig_04_01_robustness_part2.png` | `04_exp3_robustness.py` / `04_exp3_robustness_ext.py` | Two-panel figure: (top) BER vs. TSM ratio (×0.9, ×0.95, ×1.05, ×1.1); (bottom) BER vs. PGD iteration steps (T = 5, 10, 20). Shows TSM vulnerability and near-linear adversarial BER growth within PEAQ ODG > −1.0 constraint. |
| `fig_04_03_drift_sensitivity.png` | `04_exp3_robustness_ext.py` | BER vs. leading sample drift in milliseconds (0, 0.625, 3.125, 6.25, 31.25, 62.5 ms). Symlog x-axis to spread the near-zero cliff region. Reference lines at BER = 5% and BER = 10%; vertical dashed line at STFT hop boundary (32 ms). Demonstrates near-instantaneous synchronisation cliff at 10 samples. |
| `fig_05_01_genre_ber_final.png` | `05_exp5_genre_analysis.py` | Box plots of BER under MP3-128 stratified by musical genre (Classical, Jazz, Pop, Electronic, Metal), 200 tracks per genre. Illustrates the harmonic clarity dependency: Jazz near-zero BER, Electronic catastrophic outliers reaching 59.38%. |
| `fig_06_01_baseline_robustness.png` | `06_unified_baselines.py` / `07_baseline_analysis.py` | Grouped box plots comparing BER distributions for IPC (proposed), Spread-Spectrum, and WavMark across four degradation conditions: Clean, MP3-64, TSM-0.9×, and Combined (MP3-64 + TSM-0.9×). Red dashed reference line at BER = 0.5 (random chance). |
| `fig_07_01_confusion_4class.png` | `07_exp7_generalisation.py` | 4×4 confusion matrix expanding the 3-class attribution task to include MusicGen-large as a holdout generator. Columns include an "unattributed" class for clips rejected by the τ threshold. Macro-F1 = 0.9962. |
| `fig_08_01_delta_sweep.png` | `08_exp8_delta_sweep.py` | Dual-axis line plot: BER % (left axis, blue) and PEAQ ODG (right axis, red dashed) vs. Δ_max ∈ {π/16, π/8, π/4, π/2, π}. Error bars show ±1 std across 200 clips. Reference lines at BER = 5% and ODG = −1.0. Vertical dashed line marks selected operating point Δ* = π/4. |

---

## 200 Selected Prompts

`00_data_generation.py` randomly selects 200 prompts from MusicCaps (uses `random_state=42` for reproducibility) and stores them to `00_generation_checkpoint.json` in the Google Drive project folder. These selected prompts are made available in `200-musiccaps_prompts.xlsx` for reproducibility.
```
