# ==============================================================================
# Program Name: 00_data_generation.py
# Version: 1.0
# Description: Extracts 200 prompts from MusicCaps and generates 1,000 10-second 
#              audio clips each from MusicGen, AudioLDM-2, and Stable Audio Open.
#              Outputs are saved directly to the Google Drive project folder.
#             1.0 - Initial version with Google Drive  checkpointing and VRAM management.
# GPU Required: YES (T4 minimum, A100 recommended for speed)
# ==============================================================================

!pip install -q torch torchvision torchaudio torchsde transformers diffusers accelerate scipy soundfile pandas 

import sys
import os
import gc
import json
import zipfile
import shutil
import torch
import pandas as pd
import soundfile as sf
from google.colab import drive

drive.mount('/content/drive')

# --- 1. GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU.")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# --- 2. Configuration ---
DRIVE_DIR = "/content/drive/MyDrive/datasets"
PROJECT_DIR = "/content/drive/MyDrive/paper/ipc-watermark/"  # Persistent storage for generated audio
MUSICCAPS_ZIP = os.path.join(DRIVE_DIR, "MusicCaps.zip")
LOCAL_TEMP_DIR = "/content/temp_data"

TARGET_PROMPTS = 200
VARIATIONS_PER_PROMPT = 5
DURATION_SEC = 10

CHECKPOINT_FILE = "00_generation_checkpoint.json"

# --- 3. Google Drive Helper Functions (replacing FTP) ---
def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_filepath, remote_filename):
    """
    Copy a local file to Google Drive project folder.
    remote_filename: filename in Drive (used as-is in PROJECT_DIR).
    """
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_filename)
    try:
        shutil.copy2(local_filepath, dest_path)
        print(f"  [DRIVE OK] {local_filepath}  →  {dest_path}")
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_filepath}: {e}")

def load_from_drive(remote_filename, local_filepath):
    """
    Copy a file from Google Drive project folder to local path.
    Returns True if successful, False otherwise.
    """
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
    local_cp = "/content/temp_checkpoint.json"
    if load_from_drive(CHECKPOINT_FILE, local_cp):
        with open(local_cp, "r") as f:
            return json.load(f)
    # Return empty state if checkpoint doesn't exist
    return {"prompts_selected": False, "musiccaps_indices": [], "completed": {}}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local_cp = "/content/temp_checkpoint.json"
    with open(local_cp, "w") as f:
        json.dump(state, f)
    save_to_drive(local_cp, CHECKPOINT_FILE)
    os.remove(local_cp)  # Clean up local temp

# --- 4. Initialization & Data Loading ---
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

state = load_checkpoint()

if not state.get("prompts_selected"):
    print("Extracting MusicCaps...")
    with zipfile.ZipFile(MUSICCAPS_ZIP, 'r') as zip_ref:
        zip_ref.extract("musiccaps-public.csv", LOCAL_TEMP_DIR)
    
    df = pd.read_csv(os.path.join(LOCAL_TEMP_DIR, "musiccaps-public.csv"))
    # Sample 200 random prompts and save their indices
    sampled_df = df.sample(n=TARGET_PROMPTS, random_state=42)
    state["musiccaps_indices"] = sampled_df.index.tolist()
    state["prompts"] = sampled_df['caption'].tolist()
    state["prompts_selected"] = True
    save_checkpoint(state)
    print(f"Sampled {TARGET_PROMPTS} prompts.")
else:
    print("Loaded existing prompt selections from Drive checkpoint.")

prompts = state["prompts"]

# --- 5. Generators ---

# Helper to clear VRAM
def flush_vram():
    gc.collect()
    torch.cuda.empty_cache()

# --- A. MusicGen ---
if "musicgen" not in state["completed"] or state["completed"]["musicgen"] < TARGET_PROMPTS * VARIATIONS_PER_PROMPT:
    print("\n--- Starting MusicGen Generation ---")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium").to("cuda")
    
    completed_count = state["completed"].get("musicgen", 0)
    
    for i, prompt in enumerate(prompts):
        for var in range(VARIATIONS_PER_PROMPT):
            global_idx = (i * VARIATIONS_PER_PROMPT) + var
            if global_idx < completed_count:
                continue
                
            print(f"MusicGen: Prompt {i+1}/{TARGET_PROMPTS}, Var {var+1}/{VARIATIONS_PER_PROMPT}")
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
            # 256 tokens is roughly 5 seconds; 512 is roughly 10 seconds for MusicGen
            audio_values = model.generate(**inputs, max_new_tokens=512)
            
            audio_data = audio_values[0, 0].cpu().numpy()
            sample_rate = model.config.audio_encoder.sampling_rate
            
            filename = f"gen_musicgen_p{i:03d}_v{var}.wav"
            local_path = os.path.join(LOCAL_TEMP_DIR, filename)
            sf.write(local_path, audio_data, sample_rate)
            
            save_to_drive(local_path, filename)
            os.remove(local_path)  # Clean up local disk
            
            state.setdefault("completed", {})["musicgen"] = global_idx + 1
            save_checkpoint(state)

    del model
    del processor
    flush_vram()

# --- B. AudioLDM-2 ---
if "audioldm2" not in state["completed"] or state["completed"]["audioldm2"] < TARGET_PROMPTS * VARIATIONS_PER_PROMPT:
    print("\n--- Starting AudioLDM-2 Generation ---")
    from diffusers import AudioLDM2Pipeline
    from transformers import AutoModelForCausalLM
    
    # 1. Explicitly force the correct Causal LM model class IN FLOAT16 to match the pipeline
    correct_lm = AutoModelForCausalLM.from_pretrained(
        "cvssp/audioldm2", 
        subfolder="language_model",
        torch_dtype=torch.float16
    )
    
    # 2. Pass it into the pipeline
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", 
        language_model=correct_lm, 
        torch_dtype=torch.float16
    ).to("cuda")
    
    completed_count = state["completed"].get("audioldm2", 0)
    
    for i, prompt in enumerate(prompts):
        for var in range(VARIATIONS_PER_PROMPT):
            global_idx = (i * VARIATIONS_PER_PROMPT) + var
            if global_idx < completed_count:
                continue
                
            print(f"AudioLDM-2: Prompt {i+1}/{TARGET_PROMPTS}, Var {var+1}/{VARIATIONS_PER_PROMPT}")
            audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=DURATION_SEC).audios[0]
            
            filename = f"gen_audioldm2_p{i:03d}_v{var}.wav"
            local_path = os.path.join(LOCAL_TEMP_DIR, filename)
            sf.write(local_path, audio, 16000) 
            
            save_to_drive(local_path, filename)
            os.remove(local_path)
            
            state.setdefault("completed", {})["audioldm2"] = global_idx + 1
            save_checkpoint(state)

    del pipe
    del correct_lm
    flush_vram()
    

# --- C. Stable Audio Open ---
if "stableaudio" not in state["completed"] or state["completed"]["stableaudio"] < TARGET_PROMPTS * VARIATIONS_PER_PROMPT:
    print("\n--- Starting Stable Audio Generation ---")
    
    # --- ADDED AUTHENTICATION ---
    from huggingface_hub import login
    # PASTE YOUR TOKEN HERE:
    login(token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") 
    
    from diffusers import StableAudioPipeline
    
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
    
    completed_count = state["completed"].get("stableaudio", 0)
    
    for i, prompt in enumerate(prompts):
        for var in range(VARIATIONS_PER_PROMPT):
            global_idx = (i * VARIATIONS_PER_PROMPT) + var
            if global_idx < completed_count:
                continue
                
            print(f"StableAudio: Prompt {i+1}/{TARGET_PROMPTS}, Var {var+1}/{VARIATIONS_PER_PROMPT}")
            audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=DURATION_SEC).audios[0]
            
            filename = f"gen_stableaudio_p{i:03d}_v{var}.wav"
            local_path = os.path.join(LOCAL_TEMP_DIR, filename)
            sf.write(local_path, audio.cpu().to(torch.float32).numpy().T, 44100)
            
            save_to_drive(local_path, filename)
            os.remove(local_path)
            
            state.setdefault("completed", {})["stableaudio"] = global_idx + 1
            save_checkpoint(state)

    del pipe
    flush_vram()
    

print("\n[SUCCESS] All generative datasets created and saved to Google Drive!")