import pandas as pd
import numpy as np
import pickle
import os
import ast
from sklearn.model_selection import train_test_split
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp

# Global cache for audio files (per process)
audio_cache = {}

def load_audio_cached(file_path, sr=16000, start_sample=0):
    """Load audio file with caching to avoid repeated disk I/O"""
    cache_key = (file_path, start_sample)

    if cache_key in audio_cache:
        return audio_cache[cache_key].copy()

    try:
        # Calculate 2 minutes of audio at 16000 Hz sampling rate
        duration_samples = sr * 120  # 16000 * 120 = 1920000 samples (2 minutes)

        # Read only the specified segment starting from start_sample
        audio, _ = sf.read(file_path, start=start_sample, frames=duration_samples)

        # Cache the loaded audio
        audio_cache[cache_key] = audio.copy()
        return audio
    except Exception as e:
        # print(f"Error loading {file_path}: {e}")
        return None

def load_audio(file_path, sr=16000, start_sample=0):
    """Load audio file using soundfile with optional start position and fixed 2-minute duration"""
    return load_audio_cached(file_path, sr, start_sample)

def normalize_audio_loudness(audio, target_lufs=-23.0):
    """Normalize audio loudness to target LUFS level using RMS-based approach"""
    if audio is None or len(audio) == 0:
        return audio

    # Calculate RMS of the audio
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        return audio

    # Convert RMS to LUFS-like scale
    current_lufs = 20 * np.log10(rms)

    # Calculate gain needed to reach target loudness
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain
    normalized_audio = audio * gain_linear

    # Prevent clipping by checking peak after normalization
    max_val = np.max(np.abs(normalized_audio))
    if max_val > 1.0:
        # If clipping would occur, reduce gain to prevent it
        clipping_gain = 0.95 / max_val  # Leave some headroom
        normalized_audio = normalized_audio * clipping_gain

    return normalized_audio

def process_single_sample(row_data):
    """Process a single row from the dataframe"""
    idx, row = row_data
    
    try:
        mix_path = row['mix']
        source1_path = row['source1']
        source2_path = row['source2']
        source1_start = int(row['source1_start'])
        source2_start = int(row['source2_start'])
        
        # Handle vector parsing safely
        s1_vec_raw = row['source1_vector']
        s2_vec_raw = row['source2_vector']
        
        source1_vector = ast.literal_eval(s1_vec_raw) if isinstance(s1_vec_raw, str) else s1_vec_raw
        source2_vector = ast.literal_eval(s2_vec_raw) if isinstance(s2_vec_raw, str) else s2_vec_raw

        # Load source audios with start positions
        source1 = load_audio(source1_path, start_sample=source1_start)
        source2 = load_audio(source2_path, start_sample=source2_start)

        # Skip if any audio failed to load
        if source1 is None or source2 is None:
            return None

        # Load mix audio (if exists, otherwise create from sources)
        if os.path.exists(mix_path):
            if "_CH/" in mix_path:
                # For CH files, start from 3 seconds (16000*3) and read 2 minutes
                mix_start = 16000 * 3
                mix = load_audio(mix_path, start_sample=mix_start)
            else:
                # Read from beginning for 2 minutes
                mix = load_audio(mix_path, start_sample=0)
            if mix is None:
                return None
        else:
            # Create mix by adding sources
            mix = source1 + source2
        
        # Normalize audio loudness
        source1 = normalize_audio_loudness(source1)
        source2 = normalize_audio_loudness(source2)
        mix = normalize_audio_loudness(mix)

        # Ensure all audios have the expected length (2 minutes at 16000 Hz)
        expected_length = 16000 * 120  # 1920000 samples
        if len(source1) != expected_length or len(source2) != expected_length or len(mix) != expected_length:
            return None

        # Create data sample
        sample = {
            'source1': source1,
            'source2': source2,
            'mix': mix,
            'mix_path': mix_path,
            'source1_path': source1_path,
            'source2_path': source2_path,
            'source1_vector': source1_vector,
            'source2_vector': source2_vector,
            'length': len(mix)
        }
        return sample
        
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None

def process_and_save_split(df, output_path, desc):
    """Process a dataframe split and save to pickle"""
    print(f"\nProcessing {desc} ({len(df)} samples)...")
    
    # Convert dataframe rows to list of tuples for map
    rows = list(df.iterrows())
    
    if len(rows) == 0:
        print(f"Skipping {desc} (empty)")
        return

    # Use max CPUs available but cap at 16 to avoid excessive overhead/memory
    # Adjust this based on your system memory
    num_workers = min(mp.cpu_count(), 16) 
    
    # Create output directory if it doesn't exist (handle nested dirs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        results = list(tqdm(executor.map(process_single_sample, rows), total=len(rows)))
        
    # Filter out None results
    valid_samples = [s for s in results if s is not None]
    print(f"Successfully processed {len(valid_samples)}/{len(df)} samples for {desc}")
    
    # Save to pickle
    print(f"Saving {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(valid_samples, f)
    print("Done.")

def create_dataset(csv_path, output_dir='data'):
    """Create dataset from CSV file with train/val/test split (8:1:1)"""
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        return

    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split dataset (8:1:1) FIRST to save memory
    # Check if we have enough samples to split
    if len(df) >= 10:
        train_val, test = train_test_split(df, test_size=0.1, random_state=42)
        train, val = train_test_split(train_val, test_size=0.111, random_state=42)  # 0.111 * 0.9 ≈ 0.1
    else:
        print("Warning: Dataset too small for splitting, putting everything in train")
        train, val, test = df, pd.DataFrame(), pd.DataFrame()

    print(f"Split sizes -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Process and save each split separately to manage memory
    process_and_save_split(train, os.path.join(output_dir, 'train.pkl'), "Train Set")
    if not val.empty:
        process_and_save_split(val, os.path.join(output_dir, 'valid.pkl'), "Validation Set")
    if not test.empty:
        process_and_save_split(test, os.path.join(output_dir, 'test.pkl'), "Test Set")

    print("All datasets saved successfully!")

if __name__ == "__main__":
    csv_path = "mix_sources.csv"
    create_dataset(csv_path)
