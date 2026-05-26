import os
import copy
import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAESTRO_CSV = os.path.join(PROJECT_ROOT, "data/raw/maestro-v3.0.0/maestro-v3.0.0.csv")
MAESTRO_ROOT = os.path.join(PROJECT_ROOT, "data/raw/maestro-v3.0.0")

SAVE_PATH = os.path.join(PROJECT_ROOT, "data/processed/vae_dataset.npz")

CHUNK_SIZE = 256       
PIANO_ROLL_FS = 24  
RHYTHM_PITCH = 60     
QUANTIZE_STEP_16TH = 0.125 


def chunk_piano_roll(pr, chunk_size):
    pr_transposed = pr.T
    
    num_chunks = len(pr_transposed) // chunk_size
    chunks = []
    
    for i in range(num_chunks):
        chunk = pr_transposed[i * chunk_size : (i + 1) * chunk_size]
        chunks.append(chunk)
        
    return chunks

def process_midi_file(file_path, fs, chunk_size, rhythm_pitch, quant_step):
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
        
        pr_original = pm.get_piano_roll(fs=fs)
        pr_original_bin = (pr_original > 0).astype(np.float32)
        original_chunks = chunk_piano_roll(pr_original_bin, chunk_size)

        pm_rhythm = copy.deepcopy(pm)
        for instrument in pm_rhythm.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note.pitch = rhythm_pitch
        
        pr_rhythm = pm_rhythm.get_piano_roll(fs=fs)
        pr_rhythm_bin = (pr_rhythm > 0).astype(np.float32) 
        rhythm_chunks = chunk_piano_roll(pr_rhythm_bin, chunk_size)

        pm_pitch = copy.deepcopy(pm)
        for instrument in pm_pitch.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note.start = round(note.start / quant_step) * quant_step
                    note.end = round(note.end / quant_step) * quant_step
                    
                    if note.end <= note.start:
                         note.end = note.start + quant_step
        
        pr_pitch = pm_pitch.get_piano_roll(fs=fs)
        pr_pitch_bin = (pr_pitch > 0).astype(np.float32) 
        pitch_chunks = chunk_piano_roll(pr_pitch_bin, chunk_size)
        

        min_chunks = min(len(original_chunks), len(rhythm_chunks), len(pitch_chunks))
        
        return (original_chunks[:min_chunks], 
                rhythm_chunks[:min_chunks], 
                pitch_chunks[:min_chunks])

    except Exception as e:
        warnings.warn(f"Could not process file {file_path}: {e}")
        return None, None, None

def main():
    print("Starting VAE data preprocessing...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    try:
        df = pd.read_csv(MAESTRO_CSV)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {MAESTRO_CSV}")
        return

    composers_to_process = [
        "Johann Sebastian Bach",
        "Frédéric Chopin",
        "Ludwig van Beethoven",
        "Wolfgang Amadeus Mozart"
    ]
    df_filtered = df[df['canonical_composer'].isin(composers_to_process)]
    
    if len(df_filtered) == 0:
        print(f"Error: No files found for the specified composers: {composers_to_process}")
        print("Check composer names in the MAESTRO_CSV or in the script.")
        return


    all_original = []
    all_rhythm = []
    all_pitch = []

    print(f"Found {len(df_filtered)} MIDI files to process (out of {len(df)} total)...")

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing MIDI files"):
        file_path = os.path.join(MAESTRO_ROOT, row['midi_filename'])
        
        if not os.path.exists(file_path):
            warnings.warn(f"File not found: {file_path}")
            continue
            
        original_chunks, rhythm_chunks, pitch_chunks = process_midi_file(
            file_path, PIANO_ROLL_FS, CHUNK_SIZE, RHYTHM_PITCH, QUANTIZE_STEP_16TH
        )
        
        if original_chunks:
            all_original.extend(original_chunks)
            all_rhythm.extend(rhythm_chunks)
            all_pitch.extend(pitch_chunks)

    print(f"\nTotal chunks created: {len(all_original)}")

    X_original = np.array(all_original)
    X_rhythm = np.array(all_rhythm)
    X_pitch = np.array(all_pitch)

    print(f"Shape of X_original: {X_original.shape}")
    print(f"Shape of X_rhythm: {X_rhythm.shape}")
    print(f"Shape of X_pitch: {X_pitch.shape}")

    print(f"Saving dataset to {SAVE_PATH}...")
    np.savez_compressed(
        SAVE_PATH, 
        X_original=X_original, 
        X_rhythm=X_rhythm, 
        X_pitch=X_pitch
    )

    print("Preprocessing complete. ")

if __name__ == "__main__":
    warnings.filterwarnings('ignore', 'Trying to estimate tuning from empty frequency set')
    main()
    

