import os 
import numpy as np
from tqdm import tqdm
import sys


sys.path.append(os.getcwd())
from src.data.midi_tokenizer import MIDITokenizer

INPUT_DIR = "data/processed/fully_processed"
OUTPUT_DIR = "data/processed/tokenized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def tokenize_dataset():
    tokenizer = MIDITokenizer()
    midi_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('midi', '.mid'))]
    
    print(f"Found {len(midi_files)} MIDI files to tokenize")
    
    for filename in tqdm(midi_files, desc="Tokenizing dataset"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".npy"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if os.path.exists(output_path):
            continue  
            
        try:
            tokens = tokenizer.midi_to_tokens(input_path)
            
            np.save(output_path, np.array(tokens, dtype=np.uint16))
        except Exception as e:
            print(f"\nFailed to tokenize {filename}: {e}")
            
            
if __name__ == "__main__":
    tokenize_dataset()
    print("\nDataset Tokenization complete")
    print(f"Tokenized files are saved in: {OUTPUT_DIR}")