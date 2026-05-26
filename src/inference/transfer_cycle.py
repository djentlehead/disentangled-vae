import os
import sys
import torch
import pandas as pd
import random


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.train_cycle import CycleTrainer
from src.data.midi_tokenizer import MIDITokenizer

if __name__ == '__main__':

    CHECKPOINT_PATH = "lightning_logs/version_9/checkpoints/epoch=5-step=1212.ckpt"
    TRANSFER_DIRECTION = 'Bach_to_Chopin'
    OUTPUT_MIDI_PATH = "cycle_transfer_output.mid"
    MAX_SEQ_LEN = 2048 


    print("--- Finding MIDI files from metadata ---")
    METADATA_PATH = os.path.join(project_root, "data/raw/maestro-v3.0.0/maestro-v3.0.0.csv")
    MIDI_ROOT_DIR = os.path.join(project_root, "data/raw/maestro-v3.0.0")
    df = pd.read_csv(METADATA_PATH)
    bach_files = [os.path.join(MIDI_ROOT_DIR, f) for f in df[df['canonical_composer'] == 'Johann Sebastian Bach']['midi_filename']]
    chopin_files = [os.path.join(MIDI_ROOT_DIR, f) for f in df[df['canonical_composer'] == 'Frédéric Chopin']['midi_filename']]
    print(f"Found {len(bach_files)} Bach files and {len(chopin_files)} Chopin files.")
    
    if TRANSFER_DIRECTION == 'Bach_to_Chopin':
        generator_key = 'G_AB'
        CONTENT_MIDI_PATH = random.choice(bach_files)
    else:
        generator_key = 'G_BA'
        CONTENT_MIDI_PATH = random.choice(chopin_files)
    print(f"Selected file for content: {os.path.basename(CONTENT_MIDI_PATH)}")

    print("\n--- Loading model and tokenizer ---")
    trained_model = CycleTrainer.load_from_checkpoint(CHECKPOINT_PATH)
    generator = getattr(trained_model, generator_key)
    tokenizer = MIDITokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()

    print(f"--- Performing style transfer ---")

    with torch.no_grad():
        content_tokens = tokenizer.midi_to_tokens(CONTENT_MIDI_PATH)
        
        if len(content_tokens) > MAX_SEQ_LEN:
            print(f"Warning: Input sequence is too long ({len(content_tokens)}). Truncating to {MAX_SEQ_LEN} tokens.")
            content_tokens = content_tokens[:MAX_SEQ_LEN]

        content_tensor = torch.tensor([content_tokens], dtype=torch.long).to(device)
        output_logits = generator(content_tensor)
        output_tokens = output_logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

    print("--- Converting tokens back to MIDI ---")
    transferred_midi = tokenizer.tokens_to_midi(output_tokens)
    transferred_midi.write(OUTPUT_MIDI_PATH)
    
    print(f"Style transfer complete! Output saved to {OUTPUT_MIDI_PATH}")