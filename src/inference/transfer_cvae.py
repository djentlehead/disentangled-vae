import os 
import sys
import torch
import numpy as np


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)


from src.training.train_cvae import CVAETrainer
from src.data.midi_tokenizer import MIDITokenizer

if __name__ == "__main__":
    CVAE_CHECKPOINT_PATH = "lightning_logs/version_0/checkpoints/epoch=999-step=60000.ckpt"

    CONTENT_MIDI_PATH = "data/raw/maestro-v3.0.0/2013/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--1.midi"  
    
    TARGET_STYLE_NAME = "Frederic Chopin" 

    OUTPUT_MIDI_PATH = "style_transfer_output.mid"
    
    STYLE_MAP = {
        "Johann Sebastian Bach": 0, "Frederic Chopin": 1,
        "Ludwig Van Beethoven": 2, "Franz Schubert": 3
    }
    
    print("--- Loading model ---")
    model = CVAETrainer.load_from_checkpoint(CVAE_CHECKPOINT_PATH).cvae
    tokenizer = MIDITokenizer()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    
    print("--- Performing Style Transfer ---")
    print(f"Content Source: {os.path.basename(CONTENT_MIDI_PATH)}")
    print(f"Target Style: {TARGET_STYLE_NAME}")
    
    with torch.no_grad():
        content_tokens = tokenizer.midi_to_tokens(CONTENT_MIDI_PATH)
        content_tensor = torch.tensor([content_tokens], dtype=torch.long).to(device)
        
        mu, log_var = model.encode(content_tensor)
        z_c = model.reparameterize(mu, log_var)
        
        target_style_idx = STYLE_MAP[TARGET_STYLE_NAME]
        style_tensor = torch.tensor([target_style_idx], dtype=torch.long).to(device)
        
        seq_len = content_tensor.size(1)
        output_logits = model.decode(z_c, style_tensor, seq_len)
        
        output_tokens = output_logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()
        
        print(f"First 50 generated tokens: {output_tokens[:50]}")
        transferred_midi = tokenizer.tokens_to_midi(output_tokens)
    
    print("--- Converting tokens to MIDI ---")
    transferred_midi = tokenizer.tokens_to_midi(output_tokens)
    transferred_midi.write(OUTPUT_MIDI_PATH)
    
    print(f"Style transfer complete. Output saved to {OUTPUT_MIDI_PATH}")       