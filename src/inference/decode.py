import torch
import sys
import os
import numpy as np


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from src.training.train_baseline import MusicLanguageModel
from src.data.midi_tokenizer import MIDITokenizer

def generate_sequence(model, tokenizer, prompt_tokens, max_length=1024):
   
    device = next(model.parameters()).device

    generated_tokens = prompt_tokens.copy()

    with torch.no_grad():
        for _ in range(max_length - len(prompt_tokens)):

            input_tensor = torch.tensor([generated_tokens], dtype=torch.long).to(device)

            output_logits = model(input_tensor)

            next_token_logits = output_logits[0, -1, :]

            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            next_token = torch.multinomial(next_token_probs, num_samples=1).item()

            generated_tokens.append(next_token)

    return generated_tokens

if __name__ == '__main__':

    CHECKPOINT_PATH = "lightning_logs/version_2/checkpoints/epoch=9-step=1600.ckpt"
    OUTPUT_MIDI_PATH = "generated_baseline.mid"

    print("Loading model and tokenizer")

    model = MusicLanguageModel.load_from_checkpoint(CHECKPOINT_PATH)
    tokenizer = MIDITokenizer()


    prompt_str = ["Velocity_20", "Note-On_60"]
    prompt_tokens = [tokenizer.event_to_int[ev] for ev in prompt_str]

    print(f"Generating sequence of max length 1024")
    generated_tokens = generate_sequence(model, tokenizer, prompt_tokens)


    print(f"Converting {len(generated_tokens)} tokens back to MIDI ")
    generated_midi = tokenizer.tokens_to_midi(generated_tokens)


    generated_midi.write(OUTPUT_MIDI_PATH)
    print(f"Successfully saved generated music to {OUTPUT_MIDI_PATH}")