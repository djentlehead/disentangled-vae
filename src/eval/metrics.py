import torch
import os
import sys
import numpy as np
import pretty_midi
from Levenshtein import distance as levenshtein_distance
from src.training.train_classifier import ClassifierTrainer
from src.data.midi_tokenizer import MIDITokenizer

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    


def calculate_style_accuracy(midi_files, target_style_label, classifier_ckpt_path):
    print("Calculating Style Match Accuracy")
    model = ClassifierTrainer.load_from_checkpoint(classifier_ckpt_path)
    model.eval()
    
    tokenizer = MIDITokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct_predictions = 0
    total_files = len(midi_files)

    for midi_path in midi_files:
        try:
            tokens = tokenizer.midi_to_tokens(midi_path)
            token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(token_tensor)
                prediction = torch.argmax(logits, dim=1).item()
            
            if prediction == target_style_label:
                correct_predictions += 1
        except Exception as e:
            print(f"Could not process {midi_path}: {e}")
            total_files -= 1
            
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    print(f"Result: {correct_predictions}/{total_files} files correctly classified.")
    return accuracy


def extract_melody_pitch_sequence(midi_path, resolution=4):
    pm = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = pm.get_piano_roll(fs=resolution) 
    melody_sequence = []
    for time_slice in piano_roll.T:
        active_pitches = np.where(time_slice > 0)[0]
        if len(active_pitches) > 0:
            melody_sequence.append(np.max(active_pitches))
        else:
            melody_sequence.append(-1) 
    return melody_sequence

def calculate_melody_distance(original_midi_path, transferred_midi_path):
    
    original_melody = extract_melody_pitch_sequence(original_midi_path)
    transferred_melody = extract_melody_pitch_sequence(transferred_midi_path)
    
    dist = levenshtein_distance(original_melody, transferred_melody)
    
    max_len = max(len(original_melody), len(transferred_melody))
    similarity = 1.0 - (dist / max_len) if max_len > 0 else 0
    
    return similarity