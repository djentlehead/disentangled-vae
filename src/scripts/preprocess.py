import os
import pretty_midi
import music21
import pandas as pd
from tqdm import tqdm
import io


DATA_ROOT = "data/raw/maestro-v3.0.0"
METADATA_PATH = os.path.join(DATA_ROOT, "maestro-v3.0.0.csv")
OUTPUT_DIR = "data/processed/normalized_midi"


os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(METADATA_PATH)

def normalize_midi_key(midi_obj: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:

    midi_bytes = io.BytesIO()
    midi_obj.write(midi_bytes)
    midi_bytes.seek(0)

    score = music21.converter.parse(midi_bytes.read(), format='midi')
    key = score.analyze('key')
    target_key = music21.key.Key('C') if key.mode == 'major' else music21.key.Key('a')
    interval = music21.interval.Interval(key.tonic, target_key.tonic)
    
    
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += interval.semitones
    
    return midi_obj


print(f"Starting key normalization for {len(df)} files...")

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    midi_path_relative = row['midi_filename']
    midi_path_full = os.path.join(DATA_ROOT, midi_path_relative)
    output_filename = os.path.basename(midi_path_relative)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if os.path.exists(output_path):
        continue

    try:
        midi_obj = pretty_midi.PrettyMIDI(midi_path_full)
        normalized_midi = normalize_midi_key(midi_obj)
        normalized_midi.write(output_path)
    except Exception as e:
        print(f"Failed to process {midi_path_full}: {e}")

print("Key normalization complete!")