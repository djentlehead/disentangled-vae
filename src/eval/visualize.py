import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

FLUIDSYNTH_PATH = r"C:\Users\sharm\Downloads\fluidsynth-2.4.8-win10-x64\bin\fluidsynth.exe"

SOUNDFONT_PATH = r"C:\Users\sharm\Downloads\FluidR3_GM\FluidR3_GM.sf2"

ORIGINAL_MIDI_PATH = r"C:\Users\sharm\Downloads\Programming\Style Transfer\data\raw\maestro-v3.0.0\2008\MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi"

TRANSFERRED_MIDI_PATH = r"C:\Users\sharm\Downloads\Programming\Style Transfer\cycle_transfer_output.mid"


def plot_spectrogram_direct(midi_path, title, fluidsynth_path, soundfont_path):

    wav_path = "temp_audio.wav"
    try:
        if not os.path.exists(fluidsynth_path):
            print(f"Error: FluidSynth executable not found at '{fluidsynth_path}'")
            return
        if not os.path.exists(soundfont_path):
            print(f"Error: SoundFont file not found at '{soundfont_path}'")
            return
        if not os.path.exists(midi_path):
            print(f"Error: MIDI file not found at '{midi_path}'")
            return

        print(f"Synthesizing '{os.path.basename(midi_path)}' using a direct command...")
        command = [
            fluidsynth_path,
            '-niq', 
            soundfont_path,
            midi_path,
            '-F',
            wav_path,
            '-r',
            '44100'
        ]

        print(f"Running command: {' '.join(command)}") 

        try:
            process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            print("FluidSynth command completed.")
            if process.returncode != 0:
                print(f"FluidSynth Error Output:\n{process.stderr}")
                return
        except subprocess.TimeoutExpired:
            print("\n--- ERROR ---")
            print("FluidSynth command timed out after 2 minutes. It is likely hanging.")
            print("Try running the command directly in your terminal to see if it produces errors:")
            print(' '.join(command))
            return 
        except subprocess.CalledProcessError as e:
            print("\n--- ERROR ---")
            print("FluidSynth command failed.")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Error output:\n{e.stderr}")
            return

        print("Loading WAV file...")
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
             print(f"Error: Temporary WAV file '{wav_path}' was not created or is empty.")
             return

        audio_data, sr = librosa.load(wav_path, sr=44100)
        print("WAV loaded. Creating spectrogram...")

        stft = librosa.stft(audio_data)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        print("Spectrogram created.")

        print("Plotting spectrogram...")
        fig, ax = plt.subplots(figsize=(12, 6))
        img = librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity (dB)')
        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        print("Plotting complete.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
            print(f"Cleaned up temporary file: {wav_path}")

print("--- Creating Spectrogram for Original MIDI ---")
plot_spectrogram_direct(ORIGINAL_MIDI_PATH, "Original Piece Spectrogram", FLUIDSYNTH_PATH, SOUNDFONT_PATH)

print("\n--- Creating Spectrogram for Generated MIDI ---")
plot_spectrogram_direct(TRANSFERRED_MIDI_PATH, "Generated Piece Spectrogram", FLUIDSYNTH_PATH, SOUNDFONT_PATH)