"""
Audio rendering for generated MIDI -- shared by app.py and server/main.py.

Tries FluidSynth with the bundled soundfont (FluidR3_GM.sf2 in the project
root) for real piano tone; falls back to a simple sine-wave synthesis (no
extra native dependencies) if FluidSynth isn't available.

On Windows, set FLUIDSYNTH_PATH to the FluidSynth bin directory
(e.g. C:\\...\\fluidsynth-2.4.8-win10-x64\\bin) or add it to your system PATH.

This is rendering for in-app preview only -- the MIDI file itself (what you
actually download / drag out) doesn't go through this and is unaffected.
"""
import io
import os
import sys
import shutil
import wave

import numpy as np
import pretty_midi

SAMPLE_RATE = 44100


def _add_fluidsynth_dll_dir() -> None:
    """On Windows, add the FluidSynth bin directory to the DLL search path.

    Checks (in order):
      1. FLUIDSYNTH_PATH env var (directory containing libfluidsynth-3.dll)
      2. Parent directory of `fluidsynth.exe` found on PATH
    """
    if sys.platform != "win32":
        return
    candidates: list[str] = []
    env = os.environ.get("FLUIDSYNTH_PATH", "")
    if env:
        candidates.append(env)
    exe = shutil.which("fluidsynth")
    if exe:
        candidates.append(os.path.dirname(os.path.abspath(exe)))
    for d in candidates:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                return
            except Exception:
                pass


def _to_wav_bytes(audio: np.ndarray) -> bytes | None:
    if audio is None or len(audio) == 0:
        return None
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    pcm = np.int16(audio / peak * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()


def synthesize_wav(pm: pretty_midi.PrettyMIDI, project_root: str) -> tuple[bytes | None, str]:
    """Render `pm` to a WAV preview.

    Returns (wav_bytes, method) on success, where method is "fluidsynth" or
    "sine". Returns (None, error_message) if nothing could be synthesized.
    `project_root` is where to look for a bundled soundfont.
    """
    sf2 = None
    for cand in ("soundfont.sf2", "assets/soundfont.sf2", "SalC5Light2.sf2", "FluidR3_GM.sf2"):
        p = os.path.join(project_root, cand)
        if os.path.exists(p):
            sf2 = p
            break

    if sf2:
        _add_fluidsynth_dll_dir()
        try:
            audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=sf2)
            wav = _to_wav_bytes(audio)
            if wav:
                return wav, f"fluidsynth:{os.path.basename(sf2)}"
        except Exception:
            pass  # native FluidSynth lib not found -- fall through

    try:
        audio = pm.synthesize(fs=SAMPLE_RATE)
        wav = _to_wav_bytes(audio)
        if wav:
            return wav, "sine"
    except Exception as e:
        return None, str(e)
    return None, "no audio produced"
