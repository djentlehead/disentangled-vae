"""
Audio rendering for generated MIDI -- shared by app.py and server/main.py.

Tries FluidSynth with the bundled soundfont (FluidR3_GM.sf2 in the project
root) for real piano tone; falls back to a simple sine-wave synthesis (no
extra native dependencies) if FluidSynth isn't available. See QUICKSTART.md
for how to get FluidSynth working on Windows.

This is rendering for in-app preview only -- the MIDI file itself (what you
actually download / drag out) doesn't go through this and is unaffected by
which path was used.
"""
import io
import os
import wave

import numpy as np
import pretty_midi

SAMPLE_RATE = 44100


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
    for cand in ("soundfont.sf2", "assets/soundfont.sf2", "FluidR3_GM.sf2"):
        p = os.path.join(project_root, cand)
        if os.path.exists(p):
            sf2 = p
            break

    if sf2:
        try:
            audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=sf2)
            wav = _to_wav_bytes(audio)
            if wav:
                return wav, "fluidsynth"
        except Exception:
            pass  # native FluidSynth lib probably isn't installed -- fall through

    try:
        audio = pm.synthesize(fs=SAMPLE_RATE)
        wav = _to_wav_bytes(audio)
        if wav:
            return wav, "sine"
    except Exception as e:
        return None, str(e)
    return None, "no audio produced"
