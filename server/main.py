"""
Style Transfer Studio -- local web server.

Serves the custom frontend (web/) and exposes the DisentangledVAE
recombination pipeline (src/inference/recombine.py) over a small JSON API.
This is now the primary way to use the tool, replacing the Streamlit app
(app.py is kept around as a reference/fallback -- it stays behaviorally
identical via the same shared pipeline module).

Run with:
    .venv\\Scripts\\python server\\main.py

then open http://127.0.0.1:8000 -- the launcher scripts do this for you
and wait for the model to finish loading before opening the browser.
"""
import base64
import io
import os
import sys
import threading
import time
import webbrowser
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import pretty_midi
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.inference.audio import synthesize_wav
from src.inference.recombine import (
    FS, N_PITCH, PITCH_LO, SEQ_LEN,
    decode_latents, encode_roll, find_checkpoint, load_model,
    midi_to_roll, postprocess_roll, roll_to_midi,
)

EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

app = FastAPI(title="Style Transfer Studio")

# The frontend is served from the same origin (StaticFiles, below), but CORS
# is enabled anyway -- harmless locally, and means the API still works if
# someone opens web/index.html directly or serves it from a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_error = None
_model_lock = threading.Lock()


def get_model():
    global _model, _model_error
    with _model_lock:
        if _model is None and _model_error is None:
            try:
                ckpt = find_checkpoint(PROJECT_ROOT)
                _model = load_model(ckpt)
            except Exception as e:
                _model_error = str(e)
    return _model, _model_error


@app.on_event("startup")
def _startup():
    # Load eagerly so the first real request isn't the one that pays for it.
    get_model()


@app.get("/api/health")
def health():
    model, err = get_model()
    return {
        "ok": model is not None,
        "error": err,
        "params": sum(p.numel() for p in model.parameters()) if model else 0,
        "latent_dim": model.hparams.latent_dim if model else None,
        "seq_len": SEQ_LEN,
        "fs": FS,
        "pitch_lo": PITCH_LO,
        "n_pitch": N_PITCH,
    }


@app.get("/api/examples")
def list_examples():
    if not os.path.isdir(EXAMPLES_DIR):
        return {"examples": []}
    files = sorted(
        f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith((".mid", ".midi"))
    )
    return {"examples": files}


def _load_pm(upload: Optional[UploadFile], example: Optional[str], label: str) -> Optional[pretty_midi.PrettyMIDI]:
    """Load a PrettyMIDI either from an uploaded file or a bundled example name."""
    if upload is not None and upload.filename:
        data = upload.file.read()
        try:
            return pretty_midi.PrettyMIDI(io.BytesIO(data))
        except Exception as e:
            raise HTTPException(400, f"Couldn't read the {label} MIDI file: {e}")
    if example:
        path = os.path.join(EXAMPLES_DIR, example)
        if not os.path.isfile(path):
            raise HTTPException(400, f"Unknown example: {example!r}")
        try:
            return pretty_midi.PrettyMIDI(path)
        except Exception as e:
            raise HTTPException(400, f"Couldn't read example {example!r}: {e}")
    return None


@app.post("/api/roll-preview")
async def roll_preview(
    midi: Optional[UploadFile] = File(None),
    example: Optional[str] = Form(None),
):
    """Returns the binarized input piano roll for a MIDI file/example, so the
    frontend can show it immediately, before the user clicks Generate."""
    pm = _load_pm(midi, example, "input")
    if pm is None:
        raise HTTPException(400, "Provide a midi file or an example name.")
    roll = midi_to_roll(pm)
    return {"roll": roll.astype(int).tolist(), "notes": int(roll.sum())}


@app.post("/api/generate")
async def generate(
    content_midi: Optional[UploadFile] = File(None),
    content_example: Optional[str] = Form(None),
    style_midi: Optional[UploadFile] = File(None),
    style_example: Optional[str] = Form(None),
    exploration: bool = Form(False),
    threshold: float = Form(0.35),
    min_len: int = Form(2),
    gap_merge: int = Form(2),
    beat_steps: int = Form(0),
    bpm: float = Form(120.0),
    rhythm_scale: float = Form(1.0),
    rhythm_noise: float = Form(0.0),
    pitch_scale: float = Form(1.0),
    pitch_noise: float = Form(0.0),
    rhythm_sigma: float = Form(1.0),
    pitch_sigma: float = Form(1.0),
    seed: int = Form(42),
    max_polyphony: int = Form(8),
):
    model, err = get_model()
    if model is None:
        raise HTTPException(500, f"Model not loaded: {err}")

    if exploration:
        dim = model.hparams.latent_dim
        g = torch.Generator().manual_seed(int(seed))
        z_r = torch.randn(1, dim, generator=g) * rhythm_sigma
        z_p = torch.randn(1, dim, generator=g) * pitch_sigma
    else:
        content_pm = _load_pm(content_midi, content_example, "content")
        if content_pm is None:
            raise HTTPException(400, "Provide a content MIDI file or example for Reconstruct/Recombine mode.")
        style_pm = _load_pm(style_midi, style_example, "style")

        mu_r, mu_p = encode_roll(model, midi_to_roll(content_pm))
        if style_pm is not None:
            _, mu_p = encode_roll(model, midi_to_roll(style_pm))

        z_r = mu_r * rhythm_scale + (torch.randn_like(mu_r) * rhythm_noise if rhythm_noise else 0)
        z_p = mu_p * pitch_scale + (torch.randn_like(mu_p) * pitch_noise if pitch_noise else 0)

    # Same decode -> postprocess -> MIDI steps recombine() runs internally,
    # done inline here so we keep pr_bin around for the frontend's piano-roll
    # canvas instead of re-deriving it with a lossy MIDI round-trip.
    pr_raw = decode_latents(model, z_r, z_p)
    pr_bin = postprocess_roll(pr_raw, threshold, min_len, gap_merge, beat_steps, max_polyphony)
    pm_out = roll_to_midi(pr_bin, bpm=float(bpm))
    note_count = len(pm_out.instruments[0].notes) if pm_out.instruments else 0
    density = float(pr_bin.mean())

    midi_buf = io.BytesIO()
    pm_out.write(midi_buf)
    midi_b64 = base64.b64encode(midi_buf.getvalue()).decode("ascii")

    audio_b64, audio_method = None, None
    if note_count > 0:
        wav, method = synthesize_wav(pm_out, PROJECT_ROOT)
        if wav:
            audio_b64 = base64.b64encode(wav).decode("ascii")
            audio_method = method
        else:
            audio_method = method  # error string, surfaced to the UI

    return JSONResponse({
        "roll": pr_bin.astype(int).tolist(),
        "note_count": note_count,
        "density": density,
        "z_rhythm": z_r.squeeze(0).detach().tolist(),
        "z_pitch": z_p.squeeze(0).detach().tolist(),
        "midi_base64": midi_b64,
        "audio_base64": audio_b64,
        "audio_method": audio_method,
    })


# Serve the custom frontend. Mounted last so it doesn't shadow /api/*.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def _open_browser_when_ready(url: str):
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    webbrowser.open(url)


if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
