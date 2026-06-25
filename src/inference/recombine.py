"""
Shared DisentangledVAE recombination pipeline.

This is the single source of truth for "encode a MIDI clip into rhythm/pitch
latents, optionally swap one half with another clip's, decode back to a piano
roll, clean it up, and write MIDI" -- used by both:

  - app.py                          (Streamlit desktop app)
  - plugin/server/inference_server.py   (HTTP server for the JUCE plugin)

Keeping this logic in one place means the plugin and the desktop app can never
drift apart in behavior. If you change the model's pre/post-processing, change
it here once.
"""
import os
import inspect

import numpy as np
import torch
import pretty_midi

from src.models.disentangled_vae import DisentangledVAE

FS       = 8
SEQ_LEN  = 256
PITCH_LO = 40
N_PITCH  = 32


def find_checkpoint(project_root: str) -> str | None:
    """Look in lightning_logs/disentangled_vae for the newest .ckpt file."""
    ckpt_dir = os.path.join(project_root, "lightning_logs", "disentangled_vae")
    if not os.path.isdir(ckpt_dir):
        return None
    cks = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not cks:
        return None
    cks.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)), reverse=True)
    return os.path.join(ckpt_dir, cks[0])


def load_model(checkpoint_path: str) -> DisentangledVAE:
    """Load a DisentangledVAE checkpoint, handling both raw Lightning checkpoints
    and plain state_dict-style dumps. Raises FileNotFoundError / RuntimeError on
    failure -- callers decide how to surface that (Streamlit error vs HTTP 500)."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")

    ck = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ck, dict) and "state_dict" in ck:
        hp = dict(ck.get("hyper_parameters", {}))
        valid = set(inspect.signature(DisentangledVAE.__init__).parameters)
        hp = {k: v for k, v in hp.items() if k in valid}
        model = DisentangledVAE(**hp)
        model.load_state_dict(ck["state_dict"], strict=True)
    else:
        model = DisentangledVAE.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def midi_to_roll(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    """First SEQ_LEN steps (at FS Hz) of pitches [PITCH_LO, PITCH_LO+N_PITCH),
    binarized. Zero-padded if the clip is shorter than SEQ_LEN steps."""
    full = (pm.get_piano_roll(fs=FS) > 0).astype(np.float32)
    roll = full[PITCH_LO:PITCH_LO + N_PITCH].T
    t = roll.shape[0]
    if t >= SEQ_LEN:
        return roll[:SEQ_LEN]
    return np.concatenate([roll, np.zeros((SEQ_LEN - t, N_PITCH), np.float32)], axis=0)


def roll_to_midi(pr: np.ndarray, bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)
    step_s = 1.0 / FS
    for ch in range(pr.shape[1]):
        col = pr[:, ch].astype(int)
        padded = np.concatenate([[0], col, [0]])
        diffs = np.diff(padded)
        for on, off in zip(np.where(diffs == 1)[0], np.where(diffs == -1)[0]):
            start, end = on * step_s, off * step_s
            if end > start:
                inst.notes.append(pretty_midi.Note(
                    velocity=90, pitch=PITCH_LO + ch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


def postprocess_roll(prob: np.ndarray, threshold: float, min_len: int,
                      gap_merge: int, beat_steps: int) -> np.ndarray:
    roll = (prob >= threshold).astype(np.float32)

    def runs(col):
        out, i, n = [], 0, len(col)
        while i < n:
            if col[i] > 0:
                j = i
                while j < n and col[j] > 0:
                    j += 1
                out.append((i, j)); i = j
            else:
                i += 1
        return out

    for ch in range(roll.shape[1]):
        col = roll[:, ch]
        if not col.any():
            continue
        if min_len > 1:
            for a, b in runs(col):
                if (b - a) < min_len:
                    col[a:b] = 0.0
        if gap_merge > 0:
            rs = runs(col)
            for (a1, b1), (a2, b2) in zip(rs[:-1], rs[1:]):
                if (a2 - b1) <= gap_merge:
                    col[b1:a2] = 1.0
        roll[:, ch] = col

    if beat_steps > 1:
        snapped = np.zeros_like(roll)
        n = roll.shape[0]
        for ch in range(roll.shape[1]):
            for a, b in runs(roll[:, ch]):
                g = int(round(a / beat_steps) * beat_steps)
                g = max(0, min(g, n - 1))
                snapped[g:min(g + (b - a), n), ch] = 1.0
        roll = snapped

    return roll


@torch.no_grad()
def encode_roll(model: DisentangledVAE, roll: np.ndarray):
    x = torch.from_numpy(roll).float().unsqueeze(0)
    mu_r, _, mu_p, _ = model.encode(x)
    return mu_r, mu_p


@torch.no_grad()
def decode_latents(model: DisentangledVAE, z_r: torch.Tensor, z_p: torch.Tensor) -> np.ndarray:
    """Additive merge: decoder takes z_r + z_p and returns probabilities."""
    probs = torch.sigmoid(model.decoder(z_r + z_p, return_logits=True))
    return probs.squeeze(0).numpy()


def recombine(model: DisentangledVAE, content_pm: pretty_midi.PrettyMIDI,
              style_pm: "pretty_midi.PrettyMIDI | None" = None,
              threshold: float = 0.35, min_len: int = 2, gap_merge: int = 2,
              beat_steps: int = 0, bpm: float = 120.0,
              rhythm_scale: float = 1.0, rhythm_noise: float = 0.0,
              pitch_scale: float = 1.0, pitch_noise: float = 0.0) -> tuple[pretty_midi.PrettyMIDI, int, float]:
    """End-to-end: content (+ optional style) MIDI in, generated MIDI out.
    Returns (generated_midi, note_count, density). This is what both the
    Streamlit app and the plugin's inference server call."""
    mu_r, mu_p = encode_roll(model, midi_to_roll(content_pm))

    if style_pm is not None:
        _, mu_p = encode_roll(model, midi_to_roll(style_pm))

    z_r = mu_r * rhythm_scale + (torch.randn_like(mu_r) * rhythm_noise if rhythm_noise else 0)
    z_p = mu_p * pitch_scale + (torch.randn_like(mu_p) * pitch_noise if pitch_noise else 0)

    pr_raw = decode_latents(model, z_r, z_p)
    pr_bin = postprocess_roll(pr_raw, threshold, min_len, gap_merge, beat_steps)
    pm_out = roll_to_midi(pr_bin, bpm=bpm)

    note_count = len(pm_out.instruments[0].notes) if pm_out.instruments else 0
    density = float(pr_bin.mean())
    return pm_out, note_count, density
