import io
import os
import sys

import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import streamlit as st
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.inference.recombine import (
    FS, SEQ_LEN, PITCH_LO, N_PITCH,
    find_checkpoint, load_model as _load_model_from_ckpt,
    midi_to_roll, roll_to_midi, postprocess_roll,
    encode_roll, decode_latents,
)
from src.inference.audio import synthesize_wav as _synthesize_wav



PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_PATH = find_checkpoint(PROJECT_ROOT)


@st.cache_resource(show_spinner="Loading DisentangledVAE checkpoint…")
def load_model():
    try:
        return _load_model_from_ckpt(CHECKPOINT_PATH), None
    except Exception as e:
        return None, str(e)


def synthesize_wav(pm: pretty_midi.PrettyMIDI):
    return _synthesize_wav(pm, PROJECT_ROOT)


_BG = "#0e1117"


def _style_ax(ax, title, xlabel="Time step", ylabel="Pitch (MIDI 40+)"):
    ax.set_facecolor(_BG)
    ax.set_title(title, color="white", fontsize=9, pad=3)
    ax.set_xlabel(xlabel, color="#888", fontsize=7)
    ax.set_ylabel(ylabel, color="#888", fontsize=7)
    ax.tick_params(colors="#777", labelsize=6)
    for s in ax.spines.values():
        s.set_edgecolor("#333")


def roll_fig(pr: np.ndarray, title: str, cmap: str = "magma") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2))
    fig.patch.set_facecolor(_BG)
    ax.imshow(pr.T, aspect="auto", origin="lower", cmap=cmap,
              vmin=0, vmax=1, interpolation="nearest")
    _style_ax(ax, title)
    plt.tight_layout(pad=0.4)
    return fig


def latent_bar_fig(z_r: torch.Tensor, z_p: torch.Tensor) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 1.6))
    fig.patch.set_facecolor(_BG)
    for ax, z, label, col in [
        (ax1, z_r, "z_rhythm", "#4c9be8"),
        (ax2, z_p, "z_pitch",  "#e87c4c"),
    ]:
        vals = z.squeeze(0).numpy()
        ax.set_facecolor(_BG)
        ax.bar(range(len(vals)), vals, color=col, width=1.0)
        ax.axhline(0, color="#555", linewidth=0.5)
        _style_ax(ax, label, xlabel="Dimension", ylabel="Value")
        ax.set_xlim(0, len(vals))
    plt.tight_layout(pad=0.4)
    return fig



st.set_page_config(page_title="Disentangled Music Studio", layout="wide", page_icon="🎹")
st.title("🎹 Polyphonic Music Style Transfer")
st.markdown(
    "**Disentangled VAE** - pitch and rhythm live in *separate* 64-dim latent spaces, "
    "merged additively. Recombine the rhythm of one piece with the pitch of another, "
    "explore the prior, or reconstruct a single clip."
)

with st.expander("ℹ️ What this tool actually does (read before uploading)"):
    st.markdown(
        f"""
This is a **rhythm/pitch recombination** tool, not a "make it sound like Chopin" composer
filter. It encodes a MIDI clip into two separate latent codes - one for rhythm, one for
pitch - and lets you swap either half with another clip's. Try a Bach rhythm with a
Chopin melody contour, or just reconstruct a single piece to hear what the model keeps
and loses.

**Current limits, so results make sense:**
- Only the first **{SEQ_LEN / FS:.0f} seconds** of any uploaded MIDI are used - the rest is ignored.
- Only notes in MIDI pitch range **{PITCH_LO}–{PITCH_LO + N_PITCH - 1}** (roughly E2–G5) are seen by the model.
- Output is a single instrument, binary on/off piano roll - no velocity or pedal nuance.
- True composer-conditioned transfer (pick "Chopin," get Chopin-flavored output) is a
  planned v2, built on a different model that isn't wired up in this app yet.
"""
    )

model, err = load_model()
if err:
    st.error(f"**Model load failed.** {err}")
    st.stop()

DIM = model.hparams.latent_dim
st.success(
    f"Model loaded - {sum(p.numel() for p in model.parameters()):,} params · "
    f"z_rhythm & z_pitch: {DIM}-dim each · additive merge"
)


def _list_examples():
    d = os.path.join(os.path.dirname(__file__), "examples")
    if os.path.isdir(d):
        return sorted(f for f in os.listdir(d) if f.lower().endswith((".mid", ".midi")))
    return []

with st.sidebar:
    st.header("Input")
    examples = _list_examples()
    example_choice = None
    if examples:
        example_choice = st.selectbox(
            "Or load an example", ["— none —", *examples],
            help="No MIDI handy? Pick a bundled example to try the model instantly.")
        if example_choice == "— none —":
            example_choice = None
    content_file = st.file_uploader("Content MIDI (rhythm source)", type=["mid", "midi"])
    style_file   = st.file_uploader("Style MIDI (pitch source, optional)", type=["mid", "midi"],
                                    help="Recombination: rhythm from Content, pitch from Style.")
    st.caption(
        f"Only the first {SEQ_LEN / FS:.0f}s and MIDI pitches {PITCH_LO}–{PITCH_LO + N_PITCH - 1} "
        f"are used — see the **ℹ️ What this tool does** panel at the top of the page."
    )

    st.divider()
    st.header("Generation mode")
    mode = st.radio(
        "Mode",
        ["Reconstruct / Recombine — encode MIDI", "Exploration — sample from prior"],
        help=("**Reconstruct/Recombine**: encodes your MIDI. With a Style file, takes "
              "z_rhythm from Content and z_pitch from Style.\n\n"
              "**Exploration**: ignores uploads; samples z_r, z_p from N(0, σ)."),
    )
    exploration_mode = mode.startswith("Exploration")

    st.divider()
    st.header("Latent controls")
    if exploration_mode:
        st.caption("Sample z independently from the prior N(0, σ)")
        rhythm_sigma = st.slider("z_rhythm σ", 0.1, 3.0, 1.0, 0.1)
        pitch_sigma  = st.slider("z_pitch σ",  0.1, 3.0, 1.0, 0.1)
        seed = st.number_input("Random seed", 0, 9999, 42, 1)
    else:
        st.markdown("**Rhythm** `z_r`")
        rhythm_scale = st.slider("Scale", 0.0, 3.0, 1.0, 0.05, key="r_scale")
        rhythm_noise = st.slider("Noise σ", 0.0, 2.0, 0.0, 0.05, key="r_noise")
        st.markdown("**Pitch** `z_p`")
        pitch_scale  = st.slider("Scale", 0.0, 3.0, 1.0, 0.05, key="p_scale")
        pitch_noise  = st.slider("Noise σ", 0.0, 2.0, 0.0, 0.05, key="p_noise")

    st.divider()
    st.header("Cleanup (post-processing)")
    threshold = st.slider("Binary threshold", 0.05, 0.95, 0.35, 0.05,
                          help="Lower = denser. Tune so output density matches your input.")
    min_len = st.slider("Min note length (steps)", 0, 8, 2, 1,
                        help="Drop notes shorter than this — removes scattered single-frame noise.")
    gap_merge = st.slider("Merge gaps ≤ (steps)", 0, 8, 2, 1,
                          help="Bridge tiny silences within a pitch so stuttering notes sustain.")
    beat_steps = st.slider("Snap onsets to grid (steps)", 0, 8, 0, 1,
                           help="Quantise note starts to a beat grid. 0 = off. fs=8, so 2≈8th, 4≈quarter.")
    bpm = st.number_input("Output BPM", 40, 240, 120, 5)
    generate_btn = st.button("Generate", type="primary", use_container_width=True)


def _example_bytes(name):
    with open(os.path.join(os.path.dirname(__file__), "examples", name), "rb") as f:
        return f.read()

content_bytes = None
if content_file is not None:
    content_bytes = content_file.read()
elif example_choice:
    content_bytes = _example_bytes(example_choice)

style_bytes = style_file.read() if style_file is not None else None


def _safe_load_midi(raw_bytes: bytes, label: str):
    """Parse uploaded MIDI bytes, surfacing a friendly error instead of a stack trace."""
    try:
        return pretty_midi.PrettyMIDI(io.BytesIO(raw_bytes))
    except Exception as e:
        st.error(f"**Couldn't read the {label} MIDI file.** It may be corrupt or not a "
                  f"standard MIDI file. ({e})")
        st.stop()


if content_bytes and not exploration_mode:
    pm_vis = _safe_load_midi(content_bytes, "Content")
    roll_vis = midi_to_roll(pm_vis)
    if roll_vis.sum() == 0:
        st.warning(
            f"No notes found in MIDI pitch range {PITCH_LO}–{PITCH_LO + N_PITCH - 1} "
            f"(roughly E2–G5) in the first {SEQ_LEN / FS:.0f}s of this file. Generation will "
            f"likely produce silence — try a different clip or one with more mid-range notes."
        )
    st.subheader("Input (MIDI 40–71 window, fs=8)")
    fig = roll_fig(roll_vis, "Content piano roll", cmap="Blues")
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.divider()


if not generate_btn:
    st.info("Set a mode and controls in the sidebar, then click **Generate**. "
            "Reconstruct/Recombine needs a Content MIDI (upload one or load an example).")
    st.stop()

if exploration_mode:
    g = torch.Generator().manual_seed(int(seed))
    z_r = torch.randn(1, DIM, generator=g) * rhythm_sigma
    z_p = torch.randn(1, DIM, generator=g) * pitch_sigma
    label = f"Exploration (σ_r={rhythm_sigma}, σ_p={pitch_sigma}, seed={seed})"
else:
    if not content_bytes:
        st.error("Upload a Content MIDI (or load an example) for Reconstruct/Recombine mode."); st.stop()
    pm_c = _safe_load_midi(content_bytes, "Content")
    mu_r, mu_p = encode_roll(model, midi_to_roll(pm_c))

    if style_bytes:                      # recombination: pitch from style piece
        pm_s = _safe_load_midi(style_bytes, "Style")
        _, mu_p = encode_roll(model, midi_to_roll(pm_s))
        label = "Recombination (rhythm = Content, pitch = Style)"
    else:
        label = "Reconstruction (Content)"

    z_r = mu_r * rhythm_scale + (torch.randn_like(mu_r) * rhythm_noise if rhythm_noise else 0)
    z_p = mu_p * pitch_scale  + (torch.randn_like(mu_p) * pitch_noise  if pitch_noise  else 0)

pr_raw = decode_latents(model, z_r, z_p)
pr_bin = postprocess_roll(pr_raw, threshold, min_len, gap_merge, beat_steps)
pm_out = roll_to_midi(pr_bin, bpm=float(bpm))
note_count = len(pm_out.instruments[0].notes) if pm_out.instruments else 0
density = pr_bin.mean()


st.subheader(f"Generated - {label}")
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**Latent codes used**")
    fig = latent_bar_fig(z_r, z_p); st.pyplot(fig, use_container_width=True); plt.close(fig)
with col_r:
    st.markdown(f"**Output** ({note_count} notes · density {density:.3f} · threshold {threshold})")
    fig = roll_fig(pr_bin, "Decoder output (thresholded)", cmap="magma")
    st.pyplot(fig, use_container_width=True); plt.close(fig)

st.divider()


if note_count > 0:
    with st.spinner("Synthesizing audio…"):
        wav, method = synthesize_wav(pm_out)
    if wav:
        st.audio(wav, format="audio/wav")
        if method == "sine":
            st.caption(
                "Sine-wave preview - FluidSynth wasn't found, so this is a rough approximation, "
                "not real piano tone. The downloaded MIDI is unaffected - it'll sound correct in any DAW or player."
            )
    else:
        st.caption(f"Audio synthesis unavailable ({method}). Download the MIDI to listen.")
else:
    st.caption("No notes generated - lower the threshold and regenerate to hear audio.")

buf = io.BytesIO(); pm_out.write(buf); buf.seek(0)
st.download_button("Download Generated MIDI", data=buf,
                   file_name="generated.mid", mime="audio/midi",
                   use_container_width=True)
st.caption(f"Duration {SEQ_LEN/FS:.1f}s · fs={FS} · BPM={bpm} · {note_count} notes · density {density:.3f}")