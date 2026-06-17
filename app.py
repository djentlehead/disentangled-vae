# source "/c/Users/sharm/Downloads/Programming/Style Transfer/.venv/Scripts/activate"
import io
import os
import sys

import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import streamlit as st
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.models.disentangled_vae import DisentangledVAE

FS         = 8     
SEQ_LEN    = 256
PITCH_LO   = 40    
N_PITCH    = 32

def _find_checkpoint():
    ckpt_dir = os.path.join(os.path.dirname(__file__), "lightning_logs", "disentangled_vae")
    if os.path.isdir(ckpt_dir):
        cks = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if cks:
            cks.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)), reverse=True)
            return os.path.join(ckpt_dir, cks[0])
    return None

CHECKPOINT_PATH = _find_checkpoint()


# ── Model loading (handles weights-only AND full checkpoints) ───────────────
@st.cache_resource(show_spinner="Loading DisentangledVAE checkpoint…")
def load_model():
    if CHECKPOINT_PATH is None or not os.path.exists(CHECKPOINT_PATH):
        return None, (

        )

    ck = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(ck, dict) and "state_dict" in ck:
        # Weights-only export (what the notebook produces) OR full Lightning ckpt.
        import inspect
        hp = dict(ck.get("hyper_parameters", {}))
        valid = set(inspect.signature(DisentangledVAE.__init__).parameters)
        hp = {k: v for k, v in hp.items() if k in valid}
        model = DisentangledVAE(**hp)
        model.load_state_dict(ck["state_dict"], strict=True)
    else:
        model = DisentangledVAE.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")
    model.eval()
    return model, None


# ── MIDI <-> roll (32-pitch window, MIDI 40..71, fs=8) ──────────────────────
def midi_to_roll(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    """Single binary roll (SEQ_LEN, N_PITCH). The model derives v_r / v_p itself."""
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


# ── Post-processing: turn a raw probability map into a musical roll ─────────
def postprocess_roll(prob: np.ndarray, threshold: float, min_len: int,
                     gap_merge: int, beat_steps: int) -> np.ndarray:


    roll = (prob >= threshold).astype(np.float32)

    def runs(col):
        """Yield (start, end_exclusive) for each contiguous active run."""
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

    # 3) snap onsets to a beat grid, preserving each note's length.
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


# ── Inference (additive-merge model) ────────────────────────────────────────
@torch.no_grad()
def encode_roll(model, roll: np.ndarray):
    """roll (SEQ_LEN,N_PITCH) -> (mu_r, mu_p). The model builds v_r/v_p + main feat."""
    x = torch.from_numpy(roll).float().unsqueeze(0)
    mu_r, _, mu_p, _ = model.encode(x)
    return mu_r, mu_p


@torch.no_grad()
def decode_latents(model, z_r: torch.Tensor, z_p: torch.Tensor) -> np.ndarray:
    """Additive merge: decoder takes z_r + z_p and returns probabilities."""
    probs = torch.sigmoid(model.decoder(z_r + z_p, return_logits=True))
    return probs.squeeze(0).numpy()


# ── Visualisation ────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Disentangled Music Studio", layout="wide", page_icon="🎹")
st.title("🎹 Polyphonic Music Style Transfer")
st.markdown(
    "**Disentangled VAE** — pitch and rhythm live in *separate* 64-dim latent spaces, "
    "merged additively. Recombine the rhythm of one piece with the pitch of another, "
    "explore the prior, or reconstruct a single clip."
)

model, err = load_model()
if err:
    st.error(f"**Model load failed.** {err}")
    st.stop()

DIM = model.hparams.latent_dim
st.success(
    f"Model loaded — {sum(p.numel() for p in model.parameters()):,} params · "
    f"z_rhythm & z_pitch: {DIM}-dim each · additive merge"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Input")
    content_file = st.file_uploader("Content MIDI (rhythm source)", type=["mid", "midi"])
    style_file   = st.file_uploader("Style MIDI (pitch source, optional)", type=["mid", "midi"],
                                    help="Recombination: rhythm from Content, pitch from Style.")

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

# ── Preprocessing preview ─────────────────────────────────────────────────────
if content_file and not exploration_mode:
    raw = content_file.read(); content_file.seek(0)
    pm_vis = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    roll_vis = midi_to_roll(pm_vis)
    st.subheader("Input (MIDI 40–71 window, fs=8)")
    fig = roll_fig(roll_vis, "Content piano roll", cmap="Blues")
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.divider()

# ── Generation ────────────────────────────────────────────────────────────────
if not generate_btn:
    st.info("Set a mode and controls in the sidebar, then click **Generate**. "
            "Reconstruct/Recombine needs a Content MIDI.")
    st.stop()

if exploration_mode:
    g = torch.Generator().manual_seed(int(seed))
    z_r = torch.randn(1, DIM, generator=g) * rhythm_sigma
    z_p = torch.randn(1, DIM, generator=g) * pitch_sigma
    label = f"Exploration (σ_r={rhythm_sigma}, σ_p={pitch_sigma}, seed={seed})"
else:
    if not content_file:
        st.error("Upload a Content MIDI for Reconstruct/Recombine mode."); st.stop()
    raw = content_file.read()
    pm_c = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    mu_r, mu_p = encode_roll(model, midi_to_roll(pm_c))

    if style_file:                       # recombination: pitch from style piece
        pm_s = pretty_midi.PrettyMIDI(io.BytesIO(style_file.read()))
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

# ── Display ───────────────────────────────────────────────────────────────────
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
buf = io.BytesIO(); pm_out.write(buf); buf.seek(0)
st.download_button("⬇ Download Generated MIDI", data=buf,
                   file_name="generated.mid", mime="audio/midi",
                   use_container_width=True)
st.caption(f"Duration {SEQ_LEN/FS:.1f}s · fs={FS} · BPM={bpm} · {note_count} notes · density {density:.3f}")




    











