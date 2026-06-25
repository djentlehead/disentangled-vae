# Quickstart

## Launch it

Double-click **`Launch Style Transfer Studio.vbs`**. No terminal window — your browser
opens automatically in a few seconds at `http://127.0.0.1:8000`.

Nothing happens? Double-click **`Launch (visible).bat`** instead — it shows a console
window with whatever error came up. (Also check `launch.log`, which the silent launcher
writes to.)

To stop the app: close the browser tab, then end the `python.exe` process in Task
Manager (the silent launcher has no window to close).

## Using it

1. Pick a mode: **Recombine** (top of the left panel) encodes MIDI you provide;
   **Explore** ignores any files and samples random latents straight from the model's
   prior — good for hearing what it can generate without input.
2. In Recombine mode, either drag a **Content** MIDI file into the drop zone or pick one
   of the bundled examples (`bach_invention.mid`, `chopin_nocturne.mid`) from the dropdown
   underneath it. Add a **Style** file too if you want recombination — that takes the
   *pitch* from Style and the *rhythm* from Content. Leave Style empty for a plain
   reconstruction of Content.
3. Tune the sliders if the output is too sparse/dense (**Binary threshold**, under
   "Cleanup & output"), too choppy (**Min note length**, **Merge gaps**), or needs to
   snap to a grid (**Snap onsets**).
4. Click **Generate**, listen to the preview, click **Download MIDI**.

Read the **"What this tool actually does"** banner near the top of the page before
judging the output — it explains the model's real limits (32-second window, narrow pitch
range, no per-composer control yet).

## Getting real piano audio instead of the sine-wave preview

The app can render through the bundled `FluidR3_GM.sf2` soundfont, but that needs the
native FluidSynth library installed — `pip install pyfluidsynth` alone isn't enough.

On Windows:
1. Download a prebuilt FluidSynth release for Windows from the official project's GitHub
   releases page (search "FluidSynth releases," grab the `win10-x64` zip).
2. Unzip it, and copy the `.dll` files from its `bin` folder into this project's
   `.venv\Scripts` folder (next to `python.exe`) — or add that `bin` folder to your
   system `PATH`.
3. Install the Python binding: `.venv\Scripts\pip install pyfluidsynth`
4. Relaunch the app. If FluidSynth loads correctly, the audio preview caption under the
   player will no longer say "Sine-wave preview."

If you skip this, the downloaded MIDI file is unaffected — it'll sound correct in any
DAW or MIDI player regardless.

## Troubleshooting

- **Status pill in the top-right says "Server offline"** — the backend isn't running
  or hasn't finished starting yet; wait a few seconds, or relaunch.
- **Status pill says "Model failed to load"** — the checkpoint at
  `lightning_logs/disentangled_vae/disentangled_vae_paper_weights.ckpt` is missing or
  moved. Don't relocate that folder relative to the project root.
- **"No notes found..." / silent output** — your MIDI's notes fall outside MIDI pitch
  40–71 (roughly E2–G5), or outside the first 32 seconds. Try a different clip, or lower
  the binary threshold.
- **Port already in use** — something else is already listening on 8000. Edit the last
  line of `server/main.py` (`uvicorn.run(app, host="127.0.0.1", port=8000, ...)`) to a
  free port, and update the URL the launcher opens to match
  (`_open_browser_when_ready` near the bottom of the same file).

## The old Streamlit app

`app.py` (the original Streamlit UI) still works and shares the exact same model code
(`src/inference/recombine.py`, `src/inference/audio.py`) as the new web app, but it's no
longer the primary interface. Run it manually if you ever want it back:

```bat
.venv\Scripts\streamlit.exe run app.py
```

## What's next (v2)

This app currently ships only the **DisentangledVAE recombination** model. The
composer-conditioned models (CVAE, cycle-consistent Transformer — "make it sound like
Chopin") exist in `src/models/` and `src/training/` but their trained checkpoints aren't
in this repo; they need to be retrained before they can be wired into the app.

A DAW plugin (so this runs inside your DAW instead of a browser tab) exists as
**unbuilt source** in `plugin/` — see `plugin/README.md` for what it does and
`plugin/BUILD.md` to compile it yourself. It wasn't compiled or tested by the
tool that wrote it (no Windows build environment available), so treat it as a
starting point, not a finished build.
