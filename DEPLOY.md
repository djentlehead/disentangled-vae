# Deploying to Hugging Face Spaces

This repo is set up to run as a Docker Space: `Dockerfile`, `.dockerignore`,
and `requirements-docker.txt` build a container around `server/main.py` +
`web/`, and `README.md` has the metadata block Spaces needs at the top.

## One-time setup

1. **Install Git LFS** (if you haven't already): https://git-lfs.com, then
   run `git lfs install` once per machine. The checkpoint
   (`lightning_logs/disentangled_vae/disentangled_vae_paper_weights.ckpt`,
   ~150MB) is already configured for LFS via `.gitattributes`.

2. **Create the Space**: go to https://huggingface.co/new-space, pick a name,
   set **Docker** as the SDK, and choose the free **CPU basic** hardware.
   Don't initialize it with any files — you're pushing this repo's content.

3. **Add it as a git remote** (from this repo):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   ```

## Every time you want to (re)deploy

1. **Check what's changed and commit.** This working tree currently has
   uncommitted changes from building the web app — review with `git status`
   and commit what you want to ship:
   ```bash
   git add -A
   git commit -m "Add web app + Docker deployment"
   ```

2. **Push to the Space:**
   ```bash
   git push space main
   ```
   Git LFS uploads the checkpoint separately from the regular push — for a
   150MB file on a normal connection this can take a few minutes.

3. **Watch the build**: open your Space's page on huggingface.co and check
   the **Logs** tab. It installs CPU-only torch + the rest of
   `requirements-docker.txt`, copies in the model, and starts the server.
   First build typically takes a few minutes; later ones are faster if the
   dependency layer is cached.

4. Once it says **Running**, your Space's URL is live and public.

## Things to know

- **Cold starts**: free CPU Spaces sleep after 48 hours with no traffic. The
  next visitor waits through a cold start (container boot + model load,
  usually well under a minute) — there's no way around this on the free tier.
- **Audio preview quality**: the 142MB FluidSynth soundfont isn't bundled (see
  `.dockerignore`), so the hosted version's audio preview always uses the
  built-in sine-wave fallback (`src/inference/audio.py`). The downloaded MIDI
  file is unaffected — this only touches the in-browser preview player.
- **Updating later**: repeat steps 1-2 above. Spaces rebuilds on every push.
- **Local Docker testing** (optional, before pushing): `docker build -t style-transfer .`
  then `docker run -p 7860:7860 style-transfer` and open
  `http://localhost:7860`. Not required — pushing directly and watching the
  Spaces build logs works fine too.
