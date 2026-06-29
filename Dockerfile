# Hosts the FastAPI + custom web frontend (server/main.py + web/) that wraps
# the trained DisentangledVAE. Built for Hugging Face Spaces' Docker SDK, but
# is a plain container -- works on Render/Fly/any other Docker host too.
#
# CPU-only: inference is small and fast, no GPU needed.
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

# All deps below (torch CPU build, pytorch-lightning, pretty_midi, fastapi,
# uvicorn) ship manylinux wheels for this base image, so no compiler toolchain
# is needed here -- keeps the image smaller and the build faster.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-docker.txt

# Only what server/main.py actually needs at runtime. Training code
# (scripts/, configs/, data/), notebooks, and the unbuilt plugin/ source are
# intentionally left out -- see .dockerignore.
COPY src/ src/
COPY server/ server/
COPY web/ web/
COPY examples/ examples/
COPY lightning_logs/disentangled_vae/*.ckpt lightning_logs/disentangled_vae/
COPY SalC5Light2.sf2 .

# Hugging Face Spaces' Docker SDK expects the app on port 7860 by default.
EXPOSE 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
