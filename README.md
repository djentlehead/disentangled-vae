# Polyphonic Music Style Transfer

> This repository contains the code for a Master's dissertation project focused on transferring the musical style between polyphonic composers (e.g., making a piece by Bach sound as if it were composed by Chopin).

## Core Models Implemented

* **Conditional Variational Autoencoder (CVAE):** A generative model that learns to disentangle musical content from style.
* **Cycle-consistent Transformer:** A GAN-style architecture with two Transformer models that learn to translate between two musical styles in a cycle.
* **Baseline Transformer LM:** A simple language model used to verify the data representation and for basic music generation.
* **Style Classifier:** An LSTM-based model trained to identify a composer's style, used as a loss component for training the CVAE.

## Tech Stack

* Python 3.10+
* PyTorch & PyTorch Lightning
* pretty_midi & music21 (for MIDI processing)
* librosa (for audio analysis)
* pandas & numpy

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/djentlehead/polyphonic-style-transfer.git]
    cd polyphonic-style-transfer
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt
    ```

3.  **Download Data:**
    Download the [MAESTRO Dataset v3.0.0](https://magenta.tensorflow.org/datasets/maestro) and place the `maestro-v3.0.0` folder inside the `data/raw/` directory.

