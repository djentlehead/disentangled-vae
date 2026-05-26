import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models.disentangled_vae import DisentangledVAE


DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/vae_dataset.npz")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "lightning_logs", "disentangled_vae")

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100 
SEQ_LEN = 256
LATENT_DIM_RHYTHM = 64
LATENT_DIM_PITCH = 64
BETA_R = 1.0  
BETA_P = 1.0  
CHANNELS = 32

class VAEDataset(Dataset):
    """
    Loads the preprocessed .npz file containing the parallel piano-rolls.
    """
    def __init__(self, npz_path):
        print(f"Loading dataset from {npz_path}...")
        try:
            data = np.load(npz_path)
            
            self.X_original = torch.from_numpy(data['X_original']).float()
            self.X_rhythm = torch.from_numpy(data['X_rhythm']).float()
            self.X_pitch = torch.from_numpy(data['X_pitch']).float()
            
            if not (len(self.X_original) == len(self.X_rhythm) == len(self.X_pitch)):
                raise ValueError("All data arrays must have the same number of samples.")
                
            print(f"Dataset loaded. Number of samples: {len(self.X_original)}")
        except FileNotFoundError:
            print(f"--- ERROR ---")
            print(f"Data file not found at: {npz_path}")
            print("Please run 'python scripts/preprocess_vae_data.py' first.")
            sys.exit()
        except KeyError:
            print(f"--- ERROR ---")
            print(f"Data file at {npz_path} is corrupt or missing required keys.")
            print("Please re-run 'python scripts/preprocess_vae_data.py'.")
            sys.exit()

    def __len__(self):
        return len(self.X_original)

    def __getitem__(self, idx):
        return (
            self.X_original[idx],
            self.X_rhythm[idx],
            self.X_pitch[idx]
        )

if __name__ == "__main__":
    
    dataset = VAEDataset(DATA_PATH)
    
    if len(dataset) == 0:
        print("\n--- ABORTING TRAINING ---")
        print("The loaded dataset is empty (contains 0 samples).")
        print("This likely means 'preprocess_vae_data.py' did not find any MIDI files for the specified composers.")
        print("Please check two things:")
        print(f"  1. The composer names in 'composers_to_process' list in 'preprocess_vae_data.py'.")
        # print(f"  2. That the 'canonical_composer' column in your '{MAESTRO_CSV}' file matches those names.")
        sys.exit() 


    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4  
    )
    

    model = DisentangledVAE(
        seq_len=SEQ_LEN,
        latent_dim_rhythm=LATENT_DIM_RHYTHM,
        latent_dim_pitch=LATENT_DIM_PITCH,
        lr=LEARNING_RATE,
        beta_r=BETA_R,
        beta_p=BETA_P,
        channels=CHANNELS
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='best_model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='train_loss', 
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir=CHECKPOINT_DIR,
        log_every_n_steps=10
    )
    

    print("--- Starting Disentangled VAE Training (Task 1.4) ---")
    trainer.fit(model, train_loader)
    print("--- Training Complete ---")