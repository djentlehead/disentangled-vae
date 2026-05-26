import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from src.models.cvae import CVAE
from src.models.style_classifier import StyleClassifier

class StyleDataset(Dataset):
    def __init__(self, data_dir, metadata_path, style_map):
        self.file_paths = []
        self.labels = []
        df = pd.read_csv(metadata_path)
        df['basename'] = df['midi_filename'].apply(os.path.basename)
        filename_to_composer = pd.Series(df.canonical_composer.values, index=df.basename).to_dict()
        for f in os.listdir(data_dir):
            if f.endswith(".npy"):
                midi_basename = os.path.splitext(f)[0] + ".midi"
                if midi_basename in filename_to_composer:
                    composer = filename_to_composer[midi_basename]
                    if composer in style_map:
                        self.file_paths.append(os.path.join(data_dir, f))
                        self.labels.append(style_map[composer])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tokens = np.load(self.file_paths[idx]).astype(np.int64)
        fixed_length = 2048
        if len(tokens) > fixed_length:
            start_idx = np.random.randint(0, len(tokens) - fixed_length)
            tokens = tokens[start_idx : start_idx + fixed_length]
        else:
            tokens = np.pad(tokens, (0, fixed_length - len(tokens)), 'constant', constant_values=0)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class CVAETrainer(pl.LightningModule):

    def __init__(self, vocab_size, style_vocab_size, embedding_dim, hidden_dim, latent_dim, 
                 style_embedding_dim, learning_rate, batch_size, kl_anneal_steps,
                 w_recon, w_kl, w_style, w_content):
        super().__init__()

        self.save_hyperparameters()
        
        self.cvae = CVAE(
            vocab_size=self.hparams.vocab_size,
            style_vocab_size=self.hparams.style_vocab_size,
            embedding_dim=self.hparams.embedding_dim,
            hidden_dim=self.hparams.hidden_dim,
            latent_dim=self.hparams.latent_dim,
            style_embedding_dim=self.hparams.style_embedding_dim
        )
        
        self.style_classifier = StyleClassifier(self.hparams.vocab_size, self.hparams.style_vocab_size)
        classifier_ckpt_path = os.path.join(project_root, "checkpoints/style-classifier-best.ckpt")
        checkpoint = torch.load(classifier_ckpt_path, map_location=self.device)
        new_state_dict = {key.replace('model.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        self.style_classifier.load_state_dict(new_state_dict)
        self.style_classifier.eval()
        for param in self.style_classifier.parameters():
            param.requires_grad = False
            
        self.recon_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def training_step(self, batch, batch_idx):
        x, style_original = batch
        style_target = torch.randint_like(style_original, 0, self.hparams.style_vocab_size)
        for i in range(len(style_target)):
            while style_target[i] == style_original[i]:
                style_target[i] = random.randint(0, self.hparams.style_vocab_size - 1)

        logits_recon, mu, log_var = self.cvae(x, style_original)
        z = self.cvae.reparameterize(mu, log_var)
        logits_transfer = self.cvae.decode(z, style_target, x.size(1))
        
        loss_recon = self.recon_loss_fn(logits_recon.view(-1, self.hparams.vocab_size), x.view(-1))
        
        kl_weight = min(1.0, self.global_step / self.hparams.kl_anneal_steps) * self.hparams.w_kl
        loss_kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        with torch.no_grad():
            pred_style_logits = self.style_classifier(logits_transfer.argmax(dim=-1).detach())
        loss_style = F.cross_entropy(pred_style_logits, style_target)
        
        with torch.no_grad():
            mu_content, _ = self.cvae.encode(logits_recon.argmax(dim=-1).detach())
        loss_content = F.l1_loss(mu, mu_content)
        
        total_loss = (self.hparams.w_recon * loss_recon + kl_weight * loss_kl +
                      self.hparams.w_style * loss_style + self.hparams.w_content * loss_content)
        
        self.log_dict({'loss': total_loss, 'recon': loss_recon, 'kl': loss_kl, 'kl_w': kl_weight}, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.cvae.parameters(), lr=self.hparams.learning_rate)

if __name__ == '__main__':
    hparams = {
        "vocab_size": 388, "style_vocab_size": 4, "embedding_dim": 256,
        "hidden_dim": 512, "latent_dim": 256, "style_embedding_dim": 64,
        "learning_rate": 1e-4, "batch_size": 8, "kl_anneal_steps": 10000,
        "w_recon": 1.0, "w_kl": 0.1, "w_style": 1.0, "w_content": 1.0
    }
    
    TOKENIZED_DIR = os.path.join(project_root, "data/processed/tokenized")
    METADATA_PATH = os.path.join(project_root, "data/raw/maestro-v3.0.0/maestro-v3.0.0.csv")
    STYLE_MAP = {
        "Johann Sebastian Bach": 0, "Frederic Chopin": 1,
        "Ludwig van Beethoven": 2, "Franz Schubert": 3
    }

    dataset = StyleDataset(TOKENIZED_DIR, METADATA_PATH, STYLE_MAP)
    train_loader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=0)

    model = CVAETrainer(**hparams)
    trainer = pl.Trainer(max_epochs=1000, accelerator="auto", log_every_n_steps=10)
    
    print("--- Starting training for the CVAE Style Transfer Model (with KL Annealing) ---")
    trainer.fit(model, train_loader)
    print("--- CVAE Training finished ---")
