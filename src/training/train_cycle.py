import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random
import itertools


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from src.models.cyc_transformer import CycleTransformer, Discriminator

class PairedStyleDataset(Dataset):
    def __init__(self, data_dir, metadata_path, style_map, style_A_name, style_B_name):
        df = pd.read_csv(metadata_path)
        df['basename'] = df['midi_filename'].apply(os.path.basename)
        filename_to_composer = pd.Series(df.canonical_composer.values, index=df.basename).to_dict()
        self.paths_A = []
        self.paths_B = []
        style_A_label = style_map[style_A_name]
        style_B_label = style_map[style_B_name]
        for f in os.listdir(data_dir):
            if f.endswith(".npy"):
                midi_basename = os.path.splitext(f)[0] + ".midi"
                if midi_basename in filename_to_composer:
                    composer = filename_to_composer[midi_basename]
                    if composer in style_map:
                        full_path = os.path.join(data_dir, f)
                        if style_map[composer] == style_A_label:
                            self.paths_A.append(full_path)
                        elif style_map[composer] == style_B_label:
                            self.paths_B.append(full_path)
        print(f"Found {len(self.paths_A)} files for style A ({style_A_name})")
        print(f"Found {len(self.paths_B)} files for style B ({style_B_name})")
    def _load_and_process_token_file(self, path):
        tokens = np.load(path).astype(np.int64)
        fixed_length = 2048
        if len(tokens) > fixed_length:
            start_idx = np.random.randint(0, len(tokens) - fixed_length)
            tokens = tokens[start_idx : start_idx + fixed_length]
        else:
            tokens = np.pad(tokens, (0, fixed_length - len(tokens)), 'constant', constant_values=0)
        return torch.tensor(tokens, dtype=torch.long)
    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))
    def __getitem__(self, index):
        path_A = self.paths_A[index % len(self.paths_A)]
        path_B = random.choice(self.paths_B)
        tokens_A = self._load_and_process_token_file(path_A)
        tokens_B = self._load_and_process_token_file(path_B)
        return tokens_A, tokens_B



class CycleTrainer(pl.LightningModule):
    def __init__(self, vocab_size, model_params, disc_params, lr, lambda_id, lambda_cyc):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.G_AB = CycleTransformer(self.hparams.vocab_size, **self.hparams.model_params)
        self.G_BA = CycleTransformer(self.hparams.vocab_size, **self.hparams.model_params)
        self.D_A = Discriminator(self.hparams.vocab_size, **self.hparams.disc_params)
        self.D_B = Discriminator(self.hparams.vocab_size, **self.hparams.disc_params)
        self.cyc_id_loss = torch.nn.CrossEntropyLoss()
        self.adv_loss = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        real_A, real_B = batch

        # --- Train Generators ---
        g_opt.zero_grad()
        
        # Identity loss
        id_A_logits = self.G_BA(real_A)
        loss_id_A = self.cyc_id_loss(id_A_logits.view(-1, self.hparams.vocab_size), real_A.view(-1))
        id_B_logits = self.G_AB(real_B)
        loss_id_B = self.cyc_id_loss(id_B_logits.view(-1, self.hparams.vocab_size), real_B.view(-1))
        loss_identity = (loss_id_A + loss_id_B) * self.hparams.lambda_id

        # Adversarial loss
        fake_B_logits = self.G_AB(real_A)
        pred_fake_B = self.D_B(fake_B_logits.argmax(dim=-1))
        loss_adv_AB = self.adv_loss(pred_fake_B, torch.ones_like(pred_fake_B))
        
        fake_A_logits = self.G_BA(real_B)
        pred_fake_A = self.D_A(fake_A_logits.argmax(dim=-1))
        loss_adv_BA = self.adv_loss(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_adversarial = loss_adv_AB + loss_adv_BA

        # Cycle-consistency loss
        cyc_A_logits = self.G_BA(fake_B_logits.argmax(dim=-1))
        loss_cyc_A = self.cyc_id_loss(cyc_A_logits.view(-1, self.hparams.vocab_size), real_A.view(-1))
        cyc_B_logits = self.G_AB(fake_A_logits.argmax(dim=-1))
        loss_cyc_B = self.cyc_id_loss(cyc_B_logits.view(-1, self.hparams.vocab_size), real_B.view(-1))
        loss_cycle = (loss_cyc_A + loss_cyc_B) * self.hparams.lambda_cyc

        g_loss = loss_identity + loss_adversarial + loss_cycle
        
        self.manual_backward(g_loss)
        g_opt.step()

        # --- Train Discriminators ---
        d_opt.zero_grad()
        
        pred_real_A = self.D_A(real_A)
        loss_d_real_A = self.adv_loss(pred_real_A, torch.ones_like(pred_real_A))
        pred_real_B = self.D_B(real_B)
        loss_d_real_B = self.adv_loss(pred_real_B, torch.ones_like(pred_real_B))
        
        pred_fake_A = self.D_A(fake_A_logits.detach().argmax(dim=-1))
        loss_d_fake_A = self.adv_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
        pred_fake_B = self.D_B(fake_B_logits.detach().argmax(dim=-1))
        loss_d_fake_B = self.adv_loss(pred_fake_B, torch.zeros_like(pred_fake_B))

        d_loss = (loss_d_real_A + loss_d_fake_A + loss_d_real_B + loss_d_fake_B) * 0.5

        self.manual_backward(d_loss)
        d_opt.step()
        
        # Log all losses
        self.log_dict({'g_loss': g_loss, 'd_loss': d_loss, 'id_loss': loss_identity, 
                       'adv_loss': loss_adversarial, 'cyc_loss': loss_cycle}, prog_bar=True)

    def configure_optimizers(self):
        g_params = itertools.chain(self.G_AB.parameters(), self.G_BA.parameters())
        d_params = itertools.chain(self.D_A.parameters(), self.D_B.parameters())
        
        g_opt = torch.optim.Adam(g_params, lr=self.hparams.lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(d_params, lr=self.hparams.lr, betas=(0.5, 0.999))
        
        return g_opt, d_opt


if __name__ == '__main__':

    BATCH_SIZE = 2

    hparams = {
        "model_params": { "d_model": 256, "nhead": 4, "d_hid": 512, "nlayers": 4, "dropout": 0.1 },
        "disc_params": { "embedding_dim": 128, "hidden_dim": 256 },
        "lr": 2e-4, 
        "lambda_id": 5.0, 
        "lambda_cyc": 10.0
    }
    
    TOKENIZED_DIR = os.path.join(project_root, "data/processed/tokenized")
    METADATA_PATH = os.path.join(project_root, "data/raw/maestro-v3.0.0/maestro-v3.0.0.csv")
    STYLE_MAP = {"Johann Sebastian Bach": 0, "Frédéric Chopin": 1}

    dataset = PairedStyleDataset(
        data_dir=TOKENIZED_DIR, metadata_path=METADATA_PATH,
        style_map=STYLE_MAP, style_A_name="Johann Sebastian Bach",
        style_B_name="Frédéric Chopin"
    )

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Pass vocab_size and the hparams dictionary
    model = CycleTrainer(vocab_size=388, **hparams)
    
    trainer = pl.Trainer(max_epochs=200, accelerator="auto", log_every_n_steps=10)
    
    print("--- Starting training for the Cycle Transformer Model ---")
    trainer.fit(model, train_loader)
    print("--- Cycle Training finished ---")