import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

script_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

sys.path.append(project_root)

from src.models.baseline_lm import BaselineTransformer
from src.data.midi_dataset import MIDIDataset

VOCAB_SIZE = 388
D_MODEL = 256
N_HEAD = 4
D_HID = 512
N_LAYERS = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 8


DATA_DIR = os.path.join(project_root, "data/processed/tokenized")

class MusicLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model, nhead, d_hid, nlayers, dropout, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = BaselineTransformer(vocab_size, d_model, nhead, d_hid, nlayers, dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        dataset = MIDIDataset(data_dir=DATA_DIR)

        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

if __name__ == '__main__':
    model = MusicLanguageModel(VOCAB_SIZE, D_MODEL, N_HEAD, D_HID, N_LAYERS, DROPOUT, LEARNING_RATE)
    
    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator="auto",
        log_every_n_steps=10
    )
    

    print("--- Starting training for the baseline language model ---")
    trainer.fit(model)
    print("--- Training finished ---")