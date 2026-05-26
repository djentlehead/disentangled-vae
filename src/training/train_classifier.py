import os
import sys
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

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
            tokens = tokens[:fixed_length]
        else:
            tokens = np.pad(tokens, (0, fixed_length - len(tokens)), 'constant', constant_values=0)
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class ClassifierTrainer(pl.LightningModule):
    def __init__(self, vocab_size, style_vocab_size, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = StyleClassifier(vocab_size, style_vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':

    TOKENIZED_DIR = os.path.join(project_root, "data/processed/tokenized")
    METADATA_PATH = os.path.join(project_root, "data/raw/maestro-v3.0.0/maestro-v3.0.0.csv")
    STYLE_MAP = {
        "Johann Sebastian Bach": 0,
        "Frédéric Chopin": 1,
        "Ludwig van Beethoven": 2,
        "Franz Schubert": 3 

    }
    

    dataset = StyleDataset(TOKENIZED_DIR, METADATA_PATH, STYLE_MAP)
    print(f"Found {len(dataset)} files for the specified composers.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=16, num_workers=0)

    model = ClassifierTrainer(vocab_size=388, style_vocab_size=len(STYLE_MAP))
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="style-classifier-best",
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )

    trainer = pl.Trainer(max_epochs=10, accelerator="auto", callbacks=[checkpoint_callback])
    print("--- Starting training for the Style Classifier ---")
    trainer.fit(model, train_loader, val_loader)