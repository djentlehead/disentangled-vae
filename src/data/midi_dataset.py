import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List

class MIDIDataset(Dataset):
    def __init__(self, data_dir: str, max_sequence_length: int = 2048):
        self.data_dir = data_dir
        self.max_sequence_length = max_sequence_length
        self.file_paths: List[str] = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        try:
            loaded_data = np.load(file_path, allow_pickle=True)

            if not isinstance(loaded_data, np.ndarray):
                raise TypeError(f"Corrupted file detected: {file_path} contains {type(loaded_data)}, not numpy.ndarray")

            full_sequence = loaded_data.astype(np.int64)
            
            seq_len = len(full_sequence)
            if seq_len > self.max_sequence_length + 1:
                start_idx = np.random.randint(0, seq_len - (self.max_sequence_length + 1))
                chunk = full_sequence[start_idx : start_idx + self.max_sequence_length + 1]
            else:
                chunk = full_sequence
            
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            
            x_len = len(x)
            if x_len < self.max_sequence_length:
                pad_amount = self.max_sequence_length - x_len
                x = torch.nn.functional.pad(x, (0, pad_amount), 'constant', 0)
                y = torch.nn.functional.pad(y, (0, pad_amount), 'constant', 0)
            
            return x, y

        except Exception as e:
            print(f"--- FAILED TO PROCESS FILE ---")
            print(f"File path: {file_path}")
            print(f"Error: {e}")
            raise e