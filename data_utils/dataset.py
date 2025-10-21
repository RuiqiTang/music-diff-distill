import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    def __init__(self, latent_dir, transform=None):
        self.latent_dir = Path(latent_dir)
        self.files = sorted(list(self.latent_dir.glob('*.npz')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        data = np.load(str(p))
        keys = [k for k in data.files if k not in ('filename','start')]
        if len(keys) == 0:
            raise RuntimeError(f"No latent in {p}")
        if all(k.startswith('code_') for k in keys):
            codes = [data[k] for k in keys]
            latent = np.concatenate([c.astype(np.float32) for c in codes], axis=0)
        elif 'latent' in keys:
            latent = data['latent'].astype(np.float32)
        elif 'raw' in keys:
            raw = data['raw'].astype(np.float32)
            latent = np.expand_dims(raw, 0)
        else:
            latent = np.concatenate([data[k].astype(np.float32) for k in keys], axis=0)
        tensor = torch.from_numpy(latent)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor
