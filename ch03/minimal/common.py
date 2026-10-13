"""Small shared pieces for the Chapter 3 teaching examples."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]


class Frames(Dataset):
    """Episode-balanced sample of cached frames."""
    def __init__(self, split="train", limit=4000):
        path = ROOT / "data" / f"dmc_cartpole_balance_{split}.npz"
        with np.load(path) as data:
            self.frames = data["frames"]
            self.episodes = data["episode_ids"]
        groups = [np.flatnonzero(self.episodes == e) for e in np.unique(self.episodes)]
        chosen = []
        while len(chosen) < min(limit, len(self.frames)):
            changed = False
            for group in groups:
                if not len(group):
                    continue
                position = len(chosen) // max(1, len(groups))
                if position < len(group):
                    chosen.append(group[position])
                    changed = True
                    if len(chosen) == min(limit, len(self.frames)):
                        break
            if not changed:
                break
        self.indices = np.asarray(chosen, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image = self.frames[self.indices[i]]
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255


class Sequences(Dataset):
    """Contiguous windows that never cross an episode boundary."""
    def __init__(self, split="train", length=16, limit=2000):
        path = ROOT / "data" / f"dmc_cartpole_balance_{split}.npz"
        with np.load(path) as data:
            self.frames = data["frames"]
            self.episodes = data["episode_ids"]
        self.length = length
        starts = [i for i in range(len(self.frames) - length + 1)
                  if np.all(self.episodes[i:i + length] == self.episodes[i])]
        if len(starts) > limit:
            starts = np.linspace(0, len(starts) - 1, limit, dtype=int).tolist()
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start = self.starts[i]
        frames = self.frames[start:start + self.length]
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255


class Encoder(nn.Module):
    """Four stride-2 convolutions map a 64x64 RGB frame to one latent vector."""
    def __init__(self, dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Linear(256 * 4 * 4, dim)

    def forward(self, x):
        return self.head(self.conv(x))


class Views:
    """Small crop and brightness/contrast perturbations."""
    def __call__(self, x):
        return self.augment(x), self.augment(x)

    def augment(self, x):
        pad = F.pad(x, (3, 3, 3, 3), mode="replicate")
        shifted = torch.stack([
            pad[i, :, dy:dy + x.shape[-2], dx:dx + x.shape[-1]]
            for i, (dy, dx) in enumerate(torch.randint(0, 7, (len(x), 2)).tolist())
        ])
        contrast = 1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=x.device) - 0.5)
        brightness = 0.2 * (torch.rand(len(x), 1, 1, 1, device=x.device) - 0.5)
        return (shifted * contrast + brightness).clamp(0, 1)


def ntxent(a, b, temperature=0.5):
    """Cross-entropy: each view must identify its matching view."""
    n = len(a)
    if n < 2:
        raise ValueError("NT-Xent needs at least two examples")
    z = F.normalize(torch.cat([a, b]), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(float("-inf"))
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(logits, labels)


def vicreg(a, b, sim=25, std=25, cov=1):
    """Invariance + variance floor + covariance penalty."""
    def variance(z):
        return F.relu(1 - torch.sqrt(z.var(0, unbiased=False) + 1e-4)).mean()

    def covariance(z):
        z = z - z.mean(0)
        c = z.T @ z / (len(z) - 1)
        off_diagonal = ~torch.eye(c.shape[0], dtype=torch.bool, device=z.device)
        return c[off_diagonal].square().sum() / c.shape[0]

    invariance = F.mse_loss(a, b)
    spread = variance(a) + variance(b)
    redundancy = covariance(a) + covariance(b)
    return sim * invariance + std * spread + cov * redundancy
