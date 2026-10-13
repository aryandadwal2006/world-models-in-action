"""A compact masked-pixel reconstruction example (not the full Transformer MAE)."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import Encoder, Frames


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = nn.Sequential(
            nn.Linear(16, 256 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        # Hide 75% of non-overlapping 8x8 patches.
        b, _, h, w = x.shape
        mask = (torch.rand(b, 1, h // 8, w // 8, device=x.device) < 0.75)
        mask = mask.repeat_interleave(8, -2).repeat_interleave(8, -1)
        masked = x.masked_fill(mask, 0)
        prediction = self.decoder(self.encoder(masked))
        loss = ((prediction - x).square() * mask).sum() / (mask.sum() * x.shape[1])
        return loss, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(Frames(), batch_size=32, shuffle=True)
    model = MaskedAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total = 0
        for x in loader:
            x = x.to(device)
            loss, _ = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: masked_mse={total / len(loader):.4f}")

    torch.save(model.state_dict(), "ch03/minimal/masked_encoder.pt")


if __name__ == "__main__":
    main()
