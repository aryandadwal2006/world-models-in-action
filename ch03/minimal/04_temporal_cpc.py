"""Minimal causal CPC: predict future latents from a GRU context."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import Encoder, Sequences


class CPC(nn.Module):
    def __init__(self, dim=16, context=32, steps=3):
        super().__init__()
        self.encoder = Encoder(dim)
        self.gru = nn.GRU(dim, context, batch_first=True)
        self.predict = nn.ModuleList([nn.Linear(context, dim) for _ in range(steps)])

    def forward(self, frames):
        b, t, c, h, w = frames.shape
        z = self.encoder(frames.reshape(b * t, c, h, w)).reshape(b, t, -1)
        z = F.normalize(z, dim=-1)
        context, _ = self.gru(z)
        losses = []

        # At each time, the matching future frame is the positive; other
        # sequences in this batch provide negatives.
        for step, head in enumerate(self.predict, start=1):
            for time in range(t - step):
                pred = F.normalize(head(context[:, time]), dim=-1)
                target = z[:, time + step].detach()
                logits = pred @ target.T / 0.5
                losses.append(F.cross_entropy(logits, torch.arange(b, device=frames.device)))

        return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(Sequences(length=16), batch_size=32, shuffle=True, drop_last=True)
    model = CPC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total = 0
        for frames in loader:
            frames = frames.to(device)
            loss = model(frames)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: cpc_loss={total / len(loader):.3f}")

    torch.save(model.state_dict(), "ch03/minimal/cpc_encoder.pt")


if __name__ == "__main__":
    main()
