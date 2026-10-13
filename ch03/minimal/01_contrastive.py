"""Minimal NT-Xent training loop. Run from the repository root."""

import argparse

import torch
from torch.utils.data import DataLoader

from common import Encoder, Frames, Views, ntxent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(Frames(), batch_size=64, shuffle=True, drop_last=True)
    encoder, projector = Encoder().to(device), torch.nn.Sequential(
        torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16)
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=1e-3
    )
    views = Views()

    for epoch in range(args.epochs):
        total = 0
        for x in loader:
            x = x.to(device)
            x1, x2 = views(x)
            loss = ntxent(projector(encoder(x1)), projector(encoder(x2)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(loader):.3f}")

    torch.save(encoder.state_dict(), "ch03/minimal/contrastive_encoder.pt")


if __name__ == "__main__":
    main()
