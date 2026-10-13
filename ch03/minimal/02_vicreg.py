"""Minimal VICReg training loop. Run from the repository root."""

import argparse

import torch
from torch.utils.data import DataLoader

from common import Encoder, Frames, Views, vicreg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(Frames(), batch_size=64, shuffle=True, drop_last=True)
    encoder = Encoder().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    views = Views()

    for epoch in range(args.epochs):
        total = 0
        for x in loader:
            x = x.to(device)
            x1, x2 = views(x)
            loss = vicreg(encoder(x1), encoder(x2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(loader):.3f}")

    torch.save(encoder.state_dict(), "ch03/minimal/vicreg_encoder.pt")


if __name__ == "__main__":
    main()
