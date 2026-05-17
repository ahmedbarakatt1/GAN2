"""train_gan.py — cGAN training"""
from __future__ import annotations
import argparse, random, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders, DateTokenizer
from models.gan_model import CGANGenerator, CGANDiscriminator

NOISE_DIM = 64
SEQ_LEN   = 12


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pad_to(target, seq_len, device):
    if target.size(1) < seq_len:
        pad = torch.zeros(target.size(0), seq_len - target.size(1), dtype=torch.long, device=device)
        target = torch.cat([target, pad], dim=1)
    return target[:, :seq_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="../data/data.txt")
    parser.add_argument("--save-dir",   default="weights")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DateTokenizer()
    tr_l, _, _ = get_dataloaders(args.data, args.batch_size, args.seed)

    G = CGANGenerator(tokenizer.cond_vocab_size, tokenizer.date_vocab_size,
                      noise_dim=NOISE_DIM, seq_len=SEQ_LEN).to(device)
    D = CGANDiscriminator(tokenizer.cond_vocab_size, tokenizer.date_vocab_size,
                          seq_len=SEQ_LEN).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce   = nn.BCEWithLogitsLoss()
    save_dir = Path(args.save_dir); save_dir.mkdir(exist_ok=True)

    g_losses, d_losses = [], []
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        g_ep, d_ep, n = 0.0, 0.0, 0
        for batch in tqdm(tr_l, leave=False):
            cond   = batch["conditions"].to(device)
            target = pad_to(batch["target"].to(device), SEQ_LEN, device)
            B      = cond.size(0)
            real   = F.one_hot(target, tokenizer.date_vocab_size).float()
            ones   = torch.ones(B, 1, device=device)
            zeros  = torch.zeros(B, 1, device=device)

            # Train D
            noise    = torch.randn(B, NOISE_DIM, device=device)
            fake     = G(noise, cond).detach()
            d_loss   = (bce(D(cond, real), ones) + bce(D(cond, fake), zeros)) / 2
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # Train G
            noise    = torch.randn(B, NOISE_DIM, device=device)
            fake     = G(noise, cond)
            g_loss   = bce(D(cond, fake), ones)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_ep += g_loss.item() * B; d_ep += d_loss.item() * B; n += B

        g_losses.append(g_ep / n); d_losses.append(d_ep / n)
        print(f"Epoch {epoch:03d} | G {g_losses[-1]:.4f} | D {d_losses[-1]:.4f}")

    torch.save(G.state_dict(), save_dir / "gan_generator_best.pt")
    torch.save(D.state_dict(), save_dir / "gan_discriminator_best.pt")
    plt.figure()
    plt.plot(g_losses, label="Generator"); plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.legend()
    plt.savefig(save_dir / "gan_loss_curve.png"); plt.close()


if __name__ == "__main__":
    main()
