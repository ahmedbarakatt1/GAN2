"""train_cvae.py — cVAE training"""
from __future__ import annotations
import argparse, random, torch, numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders, DateTokenizer
from models.cvae_model import CVAE

SEQ_LEN = 12


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pad_to(target, seq_len, device):
    if target.size(1) < seq_len:
        pad = torch.zeros(target.size(0), seq_len - target.size(1), dtype=torch.long, device=device)
        target = torch.cat([target, pad], dim=1)
    return target[:, :seq_len]


def cvae_loss(logits, target, mu, logvar, beta, pad_id):
    recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            target.reshape(-1), ignore_index=pad_id)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def run_epoch(model, loader, optimizer, device, tokenizer, beta, train):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            cond   = batch["conditions"].to(device)
            target = pad_to(batch["target"].to(device), SEQ_LEN, device)
            logits, mu, logvar = model(cond, target)
            loss, _, _ = cvae_loss(logits, target, mu, logvar, beta, tokenizer.pad_id)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * cond.size(0); n += cond.size(0)
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="../data/data.txt")
    parser.add_argument("--save-dir",   default="weights")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--beta",       type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DateTokenizer()
    tr_l, vl_l, _ = get_dataloaders(args.data, args.batch_size, args.seed)

    model     = CVAE(cond_vocab_size=tokenizer.cond_vocab_size,
                     date_vocab_size=tokenizer.date_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_dir  = Path(args.save_dir); save_dir.mkdir(exist_ok=True)

    best_val, tr_losses, vl_losses = float("inf"), [], []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, tr_l, optimizer, device, tokenizer, args.beta, True)
        vl = run_epoch(model, vl_l, optimizer, device, tokenizer, args.beta, False)
        tr_losses.append(tr); vl_losses.append(vl)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {vl:.4f}")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_dir / "cvae_best.pt")

    plt.figure()
    plt.plot(tr_losses, label="train"); plt.plot(vl_losses, label="val")
    plt.xlabel("Epoch"); plt.ylabel("ELBO Loss"); plt.legend()
    plt.savefig(save_dir / "cvae_loss_curve.png"); plt.close()


if __name__ == "__main__":
    main()
