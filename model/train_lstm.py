"""train_lstm.py — Seq2Seq LSTM training"""
from __future__ import annotations
import argparse, random, torch, numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders, DateTokenizer
from models.lstm_model import LSTMDateGenerator


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, criterion, device, train: bool) -> float:
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            cond   = batch["conditions"].to(device)
            target = batch["target"].to(device)
            inp    = target[:, :-1]
            lbl    = target[:, 1:]
            logits = model(cond, inp)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), lbl.reshape(-1))
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * cond.size(0); n += cond.size(0)
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="../data/data.txt")
    parser.add_argument("--save-dir",   default="weights")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DateTokenizer()
    tr_l, vl_l, _ = get_dataloaders(args.data, args.batch_size, args.seed)

    model     = LSTMDateGenerator(tokenizer.cond_vocab_size, tokenizer.date_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    save_dir  = Path(args.save_dir); save_dir.mkdir(exist_ok=True)

    best_val, tr_losses, vl_losses = float("inf"), [], []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, tr_l, optimizer, criterion, device, True)
        vl = run_epoch(model, vl_l, optimizer, criterion, device, False)
        tr_losses.append(tr); vl_losses.append(vl)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {vl:.4f}")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_dir / "lstm_best.pt")

    plt.figure()
    plt.plot(tr_losses, label="train"); plt.plot(vl_losses, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(save_dir / "lstm_loss_curve.png"); plt.close()
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
