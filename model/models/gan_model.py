"""gan_model.py — Conditional GAN for date generation"""
from __future__ import annotations
import torch, torch.nn as nn


class CGANGenerator(nn.Module):
    def __init__(
        self,
        cond_vocab_size: int,
        date_vocab_size: int,
        noise_dim:   int   = 64,
        embed_dim:   int   = 32,
        hidden_dim:  int   = 256,
        seq_len:     int   = 12,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len     = seq_len
        self.noise_dim   = noise_dim
        self.date_vocab  = date_vocab_size
        self.temperature = temperature
        self.cond_embed  = nn.Embedding(cond_vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 4 * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),                nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * date_vocab_size),
        )

    def forward(self, noise: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        cond_flat = self.cond_embed(conditions).view(conditions.size(0), -1)
        out       = self.fc(torch.cat([noise, cond_flat], -1))
        return torch.softmax(out.view(-1, self.seq_len, self.date_vocab)
                             / self.temperature, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        conditions: torch.Tensor,
        eos_id: int,
    ) -> list[list[int]]:
        B      = conditions.size(0)
        noise  = torch.randn(B, self.noise_dim, device=conditions.device)
        probs  = self.forward(noise, conditions)          # (B, seq_len, V)
        tokens = probs.argmax(-1)                         # (B, seq_len)
        results = []
        for i in range(B):
            seq = []
            for tid in tokens[i].tolist():
                if tid == eos_id:
                    break
                seq.append(tid)
            results.append(seq)
        return results


class CGANDiscriminator(nn.Module):
    def __init__(
        self,
        cond_vocab_size: int,
        date_vocab_size: int,
        embed_dim:  int = 32,
        hidden_dim: int = 256,
        seq_len:    int = 12,
    ) -> None:
        super().__init__()
        self.cond_embed = nn.Embedding(cond_vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(4 * embed_dim + seq_len * date_vocab_size, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),                                nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, conditions: torch.Tensor, date_seq: torch.Tensor) -> torch.Tensor:
        cond_flat = self.cond_embed(conditions).view(conditions.size(0), -1)
        return self.fc(torch.cat([cond_flat, date_seq.view(conditions.size(0), -1)], -1))
