"""lstm_model.py — Seq2Seq LSTM for conditional date generation"""
from __future__ import annotations
import torch, torch.nn as nn


class LSTMDateGenerator(nn.Module):
    def __init__(
        self,
        cond_vocab_size: int,
        date_vocab_size: int,
        embed_dim:   int   = 32,
        hidden_dim:  int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.cond_embed  = nn.Embedding(cond_vocab_size, embed_dim, padding_idx=0)
        self.encoder     = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=dropout)
        self.date_embed  = nn.Embedding(date_vocab_size, embed_dim, padding_idx=0)
        self.decoder     = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, date_vocab_size)

    def forward(
        self,
        conditions: torch.Tensor,   # (B, 4)
        target:     torch.Tensor,   # (B, T)  — teacher-forced input
    ) -> torch.Tensor:              # (B, T, date_vocab_size) logits
        _, (h, c) = self.encoder(self.cond_embed(conditions))
        out, _    = self.decoder(self.date_embed(target), (h, c))
        return self.output_proj(out)

    @torch.no_grad()
    def generate(
        self,
        conditions: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 14,
    ) -> list[list[int]]:
        B = conditions.size(0)
        _, (h, c) = self.encoder(self.cond_embed(conditions))
        token   = torch.full((B, 1), bos_id, dtype=torch.long, device=conditions.device)
        results = [[] for _ in range(B)]
        done    = [False] * B
        for _ in range(max_len):
            out, (h, c) = self.decoder(self.date_embed(token), (h, c))
            token = self.output_proj(out.squeeze(1)).argmax(-1, keepdim=True)
            for i in range(B):
                if not done[i]:
                    tid = token[i, 0].item()
                    if tid == eos_id: done[i] = True
                    else: results[i].append(tid)
            if all(done): break
        return results
