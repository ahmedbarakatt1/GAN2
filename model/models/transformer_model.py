"""transformer_model.py — Encoder-Decoder Transformer for date generation"""
from __future__ import annotations
import torch, torch.nn as nn


class TransformerDateGenerator(nn.Module):
    def __init__(
        self,
        cond_vocab_size: int,
        date_vocab_size: int,
        embed_dim:      int   = 64,
        nhead:          int   = 4,
        num_enc_layers: int   = 2,
        num_dec_layers: int   = 2,
        ff_dim:         int   = 256,
        dropout:        float = 0.1,
        max_seq_len:    int   = 16,
    ) -> None:
        super().__init__()
        self.cond_embed  = nn.Embedding(cond_vocab_size, embed_dim, padding_idx=0)
        self.date_embed  = nn.Embedding(date_vocab_size, embed_dim, padding_idx=0)
        self.pos_enc     = nn.Embedding(max_seq_len, embed_dim)
        enc_layer        = nn.TransformerEncoderLayer(embed_dim, nhead, ff_dim, dropout, batch_first=True)
        dec_layer        = nn.TransformerDecoderLayer(embed_dim, nhead, ff_dim, dropout, batch_first=True)
        self.encoder     = nn.TransformerEncoder(enc_layer, num_enc_layers)
        self.decoder     = nn.TransformerDecoder(dec_layer, num_dec_layers)
        self.output_proj = nn.Linear(embed_dim, date_vocab_size)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def _encode(self, conditions: torch.Tensor) -> torch.Tensor:
        pos    = torch.arange(4, device=conditions.device)
        return self.encoder(self.cond_embed(conditions) + self.pos_enc(pos))

    def forward(self, conditions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory = self._encode(conditions)
        T      = target.size(1)
        pos    = torch.arange(T, device=target.device)
        dec    = self.decoder(
            self.date_embed(target) + self.pos_enc(pos),
            memory,
            tgt_mask=self._causal_mask(T, target.device),
        )
        return self.output_proj(dec)

    @torch.no_grad()
    def generate(
        self, conditions: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 14,
    ) -> list[list[int]]:
        B, device = conditions.size(0), conditions.device
        memory    = self._encode(conditions)
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        results   = [[] for _ in range(B)]
        done      = [False] * B
        for _ in range(max_len):
            T      = generated.size(1)
            pos    = torch.arange(T, device=device)
            dec    = self.decoder(
                self.date_embed(generated) + self.pos_enc(pos),
                memory, tgt_mask=self._causal_mask(T, device),
            )
            next_tok  = self.output_proj(dec[:, -1]).argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            for i in range(B):
                if not done[i]:
                    tid = next_tok[i, 0].item()
                    if tid == eos_id: done[i] = True
                    else: results[i].append(tid)
            if all(done): break
        return results
