"""cvae_model.py — Conditional VAE for date generation"""
from __future__ import annotations
import torch, torch.nn as nn

SEQ_LEN = 12


class CVAEEncoder(nn.Module):
    def __init__(self, cond_vocab_size, date_vocab_size,
                 embed_dim=32, hidden_dim=128, latent_dim=32) -> None:
        super().__init__()
        self.cond_embed = nn.Embedding(cond_vocab_size, embed_dim, padding_idx=0)
        self.date_embed = nn.Embedding(date_vocab_size, embed_dim, padding_idx=0)
        self.fc         = nn.Sequential(
            nn.Linear(4 * embed_dim + SEQ_LEN * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),                           nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, conditions, date_seq):
        x  = torch.cat([
            self.cond_embed(conditions).view(conditions.size(0), -1),
            self.date_embed(date_seq).view(date_seq.size(0), -1),
        ], dim=-1)
        h  = self.fc(x)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    def __init__(self, cond_vocab_size, date_vocab_size,
                 embed_dim=32, hidden_dim=128, latent_dim=32) -> None:
        super().__init__()
        self.latent_dim  = latent_dim
        self.date_vocab  = date_vocab_size
        self.cond_embed  = nn.Embedding(cond_vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(4 * embed_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),                  nn.ReLU(),
            nn.Linear(hidden_dim, SEQ_LEN * date_vocab_size),
        )

    def forward(self, conditions, z):
        cond_flat = self.cond_embed(conditions).view(conditions.size(0), -1)
        return self.fc(torch.cat([cond_flat, z], -1)).view(-1, SEQ_LEN, self.date_vocab)


class CVAE(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = CVAEEncoder(**kwargs)
        self.decoder = CVAEDecoder(**kwargs)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def forward(self, conditions, date_seq):
        mu, logvar = self.encoder(conditions, date_seq)
        return self.decoder(conditions, self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def generate(self, conditions, eos_id: int, num_samples: int = 1) -> list[list[int]]:
        B, device = conditions.size(0), conditions.device
        results   = []
        for i in range(B):
            cond_i = conditions[i:i+1].expand(num_samples, -1)
            z      = torch.randn(num_samples, self.decoder.latent_dim, device=device)
            logits = self.decoder(cond_i, z)       # (S, seq_len, V)
            tokens = logits.argmax(-1)[0].tolist() # first sample, list of int ids
            seq = []
            for tid in tokens:
                if tid == eos_id:
                    break
                seq.append(tid)
            results.append(seq)
        return results
