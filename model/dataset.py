"""dataset.py — tokenizer, dataset, and dataloaders"""
from __future__ import annotations
import re, torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional

DAYS    = ["MON","TUE","WED","THU","FRI","SAT","SUN"]
MONTHS  = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
LEAPS   = ["False","True"]
DECADES = [str(d) for d in range(180, 221)]

SPECIAL      = ["<PAD>", "<UNK>"]
COND_VOCAB   = SPECIAL + DAYS + MONTHS + LEAPS + DECADES

DATE_SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]
DATE_CHARS   = list("0123456789-")
DATE_VOCAB   = DATE_SPECIAL + DATE_CHARS   # 14 tokens


class DateTokenizer:
    def __init__(self) -> None:
        self.cond_to_id = {t: i for i, t in enumerate(COND_VOCAB)}
        self.date_to_id = {t: i for i, t in enumerate(DATE_VOCAB)}
        self.id_to_date = {i: t for t, i in self.date_to_id.items()}

    @property
    def cond_vocab_size(self) -> int: return len(COND_VOCAB)
    @property
    def date_vocab_size(self) -> int: return len(DATE_VOCAB)
    @property
    def pad_id(self)  -> int: return self.date_to_id["<PAD>"]
    @property
    def bos_id(self)  -> int: return self.date_to_id["<BOS>"]
    @property
    def eos_id(self)  -> int: return self.date_to_id["<EOS>"]

    def encode_conditions(self, line: str) -> list[int]:
        """Encodes the 4 condition tokens from a line (strips brackets)."""
        tokens = re.findall(r'\[([^\]]+)\]', line)[:4]
        return [self.cond_to_id.get(t, 1) for t in tokens]   # 1 = UNK

    def encode_date(self, date_str: str) -> list[int]:
        """Encodes dd-mm-yyyy char by char, wrapped with BOS and EOS."""
        ids = [self.bos_id]
        for ch in date_str.strip():
            ids.append(self.date_to_id.get(ch, 0))
        ids.append(self.eos_id)
        return ids

    def decode_date(self, token_ids: list[int]) -> str:
        """Decodes token ids back to a date string, stopping at EOS."""
        chars = []
        for tid in token_ids:
            tok = self.id_to_date.get(tid, "")
            if tok == "<EOS>":
                break
            if tok not in ("<PAD>", "<BOS>", "<UNK>"):
                chars.append(tok)
        return "".join(chars)


class DateDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Optional[DateTokenizer] = None) -> None:
        self.tokenizer = tokenizer or DateTokenizer()
        self.samples: list[dict] = []
        for line in Path(data_path).read_text().strip().splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            conditions_str, date_str = parts
            self.samples.append({
                "conditions":     self.tokenizer.encode_conditions(conditions_str),
                "target":         self.tokenizer.encode_date(date_str),
                "raw_conditions": conditions_str.strip(),
                "raw_date":       date_str.strip(),
            })

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "conditions": torch.tensor(s["conditions"], dtype=torch.long),
            "target":     torch.tensor(s["target"],     dtype=torch.long),
        }

    @staticmethod
    def split(
        dataset: "DateDataset",
        train: float = 0.8,
        val:   float = 0.1,
        test:  float = 0.1,
        seed:  int   = 42,
    ):
        n       = len(dataset)
        n_train = int(n * train)
        n_val   = int(n * val)
        n_test  = n - n_train - n_val
        gen     = torch.Generator().manual_seed(seed)
        return random_split(dataset, [n_train, n_val, n_test], generator=gen)


def collate_fn(batch: list[dict]) -> dict:
    conditions = torch.stack([b["conditions"] for b in batch])
    targets    = [b["target"] for b in batch]
    max_len    = max(t.size(0) for t in targets)
    padded     = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, :t.size(0)] = t
    return {"conditions": conditions, "target": padded}


def get_dataloaders(
    data_path:  str,
    batch_size: int = 64,
    seed:       int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset                        = DateDataset(data_path)
    train_set, val_set, test_set   = DateDataset.split(dataset, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
