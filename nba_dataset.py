# nba_dataset.py – v3 (curriculum + player‑shuffle, no mirroring)
"""
PyTorch Dataset utilities with **chronological curriculum** and **player‑order
shuffling**, but **no team‑swap mirroring** (per user request).

Changes vs. original:
* Data are **sorted by season → game clock** so embeddings naturally carry over.
* Training split augments each sample by randomly permuting the 5 offensive and
  5 defensive player indices independently.
* Returns an extra tensor `team_ids` `[10]` (0 = offence, 1 = defence) for the
  transformer’s same‑team bias embedding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

################################################################################
# Outcome vocab
################################################################################

outcome_vocab = {
    "made_3pt_FG": 0,
    "made_2pt_FG": 1,
    "missed_3pt_FG": 2,
    "missed_2pt_FG": 3,
    "FT": 4,
    "turnover": 5,
    "other": 6,
}
NUM_OUTCOMES = 7


################################################################################
# Dataset
################################################################################

class PossessionDataset(Dataset):
    """Chronological curriculum + in‑place player shuffling (train split only)."""

    def __init__(
        self,
        parquet_path: str | Path,
        split: str = "train",
        state_norm: bool = True,
        shuffle_players: Optional[bool] = None,
    ) -> None:
        if shuffle_players is None:
            shuffle_players = (split == "train")

        cols = [f"p{i}" for i in range(10)] + ["sec_remaining_game", "score_diff", "period", "season", "outcome"]
        df = pd.read_parquet(parquet_path, columns=cols)

        # curriculum order
        df = df.sort_values(["season", "sec_remaining_game"], ascending=[True, True])

        # season‑based split
        if split == "train":
            df = df[df["season"] < 2019]
        elif split == "val":
            df = df[df["season"].between(2019, 2020)]
        elif split == "test":
            df = df[df["season"] >= 2021]
        else:
            raise ValueError("split must be train/val/test")
        df = df.reset_index(drop=True)

        # core arrays
        self.players = df[[f"p{i}" for i in range(10)]].to_numpy(np.int64)
        state = df[["sec_remaining_game", "score_diff", "period"]].to_numpy(np.float32)
        if state_norm:
            state[:, 0] /= 3600
            state[:, 1] /= 20.0
            state[:, 2] /= 10.0
        self.state = state
        self.label = df["outcome"].map(outcome_vocab).to_numpy(np.int64)

        # fixed team ids (first 5 = offence = 0, next 5 = defence = 1)
        self.base_team_ids = np.concatenate([np.zeros(5, np.int64), np.ones(5, np.int64)])

        self.shuffle_players = shuffle_players

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        players = self.players[idx].copy()
        state   = self.state[idx].copy()
        label   = self.label[idx]
        team_ids = self.base_team_ids

        if self.shuffle_players:
            perm_off = np.random.permutation(5)
            perm_def = np.random.permutation(5) + 5
            players[:5] = players[perm_off]
            players[5:] = players[perm_def]
            # team_ids remain correctly aligned since we permuted within blocks

        return (
            torch.from_numpy(players),           # [10] int64
            torch.from_numpy(state),             # [3]  float32
            torch.from_numpy(team_ids),          # [10] int64 (0/1)
            torch.tensor(label, dtype=torch.long),
        )

################################################################################
# Dataloader helpers
################################################################################

def get_default_dataloaders(parquet_path: str | Path, batch_size: int = 1024):
    train_ds = PossessionDataset(parquet_path, split="train")
    val_ds   = PossessionDataset(parquet_path, split="val", shuffle_players=False)
    test_ds  = PossessionDataset(parquet_path, split="test", shuffle_players=False)

    def _dl(ds):
        # curriculum ⇒ no random shuffle; let sequential sampler iterate in order
        return DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return _dl(train_ds), _dl(val_ds), _dl(test_ds)
