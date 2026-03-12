# stint_dataset.py – v6 (Stint-level dataset for Relation Network)
"""
PyTorch Dataset for Unicorn v6 stint-level training.

A stint is a consecutive sequence of possessions with the same 10-player lineup.
Each stint has aggregated outcome counts that become Bayesian-smoothed distributional
targets — same formula as v3.2 but per-stint instead of per-lineup-across-games.

Key features:
* Stint-level aggregation reduces noise (3-15 possessions per stint)
* Bayesian-smoothed distributional targets from per-stint outcome counts
* Player-order shuffling within offense/defense (train split only)
* Returns `(players[10], state[3], team_ids[10], target_dist[9])`
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from nba_dataset import OUTCOME_VOCAB, NUM_OUTCOMES, OUTCOME_NAMES, get_num_players, DEFAULT_LOOKUP


class StintDataset(Dataset):
    """Stint-level dataset with Bayesian-smoothed distributional targets.

    Each stint has outcome_counts[9] from which we compute:
        empirical = outcome_counts / n_poss
        target = (n_poss * empirical + alpha * global_prior) / (n_poss + alpha)
    """

    def __init__(
        self,
        stints_path: str | Path,
        split: str = "train",
        state_norm: bool = True,
        shuffle_players: Optional[bool] = None,
        lookup_csv: str | Path = DEFAULT_LOOKUP,
        prior_strength: float = 10.0,
    ) -> None:
        if shuffle_players is None:
            shuffle_players = (split == "train")

        df = pd.read_parquet(stints_path)

        # Season-based split
        if split == "train":
            df = df[df["season"] < 2019]
        elif split == "val":
            df = df[df["season"].between(2019, 2020)]
        elif split == "test":
            df = df[df["season"] >= 2021]
        else:
            raise ValueError("split must be train/val/test")
        df = df.reset_index(drop=True)

        # Player IDs
        self.players = df[[f"p{i}" for i in range(10)]].to_numpy(np.int64)

        # Game state (start of stint)
        state = df[["sec_remaining_game", "score_diff", "period"]].to_numpy(np.float32).copy()
        if state_norm:
            state[:, 0] /= 3600
            state[:, 1] /= 20.0
            state[:, 2] /= 10.0
        self.state = state

        # Outcome counts
        outcome_counts = df[[f"oc{i}" for i in range(NUM_OUTCOMES)]].to_numpy(np.float32)
        n_poss = df["n_poss"].to_numpy(np.float32)

        # Global prior from all stints in this split
        global_counts = outcome_counts.sum(axis=0)
        global_prior = global_counts / global_counts.sum()
        self.global_prior = global_prior

        # Bayesian-smoothed distributional targets per stint
        alpha = prior_strength
        empirical = outcome_counts / n_poss[:, None]
        self.target_dist = (n_poss[:, None] * empirical + alpha * global_prior) / (n_poss[:, None] + alpha)
        self.target_dist = self.target_dist.astype(np.float32)

        # Hard labels = argmax of empirical distribution (for accuracy reporting)
        self.label = outcome_counts.argmax(axis=1).astype(np.int64)

        # Metadata
        self.n_poss = n_poss.astype(np.int64)
        self.num_players = get_num_players(lookup_csv)
        self.base_team_ids = np.concatenate([np.zeros(5, np.int64), np.ones(5, np.int64)])
        self.shuffle_players = shuffle_players

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        players = self.players[idx].copy()
        state = self.state[idx].copy()
        target_dist = self.target_dist[idx].copy()
        team_ids = self.base_team_ids

        if self.shuffle_players:
            perm_off = np.random.permutation(5)
            perm_def = np.random.permutation(5) + 5
            players[:5] = players[perm_off]
            players[5:] = players[perm_def]

        return (
            torch.from_numpy(players),       # [10] int64
            torch.from_numpy(state),         # [3]  float32
            torch.from_numpy(team_ids),      # [10] int64 (0/1)
            torch.from_numpy(target_dist),   # [9]  float32
        )


def get_stint_dataloaders(
    stints_path: str | Path,
    batch_size: int = 2048,
    lookup_csv: str | Path = DEFAULT_LOOKUP,
    shuffle_train: bool = True,
    prior_strength: float = 10.0,
):
    """Create train/val/test dataloaders for stint-level data."""
    train_ds = StintDataset(stints_path, split="train", lookup_csv=lookup_csv,
                            prior_strength=prior_strength)
    val_ds = StintDataset(stints_path, split="val", shuffle_players=False,
                          lookup_csv=lookup_csv, prior_strength=prior_strength)
    test_ds = StintDataset(stints_path, split="test", shuffle_players=False,
                           lookup_csv=lookup_csv, prior_strength=prior_strength)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        pin_memory=True, num_workers=4, persistent_workers=True,
    )
    eval_dl_fn = lambda ds: DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=4, persistent_workers=True,
    )

    return train_dl, eval_dl_fn(val_ds), eval_dl_fn(test_ds)
