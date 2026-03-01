# nba_dataset.py – v4 (9-class taxonomy, vocab fix, epoch shuffle support)
"""
PyTorch Dataset for Unicorn NBA possession outcome prediction.

Key features:
* **9-class outcome taxonomy** with shot-distance and turnover-type granularity.
* **Chronological curriculum** (sorted by season → game clock) for first epoch.
* **Player-order shuffling** (train split) for permutation invariance.
* **Fixed vocab size** from lookup table (not max of training data).
* Returns `(players[10], state[3], team_ids[10], label)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

################################################################################
# Outcome vocab (9-class, no "other")
################################################################################

OUTCOME_VOCAB = {
    "made_3pt":         0,
    "missed_3pt":       1,
    "made_2pt_close":   2,
    "made_2pt_mid":     3,
    "missed_2pt_close": 4,
    "missed_2pt_mid":   5,
    "FT":               6,
    "turnover_live":    7,
    "turnover_dead":    8,
}
NUM_OUTCOMES = len(OUTCOME_VOCAB)
OUTCOME_NAMES = {v: k for k, v in OUTCOME_VOCAB.items()}

# Default lookup table path (written by preprocessing pipeline)
DEFAULT_LOOKUP = "player_season_lookup.csv"


def get_num_players(lookup_csv: str | Path = DEFAULT_LOOKUP) -> int:
    """Return total player-season vocab size from the lookup table."""
    return len(pd.read_csv(lookup_csv))


################################################################################
# Dataset
################################################################################

class PossessionDataset(Dataset):
    """Chronological curriculum + in-place player shuffling (train split only)."""

    def __init__(
        self,
        parquet_path: str | Path,
        split: str = "train",
        state_norm: bool = True,
        shuffle_players: Optional[bool] = None,
        filter_other: bool = True,
        lookup_csv: str | Path = DEFAULT_LOOKUP,
    ) -> None:
        if shuffle_players is None:
            shuffle_players = (split == "train")

        cols = [f"p{i}" for i in range(10)] + [
            "sec_remaining_game", "score_diff", "period", "season", "outcome"
        ]
        df = pd.read_parquet(parquet_path, columns=cols)

        # filter out "other" class (defensive rebounds, period starts, jump balls)
        if filter_other:
            df = df[df["outcome"] != "other"]

        # curriculum order
        df = df.sort_values(["season", "sec_remaining_game"], ascending=[True, True])

        # season-based split
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
        self.label = df["outcome"].map(OUTCOME_VOCAB).to_numpy(np.int64)

        # vocab size from lookup table (not max of training data)
        self.num_players = get_num_players(lookup_csv)

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

        return (
            torch.from_numpy(players),           # [10] int64
            torch.from_numpy(state),             # [3]  float32
            torch.from_numpy(team_ids),          # [10] int64 (0/1)
            torch.tensor(label, dtype=torch.long),
        )


################################################################################
# Dataloader helpers
################################################################################

def get_default_dataloaders(
    parquet_path: str | Path,
    batch_size: int = 1024,
    lookup_csv: str | Path = DEFAULT_LOOKUP,
    shuffle_train: bool = False,
):
    """Create train/val/test dataloaders.

    Args:
        shuffle_train: If True, shuffle training data (recommended after epoch 1
                       to break curriculum ordering). If False, use sequential
                       order for curriculum learning.
    """
    train_ds = PossessionDataset(parquet_path, split="train", lookup_csv=lookup_csv)
    val_ds   = PossessionDataset(parquet_path, split="val", shuffle_players=False, lookup_csv=lookup_csv)
    test_ds  = PossessionDataset(parquet_path, split="test", shuffle_players=False, lookup_csv=lookup_csv)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        pin_memory=True, num_workers=4,
    )
    eval_dl_fn = lambda ds: DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=4,
    )

    return train_dl, eval_dl_fn(val_ds), eval_dl_fn(test_ds)
