# prior_year_init.py
"""
Prior-year embedding initialization and mapping utilities for Unicorn.

Builds mappings between player-season IDs and base player IDs for the
composed embedding architecture (base + delta). Also provides prior/next
year maps for temporal augmentation and embedding warm-start.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch


def build_prior_year_map(lookup_csv: str | Path = "player_season_lookup.csv") -> dict[int, int]:
    """Build {current_player_season_id: prior_year_player_season_id} mapping.

    For each (player, season) pair, looks for the same player in (season - 1).
    Returns only IDs that have a valid prior-year counterpart.
    """
    lu = pd.read_csv(lookup_csv)
    id_map = lu.set_index(["player", "season"])["player_season_id"].to_dict()

    prior_map = {}
    for (player, season), current_id in id_map.items():
        prior_key = (player, season - 1)
        if prior_key in id_map:
            prior_map[current_id] = id_map[prior_key]

    return prior_map


def build_next_year_map(lookup_csv: str | Path = "player_season_lookup.csv") -> dict[int, int]:
    """Build {current_player_season_id: next_year_player_season_id} mapping.

    For each (player, season) pair, looks for the same player in (season + 1).
    Returns only IDs that have a valid next-year counterpart.
    Used for temporal augmentation (swap to adjacent season).
    """
    lu = pd.read_csv(lookup_csv)
    id_map = lu.set_index(["player", "season"])["player_season_id"].to_dict()

    next_map = {}
    for (player, season), current_id in id_map.items():
        next_key = (player, season + 1)
        if next_key in id_map:
            next_map[current_id] = id_map[next_key]

    return next_map


def build_base_player_mapping(
    lookup_csv: str | Path = "player_season_lookup.csv",
) -> tuple[dict[int, int], int]:
    """Build mapping from player_season_id to base_player_id.

    Each unique player (e.g., "jamesle01") gets a single base_player_id
    shared across all their seasons. Returns the mapping dict and total
    number of unique base players.

    Returns:
        (ps_to_base, num_base_players):
            ps_to_base: {player_season_id: base_player_id}
            num_base_players: total unique players (2,310)
    """
    lu = pd.read_csv(lookup_csv)
    # Drop rows with missing player names
    lu = lu.dropna(subset=["player"])
    # Assign unique integer IDs to each player string
    unique_players = lu["player"].unique()
    player_to_base = {p: i for i, p in enumerate(sorted(unique_players))}

    ps_to_base = {}
    for _, row in lu.iterrows():
        ps_to_base[int(row["player_season_id"])] = player_to_base[row["player"]]

    return ps_to_base, len(unique_players)


def build_ps_to_base_tensor(
    num_player_seasons: int,
    lookup_csv: str | Path = "player_season_lookup.csv",
) -> tuple[torch.Tensor, int]:
    """Build a tensor mapping player_season_id → base_player_id.

    Returns:
        (tensor, num_base_players):
            tensor: [num_player_seasons] LongTensor, where tensor[ps_id] = base_id
            num_base_players: total unique players
    """
    ps_to_base, num_base = build_base_player_mapping(lookup_csv)
    t = torch.zeros(num_player_seasons, dtype=torch.long)
    for ps_id, base_id in ps_to_base.items():
        if ps_id < num_player_seasons:
            t[ps_id] = base_id
    return t, num_base


def build_temporal_swap_tensor(
    num_player_seasons: int,
    lookup_csv: str | Path = "player_season_lookup.csv",
) -> torch.Tensor:
    """Build a tensor for temporal augmentation: maps each ps_id to a random adjacent season.

    For each player-season ID, stores the ID of an adjacent (prior or next) season
    for the same player. If both exist, picks one randomly. If neither exists,
    maps to self (no swap).

    Returns:
        [num_player_seasons] LongTensor
    """
    prior = build_prior_year_map(lookup_csv)
    nxt = build_next_year_map(lookup_csv)

    t = torch.arange(num_player_seasons, dtype=torch.long)  # default: map to self
    for ps_id in range(num_player_seasons):
        candidates = []
        if ps_id in prior:
            candidates.append(prior[ps_id])
        if ps_id in nxt:
            candidates.append(nxt[ps_id])
        if candidates:
            t[ps_id] = candidates[np.random.randint(len(candidates))]

    return t


def init_embeddings_from_prior(
    model: torch.nn.Module,
    prior_map: dict[int, int],
    prior_checkpoint_path: str | Path,
    embedding_key: str = "player_emb.weight",
) -> int:
    """Initialize player embeddings from a prior training run's checkpoint.

    For each current_id → prior_id mapping, copies the prior checkpoint's
    embedding into the current model's embedding table. IDs without a prior
    mapping keep their random initialization.

    Supports both v1 (player_emb) and v2 (base_player_emb + delta_emb) architectures.

    Args:
        model: The model whose embeddings to initialize.
        prior_map: {current_id: prior_year_id} from build_prior_year_map.
        prior_checkpoint_path: Path to a previously trained checkpoint.
        embedding_key: Key in the state_dict for the player embedding weight.

    Returns:
        Number of embeddings successfully initialized.
    """
    ckpt = torch.load(prior_checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    prior_emb = state_dict[embedding_key]  # [V, d_model]

    # v2 composed embeddings: model has base_player_emb + delta_emb
    if hasattr(model, "base_player_emb"):
        # For v2, we can only warm-start delta from prior delta (same ps_id space)
        current_emb = model.delta_emb.weight.data
        if embedding_key in state_dict:
            prior_emb = state_dict[embedding_key]
        elif "delta_emb.weight" in state_dict:
            prior_emb = state_dict["delta_emb.weight"]
        else:
            return 0
    else:
        # v1: direct player_emb
        current_emb = model.player_emb.weight.data

    if prior_emb.shape[1] != current_emb.shape[1]:
        raise ValueError(
            f"d_model mismatch: prior checkpoint has {prior_emb.shape[1]}, "
            f"current model has {current_emb.shape[1]}"
        )

    count = 0
    for current_id, prior_id in prior_map.items():
        if current_id < current_emb.shape[0] and prior_id < prior_emb.shape[0]:
            current_emb[current_id] = prior_emb[prior_id]
            count += 1

    return count


if __name__ == "__main__":
    # Quick diagnostic
    pm = build_prior_year_map()
    nm = build_next_year_map()
    ps_to_base, num_base = build_base_player_mapping()
    lu = pd.read_csv("player_season_lookup.csv")
    total = len(lu)
    print(f"Player-season IDs: {total:,}")
    print(f"Base players: {num_base:,}")
    print(f"Prior-year mappings: {len(pm):,} / {total:,} ({len(pm)/total:.1%})")
    print(f"Next-year mappings: {len(nm):,} / {total:,} ({len(nm)/total:.1%})")
    # Players with both prior and next
    both = set(pm.keys()) & set(nm.keys())
    print(f"Both prior+next: {len(both):,} / {total:,} ({len(both)/total:.1%})")
