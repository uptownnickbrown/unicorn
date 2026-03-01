# prior_year_init.py
"""
Prior-year embedding initialization for Unicorn.

Builds a mapping from each player-season ID to the same player's prior-year ID,
then copies embeddings from a trained checkpoint to warm-start a new training run.
80.3% of player-season IDs have a prior-year mapping.
"""
from __future__ import annotations

from pathlib import Path
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
    current_emb = model.player_emb.weight.data  # [V, d_model]

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
    lu = pd.read_csv("player_season_lookup.csv")
    total = len(lu)
    print(f"Prior-year mappings: {len(pm):,} / {total:,} ({len(pm)/total:.1%})")
