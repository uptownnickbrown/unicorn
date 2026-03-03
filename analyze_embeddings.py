# analyze_embeddings.py
"""
Embedding analysis and visualization for Unicorn.

Produces:
1. Player similarity (nearest neighbors for notable players)
2. Position/role clustering (t-SNE / UMAP visualization)
3. Temporal trajectories (how player embeddings evolve across seasons)
4. Attention weight analysis
5. Embedding quality metrics

Usage:
```bash
python analyze_embeddings.py --ckpt pretrain_checkpoint.pt
python analyze_embeddings.py --ckpt pretrain_checkpoint.pt --output-dir plots/
```
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
# Data loading
################################################################################

def load_player_lookup(lookup_csv: str = "player_season_lookup.csv") -> pd.DataFrame:
    """Load player-season lookup table."""
    return pd.read_csv(lookup_csv)


def load_embeddings(ckpt_path: str) -> torch.Tensor:
    """Load player embeddings from checkpoint.

    Supports v1 (single player_emb), v2 (base + delta composed), and CBOW.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # v2: composed embeddings (base + delta)
    if "base_player_emb.weight" in sd and "delta_emb.weight" in sd:
        base_emb = sd["base_player_emb.weight"]
        delta_emb = sd["delta_emb.weight"]
        ps_to_base = sd["ps_to_base"]
        return base_emb[ps_to_base] + delta_emb
    # v1
    if "player_emb.weight" in sd:
        return sd["player_emb.weight"]
    # CBOW
    if "emb.weight" in sd:
        return sd["emb.weight"]
    raise KeyError("No player embedding found in checkpoint")


def make_id_to_name(lookup: pd.DataFrame) -> dict:
    """Create {player_season_id: 'PlayerName (YYYY)'} mapping."""
    return {
        row.player_season_id: f"{row.player} ({row.season})"
        for _, row in lookup.iterrows()
    }


################################################################################
# 1. Nearest Neighbors
################################################################################

def find_nearest_neighbors(
    emb: torch.Tensor,
    query_ids: list[int],
    id_to_name: dict,
    k: int = 10,
) -> dict:
    """Find k-nearest neighbors by cosine similarity for given player IDs."""
    emb_norm = F.normalize(emb, dim=1)
    results = {}

    for qid in query_ids:
        if qid >= emb.shape[0]:
            continue
        sims = emb_norm @ emb_norm[qid]
        topk = sims.topk(k + 1)  # +1 because the player is their own neighbor

        neighbors = []
        for idx, sim in zip(topk.indices.tolist(), topk.values.tolist()):
            if idx == qid:
                continue
            neighbors.append({
                "id": idx,
                "name": id_to_name.get(idx, f"ID_{idx}"),
                "similarity": round(sim, 4),
            })

        query_name = id_to_name.get(qid, f"ID_{qid}")
        results[query_name] = neighbors[:k]

    return results


def print_nearest_neighbors(results: dict):
    """Pretty-print nearest neighbor results."""
    for query, neighbors in results.items():
        print(f"\n  {query}:")
        for i, n in enumerate(neighbors, 1):
            print(f"    {i:2d}. {n['name']:35s} sim={n['similarity']:.4f}")


################################################################################
# 2. Clustering Visualization
################################################################################

def plot_embedding_clusters(
    emb: torch.Tensor,
    lookup: pd.DataFrame,
    output_dir: Path,
    max_points: int = 2000,
    method: str = "tsne",
):
    """Visualize player embeddings in 2D using t-SNE or UMAP."""
    from sklearn.manifold import TSNE

    # Use most-seen players (highest IDs are usually more recent / more data)
    n = min(max_points, emb.shape[0])
    # Sort by volume: count occurrences in parquet
    ids = list(range(n))
    emb_subset = emb[ids].numpy()

    # Get season info for coloring
    season_map = lookup.set_index("player_season_id")["season"].to_dict()
    seasons = [season_map.get(i, 0) for i in ids]

    print(f"  Running t-SNE on {n} embeddings...")
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200,
                init="pca", random_state=42, n_iter=1000)
    coords = tsne.fit_transform(emb_subset)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=seasons,
                         cmap="viridis", s=8, alpha=0.5)
    plt.colorbar(scatter, label="Season")
    ax.set_title(f"Player Embedding t-SNE ({n} players, colored by season)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Annotate some well-known players
    notable = lookup[lookup["player"].isin([
        "jamesle01", "curryst01", "duranke01", "davisan02",
        "hardeja01", "lillada01", "doncilu01", "anMDeto01",
    ])]
    name_map = make_id_to_name(lookup)
    for _, row in notable.iterrows():
        pid = row.player_season_id
        if pid < n:
            ax.annotate(name_map.get(pid, "")[:15],
                       (coords[pid, 0], coords[pid, 1]),
                       fontsize=6, alpha=0.8)

    plt.tight_layout()
    path = output_dir / "embedding_tsne.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


################################################################################
# 3. Temporal Trajectories
################################################################################

def plot_temporal_trajectories(
    emb: torch.Tensor,
    lookup: pd.DataFrame,
    output_dir: Path,
    players: list[str] | None = None,
):
    """Track how a player's embedding evolves across seasons.

    Uses cosine similarity between consecutive seasons to measure drift.
    """
    if players is None:
        # Pick players with most seasons
        season_counts = lookup.groupby("player")["season"].count()
        players = season_counts.nlargest(10).index.tolist()

    emb_norm = F.normalize(emb, dim=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for player_id in players:
        rows = lookup[lookup["player"] == player_id].sort_values("season")
        if len(rows) < 3:
            continue

        seasons = rows["season"].values
        pids = rows["player_season_id"].values

        # Cosine similarity between consecutive seasons
        sims = []
        for i in range(len(pids) - 1):
            if pids[i] < emb.shape[0] and pids[i + 1] < emb.shape[0]:
                sim = (emb_norm[pids[i]] @ emb_norm[pids[i + 1]]).item()
                sims.append(sim)
            else:
                sims.append(np.nan)

        label = player_id[:12]
        axes[0].plot(seasons[1:], sims, marker="o", markersize=3, label=label, alpha=0.7)

        # Embedding norm across seasons (proxy for "importance")
        norms = [emb[pid].norm().item() if pid < emb.shape[0] else np.nan for pid in pids]
        axes[1].plot(seasons, norms, marker="o", markersize=3, label=label, alpha=0.7)

    axes[0].set_title("Year-to-year embedding similarity (cosine)")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_ylim(0, 1)

    axes[1].set_title("Embedding norm across seasons")
    axes[1].set_ylabel("L2 norm")
    axes[1].set_xlabel("Season")
    axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = output_dir / "temporal_trajectories.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


################################################################################
# 4. Embedding Statistics
################################################################################

def compute_embedding_stats(emb: torch.Tensor, lookup: pd.DataFrame):
    """Compute and print embedding quality statistics."""
    emb_norm = F.normalize(emb, dim=1)

    norms = emb.norm(dim=1)
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {emb.shape}")
    print(f"  Norm: mean={norms.mean():.3f}, std={norms.std():.3f}, "
          f"min={norms.min():.3f}, max={norms.max():.3f}")

    # Average pairwise similarity (sample)
    n_sample = min(1000, emb.shape[0])
    sample_ids = np.random.choice(emb.shape[0], n_sample, replace=False)
    sample_emb = emb_norm[sample_ids]
    sim_matrix = sample_emb @ sample_emb.T
    # Zero out diagonal
    sim_matrix.fill_diagonal_(0)
    avg_sim = sim_matrix.sum() / (n_sample * (n_sample - 1))
    print(f"  Avg pairwise cosine sim (sample of {n_sample}): {avg_sim:.4f}")

    # Check if embeddings are collapsed (all similar)
    if avg_sim > 0.8:
        print("  WARNING: Embeddings may be collapsed (high avg similarity)")
    elif avg_sim < 0.01:
        print("  NOTE: Embeddings are nearly orthogonal (good diversity)")
    else:
        print(f"  Embedding diversity looks healthy")

    # Same-player cross-season similarity
    multi_season = lookup.groupby("player").filter(lambda x: len(x) >= 3)
    if len(multi_season) > 0:
        same_player_sims = []
        for player, group in multi_season.groupby("player"):
            pids = group["player_season_id"].values
            pids = pids[pids < emb.shape[0]]
            if len(pids) < 2:
                continue
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    sim = (emb_norm[pids[i]] @ emb_norm[pids[j]]).item()
                    same_player_sims.append(sim)

        if same_player_sims:
            print(f"  Same-player cross-season similarity: "
                  f"mean={np.mean(same_player_sims):.4f}, "
                  f"std={np.std(same_player_sims):.4f} "
                  f"(n={len(same_player_sims):,})")
            print(f"  (Should be higher than avg pairwise sim of {avg_sim:.4f})")


################################################################################
# Main
################################################################################

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup = load_player_lookup(args.lookup_csv)
    emb = load_embeddings(args.ckpt)
    id_to_name = make_id_to_name(lookup)

    print(f"Loaded embeddings: {emb.shape} from {args.ckpt}")

    # 1. Embedding statistics
    compute_embedding_stats(emb, lookup)

    # 2. Nearest neighbors for notable players
    print("\n" + "=" * 60)
    print("NEAREST NEIGHBORS")
    print("=" * 60)

    notable_players = [
        "jamesle01", "curryst01", "duranke01", "hardeja01",
        "anMDeto01", "doncilu01", "jokicni01", "lillada01",
    ]
    # Find the most recent season for each notable player
    query_ids = []
    for pid in notable_players:
        rows = lookup[lookup["player"] == pid]
        if len(rows) > 0:
            latest = rows.sort_values("season").iloc[-1]
            query_ids.append(int(latest.player_season_id))

    nn_results = find_nearest_neighbors(emb, query_ids, id_to_name, k=8)
    print_nearest_neighbors(nn_results)

    # 3. t-SNE visualization
    print("\n" + "=" * 60)
    print("EMBEDDING VISUALIZATION")
    print("=" * 60)
    plot_embedding_clusters(emb, lookup, output_dir)

    # 4. Temporal trajectories
    print("\n" + "=" * 60)
    print("TEMPORAL TRAJECTORIES")
    print("=" * 60)
    notable_long_career = [
        "jamesle01", "curryst01", "duranke01", "hardeja01",
        "paulch01", "nowitdi01", "wadedw01", "howardw01",
    ]
    plot_temporal_trajectories(emb, lookup, output_dir, notable_long_career)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Unicorn player embeddings")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--lookup-csv", default="player_season_lookup.csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    main(args)
