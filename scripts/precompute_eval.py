#!/usr/bin/env python3
"""
Comprehensive evaluation precomputation for Unicorn.

Takes a checkpoint and produces all evaluation data as separate files in an
output directory. A companion master_eval.ipynb notebook loads these files
and produces a visual report.

Usage:
    python scripts/precompute_eval.py --ckpt checkpoint.pt --output-dir eval_output/
    python scripts/precompute_eval.py --ckpt checkpoint.pt --output-dir eval_output/ --skip-slow
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from nba_dataset import (
    PossessionDataset,
    get_default_dataloaders,
    get_num_players,
    NUM_OUTCOMES,
    OUTCOME_NAMES,
)

################################################################################
# Model loading (adapted from evaluate.py)
################################################################################

def load_model(ckpt_path: str, device: torch.device):
    """Load model from checkpoint, auto-detecting architecture."""
    from prior_year_init import build_ps_to_base_tensor

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    num_ps = ckpt["num_player_seasons"]
    num_base = ckpt["num_base_players"]
    d_model = ckpt.get("d_model", 384)
    n_heads = ckpt.get("n_heads", 8)
    dropout = ckpt.get("dropout", 0.1)
    delta_dim = ckpt.get("delta_dim", 0)

    ps_to_base, _ = build_ps_to_base_tensor(num_ps)

    if ckpt.get("architecture") == "v6_relation":
        from train_v6 import RelationNetwork
        n_layers = ckpt.get("n_layers", 2)
        d_pair = ckpt.get("d_pair", 64)
        d_pair_hidden = ckpt.get("d_pair_hidden", 256)
        model = RelationNetwork(
            num_ps, num_base, ps_to_base, d_model, n_layers, n_heads, dropout,
            delta_dim=delta_dim, d_pair=d_pair, d_pair_hidden=d_pair_hidden,
        )
    else:
        from train_transformer import LineupTransformer
        n_layers = ckpt.get("n_layers", 8)
        attn_temperature = ckpt.get("attn_temperature", 1.0)
        pool_type = ckpt.get("pool_type", "static")
        pool_heads = ckpt.get("pool_heads", 4)
        pool_multi_layer = ckpt.get("pool_multi_layer", False)
        film_state = ckpt.get("film_state", False)
        model = LineupTransformer(
            num_ps, num_base, ps_to_base, d_model, n_layers, n_heads, dropout,
            delta_dim=delta_dim, attn_temperature=attn_temperature,
            pool_type=pool_type, pool_heads=pool_heads,
            pool_multi_layer=pool_multi_layer, film_state=film_state,
        )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, ckpt


def extract_embeddings(model) -> torch.Tensor:
    """Extract all composed embeddings [num_ps, d_model] from model."""
    with torch.no_grad():
        return model._all_composed_embeddings().cpu()


################################################################################
# 1. Training metrics
################################################################################

def parse_training_log(log_path: str) -> list[dict]:
    """Parse .log.jsonl into list of epoch dicts."""
    entries = []
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"  WARNING: log file {log_path} not found, skipping")
        return entries
    for line in log_file.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


################################################################################
# 2. Embedding statistics
################################################################################

def compute_embedding_stats(emb: torch.Tensor, lookup: pd.DataFrame) -> dict:
    """Compute embedding quality statistics."""
    emb_norm = F.normalize(emb, dim=1)
    norms = emb.norm(dim=1)

    stats = {
        "shape": list(emb.shape),
        "norm_mean": round(norms.mean().item(), 4),
        "norm_std": round(norms.std().item(), 4),
        "norm_min": round(norms.min().item(), 4),
        "norm_max": round(norms.max().item(), 4),
    }

    # Average pairwise similarity (sample)
    n_sample = min(1000, emb.shape[0])
    sample_ids = np.random.choice(emb.shape[0], n_sample, replace=False)
    sample_emb = emb_norm[sample_ids]
    sim_matrix = sample_emb @ sample_emb.T
    sim_matrix.fill_diagonal_(0)
    avg_sim = sim_matrix.sum() / (n_sample * (n_sample - 1))
    stats["avg_pairwise_cosine"] = round(avg_sim.item(), 4)

    # Same-player cross-season similarity
    multi_season = lookup.groupby("player").filter(lambda x: len(x) >= 3)
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
        stats["same_player_cosine_mean"] = round(np.mean(same_player_sims), 4)
        stats["same_player_cosine_std"] = round(np.std(same_player_sims), 4)
        stats["same_player_n"] = len(same_player_sims)

    return stats


################################################################################
# 3. Nearest neighbors
################################################################################

def compute_nearest_neighbors(
    emb: torch.Tensor, lookup: pd.DataFrame,
    notable_players: list[str], k: int = 10,
) -> dict:
    """Find nearest neighbors for notable players."""
    emb_norm = F.normalize(emb, dim=1)
    id_to_name = {
        row.player_season_id: f"{row.player} ({row.season})"
        for _, row in lookup.iterrows()
    }

    results = {}
    for pid in notable_players:
        rows = lookup[lookup["player"] == pid]
        if len(rows) == 0:
            continue
        latest = rows.sort_values("season").iloc[-1]
        qid = int(latest.player_season_id)
        if qid >= emb.shape[0]:
            continue

        sims = emb_norm @ emb_norm[qid]
        topk = sims.topk(k + 1)
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


################################################################################
# 4. Distributional prediction metrics
################################################################################

def compute_distributional_metrics(model, test_dl, device) -> dict:
    """Compute distributional prediction quality metrics."""
    all_logits, all_target_dists = [], []

    with torch.no_grad():
        for pl, st, tm, target_dist in tqdm(test_dl, desc="Predictions", leave=False):
            pl, st, tm = pl.to(device), st.to(device), tm.to(device)
            result = model.forward_finetune(pl, st, tm)
            logits = result[0]
            all_logits.append(logits.cpu())
            all_target_dists.append(target_dist)

    logits = torch.cat(all_logits)
    target_dists = torch.cat(all_target_dists)
    pred_dists = torch.softmax(logits, dim=1)

    # Mean predicted vs mean target
    mean_pred = pred_dists.mean(dim=0).numpy()
    mean_target = target_dists.mean(dim=0).numpy()
    ratios = mean_pred / (mean_target + 1e-8)

    # KL divergence: model-specific vs global baseline
    global_baseline = mean_target  # [9]
    kl_model = -(target_dists * (pred_dists + 1e-8).log()).sum(dim=1).mean().item()
    kl_baseline = -(target_dists * torch.tensor(global_baseline + 1e-8).log()).sum(dim=1).mean().item()
    kl_improvement = 1 - kl_model / kl_baseline if kl_baseline > 0 else 0

    # Hard-label accuracy
    hard_labels = target_dists.argmax(dim=1)
    hard_preds = logits.argmax(dim=1)
    hard_acc = (hard_preds == hard_labels).float().mean().item()

    return {
        "mean_pred_dist": [round(x, 4) for x in mean_pred.tolist()],
        "mean_target_dist": [round(x, 4) for x in mean_target.tolist()],
        "pred_target_ratios": [round(x, 4) for x in ratios.tolist()],
        "kl_model": round(kl_model, 5),
        "kl_baseline": round(kl_baseline, 5),
        "kl_improvement_pct": round(kl_improvement * 100, 2),
        "hard_label_accuracy": round(hard_acc, 4),
        "outcome_names": OUTCOME_NAMES,
        "n_samples": len(logits),
    }, logits, target_dists


################################################################################
# 5. Attention statistics
################################################################################

def compute_attention_stats(model, test_dl, device, max_batches: int = 50) -> dict:
    """Sample attention weights and compute statistics.

    For v6 (RelationNetwork): no attention pooling exists, so compute pairwise
    relation output statistics instead. Returns dummy uniform attention weights
    for notebook compatibility.
    """
    is_v6 = hasattr(model, "g_off")  # RelationNetwork has pairwise MLPs
    all_off_attn, all_def_attn = [], []

    if is_v6:
        # v6: compute pairwise relation stats instead of attention
        all_off_norms, all_def_norms, all_match_norms = [], [], []
        all_token_divs = []

        with torch.no_grad():
            for i, (pl, st, tm, _) in enumerate(tqdm(test_dl, desc="Pairwise stats", leave=False)):
                if i >= max_batches:
                    break
                pl, st, tm = pl.to(device), st.to(device), tm.to(device)
                B = pl.size(0)

                # Run encoder
                tok = model._embed_players(pl, tm)
                h = model.encoder(tok)

                # Pairwise outputs
                off_p, def_p, match_p = model._compute_pairwise(h)
                all_off_norms.append(off_p.norm(dim=1).cpu())
                all_def_norms.append(def_p.norm(dim=1).cpu())
                all_match_norms.append(match_p.norm(dim=1).cpu())

                # Token diversity (mean pairwise cosine of encoder outputs)
                h_norm = F.normalize(h, dim=2)
                cos = torch.bmm(h_norm, h_norm.transpose(1, 2))
                mask = ~torch.eye(10, device=device, dtype=torch.bool).unsqueeze(0)
                div = cos[mask.expand(B, -1, -1)].reshape(B, -1).mean(dim=1)
                all_token_divs.append(div.cpu())

                # Dummy uniform attention for compatibility
                all_off_attn.append(torch.full((B, 5), 0.2))
                all_def_attn.append(torch.full((B, 5), 0.2))

        off_norms = torch.cat(all_off_norms)
        def_norms = torch.cat(all_def_norms)
        match_norms = torch.cat(all_match_norms)
        token_div = torch.cat(all_token_divs)

        off_attn = torch.cat(all_off_attn)
        def_attn = torch.cat(all_def_attn)
        uniform_entropy = np.log(5)

        return {
            "architecture": "v6_relation",
            "off_entropy_mean": round(uniform_entropy, 4),  # uniform (no attn pool)
            "off_entropy_std": 0.0,
            "def_entropy_mean": round(uniform_entropy, 4),
            "def_entropy_std": 0.0,
            "uniform_entropy": round(uniform_entropy, 4),
            "off_mean_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "def_mean_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "off_max_weight_mean": 0.2,
            "def_max_weight_mean": 0.2,
            "n_samples": len(off_attn),
            # v6-specific pairwise stats
            "off_pair_norm_mean": round(off_norms.mean().item(), 4),
            "off_pair_norm_std": round(off_norms.std().item(), 4),
            "def_pair_norm_mean": round(def_norms.mean().item(), 4),
            "def_pair_norm_std": round(def_norms.std().item(), 4),
            "match_pair_norm_mean": round(match_norms.mean().item(), 4),
            "match_pair_norm_std": round(match_norms.std().item(), 4),
            "token_diversity_mean": round(token_div.mean().item(), 4),
            "token_diversity_std": round(token_div.std().item(), 4),
        }, off_attn, def_attn

    # v3.2+: standard attention pooling stats
    with torch.no_grad():
        for i, (pl, st, tm, _) in enumerate(tqdm(test_dl, desc="Attention", leave=False)):
            if i >= max_batches:
                break
            pl, st, tm = pl.to(device), st.to(device), tm.to(device)
            result = model.forward_finetune(pl, st, tm)
            off_attn, def_attn = result[1], result[2]
            all_off_attn.append(off_attn.cpu())
            all_def_attn.append(def_attn.cpu())

    off_attn = torch.cat(all_off_attn)
    def_attn = torch.cat(all_def_attn)

    off_entropy = -(off_attn * (off_attn + 1e-8).log()).sum(dim=1)
    def_entropy = -(def_attn * (def_attn + 1e-8).log()).sum(dim=1)
    uniform_entropy = np.log(5)  # ln(5) = 1.6094

    return {
        "off_entropy_mean": round(off_entropy.mean().item(), 4),
        "off_entropy_std": round(off_entropy.std().item(), 4),
        "def_entropy_mean": round(def_entropy.mean().item(), 4),
        "def_entropy_std": round(def_entropy.std().item(), 4),
        "uniform_entropy": round(uniform_entropy, 4),
        "off_mean_weights": [round(x, 4) for x in off_attn.mean(dim=0).tolist()],
        "def_mean_weights": [round(x, 4) for x in def_attn.mean(dim=0).tolist()],
        "off_max_weight_mean": round(off_attn.max(dim=1).values.mean().item(), 4),
        "def_max_weight_mean": round(def_attn.max(dim=1).values.mean().item(), 4),
        "n_samples": len(off_attn),
    }, off_attn, def_attn


################################################################################
# 6. Substitution sensitivity (renamed from Player Impact)
################################################################################

def compute_substitution_sensitivity(
    model, test_dl, device, lookup: pd.DataFrame,
    emb: torch.Tensor, min_possessions: int = 100,
) -> dict:
    """Compute how much swapping each player with a replacement changes predictions."""

    # Define replacement baseline: global centroid of base embeddings
    base_emb = model.base_player_emb.weight.detach().cpu()
    centroid = base_emb.mean(dim=0)
    distances = (base_emb - centroid).norm(dim=1)
    replacement_id_base = distances.argmin().item()

    # Find a player-season ID that maps to the replacement base player
    ps_to_base = model.ps_to_base.cpu()
    replacement_ps_ids = (ps_to_base == replacement_id_base).nonzero(as_tuple=True)[0]
    if len(replacement_ps_ids) == 0:
        print("  WARNING: Could not find replacement player-season ID")
        return {}
    replacement_ps_id = replacement_ps_ids[0].item()

    # Value weights for favorability score
    value_weights = torch.tensor([3.0, 0.0, 2.0, 2.0, -0.2, -0.3, 1.5, -1.0, -0.5])

    # Collect possessions per player (offense only, positions 0-4)
    player_possessions = {}  # ps_id -> list of (pl, st, tm, slot_idx)
    for pl, st, tm, _ in tqdm(test_dl, desc="Collecting possessions", leave=False):
        for b in range(pl.size(0)):
            for slot in range(5):  # offense only
                ps_id = pl[b, slot].item()
                if ps_id not in player_possessions:
                    player_possessions[ps_id] = []
                player_possessions[ps_id].append((pl[b], st[b], tm[b], slot))

    # Filter to players with enough possessions
    eligible = {pid: poss for pid, poss in player_possessions.items()
                if len(poss) >= min_possessions}
    print(f"  {len(eligible)} players with {min_possessions}+ possessions")

    id_to_name = {
        row.player_season_id: f"{row.player} ({row.season})"
        for _, row in lookup.iterrows()
    }

    results = {}
    with torch.no_grad():
        for ps_id, possessions in tqdm(eligible.items(), desc="Impact", leave=False):
            # Sample up to 500 possessions
            sample = possessions[:500]
            pl_batch = torch.stack([p[0] for p in sample]).to(device)
            st_batch = torch.stack([p[1] for p in sample]).to(device)
            tm_batch = torch.zeros(len(sample), 10, dtype=torch.long, device=device)
            tm_batch[:, 5:] = 1
            slots = [p[3] for p in sample]

            # Original predictions
            orig_logits = model.forward_finetune(pl_batch, st_batch, tm_batch)[0]
            orig_probs = torch.softmax(orig_logits, dim=1).cpu()
            orig_fav = (orig_probs * value_weights).sum(dim=1)

            # Replace player with replacement
            pl_replaced = pl_batch.clone()
            for i, slot in enumerate(slots):
                pl_replaced[i, slot] = replacement_ps_id
            repl_logits = model.forward_finetune(pl_replaced, st_batch, tm_batch)[0]
            repl_probs = torch.softmax(repl_logits, dim=1).cpu()
            repl_fav = (repl_probs * value_weights).sum(dim=1)

            delta_fav = (orig_fav - repl_fav).mean().item()
            delta_dist = (orig_probs - repl_probs).mean(dim=0).tolist()

            results[ps_id] = {
                "name": id_to_name.get(ps_id, f"ID_{ps_id}"),
                "n_possessions": len(sample),
                "impact_favorability": round(delta_fav, 5),
                "mean_dist_shift": [round(x, 5) for x in delta_dist],
            }

    # Sort by impact
    sorted_results = dict(sorted(results.items(),
                                 key=lambda x: x[1]["impact_favorability"],
                                 reverse=True))
    return {
        "replacement_player": id_to_name.get(replacement_ps_id, f"ID_{replacement_ps_id}"),
        "replacement_ps_id": replacement_ps_id,
        "value_weights": value_weights.tolist(),
        "players": sorted_results,
    }


################################################################################
# 7. Game outcome prediction
################################################################################

def compute_game_outcomes(
    model, ckpt_path: str, parquet_path: str, device: torch.device,
    emb: torch.Tensor,
) -> dict:
    """Compute game outcome prediction with multiple methods."""
    from sklearn.linear_model import LogisticRegression

    # Build game table
    cols = ["game_id", "season", "home_score", "away_score",
            *[f"p{i}" for i in range(10)]]
    df = pd.read_parquet(parquet_path, columns=cols)
    starters = df.groupby("game_id").first().reset_index()
    finals = (df.groupby("game_id")[["home_score", "away_score"]]
              .last().reset_index()
              .rename(columns={"home_score": "final_home", "away_score": "final_away"}))
    games = starters.merge(finals, on="game_id")
    games = games[(games["final_home"] > 0) | (games["final_away"] > 0)].copy()
    games["home_win"] = (games["final_home"] > games["final_away"]).astype(int)

    train_g = games[games["season"] < 2019]
    test_g = games[games["season"] >= 2021]
    y_train = train_g["home_win"].values
    y_test = test_g["home_win"].values

    results = {
        "n_train_games": len(train_g),
        "n_test_games": len(test_g),
        "home_win_rate_test": round(y_test.mean(), 4),
    }

    # Baseline 1: Home always wins
    results["home_always_acc"] = round(y_test.mean(), 4)

    # Baseline 2: Bag-of-player-IDs
    num_players = int(games[[f"p{i}" for i in range(10)]].max().max()) + 1
    X_train_ids = _bag_of_ids(train_g, num_players)
    X_test_ids = _bag_of_ids(test_g, num_players)
    lr_ids = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
    lr_ids.fit(X_train_ids, y_train)
    results["bag_of_ids_acc"] = round(lr_ids.score(X_test_ids, y_test), 4)

    # Method 3: Static embeddings (mean-pooled)
    emb_np = emb.numpy()
    X_train_emb = _static_embedding_features(train_g, emb_np)
    X_test_emb = _static_embedding_features(test_g, emb_np)
    lr_emb = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
    lr_emb.fit(X_train_emb, y_train)
    results["static_embedding_acc"] = round(lr_emb.score(X_test_emb, y_test), 4)

    # Method 4: Contextual embeddings (run through transformer)
    X_train_ctx = _contextual_features(model, train_g, device)
    X_test_ctx = _contextual_features(model, test_g, device)
    lr_ctx = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
    lr_ctx.fit(X_train_ctx, y_train)
    results["contextual_acc"] = round(lr_ctx.score(X_test_ctx, y_test), 4)

    return results


def _bag_of_ids(games: pd.DataFrame, num_players: int) -> np.ndarray:
    features = np.zeros((len(games), num_players), dtype=np.float32)
    for i, (_, row) in enumerate(games.iterrows()):
        for j in range(5):
            pid = int(row[f"p{j}"])
            if pid < num_players:
                features[i, pid] = 1.0
        for j in range(5, 10):
            pid = int(row[f"p{j}"])
            if pid < num_players:
                features[i, pid] = -1.0
    return features


def _static_embedding_features(games: pd.DataFrame, emb_np: np.ndarray) -> np.ndarray:
    d = emb_np.shape[1]
    features = np.zeros((len(games), 3 * d), dtype=np.float32)
    for i, (_, row) in enumerate(games.iterrows()):
        home_ids = [min(int(row[f"p{j}"]), emb_np.shape[0]-1) for j in range(5)]
        away_ids = [min(int(row[f"p{j}"]), emb_np.shape[0]-1) for j in range(5, 10)]
        home_mean = emb_np[home_ids].mean(axis=0)
        away_mean = emb_np[away_ids].mean(axis=0)
        features[i, :d] = home_mean
        features[i, d:2*d] = away_mean
        features[i, 2*d:] = home_mean - away_mean
    return features


def _contextual_features(model, games: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Run lineups through encoder to get contextual features.

    Supports both LineupTransformer (attention pooling) and RelationNetwork (mean pooling).
    """
    d = model.d_model
    is_v6 = hasattr(model, "g_off")  # RelationNetwork has pairwise MLPs
    features = np.zeros((len(games), 3 * d), dtype=np.float32)

    # Process in batches
    batch_size = 256
    game_rows = list(games.iterrows())

    with torch.no_grad():
        for start in range(0, len(game_rows), batch_size):
            batch = game_rows[start:start+batch_size]

            players = torch.zeros(len(batch), 10, dtype=torch.long)
            state = torch.zeros(len(batch), 3, dtype=torch.float32)
            team_ids = torch.zeros(len(batch), 10, dtype=torch.long)
            team_ids[:, 5:] = 1

            for b, (_, row) in enumerate(batch):
                for j in range(10):
                    players[b, j] = int(row[f"p{j}"])

            players = players.to(device)
            state = state.to(device)
            team_ids = team_ids.to(device)

            # Get encoder output
            tok = model._embed_players(players, team_ids)
            h = model.encoder(tok)

            if is_v6:
                # v6: mean-pool offense and defense (no attention pooling)
                off_pooled = h[:, :5, :].mean(dim=1)
                def_pooled = h[:, 5:, :].mean(dim=1)
            else:
                # v3.2+: attention pooling
                off_pooled, _ = model.attn_pool_off(h[:, :5, :])
                def_pooled, _ = model.attn_pool_def(h[:, 5:, :])

            off_np = off_pooled.cpu().numpy()
            def_np = def_pooled.cpu().numpy()

            for b in range(len(batch)):
                features[start+b, :d] = off_np[b]
                features[start+b, d:2*d] = def_np[b]
                features[start+b, 2*d:] = off_np[b] - def_np[b]

    return features


################################################################################
# 8. Lineup completion retrieval
################################################################################

def compute_lineup_completion(
    model, dataloader, device: torch.device,
    split_name: str, max_batches: int = 200,
) -> dict:
    """Hold out the 5th offensive player, score all candidates, report retrieval metrics."""
    value_weights = torch.tensor([3.0, 0.0, 2.0, 2.0, -0.2, -0.3, 1.5, -1.0, -0.5], device=device)

    # Collect unique offensive player-season IDs as candidates
    candidate_set = set()
    for pl, _, _, _ in dataloader:
        for b in range(pl.size(0)):
            for slot in range(5):
                candidate_set.add(pl[b, slot].item())
    candidates = sorted(candidate_set)
    print(f"  {split_name}: {len(candidates)} candidate players")

    ranks = []
    with torch.no_grad():
        for batch_i, (pl, st, tm, _) in enumerate(tqdm(dataloader, desc=f"Lineup completion ({split_name})", leave=False)):
            if batch_i >= max_batches:
                break
            pl, st, tm = pl.to(device), st.to(device), tm.to(device)

            # Hold out position 4 (5th offensive player)
            held_out_ids = pl[:, 4].cpu().tolist()

            # Score a random sample of candidates (full scoring is too expensive)
            n_candidates = min(200, len(candidates))
            sample_candidates = np.random.choice(candidates, n_candidates, replace=False)

            for b in range(pl.size(0)):
                true_id = held_out_ids[b]
                # Ensure true player is in candidate set
                test_candidates = list(sample_candidates)
                if true_id not in test_candidates:
                    test_candidates[0] = true_id

                scores = []
                pl_test = pl[b:b+1].expand(len(test_candidates), -1).clone()
                st_test = st[b:b+1].expand(len(test_candidates), -1)
                tm_test = tm[b:b+1].expand(len(test_candidates), -1) if tm.dim() > 1 else torch.zeros(len(test_candidates), 10, dtype=torch.long, device=device)
                tm_test = torch.zeros(len(test_candidates), 10, dtype=torch.long, device=device)
                tm_test[:, 5:] = 1

                for c_idx, cand in enumerate(test_candidates):
                    pl_test[c_idx, 4] = cand

                # Batch forward pass
                logits = model.forward_finetune(pl_test, st_test, tm_test)[0]
                probs = torch.softmax(logits, dim=1)
                favs = (probs * value_weights).sum(dim=1)

                # Rank of true player
                true_idx = test_candidates.index(true_id)
                true_fav = favs[true_idx].item()
                rank = (favs > true_fav).sum().item() + 1
                ranks.append(rank)

    if not ranks:
        return {}

    ranks_arr = np.array(ranks)
    n_cands = n_candidates  # approximate
    return {
        "split": split_name,
        "n_queries": len(ranks),
        "n_candidates": n_cands,
        "recall_at_1": round((ranks_arr == 1).mean(), 4),
        "recall_at_5": round((ranks_arr <= 5).mean(), 4),
        "recall_at_10": round((ranks_arr <= 10).mean(), 4),
        "recall_at_50": round((ranks_arr <= 50).mean(), 4),
        "mrr": round((1.0 / ranks_arr).mean(), 4),
        "mean_rank": round(ranks_arr.mean(), 2),
        "median_rank": round(float(np.median(ranks_arr)), 2),
        "random_recall_at_1": round(1.0 / n_cands, 4),
    }


################################################################################
# 9. Archetype clustering
################################################################################

def compute_archetype_clusters(emb: torch.Tensor, lookup: pd.DataFrame, k: int = 8) -> dict:
    """K-means clustering on base player embeddings."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Use base-player level embeddings (one per player, not per season)
    base_emb = emb.numpy()  # Will be the full composed embeddings
    # Get unique base players' most recent season
    latest = lookup.sort_values("season").drop_duplicates("player", keep="last")
    base_ids = latest["player_season_id"].values
    base_ids = base_ids[base_ids < emb.shape[0]]
    base_emb_subset = base_emb[base_ids]

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(base_emb_subset)
    sil = silhouette_score(base_emb_subset, labels)

    # Map player names to clusters
    player_names = latest.iloc[:len(base_ids)]["player"].values
    cluster_members = {}
    for c in range(k):
        members = player_names[labels == c].tolist()
        cluster_members[str(c)] = members[:20]  # top 20 per cluster

    return {
        "k": k,
        "silhouette_score": round(float(sil), 4),
        "cluster_sizes": [int((labels == c).sum()) for c in range(k)],
        "cluster_members": cluster_members,
    }


################################################################################
# 10. Aging curves
################################################################################

def compute_aging_curves(emb: torch.Tensor, lookup: pd.DataFrame) -> dict:
    """Compute delta norms by player age."""
    from prior_year_init import build_ps_to_base_tensor

    ps_to_base, _ = build_ps_to_base_tensor(emb.shape[0])

    # Need base embeddings separately — approximate by looking at checkpoint structure
    # For now, compute delta as: composed - mean_of_same_base_player
    base_means = {}
    for _, row in lookup.iterrows():
        pid = int(row.player_season_id)
        player = row.player
        if pid >= emb.shape[0]:
            continue
        if player not in base_means:
            base_means[player] = []
        base_means[player].append(emb[pid].numpy())

    for player in base_means:
        base_means[player] = np.mean(base_means[player], axis=0)

    # Compute delta norms by age
    # Need player birth years — approximate by using first season as "age 20"
    first_seasons = lookup.groupby("player")["season"].min()

    age_norms = {}
    for _, row in lookup.iterrows():
        pid = int(row.player_season_id)
        player = row.player
        if pid >= emb.shape[0] or player not in base_means:
            continue
        first_season = first_seasons.get(player, row.season)
        age = 20 + (row.season - first_season)  # approximate
        delta = emb[pid].numpy() - base_means[player]
        delta_norm = float(np.linalg.norm(delta))

        if age not in age_norms:
            age_norms[age] = []
        age_norms[age].append(delta_norm)

    return {
        str(int(age)): {
            "mean_delta_norm": round(float(np.mean(norms)), 4),
            "std_delta_norm": round(float(np.std(norms)), 4),
            "n_players": len(norms),
        }
        for age, norms in sorted(age_norms.items())
        if len(norms) >= 10  # filter noisy ages
    }


################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Precompute all evaluation data for Unicorn")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--parquet", default="possessions.parquet")
    parser.add_argument("--log-jsonl", default=None, help="Training log (.log.jsonl)")
    parser.add_argument("--lookup-csv", default="player_season_lookup.csv")
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument("--bs", type=int, default=2048)
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow computations (game outcome, lineup completion)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Output: {output_dir}/")
    t0 = time.time()

    # Load model and data
    print("\n[1/11] Loading model and data...")
    model, ckpt = load_model(args.ckpt, device)
    lookup = pd.read_csv(args.lookup_csv).dropna(subset=["player"])
    train_dl, val_dl, test_dl = get_default_dataloaders(args.parquet, batch_size=args.bs)
    print(f"  Model loaded from {args.ckpt}")

    # Parse training log
    print("\n[2/11] Parsing training log...")
    log_path = args.log_jsonl or str(Path(args.ckpt).with_suffix(".log.jsonl"))
    training_metrics = parse_training_log(log_path)
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"  {len(training_metrics)} epochs logged")

    # Extract embeddings
    print("\n[3/11] Extracting embeddings...")
    emb = extract_embeddings(model)
    torch.save(emb, output_dir / "embeddings.pt")
    print(f"  Shape: {list(emb.shape)}")

    # Embedding statistics
    print("\n[4/11] Computing embedding statistics...")
    emb_stats = compute_embedding_stats(emb, lookup)
    with open(output_dir / "embedding_stats.json", "w") as f:
        json.dump(emb_stats, f, indent=2)
    print(f"  Avg pairwise cosine: {emb_stats['avg_pairwise_cosine']}")

    # Nearest neighbors
    print("\n[5/11] Computing nearest neighbors...")
    notable = ["jamesle01", "curryst01", "duranke01", "hardeja01",
               "anMDeto01", "doncilu01", "jokicni01", "lillada01",
               "tatMDja01", "youngtr01"]
    nn_results = compute_nearest_neighbors(emb, lookup, notable)
    with open(output_dir / "nearest_neighbors.json", "w") as f:
        json.dump(nn_results, f, indent=2)
    print(f"  {len(nn_results)} players analyzed")

    # Distributional metrics
    print("\n[6/11] Computing distributional prediction metrics...")
    dist_metrics, pred_logits, target_dists = compute_distributional_metrics(model, test_dl, device)
    with open(output_dir / "distributional_metrics.json", "w") as f:
        json.dump(dist_metrics, f, indent=2)
    torch.save({"logits": pred_logits, "target_dists": target_dists},
               output_dir / "predictions.pt")
    print(f"  KL improvement: {dist_metrics['kl_improvement_pct']}%")

    # Attention statistics
    print("\n[7/11] Computing attention statistics...")
    attn_stats, off_attn, def_attn = compute_attention_stats(model, test_dl, device)
    with open(output_dir / "attention_stats.json", "w") as f:
        json.dump(attn_stats, f, indent=2)
    torch.save({"off_attn": off_attn, "def_attn": def_attn},
               output_dir / "attention_weights.pt")
    if attn_stats.get("architecture") == "v6_relation":
        print(f"  v6 pairwise: off={attn_stats['off_pair_norm_mean']:.4f}, "
              f"def={attn_stats['def_pair_norm_mean']:.4f}, "
              f"match={attn_stats['match_pair_norm_mean']:.4f}, "
              f"token_div={attn_stats['token_diversity_mean']:.4f}")
    else:
        print(f"  Off entropy: {attn_stats['off_entropy_mean']} (uniform={attn_stats['uniform_entropy']})")

    # Substitution sensitivity
    print("\n[8/11] Computing substitution sensitivity...")
    sub_sens = compute_substitution_sensitivity(model, test_dl, device, lookup, emb)
    with open(output_dir / "substitution_sensitivity.json", "w") as f:
        json.dump(sub_sens, f, indent=2, default=str)
    if sub_sens.get("players"):
        top = list(sub_sens["players"].values())[:3]
        for p in top:
            print(f"  {p['name']}: {p['impact_favorability']:+.4f}")

    # Archetype clustering
    print("\n[9/11] Computing archetype clusters...")
    clusters = compute_archetype_clusters(emb, lookup)
    with open(output_dir / "archetype_clusters.json", "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"  Silhouette: {clusters['silhouette_score']}, sizes: {clusters['cluster_sizes']}")

    # Aging curves
    print("\n[10/11] Computing aging curves...")
    aging = compute_aging_curves(emb, lookup)
    with open(output_dir / "aging_curves.json", "w") as f:
        json.dump(aging, f, indent=2)
    print(f"  {len(aging)} age groups")

    # Slow computations
    game_results = {}
    lineup_results = {}

    if not args.skip_slow:
        print("\n[11a/11] Computing game outcome prediction...")
        game_results = compute_game_outcomes(model, args.ckpt, args.parquet, device, emb)
        with open(output_dir / "game_outcomes.json", "w") as f:
            json.dump(game_results, f, indent=2)
        print(f"  Home-always: {game_results.get('home_always_acc', 'N/A')}")
        print(f"  Bag-of-IDs:  {game_results.get('bag_of_ids_acc', 'N/A')}")
        print(f"  Static emb:  {game_results.get('static_embedding_acc', 'N/A')}")
        print(f"  Contextual:  {game_results.get('contextual_acc', 'N/A')}")

        print("\n[11b/11] Computing lineup completion retrieval...")
        test_completion = compute_lineup_completion(model, test_dl, device, "test")
        train_sample_dl = DataLoader(
            train_dl.dataset, batch_size=args.bs, shuffle=True,
            pin_memory=True, num_workers=2,
        )
        train_completion = compute_lineup_completion(model, train_sample_dl, device, "train", max_batches=100)
        lineup_results = {"test": test_completion, "train": train_completion}
        with open(output_dir / "lineup_completion.json", "w") as f:
            json.dump(lineup_results, f, indent=2)
        if test_completion:
            print(f"  Test MRR: {test_completion['mrr']}, Recall@10: {test_completion['recall_at_10']}")
        if train_completion:
            print(f"  Train MRR: {train_completion['mrr']}, Recall@10: {train_completion['recall_at_10']}")
    else:
        print("\n[11/11] Skipping slow computations (game outcome, lineup completion)")

    # Summary
    elapsed = time.time() - t0
    summary = {
        "checkpoint": args.ckpt,
        "architecture": ckpt.get("architecture", "unknown"),
        "attn_temperature": ckpt.get("attn_temperature", 1.0),
        "attn_entropy_weight": ckpt.get("attn_entropy_weight", 0.0),
        "num_epochs_trained": len(training_metrics),
        "embedding_shape": list(emb.shape),
        "avg_pairwise_cosine": emb_stats["avg_pairwise_cosine"],
        "same_player_cosine": emb_stats.get("same_player_cosine_mean"),
        "kl_improvement_pct": dist_metrics["kl_improvement_pct"],
        "off_attn_entropy": attn_stats["off_entropy_mean"],
        "def_attn_entropy": attn_stats["def_entropy_mean"],
        "max_attn_weight": attn_stats["off_max_weight_mean"],
        "game_outcome": game_results,
        "lineup_completion": lineup_results,
        "n_substitution_players": len(sub_sens.get("players", {})),
        "silhouette_score": clusters["silhouette_score"],
        "computation_seconds": round(elapsed, 1),
    }

    # Add final training epoch metrics if available
    if training_metrics:
        last = training_metrics[-1]
        summary["final_val_loss"] = last.get("val_loss")
        summary["final_val_outcome_acc"] = last.get("val_outcome_acc")
        summary["final_temporal_top100"] = last.get("temporal_top100")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}/")
    print(f"  Summary: {output_dir}/summary.json")
    print(f"  Next: open notebooks/master_eval.ipynb")


if __name__ == "__main__":
    main()
