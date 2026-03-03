# game_outcome.py
"""
Downstream task: predict game winners from starting-lineup embeddings.

Uses frozen player embeddings from Phase A (pretrained transformer) to predict
whether the home team wins. This tests whether learned embeddings capture
team-level basketball ability beyond simple player identity.

Usage:
```bash
# Run with pretrained embeddings
python game_outcome.py --ckpt pretrain_checkpoint.pt

# Run with fine-tuned embeddings
python game_outcome.py --ckpt finetune_checkpoint.pt
```
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

################################################################################
# Data: extract per-game features from possessions
################################################################################

def build_game_table(parquet_path: str | Path) -> pd.DataFrame:
    """Build a game-level table with starting lineups and outcomes.

    For each game, extracts:
    - Starting 5 for home/away (first possession's player IDs)
    - Final score and home-win label
    - Season (for train/test split)
    """
    cols = [
        "game_id", "season", "home_score", "away_score",
        *[f"p{i}" for i in range(10)],
    ]
    df = pd.read_parquet(parquet_path, columns=cols)

    # First possession per game = starting lineup
    starters = df.groupby("game_id").first().reset_index()

    # Last row per game for final score
    finals = (
        df.groupby("game_id")[["home_score", "away_score"]]
        .last()
        .reset_index()
        .rename(columns={"home_score": "final_home", "away_score": "final_away"})
    )

    games = starters.merge(finals, on="game_id")

    # Filter out games with 0-0 scores (incomplete data)
    games = games[(games["final_home"] > 0) | (games["final_away"] > 0)].copy()

    # Home win label
    games["home_win"] = (games["final_home"] > games["final_away"]).astype(int)

    return games


################################################################################
# Embedding extraction
################################################################################

def load_embeddings(ckpt_path: str | Path) -> torch.Tensor:
    """Load frozen player embeddings from a checkpoint.

    Supports v1 (single player_emb), v2 (base + delta composed), and CBOW.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # v2: composed embeddings (base + delta)
    if "base_player_emb.weight" in state_dict and "delta_emb.weight" in state_dict:
        base_emb = state_dict["base_player_emb.weight"]  # [num_base, d]
        delta_emb = state_dict["delta_emb.weight"]        # [num_ps, d]
        ps_to_base = state_dict["ps_to_base"]              # [num_ps]
        emb = base_emb[ps_to_base] + delta_emb            # [num_ps, d]
        return emb
    # v1: single player embedding
    elif "player_emb.weight" in state_dict:
        return state_dict["player_emb.weight"]
    # CBOW
    elif "emb.weight" in state_dict:
        return state_dict["emb.weight"]
    else:
        raise KeyError("No player embedding found in checkpoint")


def games_to_embedding_features(
    games: pd.DataFrame,
    emb: torch.Tensor,
) -> np.ndarray:
    """Convert game table to embedding-based features.

    For each game, mean-pools the 5 home players and 5 away players,
    then concatenates: [home_mean; away_mean; home_mean - away_mean]
    """
    home_cols = [f"p{i}" for i in range(5)]
    away_cols = [f"p{i}" for i in range(5, 10)]

    emb_np = emb.numpy()
    d = emb_np.shape[1]

    features = np.zeros((len(games), 3 * d), dtype=np.float32)

    for i, (_, row) in enumerate(tqdm(games.iterrows(), total=len(games),
                                       desc="Embedding features", leave=False)):
        home_ids = [int(row[c]) for c in home_cols]
        away_ids = [int(row[c]) for c in away_cols]

        # Clamp IDs to valid range
        home_ids = [min(pid, emb_np.shape[0] - 1) for pid in home_ids]
        away_ids = [min(pid, emb_np.shape[0] - 1) for pid in away_ids]

        home_mean = emb_np[home_ids].mean(axis=0)
        away_mean = emb_np[away_ids].mean(axis=0)

        features[i, :d] = home_mean
        features[i, d:2*d] = away_mean
        features[i, 2*d:] = home_mean - away_mean

    return features


################################################################################
# Baselines
################################################################################

def bag_of_ids_features(games: pd.DataFrame, num_players: int) -> np.ndarray:
    """Create binary bag-of-player-IDs features (no learned embeddings).

    For each game, creates a sparse feature vector where each player's
    position is 1 if they're in the home lineup, -1 if away, 0 otherwise.
    """
    features = np.zeros((len(games), num_players), dtype=np.float32)

    home_cols = [f"p{i}" for i in range(5)]
    away_cols = [f"p{i}" for i in range(5, 10)]

    for i, (_, row) in enumerate(games.iterrows()):
        for c in home_cols:
            pid = int(row[c])
            if pid < num_players:
                features[i, pid] = 1.0
        for c in away_cols:
            pid = int(row[c])
            if pid < num_players:
                features[i, pid] = -1.0

    return features


################################################################################
# Main evaluation
################################################################################

def evaluate_model(name: str, X_train, y_train, X_test, y_test, max_iter=500):
    """Train logistic regression and report accuracy."""
    lr = LogisticRegression(max_iter=max_iter, solver="lbfgs", n_jobs=-1)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds


def main(args):
    print("Building game table...")
    games = build_game_table(args.parquet)
    print(f"  {len(games):,} games, seasons {games['season'].min()}-{games['season'].max()}")

    # Season-based split (same as possession model)
    train = games[games["season"] < 2019]
    test = games[games["season"] >= 2021]
    print(f"  Train: {len(train):,} games | Test: {len(test):,} games")

    y_train = train["home_win"].values
    y_test = test["home_win"].values

    home_rate_train = y_train.mean()
    home_rate_test = y_test.mean()
    print(f"  Home win rate: train={home_rate_train:.1%}, test={home_rate_test:.1%}")

    results = {}

    # Baseline 1: Always predict home win
    results["home_always"] = home_rate_test
    print(f"\n{'='*60}")
    print("GAME OUTCOME PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"  Home-always-wins:  {home_rate_test*100:.2f}%")

    # Baseline 2: Bag-of-player-IDs logistic regression
    print("\nTraining bag-of-player-IDs baseline...")
    num_players = int(games[[f"p{i}" for i in range(10)]].max().max()) + 1
    X_train_ids = bag_of_ids_features(train, num_players)
    X_test_ids = bag_of_ids_features(test, num_players)
    acc_ids, _ = evaluate_model("Bag-of-IDs", X_train_ids, y_train, X_test_ids, y_test)
    results["bag_of_ids"] = acc_ids
    print(f"  Bag-of-player-IDs: {acc_ids*100:.2f}%")

    # Embedding-based model
    if args.ckpt:
        print(f"\nLoading embeddings from {args.ckpt}...")
        emb = load_embeddings(args.ckpt)
        print(f"  Embedding shape: {emb.shape}")

        X_train_emb = games_to_embedding_features(train, emb)
        X_test_emb = games_to_embedding_features(test, emb)

        acc_emb, preds_emb = evaluate_model("Embeddings", X_train_emb, y_train,
                                             X_test_emb, y_test)
        results["embeddings"] = acc_emb
        print(f"  Learned embeddings: {acc_emb*100:.2f}%")

        # Embedding + IDs combined
        X_train_combo = np.concatenate([X_train_emb, X_train_ids], axis=1)
        X_test_combo = np.concatenate([X_test_emb, X_test_ids], axis=1)
        acc_combo, _ = evaluate_model("Combined", X_train_combo, y_train,
                                       X_test_combo, y_test)
        results["combined"] = acc_combo
        print(f"  Embeddings + IDs:  {acc_combo*100:.2f}%")

        # Show deltas
        delta_vs_home = (acc_emb - home_rate_test) * 100
        delta_vs_ids = (acc_emb - acc_ids) * 100
        print(f"\n  Embedding model vs home-always: {delta_vs_home:+.2f}pp")
        print(f"  Embedding model vs bag-of-IDs:  {delta_vs_ids:+.2f}pp")

        if delta_vs_ids > 0:
            print("  -> Embeddings improve over raw player IDs (generalization)")
        else:
            print("  -> Embeddings do not improve over raw IDs (may need more training)")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for name, acc in results.items():
        print(f"  {name:25s}: {acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game outcome prediction from player embeddings")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path for player embeddings")
    parser.add_argument("--parquet", default="possessions.parquet")
    args = parser.parse_args()
    main(args)
