#!/usr/bin/env python3
"""
Build stint-level training data from possessions.parquet.

A stint is a consecutive sequence of possessions with the same 10-player lineup.
Each stint aggregates outcome counts for distributional prediction.

Usage:
    python scripts/build_stints.py
    python scripts/build_stints.py --parquet possessions.parquet --output stints.parquet
    python scripts/build_stints.py --min-poss 3 --max-poss 15
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOME_VOCAB = {
    "made_3pt": 0, "missed_3pt": 1, "made_2pt_close": 2, "made_2pt_mid": 3,
    "missed_2pt_close": 4, "missed_2pt_mid": 5, "FT": 6,
    "turnover_live": 7, "turnover_dead": 8,
}
NUM_OUTCOMES = len(OUTCOME_VOCAB)


def build_stints(
    parquet_path: str | Path,
    min_poss: int = 3,
    max_poss: int = 15,
) -> pd.DataFrame:
    """Build stint dataframe from possessions.

    1. Load possessions, filter 'other', sort by (game_id, poss_id)
    2. Detect stint boundaries (lineup change within a game)
    3. Aggregate outcome counts per stint
    4. Split long stints (>max_poss) via recursive midpoint
    5. Drop short stints (<min_poss)
    """
    t0 = time.time()
    print("Loading possessions...", flush=True)
    cols = [f"p{i}" for i in range(10)] + [
        "game_id", "poss_id", "sec_remaining_game", "score_diff",
        "period", "season", "outcome",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    df = df[df["outcome"] != "other"].copy()
    df["outcome_id"] = df["outcome"].map(OUTCOME_VOCAB)
    df = df.sort_values(["game_id", "poss_id"]).reset_index(drop=True)
    print(f"  {len(df):,} possessions loaded ({time.time()-t0:.1f}s)", flush=True)

    # Detect stint boundaries: new stint when game changes or any player changes
    print("Detecting stint boundaries...", flush=True)
    player_cols = [f"p{i}" for i in range(10)]
    game_changed = df["game_id"] != df["game_id"].shift(1)
    lineup_changed = (df[player_cols] != df[player_cols].shift(1)).any(axis=1)
    df["stint_id"] = (game_changed | lineup_changed).cumsum()

    # Group by stint
    print("Aggregating stints...", flush=True)
    grouped = df.groupby("stint_id")

    records = []
    for stint_id, group in grouped:
        n_poss = len(group)
        first_row = group.iloc[0]

        # Outcome counts
        outcome_counts = np.bincount(group["outcome_id"].values, minlength=NUM_OUTCOMES)

        record = {
            "game_id": first_row["game_id"],
            "stint_id": stint_id,
            "season": first_row["season"],
            "sec_remaining_game": first_row["sec_remaining_game"],
            "score_diff": first_row["score_diff"],
            "period": first_row["period"],
            "n_poss": n_poss,
        }
        for i in range(10):
            record[f"p{i}"] = first_row[f"p{i}"]
        for i in range(NUM_OUTCOMES):
            record[f"oc{i}"] = outcome_counts[i]
        records.append(record)

    stints = pd.DataFrame(records)
    print(f"  {len(stints):,} raw stints ({time.time()-t0:.1f}s)", flush=True)

    # Split long stints (>max_poss) via recursive midpoint
    print(f"Splitting stints > {max_poss} possessions...", flush=True)
    long_mask = stints["n_poss"] > max_poss
    n_long = long_mask.sum()
    if n_long > 0:
        short_stints = stints[~long_mask].copy()
        long_stints = stints[long_mask]

        split_records = []
        for _, row in long_stints.iterrows():
            _split_stint(row, max_poss, split_records)

        split_df = pd.DataFrame(split_records)
        stints = pd.concat([short_stints, split_df], ignore_index=True)
        print(f"  Split {n_long:,} long stints → {len(split_df):,} chunks", flush=True)

    # Drop short stints
    before = len(stints)
    stints = stints[stints["n_poss"] >= min_poss].reset_index(drop=True)
    print(f"  Dropped {before - len(stints):,} stints with < {min_poss} possessions", flush=True)

    # Verify outcome counts sum matches
    total_oc = sum(stints[f"oc{i}"].sum() for i in range(NUM_OUTCOMES))
    print(f"  Final: {len(stints):,} stints, {total_oc:,} total possessions in counts", flush=True)

    # Print split stats
    for split_name, cond in [("train", stints["season"] < 2019),
                              ("val", stints["season"].between(2019, 2020)),
                              ("test", stints["season"] >= 2021)]:
        n = cond.sum()
        n_poss = stints.loc[cond, "n_poss"].sum()
        print(f"  {split_name}: {n:,} stints, {n_poss:,} possessions", flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s", flush=True)
    return stints


def _split_stint(row, max_poss: int, out: list):
    """Recursively split a stint at the midpoint until all chunks <= max_poss."""
    n = row["n_poss"]
    if n <= max_poss:
        out.append(row.to_dict())
        return

    # Split outcome counts approximately at midpoint
    total = n
    mid = total // 2
    oc_total = np.array([row[f"oc{i}"] for i in range(NUM_OUTCOMES)])

    # Proportional split of outcome counts
    oc_first = np.round(oc_total * (mid / total)).astype(int)
    oc_second = oc_total - oc_first

    base = {k: row[k] for k in row.index if not k.startswith("oc") and k != "n_poss"}

    first = {**base, "n_poss": mid}
    for i in range(NUM_OUTCOMES):
        first[f"oc{i}"] = oc_first[i]

    second = {**base, "n_poss": total - mid}
    for i in range(NUM_OUTCOMES):
        second[f"oc{i}"] = oc_second[i]

    _split_stint(pd.Series(first), max_poss, out)
    _split_stint(pd.Series(second), max_poss, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build stint-level data from possessions")
    parser.add_argument("--parquet", default="possessions.parquet")
    parser.add_argument("--output", default="stints.parquet")
    parser.add_argument("--min-poss", type=int, default=3, help="Drop stints with fewer possessions")
    parser.add_argument("--max-poss", type=int, default=15, help="Split stints longer than this")
    args = parser.parse_args()

    stints = build_stints(args.parquet, min_poss=args.min_poss, max_poss=args.max_poss)
    stints.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}", flush=True)
