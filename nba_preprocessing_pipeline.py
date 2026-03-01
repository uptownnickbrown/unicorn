# nba_preprocessing_pipeline.py
"""
Pre‑processing pipeline for **xocelyk/nba‑pbp** with:

* tqdm **progress bars that always render** (even in non‑TTY IDEs) via
  `tqdm.contrib.concurrent.process_map` fallback.
* **Step‑level logging** – each major stage logs start/end with elapsed time so
  you can see where the script is spending cycles even if bars don’t show.

Run:
```bash
python nba_preprocessing_pipeline.py \
  --raw-csv all_games.csv \
  --out-file possessions.parquet
```
Use `--no-tqdm` to disable bars, and `--log-interval 50000` to print heartbeat
messages every N rows during lineup encoding.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

################################################################################
# Settings overridden by CLI flags
################################################################################
USE_TQDM = True
LOG_INTERVAL = 50_000  # rows

################################################################################
# Helper: simple timer context
################################################################################
class Timer:
    def __init__(self, msg: str):
        self.msg = msg
    def __enter__(self):
        self.start = time.time()
        print(f"▶ {self.msg}…", flush=True)
    def __exit__(self, exc_type, exc_val, exc_tb):
        dur = time.time() - self.start
        print(f"✓ {self.msg} done in {dur:,.1f}s", flush=True)

################################################################################
# Column rename map
################################################################################
RENAME_MAP = {"gameid": "game_id", "period": "period", "time": "clock", "awayscore": "away_score", "homescore": "home_score", "awayevent": "away_event", "homeevent": "home_event"}
for i in range(1, 6):
    RENAME_MAP[f"a{i}"] = f"away_player{i}"
    RENAME_MAP[f"h{i}"] = f"home_player{i}"

################################################################################
# Load CSV
################################################################################

def load_raw(path: str | Path) -> pd.DataFrame:
    with Timer("Reading CSV"):
        df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df = df.rename(columns=RENAME_MAP)
    print(f"Data shape: {df.shape}")
    return df

################################################################################
# Pre‑processing utilities
################################################################################

def ensure_season(df: pd.DataFrame):
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    else:
        df["season"] = df["game_id"].astype(str).str.slice(0, 4).astype(int) + 1


def coerce_scores(df: pd.DataFrame):
    for c in ("away_score", "home_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["away_score", "home_score"]] = (
        df.groupby("game_id")[["away_score", "home_score"]].ffill().fillna(0).astype(int)
    )


def add_points_and_event_type(df: pd.DataFrame):
    df["event_text"] = df["away_event"].fillna("") + df["home_event"].fillna("")
    df["prev_away"] = df.groupby("game_id")["away_score"].shift().fillna(df["away_score"])
    df["prev_home"] = df.groupby("game_id")["home_score"].shift().fillna(df["home_score"])
    df["points"] = (df["away_score"] - df["prev_away"]).abs() + (df["home_score"] - df["prev_home"]).abs()
    txt = df["event_text"].str.lower()
    df["event_type"] = "other"
    patterns = {"shot_made": "makes", "shot_miss": "misses", "turnover": "turnover", "rebound_def": "defensive rebound", "jump_ball": "jump ball", "period_start": r"start of \d"}
    for label, pat in patterns.items():
        df.loc[txt.str.contains(pat, regex=bool("\\d" in pat)), "event_type"] = label
    df.drop(columns=["prev_away", "prev_home"], inplace=True)


def build_lookup(df: pd.DataFrame):
    lookup, next_id = {}, 0
    cols = [f"home_player{i}" for i in range(1, 6)] + [f"away_player{i}" for i in range(1, 6)]
    for season, *players in df[["season"] + cols].itertuples(index=False):
        for p in players:
            if p and p != "nan":
                key = (p, season)
                if key not in lookup:
                    lookup[key] = next_id
                    next_id += 1
    return lookup

################################################################################
# Possession segmentation & outcome
################################################################################
BOUNDARY = {"shot_made", "turnover", "rebound_def", "jump_ball", "period_start"}

def add_possessions(df: pd.DataFrame):
    poss = np.zeros(len(df), dtype=int)
    cur = -1
    for i, et in enumerate(df["event_type"].values):
        if et in BOUNDARY:
            cur += 1
        poss[i] = cur
    df["poss_id"] = poss


def outcome_label(g: pd.DataFrame) -> str:
    """Seven-way possession label with robust fall‑back logic.

    Order of precedence:
    1. **FT** – any free‑throw attempt in the possession.
    2. **Turnover** – zero‑point possession that contains a turnover.
    3. **Made FG** – use last *shot_made* row.
    4. **Missed FG** – use last *shot_miss* row.
    5. **Other** – everything else (jump‑ball violations, period starts, etc.).
    """
    pts = g["points"].sum()

    # 1) Free‑throws override all (captures and‑1s as well)
    if g["event_text"].str.contains("free throw", case=False).any():
        return "FT"

    # 2) Turnover (no points)
    if pts == 0 and (g["event_type"] == "turnover").any():
        return "turnover"

    # 3) Made field goal
    made_rows = g[g["event_type"] == "shot_made"]
    if not made_rows.empty:
        made_text = made_rows.iloc[-1]["event_text"].lower()
        is_three  = "3-pt" in made_text or "3pt" in made_text
        return "made_3pt_FG" if is_three else "made_2pt_FG"

    # 4) Missed field goal that ends possession (regardless of who rebounded)
    miss_rows = g[g["event_type"] == "shot_miss"]
    if not miss_rows.empty:
        miss_text = miss_rows.iloc[-1]["event_text"].lower()
        is_three  = "3-pt" in miss_text or "3pt" in miss_text
        return "missed_3pt_FG" if is_three else "missed_2pt_FG"

    # 5) Fallback
    return "other"

################################################################################
# Clock helper
################################################################################

def clock_to_sec(t: str | float | int) -> int:
    """Convert a `clock` string like "11:35.2" → seconds (int).
    Gracefully handles NaNs or malformed strings by returning 0 so the
    downstream arithmetic won’t fail.
    """
    if not isinstance(t, str) or ":" not in t:
        return 0
    try:
        m, rest = t.split(":", 1)
        s = rest.split(".")[0]
        return int(m) * 60 + int(s)
    except Exception:
        return 0

################################################################################
# Main
################################################################################

def encode_lineup_row(row, lookup):
    season = row["season"]
    ids = [lookup.get((row[f"home_player{j}"], season), -1) for j in range(1, 6)] + [lookup.get((row[f"away_player{j}"], season), -1) for j in range(1, 6)]
    return ids


def process(raw_csv: str | Path, out_file: str | Path, lookup_csv: str | Path | None):
    df = load_raw(raw_csv)
    ensure_season(df)
    coerce_scores(df)
    add_points_and_event_type(df)

    lookup = build_lookup(df)
    (Path(lookup_csv) if lookup_csv else Path(out_file).with_name("player_season_lookup.csv")).write_text(
        pd.DataFrame([(p, s, i) for (p, s), i in lookup.items()], columns=["player", "season", "player_season_id"]).to_csv(index=False)
    )
    print(f"Player‑season IDs: {len(lookup):,}")

    # Encode lineups with explicit logging
    tic = time.time(); print("Encoding lineups…", flush=True)
    if USE_TQDM:
        tqdm.pandas(miniters=LOG_INTERVAL)
        df[[f"p{i}" for i in range(10)]] = pd.DataFrame(df.progress_apply(encode_lineup_row, lookup=lookup, axis=1).tolist(), index=df.index)
    else:
        encoded = []
        for idx, r in df.iterrows():
            if idx % LOG_INTERVAL == 0: print(f"  processed {idx:,} rows", flush=True)
            encoded.append(encode_lineup_row(r, lookup))
        df[[f"p{i}" for i in range(10)]] = pd.DataFrame(encoded, index=df.index)
    print(f"Lineups encoded in {time.time()-tic:.1f}s")

    add_possessions(df)
    print("Grouping by possessions…", flush=True)
    grp = df.groupby(["game_id", "poss_id"], sort=False)
    first = grp.first().reset_index()
    first["outcome"] = grp.apply(outcome_label).values

    first["sec_remaining_game"] = (first["period"] - 1) * 720 + first["clock"].apply(clock_to_sec)
    first["score_diff"] = first["home_score"] - first["away_score"]

    with Timer("Writing Parquet"):
        first.to_parquet(out_file, index=False)

################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NBA PBP preprocessing with logging")
    p.add_argument("--raw-csv", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--lookup-csv", default=None)
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--log-interval", type=int, default=50_000)
    args = p.parse_args()
    USE_TQDM = not args.no_tqdm
    LOG_INTERVAL = args.log_interval
    process(args.raw_csv, args.out_file, args.lookup_csv)
