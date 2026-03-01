# Unicorn — Contextual NBA Player Embeddings

## Overview

Unicorn learns **contextual player embeddings** from NBA play-by-play data using a transformer with attention pooling — analogous to the Word2Vec → BERT leap in NLP. Unlike NBA2Vec (2023) which uses mean-pooling over player embeddings, Unicorn models player-player interactions via self-attention, producing representations that capture how a player operates in a specific lineup and game state.

**Key innovation:** "NBA2Vec is to Word2Vec as Unicorn is to BERT."

## Architecture

**Two-phase BERT-style training:**

1. **Phase A — Masked Player Prediction (pretraining):** Mask one of 10 players, predict identity from remaining 9 + game state. Forces embeddings to encode player roles, styles, and lineup context. Loss: cross-entropy over 12,821 player-season IDs.

2. **Phase B — Outcome Prediction (fine-tuning):** Predict possession outcome (9-class) from full lineup + game state. Aligns embeddings toward basketball impact. Uses class-weighted cross-entropy.

**Model:**
- Transformer encoder (8 layers, 8 heads, d_model=384) over 10 player tokens
- Attention pooling (learned, replaces CLS token) over player outputs
- Game state (time remaining, score diff, period) injected via separate projection, concatenated after pooling
- Team-side embeddings (offense=0, defense=1) added to player tokens
- Prior-year embedding initialization: LeBron_2020 initialized from learned LeBron_2019 embedding

## Data Pipeline

```
all_games.csv.zip (raw PBP, 261MB)
    ↓ unzip
all_games.csv (2.9GB, ~56M rows)
    ↓ nba_preprocessing_pipeline.py
possessions.parquet (~5.9M possessions, 9-class outcomes)
    ↓ nba_dataset.py (PyTorch Dataset)
Model training
```

## Key Files

| File | Purpose |
|------|---------|
| `nba_preprocessing_pipeline.py` | Raw CSV → possessions parquet with outcome labels |
| `nba_dataset.py` | PyTorch Dataset with splits, augmentation, normalization |
| `train_transformer.py` | Main model: attention-pooling transformer, Phase A & B training |
| `train_cbow.py` | CBOW baseline: mean-pool embeddings → MLP classifier |
| `prior_year_init.py` | Utility for prior-year embedding warm-start |
| `evaluate.py` | Evaluation: per-class metrics, confusion matrix, baselines |
| `game_outcome.py` | Downstream task: game win prediction from frozen embeddings |
| `player_season_lookup.csv` | Maps (player_name, season) → player_season_id |
| `requirements.txt` | Python dependencies |
| `Unicorn EDA.ipynb` | Exploratory data analysis notebook |

## Virtual Environment

```bash
source .unicorn/bin/activate
```

Python 3.12, key deps: torch, pandas, numpy, pyarrow, sklearn, tqdm, matplotlib, seaborn.

## How to Run

```bash
# 1. Unzip raw data (only needed for preprocessing)
unzip all_games.csv.zip

# 2. Preprocess (produces possessions.parquet + player_season_lookup.csv)
python nba_preprocessing_pipeline.py --raw-csv all_games.csv --out-file possessions.parquet

# 3. Phase A: Masked player prediction pretraining
python train_transformer.py --phase pretrain --parquet possessions.parquet --epochs 25

# 4. Phase B: Outcome prediction fine-tuning
python train_transformer.py --phase finetune --pretrain-ckpt pretrain_checkpoint.pt --epochs 15

# 5. Evaluate
python evaluate.py --ckpt finetune_checkpoint.pt --parquet possessions.parquet

# 6. Downstream: game outcome prediction
python game_outcome.py --ckpt pretrain_checkpoint.pt --parquet possessions.parquet
```

## Outcome Taxonomy (9 classes)

| Class | Description |
|-------|-------------|
| `made_3pt` | Made 3-point field goal |
| `missed_3pt` | Missed 3-point field goal |
| `made_2pt_close` | Made 2pt, ≤10ft (dunks, layups) |
| `made_2pt_mid` | Made 2pt, >10ft (mid-range jumpers) |
| `missed_2pt_close` | Missed 2pt, ≤10ft |
| `missed_2pt_mid` | Missed 2pt, >10ft |
| `FT` | Free throw sequence |
| `turnover_live` | Live ball turnover (steal, bad pass) |
| `turnover_dead` | Dead ball turnover (offensive foul, violation) |

"Other" (defensive rebounds, period starts, jump balls) is dropped — it's noise, not a meaningful possession outcome.

## Design Decisions

- **Attention pooling over player tokens (not CLS):** Lets the transformer focus on player-player interactions. State is injected separately. Attention weights provide interpretability.
- **Player-season tokenization:** Each (player, season) pair gets a unique ID. LeBron_2018 ≠ LeBron_2019. Allows temporal evolution of player representations.
- **Prior-year embedding init:** 80.3% of player-season IDs have a prior-year mapping. Warm-starting from prior year improves convergence and temporal coherence.
- **Two-phase training:** Masked prediction learns general representations; outcome prediction specializes them. Clean separation of objectives.
- **Season-based data splits:** Train (<2019), Val (2019-2020), Test (≥2021). No temporal leakage.

## Data Facts

- 5,878,297 total possessions (before filtering "other")
- 12,821 player-season IDs, 2,310 unique players
- Seasons: 1999–2023
- 10,298 of 12,821 IDs (80.3%) have prior-year mappings

## Current Status

- [x] Data preprocessing pipeline
- [x] EDA notebook
- [x] Literature review (research complete)
- [ ] 9-class outcome taxonomy (preprocessing update needed)
- [ ] Attention-pooling transformer with masked prediction
- [ ] Prior-year embedding initialization
- [ ] Evaluation infrastructure
- [ ] Downstream task (game outcome prediction)
- [ ] Embedding analysis and visualizations
- [ ] Literature review document
