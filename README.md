# Unicorn

**Applying the Word2Vec → BERT leap to basketball:** learning contextual NBA player embeddings — where a player's representation depends on lineup context — using a self-attention transformer trained on millions of possession-level outcomes, with embeddings warm-started from LLM-generated scouting reports.

## Why Contextual?

Prior work like [NBA2Vec](https://arxiv.org/abs/2302.13386) treats players as static vectors (mean-pooled, context-free). But basketball is fundamentally contextual — a player's impact depends on who else is on the court.

Unicorn models player-player interactions through self-attention over 10-player lineups, producing representations that capture how a player operates in a *specific* lineup and game state. The same player gets a different representation depending on their teammates and opponents.

## Architecture

<p align="center"><img src="docs/architecture.png" alt="Unicorn architecture" width="700"></p>

- **Composed embeddings**: Each player-season is represented as `base_player + delta_season`, where the base embedding captures enduring archetype and the delta (low-rank bottleneck) captures season-specific variation
- **Transformer encoder** (8 layers, 8 heads, d_model=384) over 10 player tokens
- **Split offense/defense attention pooling** for lineup-level representation
- **Dual forward pass**: masked lineup for contrastive learning, full lineup for outcome prediction
- **Distributional outcome targets**: Bayesian-smoothed lineup distributions instead of single-possession labels
- **Joint training**: masked player prediction (InfoNCE contrastive) + outcome distribution prediction simultaneously, inspired by BERT's MLM + NSP
- **LLM-seeded initialization**: GPT-4o generates play-style descriptions per player, anonymized and embedded via text-embedding-3-small, providing a semantic warm start
- **~17.4M parameters**

### Training Objectives

| Objective | What it does |
|-----------|-------------|
| **Contrastive (InfoNCE)** | Mask one of 10 players, predict their embedding from the remaining 9 + game state. Teaches player archetypes. |
| **Auxiliary classification** | Classify the masked player (2,310-way). Stabilizes early training. |
| **Outcome distribution** | Predict lineup outcome distribution (9-class) from full lineup + state. Players shift distributions, not deterministic outcomes. |

### Outcome Taxonomy (9 classes)

| Class | Description |
|-------|-------------|
| `made_3pt` | Made 3-pointer |
| `missed_3pt` | Missed 3-pointer |
| `made_2pt_close` | Made 2pt, ≤10ft |
| `made_2pt_mid` | Made 2pt, >10ft |
| `missed_2pt_close` | Missed 2pt, ≤10ft |
| `missed_2pt_mid` | Missed 2pt, >10ft |
| `FT` | Free throw sequence |
| `turnover_live` | Live ball turnover |
| `turnover_dead` | Dead ball turnover |

## Data

- **5.9M possessions** from NBA play-by-play data (1999–2023)
- **12,821 player-season IDs** across **2,310 unique players**
- Season-based splits: train (<2019), val (2019–2020), test (≥2021)

```
all_games.csv.zip (raw play-by-play, 261MB)
    → scripts/nba_preprocessing_pipeline.py
possessions.parquet (~5.9M possessions, 9-class outcomes)
    → nba_dataset.py (PyTorch Dataset)
Model training
```

## Project Structure

```
unicorn/
├── train_transformer.py          # Main model + training (joint, pretrain, finetune)
├── train_cbow.py                 # CBOW baseline model
├── nba_dataset.py                # PyTorch Dataset with splits and augmentation
├── prior_year_init.py            # Prior/next-year maps, base-player mapping
├── evaluate.py                   # Per-class metrics, confusion matrix, baselines
├── analyze_embeddings.py         # Nearest neighbors, t-SNE, temporal trajectories
├── game_outcome.py               # Downstream task: game win prediction
├── player_descriptions.jsonl     # LLM-generated player descriptions (12,821)
├── scripts/
│   ├── nba_preprocessing_pipeline.py   # Raw CSV → possessions parquet
│   └── generate_player_descriptions.py # GPT-4o description + embedding pipeline
├── notebooks/
│   ├── eda.ipynb                       # Exploratory data analysis
│   ├── explore_unicorn.ipynb           # Interactive embedding exploration
│   ├── explore_text_embeddings.ipynb   # Text embedding validation
│   └── evaluate_embeddings.ipynb       # Embedding evaluation
└── docs/
    ├── EXPERIMENTS.md                  # Experiment protocol and results
    └── TRAINING_NOTES.md               # Training run notes
```

## Quick Start

```bash
# Setup
python -m venv .unicorn && source .unicorn/bin/activate
pip install -r requirements.txt

# Preprocess data (requires all_games.csv)
python scripts/nba_preprocessing_pipeline.py --raw-csv all_games.csv --out-file possessions.parquet

# Train (joint: contrastive + distributional outcome)
python train_transformer.py --phase joint --epochs 25 --delta-dim 64 \
    --outcome-weight 1.0 --contrastive-weight 0.5 --prior-strength 10

# Evaluate
python evaluate.py --ckpt joint_v21_checkpoint.pt --phase joint

# Analyze embeddings
python analyze_embeddings.py --ckpt joint_v21_checkpoint.pt

# Downstream: game outcome prediction
python game_outcome.py --ckpt joint_v21_checkpoint.pt --parquet possessions.parquet
```

## Key Design Decisions

- **Attention pooling over CLS token**: Lets the transformer focus on player-player interactions. Attention weights provide interpretability.
- **Player-season tokenization**: LeBron_2018 ≠ LeBron_2019. Composed embeddings (base + delta) enable cross-season generalization while allowing evolution.
- **Joint training**: Contrastive + outcome simultaneously (not sequentially). Outcome signal from epoch 1 prevents over-optimizing for player ID discrimination.
- **Base-player contrastive (2,310-way)**: Rewards archetype learning, not roster memorization. Prior 12,821-way contrastive rewarded co-occurrence shortcuts.
- **LLM-seeded init**: Anonymized GPT-4o descriptions embedded at dim=384 provide semantic grounding before any play-by-play training.
- **Temporal augmentation**: 15% chance of swapping each player to an adjacent season — teaches near-interchangeability.

## Requirements

Python 3.12+. Key dependencies: PyTorch, pandas, numpy, pyarrow, scikit-learn, tqdm, matplotlib, seaborn.
