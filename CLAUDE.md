# Unicorn — Contextual NBA Player Embeddings

## Overview

Unicorn learns **contextual player embeddings** from NBA play-by-play data using a transformer with attention pooling — analogous to the Word2Vec → BERT leap in NLP. Unlike NBA2Vec (2023) which uses mean-pooling over player embeddings, Unicorn models player-player interactions via self-attention, producing representations that capture how a player operates in a specific lineup and game state.

**Key innovation:** "NBA2Vec is to Word2Vec as Unicorn is to BERT."

## Architecture

**Two-phase BERT-style training (v2 — contrastive):**

1. **Phase A — Contrastive Masked Player Prediction (pretraining):** Mask one of 10 players, predict their *embedding* (not class logits) from remaining 9 + game state via InfoNCE loss. Auxiliary 2,310-way base-player classification head. Forces embeddings to encode player archetypes, not memorize rosters.

2. **Phase B — Outcome Prediction (fine-tuning):** Predict possession outcome (9-class) from full lineup + game state. Aligns embeddings toward basketball impact. Uses class-weighted cross-entropy.

**Model (v2):**
- **Composed embeddings**: `base_player_emb[2310 × 384]` (shared across seasons) + `delta_emb[12821 × 384]` (season-specific, L2-regularized, init to zero)
- Transformer encoder (8 layers, 8 heads, d_model=384) over 10 player tokens
- Attention pooling (learned, replaces CLS token) over player outputs
- Game state (time remaining, score diff, period) injected via separate projection, concatenated after pooling
- Team-side embeddings (offense=0, defense=1) added to player tokens
- **LLM-seeded base embedding init**: GPT-4o descriptions → anonymized → text-embedding-3-small at dim=384
- **Temporal augmentation**: 15% chance per player of swapping to adjacent season during training
- **Learnable temperature** for InfoNCE (init 0.07, clamped [0.01, 1.0])

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
| `prior_year_init.py` | Utility for prior/next-year maps, base-player mapping, temporal swap |
| `generate_player_descriptions.py` | LLM pipeline: GPT-4o descriptions → anonymized → text embeddings for init |
| `explore_text_embeddings.ipynb` | Text embedding validation notebook (quality checks, clustering, era bias) |
| `bbref_name_mapping.csv` | Maps bbref IDs → display names (2,310 players, used by description pipeline) |
| `evaluate.py` | Evaluation: per-class metrics, confusion matrix, baselines |
| `game_outcome.py` | Downstream task: game win prediction from frozen embeddings |
| `analyze_embeddings.py` | Embedding analysis: nearest neighbors, t-SNE, temporal trajectories |
| `explore_unicorn.ipynb` | Interactive data exploration notebook (run while training) |
| `player_season_lookup.csv` | Maps (player_name, season) → player_season_id |
| `requirements.txt` | Python dependencies |
| `Unicorn EDA.ipynb` | Original EDA notebook (updated for 9-class taxonomy) |

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

# 3. (Optional) Generate descriptions + text embeddings (already done — see EXPERIMENTS.md)
python generate_player_descriptions.py --generate --prompt D4

# 4. Phase A: Contrastive pretraining (v2)
python train_transformer.py --phase pretrain --epochs 25
# (auto-loads base_player_text_embeddings.pt if present)

# 5. Phase B: Outcome prediction fine-tuning
python train_transformer.py --phase finetune --pretrain-ckpt pretrain_v2_checkpoint.pt --epochs 15

# 6. Evaluate
python evaluate.py --ckpt finetune_v2_checkpoint.pt --parquet possessions.parquet

# 7. Downstream: game outcome prediction
python game_outcome.py --ckpt pretrain_v2_checkpoint.pt --parquet possessions.parquet
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
- **Player-season tokenization:** Each (player, season) pair gets a unique ID. LeBron_2018 ≠ LeBron_2019. Allows temporal evolution of player representations. Prior-year init provides "momentum" (soft temporal prior) without forcing identity collapse — players genuinely change year to year.
- **Composed embeddings (v2):** `base[player] + delta[player_season]`. Base captures enduring archetype; delta captures season-specific variation (L2-regularized). Enables cross-season generalization: LeBron_2018 and LeBron_2019 share the same base embedding.
- **Contrastive loss (v2):** InfoNCE with stop-gradient targets. The model predicts *where in embedding space* a player belongs, not *which specific ID*. Similar players naturally score similarly.
- **LLM-seeded init (v2):** GPT-4o generates play-style descriptions (prompt D4: archetype + team fit), **anonymized** (names + years stripped), then embedded with text-embedding-3-small → base_player_emb initialization. Anonymization prevents name-similarity and era-bias from dominating embedding structure. Uses `bbref_name_mapping.csv` for correct player name resolution.
- **Temporal augmentation (v2):** 15% chance of swapping each player to an adjacent season during training. Teaches near-interchangeability of adjacent seasons.
- **Two-phase training:** Contrastive prediction learns general representations; outcome prediction specializes them. Clean separation of objectives.
- **Season-based data splits:** Train (<2019), Val (2019-2020), Test (≥2021). No temporal leakage.

### Phase A Validation Design (Dual Evaluation)

Player-season IDs are unique per (player, season), so season-based train/val splits have ZERO ID overlap — the model literally cannot predict an unseen ID, making standard val accuracy permanently 0%. We use two complementary evaluation approaches:

1. **Random holdout (overfitting detection):** 90/10 random split of pre-2019 data. Both splits share the same ID space, so accuracy tracks training properly. Detects memorization vs generalization within the training distribution.

2. **Temporal evaluation (generalization signal):** For each masked player from the 2019-2020 val set, measure where their prior-year ID (from training era) ranks in the model's predictions. E.g., if LeBron_2019 is masked, does LeBron_2018 appear in the top predictions? Measures whether the model learns "someone like LeBron goes here" even when the exact ID is unseen. Metrics: mean/median rank of prior-year ID, top-10/50/100 hit rates.

## Data Facts

- 5,878,297 total possessions (before filtering "other")
- 12,821 player-season IDs, 2,310 unique players
- Seasons: 1999–2023
- 10,298 of 12,821 IDs (80.3%) have prior-year mappings

## Monitoring Training

```bash
# View raw JSON log (one line per epoch)
cat pretrain_v2_checkpoint.log.jsonl

# Quick summary of all epochs (v2)
python -c "
import json
for line in open('pretrain_v2_checkpoint.log.jsonl'):
    d = json.loads(line)
    print(f'Ep {d[\"epoch\"]:2d} | loss={d[\"train_loss\"]:.4f} (c={d[\"contrastive_loss\"]:.4f} a={d[\"aux_loss\"]:.4f}) | top1={d[\"train_top1\"]*100:.2f}% top5={d[\"train_top5\"]*100:.2f}% aux={d.get(\"train_aux_acc\",0)*100:.2f}% | val top5={d[\"val_top5\"]*100:.2f}% aux={d.get(\"val_aux_acc\",0)*100:.2f}% | T={d[\"temperature\"]:.4f} | temporal={d.get(\"temporal_mean_rank\",0):.0f} top100={d.get(\"temporal_top100\",0)*100:.1f}% | delta={d[\"delta_norm_mean\"]:.4f}')
"
```

Best model is auto-saved to `pretrain_v2_checkpoint.pt` (by val top-5 accuracy). Log file: `pretrain_v2_checkpoint.log.jsonl`.

## Current Status

- [x] Data preprocessing pipeline (9-class outcome taxonomy)
- [x] EDA notebook (updated for 9-class taxonomy)
- [x] Literature review (research complete)
- [x] 9-class outcome taxonomy + possessions.parquet regenerated
- [x] v1 Attention-pooling transformer (trained 25 epochs — showed roster memorization)
- [x] **v2 Architecture: composed embeddings + contrastive pretraining**
- [x] LLM description generation pipeline (generate_player_descriptions.py)
- [x] Player name mapping (bbref_name_mapping.csv — 2,310 players, 100% coverage)
- [x] Description anonymization (strip names + years before embedding)
- [x] Prior-year/next-year mapping + base-player mapping utilities
- [x] CBOW baseline model (trained: 12% val accuracy on 9-class)
- [x] Evaluation infrastructure (evaluate.py — supports v1 + v2)
- [x] Downstream task (game_outcome.py — supports v1 + v2)
- [x] Embedding analysis script (analyze_embeddings.py — supports v1 + v2)
- [x] Interactive exploration notebook (explore_unicorn.ipynb)
- [x] Text embedding validation notebook (explore_text_embeddings.ipynb)
- [x] **Experiment 0: LLM prompt validation — PASSED (GPT-4o, prompt D4, anonymized)**
- [x] **Full LLM description generation (12,821 descriptions, base_player_text_embeddings.pt)**
- [x] **Training Phase A v2 run 1a: killed at epoch 11 — false negative bug + temp collapse**
- [x] **v2 fixes: false-neg masking, fixed temp, delta cap, temporal eval bug, checkpoint resumption**
- [ ] **Training Phase A v2 run 1b: contrastive pretraining (25 epochs) — IN PROGRESS**
- [ ] Training Phase B: outcome fine-tuning (needs Phase A v2 checkpoint)
- [ ] Run full evaluation pipeline
- [ ] Embedding analysis and visualizations
- [ ] Literature review document

## Git Discipline

**Commit frequently.** After completing any logical unit of work (new script, bug fix, architecture change, experiment results), commit before moving on. Don't let uncommitted changes pile up across multiple milestones — it makes history useless and rollbacks impossible. Aim for commits that each tell a clear story.
