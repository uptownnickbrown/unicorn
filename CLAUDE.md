# Unicorn — Contextual NBA Player Embeddings

## Overview

Unicorn learns **contextual player embeddings** from NBA play-by-play data using a transformer with attention pooling — analogous to the Word2Vec → BERT leap in NLP. Unlike NBA2Vec (2023) which uses mean-pooling over player embeddings, Unicorn models player-player interactions via self-attention, producing representations that capture how a player operates in a specific lineup and game state.

**Key innovation:** "NBA2Vec is to Word2Vec as Unicorn is to BERT."

## Architecture

**v2.0: Two-phase BERT-style training (contrastive):**

1. **Phase A — Contrastive Masked Player Prediction (pretraining):** Mask one of 10 players, predict their *embedding* (not class logits) from remaining 9 + game state via InfoNCE loss. Auxiliary 2,310-way base-player classification head. Forces embeddings to encode player archetypes, not memorize rosters.

2. **Phase B — Outcome Prediction (fine-tuning):** Predict possession outcome (9-class) from full lineup + game state. Aligns embeddings toward basketball impact. Uses class-weighted cross-entropy.

**v2.1: Joint training (contrastive + outcome simultaneously):**

Single training phase: mask one player, then **simultaneously** predict the masked player's embedding (InfoNCE), classify the base player (auxiliary), AND predict possession outcome (9-class). The outcome head sees a masked lineup during training (like input dropout); evaluation uses unmasked lineups.

**Key insight:** BERT does MLM + NSP simultaneously — v2.1 does masked player prediction + outcome prediction simultaneously, giving embeddings a reason to encode basketball impact from epoch 1.

**v3.0: Base-player contrastive (outcome-primary):**

Replaces 12,821-way player-season contrastive with **2,310-way base-player contrastive**. Contrastive targets are `base_player_emb.weight` [2310, 384] instead of `_all_composed_embeddings()` [12821, 384]. No false-negative masking needed (each base player appears exactly once). Contrastive weight is tunable via `--contrastive-weight` (default 1.0, recommended 0.5).

**Key insight:** 12,821-way contrastive rewards roster memorization (teammate co-occurrence is cheaper than understanding basketball role). 2,310-way base-player contrastive rewards archetype learning. Outcome prediction is the primary signal; contrastive is supporting.

**v3.2: Distributional outcome prediction (paradigm shift):**

Replaces single-possession hard classification with **Bayesian-smoothed distributional targets per lineup**. Players don't determine single-possession outcomes — they shift the *probability distribution* of outcomes. The model learns to predict how each lineup's outcome distribution deviates from the global mean.

Key changes from v3.0:
- **Distributional targets**: Each possession's target is a 9-dim probability distribution computed per 10-player lineup via Bayesian smoothing: `target = (n * empirical + alpha * prior) / (n + alpha)`. Default `alpha=10` (tunable via `--prior-strength`).
- **Dual forward pass**: Pass 1 masks one player for contrastive/auxiliary. Pass 2 uses full lineup for outcome distribution prediction. Fixes the v2.1-v3.0 mismatch where the outcome head trained on masked lineups.
- **Split offense/defense attention pooling**: Separate `AttentionPool` for offense (positions 0-4) and defense (positions 5-9). Outcome head sees `[off_pooled, def_pooled, state_repr]` (3×d_model).
- **Soft cross-entropy loss**: `-(target_dist * log_softmax(logits)).sum(1).mean()` — no class weights needed.
- **State concatenation** (reverted from v3.1): State projected via `nn.Linear(3, d_model)` and concatenated after pooling, NOT as transformer token.

**Model (v3.2):**
- **Composed embeddings**: `base_player_emb[2310 × 384]` (shared) + **delta bottleneck** `delta_raw[12821 × 64] → delta_proj[64 → 384]` (low-rank, L2-reg)
- Transformer encoder (8 layers, 8 heads, d_model=384) over 10 player tokens
- **Split attention pooling**: offense pool + defense pool (learned, separate)
- Game state injected via separate projection, concatenated after pooling
- Team-side embeddings (offense=0, defense=1) added to player tokens
- **Base-player contrastive**: InfoNCE against `base_player_emb.weight` [2310], not full vocab [12821]
- **LLM-seeded base embedding init**: GPT-4o descriptions → anonymized → text-embedding-3-small at dim=384
- **Temporal augmentation**: 15% chance per player of swapping to adjacent season
- **Fixed temperature** at 0.07 for InfoNCE
- **Differential LR**: base 0.1×, delta+encoder 0.3×, heads 1×
- **`--contrastive-weight`**: Tunable (0 = outcome-only, 0.5 = recommended, 1.0 = default)
- **`--prior-strength`**: Bayesian smoothing alpha (default 10). With alpha=10 and median lineup of 4 possessions, empirical gets 29% weight.
- **~17.4M params**

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
| `train_transformer.py` | Main model: attention-pooling transformer, Phase A & B training |
| `train_cbow.py` | CBOW baseline: mean-pool embeddings → MLP classifier |
| `nba_dataset.py` | PyTorch Dataset with splits, augmentation, normalization |
| `prior_year_init.py` | Utility for prior/next-year maps, base-player mapping, temporal swap |
| `evaluate.py` | Evaluation: per-class metrics, confusion matrix, baselines |
| `game_outcome.py` | Downstream task: game win prediction from frozen embeddings |
| `analyze_embeddings.py` | Embedding analysis: nearest neighbors, t-SNE, temporal trajectories |
| `player_descriptions.jsonl` | LLM-generated player descriptions (12,821) |
| `scripts/nba_preprocessing_pipeline.py` | Raw CSV → possessions parquet with outcome labels |
| `scripts/generate_player_descriptions.py` | LLM pipeline: GPT-4o descriptions → anonymized → text embeddings for init |
| `notebooks/eda.ipynb` | Original EDA notebook (updated for 9-class taxonomy) |
| `notebooks/explore_unicorn.ipynb` | Interactive data exploration notebook (run while training) |
| `notebooks/explore_text_embeddings.ipynb` | Text embedding validation notebook (quality checks, clustering, era bias) |
| `notebooks/evaluate_embeddings.ipynb` | Embedding evaluation notebook |
| `docs/EXPERIMENTS.md` | Experiment protocol and results |
| `docs/TRAINING_NOTES.md` | Training run notes |
| `bbref_name_mapping.csv` | Maps bbref IDs → display names (2,310 players, used by description pipeline) |
| `player_season_lookup.csv` | Maps (player_name, season) → player_season_id |
| `requirements.txt` | Python dependencies |

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
python scripts/nba_preprocessing_pipeline.py --raw-csv all_games.csv --out-file possessions.parquet

# 3. (Optional) Generate descriptions + text embeddings (already done — see docs/EXPERIMENTS.md)
python scripts/generate_player_descriptions.py --generate --prompt D4

# 4a. (v2.0) Phase A: Contrastive pretraining
python train_transformer.py --phase pretrain --epochs 25
# (auto-loads base_player_text_embeddings.pt if present)

# 4b. (v2.0) Phase B: Outcome prediction fine-tuning
python train_transformer.py --phase finetune --pretrain-ckpt pretrain_v2_checkpoint.pt --epochs 15

# 4c. (v3.0) Joint training: base-player contrastive + outcome
python train_transformer.py --phase joint --epochs 25 --delta-dim 64 --outcome-weight 1.0 --contrastive-weight 0.5 --prior-strength 10

# 5. Evaluate
python evaluate.py --ckpt finetune_v2_checkpoint.pt --phase finetune
python evaluate.py --ckpt joint_v21_checkpoint.pt --phase joint

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
- **Composed embeddings (v2+):** `base[player] + delta[player_season]`. Base captures enduring archetype; delta captures season-specific variation (L2-regularized). Enables cross-season generalization: LeBron_2018 and LeBron_2019 share the same base embedding.
- **Delta bottleneck (v2.1):** `delta_raw[12821 × 64] → delta_proj[64 → 384]`. Structurally limits season-specific capacity (from 4.9M to 0.8M params). Makes hard norm capping unnecessary — the bottleneck inherently constrains delta expressiveness.
- **Joint training (v2.1):** Contrastive + outcome prediction simultaneously, not sequentially. Outcome signal from epoch 1 prevents embeddings from over-optimizing for ID discrimination at the expense of basketball-impact encoding.
- **Contrastive loss (v3.0):** Base-player InfoNCE (2,310-way) against `base_player_emb.weight` with stop-gradient. No false-negative masking needed — each base player appears exactly once. Prior versions used 12,821-way player-season contrastive which rewarded roster memorization.
- **LLM-seeded init (v2+):** GPT-4o generates play-style descriptions (prompt D4: archetype + team fit), **anonymized** (names + years stripped), then embedded with text-embedding-3-small → base_player_emb initialization. Anonymization prevents name-similarity and era-bias from dominating embedding structure. Uses `bbref_name_mapping.csv` for correct player name resolution.
- **Temporal augmentation (v2+):** 15% chance of swapping each player to an adjacent season during training. Teaches near-interchangeability of adjacent seasons.
- **Two-phase training (v2.0):** Contrastive prediction learns general representations; outcome prediction specializes them. Clean separation of objectives. Superseded by joint training in v2.1.
- **Distributional targets (v3.2):** Players don't determine single-possession outcomes — they shift the probability distribution. Bayesian-smoothed targets per lineup let the model learn "how does this lineup deviate from average?" instead of predicting individual plays. With alpha=10, median lineup (4 possessions) gets 29% empirical weight.
- **Dual forward pass (v3.2):** Separate encoder passes for contrastive (masked) and outcome (unmasked) prediction. Fixes the v2.1-v3.0 mismatch where the outcome head trained on incomplete lineups but evaluated on complete lineups.
- **Split offense/defense pooling (v3.2):** After cross-attention over all 10 players, pool offense and defense separately. Gives the outcome head explicit "offensive unit capability" and "defensive unit pressure" representations. Representation hierarchy: player → interactions → unit → matchup.
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
# v2.0 pretrain log
python -c "
import json
for line in open('pretrain_v2_checkpoint.log.jsonl'):
    d = json.loads(line)
    print(f'Ep {d[\"epoch\"]:2d} | loss={d[\"train_loss\"]:.4f} (c={d[\"contrastive_loss\"]:.4f} a={d[\"aux_loss\"]:.4f}) | top1={d[\"train_top1\"]*100:.2f}% top5={d[\"train_top5\"]*100:.2f}% aux={d.get(\"train_aux_acc\",0)*100:.2f}% | val top5={d[\"val_top5\"]*100:.2f}% aux={d.get(\"val_aux_acc\",0)*100:.2f}% | T={d[\"temperature\"]:.4f} | temporal={d.get(\"temporal_mean_rank\",0):.0f} top100={d.get(\"temporal_top100\",0)*100:.1f}% | delta={d[\"delta_norm_mean\"]:.4f}')
"

# v3.0 joint training log (base-player contrastive)
python -c "
import json
for line in open('joint_v21_basecontra.log.jsonl'):
    d = json.loads(line)
    print(f'Ep {d[\"epoch\"]:2d} | loss={d[\"train_loss\"]:.4f} (c={d[\"contrastive_loss\"]:.4f} a={d[\"aux_loss\"]:.4f} o={d[\"outcome_loss\"]:.4f}) | base_top5={d[\"train_base_top5\"]*100:.2f}% out={d[\"train_outcome_acc\"]*100:.2f}% | val out={d[\"val_outcome_acc\"]*100:.2f}% | temporal={d.get(\"temporal_mean_rank\",0):.0f} top100={d.get(\"temporal_top100\",0)*100:.1f}% | delta={d[\"delta_norm_mean\"]:.4f}')
"

# v3.2 distributional training log
python -c "
import json
for line in open('joint_v32_checkpoint.log.jsonl'):
    d = json.loads(line)
    print(f'Ep {d[\"epoch\"]:2d} | loss={d[\"train_loss\"]:.4f} (c={d[\"contrastive_loss\"]:.4f} a={d[\"aux_loss\"]:.4f} o={d[\"outcome_loss\"]:.4f}) | base_top5={d[\"train_base_top5\"]*100:.2f}% out={d[\"train_outcome_acc\"]*100:.2f}% | val out={d[\"val_outcome_acc\"]*100:.2f}% | temporal={d.get(\"temporal_mean_rank\",0):.0f} top100={d.get(\"temporal_top100\",0)*100:.1f}% | delta={d[\"delta_norm_mean\"]:.4f}')
"
```

**v2.0:** Best model saved to `pretrain_v2_checkpoint.pt` (by val top-5). Log: `pretrain_v2_checkpoint.log.jsonl`.
**v2.1:** Best model saved to `joint_v21_checkpoint.pt` (by val outcome accuracy). Log: `joint_v21_checkpoint.log.jsonl`.
**v3.0:** Best model saved to `joint_v21_basecontra.pt` (by val outcome accuracy). Log: `joint_v21_basecontra.log.jsonl`.
**v3.2:** Best model saved to `joint_v32_checkpoint.pt` (by val outcome accuracy). Log: `joint_v32_checkpoint.log.jsonl`.

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
- [x] **Training Phase A v2 run 1b: contrastive pretraining — killed at epoch 6 (temporal metrics oscillating)**
- [x] Training Phase B v2.0: outcome fine-tuning on run 1b checkpoint — killed ep 1, matches CBOW
- [x] **v2.1 implementation: joint training (F7) + delta bottleneck (F6)**
- [x] **v2.1 run 1: joint training — killed ep 7 (delta explosion, 15.66% val outcome)**
- [x] **v2.1b fix: projected delta reg + hard cap**
- [x] **v3.0 implementation: base-player contrastive (2,310-way) + outcome-primary**
- [x] **v3.1 training: state token experiment — KILLED (unstable val acc oscillating 8-20%)**
- [x] **v3.2 implementation: distributional outcome prediction paradigm shift**
- [ ] **Training v3.2: distributional + split pooling + dual forward pass (25 epochs) — PENDING**
- [ ] Run full evaluation pipeline
- [ ] Embedding analysis and visualizations
- [ ] Literature review document

## Git Discipline

**Commit frequently.** After completing any logical unit of work (new script, bug fix, architecture change, experiment results), commit before moving on. Don't let uncommitted changes pile up across multiple milestones — it makes history useless and rollbacks impossible. Aim for commits that each tell a clear story.
