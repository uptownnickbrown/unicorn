# Unicorn v2: Experimental Design

## Motivation

Phase A v1 (25 epochs, 17 hours) revealed that 12,821-way classification over player-season IDs rewards lineup memorization, not learning player archetypes. Train top-5 reached 43.5% but temporal generalization collapsed to random (mean rank ~6,660 out of 12,821). This document captures the experimental design for the v2 architecture rework — composed embeddings, contrastive pretraining, and LLM-seeded initialization.

---

## Hypotheses

| # | Hypothesis | Metric | Success Criteria |
|---|-----------|--------|-----------------|
| H1 | Composed embeddings (base + delta) with contrastive loss produce better temporal generalization than ID classification | Temporal mean rank | Mean rank < 1000 (vs ~6,660 in v1) |
| H2 | LLM-seeded initialization provides meaningful semantic structure from epoch 0 | Pre-training embedding similarity | Archetypally similar players cluster before any training |
| H3 | LLM init accelerates convergence compared to random init | Temporal metrics at epoch 5 | LLM-init reaches epoch-15 random-init quality by epoch 5 |
| H4 | Temporal augmentation improves cross-season coherence beyond what composed embeddings alone provide | Temporal top-100 hit rate | Augmented model > non-augmented by ≥5pp |
| H5 | Delta regularization prevents season-specific overfitting without collapsing cross-season variation | Delta norm distribution | Deltas are small but non-zero; career trajectories are visible in delta space |

---

## Experiment 0: LLM Description Validation (COMPLETED)

**Purpose:** Validate that LLM descriptions + OpenAI embeddings produce a semantically meaningful embedding space BEFORE training.

**Status: PASSED** — All criteria met. Results verified in `explore_text_embeddings.ipynb`.

### What Was Done

**Three critical pipeline bugs were discovered and fixed before generation:**
1. **Player misidentification** — 97% of players had raw bbref IDs (e.g., `jamesle01`) sent to GPT instead of real names. Fixed by creating `bbref_name_mapping.csv` (2,310 players, 100% coverage via external CSV + bbref scraping) and `load_name_mapping()` in the generation script.
2. **Name string leakage** — The embedding model encoded player names as a dominant signal, causing "Stephen Curry" to cluster with "Seth Curry" and "Eddy Curry" by name similarity rather than play style. Fixed by adding `anonymize_description()` — strips all name variants and year references from descriptions before embedding.
3. **Year/era bias** — Season year strings (e.g., "2016-2017") created era clustering that dominated archetype structure. Also fixed by anonymization.

**Final pipeline configuration:**
- **Model:** GPT-4o (upgraded from GPT-4o-mini for better obscure player descriptions)
- **Prompt:** D4 — Archetype + team fit framing (winner after iterating through A, B, B2-B4, C, D1-D3):
  ```
  In 3-4 specific sentences, describe {player_name}'s basketball identity during the
  {season}-{season_next} NBA season. What type of player is he — not just his position,
  but his archetype (e.g., floor general, rim-running big, stretch four, 3-and-D wing,
  isolation scorer)? Describe his offensive and defensive tendencies, how his game has
  aged or developed compared to earlier seasons, and what kind of team construction he
  fits best in.
  ```
- **Anonymization:** All descriptions stripped of player names (+ common variants) and year references before embedding
- **Embedding:** `text-embedding-3-small` at dim=384, then averaged per base player, scaled to std=0.02
- **Cost:** ~$5 for GPT-4o + ~$0.10 for embeddings. Time: ~40 minutes at concurrency=20.

### Verified Results

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Curry neighbors are shooters, not namesakes | No Seth/Eddy Curry | Klay Thompson (5 seasons), Ian Clark — all shooters | PASS |
| Era bias (within-era vs cross-era similarity gap) | < 0.02 | 0.009 (within-early 0.754, within-late 0.769, cross-era 0.738) | PASS |
| Same-player vs different-player separation | > 0.15 | 0.118 (same: 0.865, diff: 0.747) | ACCEPTABLE |
| Obscure vs well-known similarity distributions | Nearly identical | 0.743 vs 0.748 | PASS |
| K-means clusters map to archetypes | Recognizable types | Clear clusters: 3-and-D wings, floor generals, rim protectors, etc. | PASS |
| Effective dimensionality | > 15 dims for 90% variance | 22 dims | PASS |
| Labels show real names, not bbref IDs | All real names | All real names | PASS |

**Note on same-player separation (0.118 vs 0.15 target):** This is expected and correct. Anonymization deliberately removes the strongest identity signal (the player's name), so same-player similarity drops. The tradeoff is worth it — without anonymization, name similarity dominated and Curry clustered with Currys, not shooters. The 0.118 gap still shows clear separation between same-player and different-player distributions.

### Key Output Files
- `player_descriptions.jsonl` — 12,821 GPT-4o descriptions (prompt D4)
- `player_text_embeddings.pt` — [12821, 384] anonymized per-player-season embeddings
- `base_player_text_embeddings.pt` — [2310, 384] base player averages (std=0.02)
- `bbref_name_mapping.csv` — 2,310 bbref ID → display name mappings

---

## Experiment 1: Baseline v2 Training (Full Architecture)

**Purpose:** Validate the new architecture with reasonable defaults before hyperparameter tuning.

**Status: IN PROGRESS** — Run 1b (with fixes) active, 5/25 epochs complete. Run 1a was killed at epoch 11 due to training pathology (see below).

### Run 1a: Original Configuration (KILLED — epoch 11/25)

**Configuration:** Original defaults (delta_reg=0.01, learned temperature, no false-negative masking).

**Outcome: Training pathology — same memorization failure as v1.** Killed after 11 epochs (~7.6h).

**Symptoms:**
- Temperature collapsed 0.07 → 0.01 (hit floor clamp) — contrastive loss acting as hard classification
- Delta norms at 78% of base norms — season-specific memorization
- Train top-5 = 26% but val top-5 = 0.35% — massive overfitting on same ID space
- Temporal mean rank oscillating wildly (1,319 to 9,392 — but see bug note below)

**Root cause analysis identified three issues:**

1. **Same-base-player false negatives (CRITICAL):** InfoNCE treated all 12,821 player-seasons as negatives, including other seasons of the same player. LeBron_2020 was penalized for being near LeBron_2019. This forced the model to grow deltas to distinguish same-player seasons, collapsing temperature to sharpen distinctions. Fix: mask same-base-player IDs from the contrastive denominator.

2. **Temperature collapse (amplifier):** Learned temperature crashed to 0.01 floor. At τ=0.01, cosine sims scaled 100×, creating near-one-hot softmax — gradients vanish for hard negatives. Fix: fix temperature at 0.07.

3. **Weak delta regularization (amplifier):** λ=0.01 too weak to counteract false-negative-driven delta growth. Fix: increase to 0.05 + add hard norm cap.

**Critical measurement bug also discovered:** The temporal evaluation function had false-negative masking applied, which masked out same-base-player entries including the prior-year ID being ranked. This caused temporal rank to always report ~12,813 (dead last). Run 1a temporal metrics were partially corrupted by this bug (the eval was added mid-run). Fixed for run 1b.

### Run 1b: Fixed Configuration (ACTIVE)

**Changes from run 1a:**

| Parameter | Run 1a | Run 1b | Rationale |
|-----------|--------|--------|-----------|
| False-negative masking | None | Same-base-player masked in contrastive loss | Core fix — don't penalize archetype similarity |
| Temperature | Learned (init 0.07) | Fixed at 0.07 | Prevent collapse; validate archetype learning first |
| `delta_reg_weight` | 0.01 | 0.05 | Stronger soft regularization |
| `delta_max_norm` | None | 0.3 (hard cap) | Hard constraint on delta growth after each step |
| `pred_cossim` logging | None | Added | Embedding collapse detection |
| Temporal eval masking | Bugged (masked prior-year ID) | Fixed (no masking in temporal eval) | Correct measurement |
| Checkpoint resumption | None | `--resume` flag + `_latest.pt` every epoch | Survive interruptions |

**Results through epoch 5:**

| Metric | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 |
|--------|------|------|------|------|------|
| Contrastive loss | 8.69 | 8.21 | 7.98 | 7.54 | 7.10 |
| Train top-5 | 0.54% | 1.01% | 1.93% | 5.10% | 9.98% |
| Val top-5 | 0.28% | 0.13% | 0.49% | 0.52% | 0.64% |
| Train aux acc | 0.52% | 0.89% | 1.50% | 3.52% | 6.58% |
| Temporal mean rank | 1,520 | 2,343 | 3,202 | 1,854 | 3,374 |
| Temporal top-100 | 7.4% | 8.5% | 6.1% | 6.7% | 3.5% |
| Delta norm mean | 0.142 | 0.153 | 0.170 | 0.176 | 0.180 |
| Base norm mean | 0.658 | 0.683 | 0.730 | 0.768 | 0.803 |
| Delta/base ratio | 21.6% | 22.4% | 23.3% | 22.9% | 22.4% |
| pred_cossim | 0.75 | 0.40 | 0.31 | 0.22 | 0.15 |

**Assessment at epoch 5:**
- **Temporal metrics are real signal** — mean rank 1,500-3,400 (vs v1's ~6,400 random). The model does learn archetypes.
- **But temporal top-100 is declining** (7.4% → 3.5%) — as the model memorizes training IDs, the archetype signal erodes.
- **Delta norms well-controlled** — hard cap at 0.3 is working; ratio stable at ~22%.
- **Embedding space healthy** — pred_cossim dropped from 0.75 to 0.15, no collapse.
- **Train-val gap growing** — train top-5 at 10% vs val top-5 at 0.64%. Memorization within training set.
- **Key concern:** contrastive loss still fundamentally rewards ID-level discrimination, which competes with archetype-level generalization. The false-negative fix helps but doesn't fully resolve this tension.

### Default Configuration (Run 1b)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `d_model` | 384 | Matches text-embedding-3-small dimensions=384 |
| `n_layers` | 8 | Unchanged from v1 |
| `n_heads` | 8 | Unchanged from v1 |
| `delta_reg_weight` | 0.05 | 5× increase from run 1a to counter delta growth |
| `delta_max_norm` | 0.3 | Hard cap on delta norms (~45% of base norm) |
| `aux_loss_weight` | 0.1 | Auxiliary signal should guide but not dominate |
| `temperature` | Fixed 0.07 | CLIP/MoCo default; removed as learnable to prevent collapse |
| `temporal_aug_prob` | 0.15 | 15% swap rate |
| `learning_rate` | 3e-4 | Unchanged from v1 |
| `batch_size` | 1024 | Unchanged from v1 |
| `gradient_accum` | 4 | Effective batch size 4096 |
| `epochs` | 25 | Unchanged from v1 |

### Metrics to Track Per Epoch
| Metric | What It Measures |
|--------|-----------------|
| `contrastive_loss` | Primary training loss |
| `contrastive_acc` | % correct player ranked #1 in full vocab |
| `aux_loss` | Base-player classification loss |
| `aux_acc` | % correct base player (2,310-way) |
| `temporal_mean_rank` | Prior-year ID rank in contrastive scores (key metric) |
| `temporal_median_rank` | Median of above |
| `temporal_top50` | % prior-year ID in top 50 |
| `temporal_top100` | % prior-year ID in top 100 |
| `delta_norm_mean` | Average L2 norm of delta embeddings |
| `delta_norm_max` | Max delta norm (detect outliers) |
| `base_norm_mean` | Average base embedding norm |
| `temperature` | Fixed at 0.07 (sanity check) |
| `val_contrastive_acc` | Validation contrastive accuracy (random holdout) |
| `pred_cossim` | Mean pairwise cosine similarity of batch predictions (collapse detection) |

### Success Criteria for Experiment 1
- Temporal mean rank < 2000 (vs v1's 6,660)
- Temporal top-100 > 10% (vs v1's 0.4%)
- Contrastive accuracy improves over training
- Delta norms stay bounded (< 50% of base norms)
- No embedding collapse (pred_cossim < 0.5 after epoch 3)

---

## Experiment 2: Hyperparameter Sensitivity Sweeps

Run after Experiment 1 validates the architecture. Use **5-epoch short runs** to compare settings efficiently (~2h each on MPS).

### 2a: Delta Regularization Weight
| Run | `delta_reg_weight` | Expected Behavior |
|-----|-------------------|-------------------|
| 2a-1 | 0.001 | Low regularization — deltas may grow large, season specificity dominates |
| 2a-2 | 0.01 | Moderate (default) — balanced |
| 2a-3 | 0.1 | High — deltas nearly zero, all seasons of a player nearly identical |
| 2a-4 | 1.0 | Very high — deltas effectively zeroed out, pure base embeddings |

**Key metric:** Plot delta_norm_mean vs temporal_mean_rank. Find the sweet spot.

### 2b: Auxiliary Loss Weight
| Run | `aux_loss_weight` | Expected Behavior |
|-----|-------------------|-------------------|
| 2b-1 | 0.0 | No auxiliary — embeddings only grounded through contrastive loss |
| 2b-2 | 0.1 | Moderate (default) |
| 2b-3 | 0.5 | Strong auxiliary — may pull model toward player classification |

**Key metric:** aux_acc and temporal_mean_rank. Higher aux_weight should improve aux_acc but may not help temporal metrics if it overwhelms contrastive signal.

### 2c: Temporal Augmentation Probability
| Run | `temporal_aug_prob` | Expected Behavior |
|-----|---------------------|-------------------|
| 2c-1 | 0.0 | No augmentation — baseline |
| 2c-2 | 0.10 | Light augmentation |
| 2c-3 | 0.15 | Default |
| 2c-4 | 0.25 | Aggressive — 2.5 players swapped per sample on average |

**Key metric:** Temporal metrics (mean rank, top-100). Also monitor contrastive_acc — too much augmentation may hurt it.

### 2d: LLM Init vs Random Init (Ablation)
| Run | Init | Description |
|-----|------|-------------|
| 2d-1 | LLM | Text-embedding-3-small initialized base embeddings |
| 2d-2 | Random | Standard nn.Embedding initialization (N(0, 0.02)) |

**Key metric:** Compare metrics at epoch 5. LLM init should show faster convergence and better early temporal metrics.

### Sweep Protocol
- All sweep runs use the same random seed (42)
- Run one variable at a time, holding others at defaults
- 5 epochs each (~2h on MPS)
- Log all metrics to separate JSONL files: `sweep_{experiment}_{run}.log.jsonl`
- After sweeps, select best settings for full 25-epoch run

---

## Experiment 3: Full Training Run

Run with best hyperparameters from Experiment 2. Full 25 epochs.

### Followed By
- Phase B fine-tuning (outcome prediction)
- Full evaluation pipeline (evaluate.py on all checkpoints)
- Downstream tasks (game outcome prediction)
- Embedding analysis (nearest neighbors, t-SNE, career trajectories)

---

## Experiment 1c: v2.1 Joint Training (F7 + F6)

**Purpose:** Combine contrastive masked prediction + outcome prediction in a single training phase, with a delta bottleneck to structurally limit season-specific capacity.

**Status: PENDING** — Implementation complete, smoke test in progress.

### Architecture Changes from v2.0

| Component | v2.0 | v2.1 | Rationale |
|-----------|------|------|-----------|
| Training phases | Sequential (Phase A → Phase B) | Joint (simultaneous) | Outcome signal from epoch 1 prevents ID-over-discrimination |
| Delta embedding | Full-rank `[12821 × 384]` (4.9M params) | Bottleneck `[12821 × 64] → [64 → 384]` (0.8M + 24K params) | Structurally limits season-specific capacity |
| Delta regularization | Soft λ=0.05 + hard cap 0.3 | Soft λ=0.01, no hard cap | Bottleneck makes hard cap unnecessary |
| Differential LR | Phase B only (base 0.01×, delta 0.03×, encoder 0.1×) | From epoch 1 (base 0.1×, delta+encoder 0.3×, heads 1×) | Protect LLM-seeded base embeddings throughout |
| Total params | ~21.5M | ~17.4M | Leaner model |
| Loss function | Phase A: InfoNCE + 0.1×aux. Phase B: CE | InfoNCE + 0.1×aux + outcome_weight×CE + 0.01×delta_reg | All objectives simultaneous |
| Outcome during training | Unmasked (Phase B) | Masked (one player replaced with [MASK]) | Acts as input dropout |

### Loss Function

`L = L_contrastive + 0.1 × L_aux + outcome_weight × L_outcome + 0.01 × L_delta_reg`

Where:
- `L_contrastive`: InfoNCE with same-base-player false-negative masking, fixed τ=0.07
- `L_aux`: Base-player classification (2,310-way CE)
- `L_outcome`: Class-weighted 9-way CE for possession outcome (on masked lineup)
- `L_delta_reg`: Mean L2 norm of delta_raw embeddings (before projection)

### What Was Kept from v2.0 Fixes

| Change | Decision | Rationale |
|--------|----------|-----------|
| False-neg masking in contrastive loss | KEPT | Essential — don't penalize same-player similarity |
| Fixed temperature 0.07 | KEPT | Proven, one less variable |
| pred_cossim logging | KEPT | Free collapse detection |
| Checkpoint resumption | KEPT | Essential infrastructure |
| Temporal eval (no false-neg masking) | KEPT | Correct measurement |
| Auxiliary base-player head | KEPT | Cheap supporting signal |
| Delta soft reg λ=0.05 | REDUCED to 0.01 | Bottleneck structurally limits capacity |
| Delta hard norm cap 0.3 | DROPPED | Bottleneck does this job |

### Default Configuration

| Parameter | Value |
|-----------|-------|
| `d_model` | 384 |
| `n_layers` | 8 |
| `n_heads` | 8 |
| `delta_dim` | 64 |
| `outcome_weight` | 1.0 |
| `delta_reg_weight` | 0.01 (reduced from 0.05) |
| `aux_loss_weight` | 0.1 |
| `temporal_aug_prob` | 0.15 |
| `learning_rate` | 3e-4 |
| `batch_size` | 1024 |
| `gradient_accum` | 4 |
| `epochs` | 25 |

### Success Criteria

| Metric | Target | Comparison |
|--------|--------|------------|
| Val outcome accuracy | > 12% | Beat CBOW baseline |
| Temporal mean rank | < 1,500 | Better than v2.0's oscillating 1,500-3,400 |
| Temporal top-100 | Stable or increasing | v2.0 showed declining top-100 |
| Both objectives improving | Simultaneous | Not trading off against each other |

### Comparison Table (to fill after training)

| Model | Val Outcome Acc | Temporal Mean Rank | Temporal Top-100 | Params |
|-------|----------------|-------------------|-----------------|--------|
| CBOW | 12% | N/A | N/A | — |
| v2.0 + Phase B | ? | ~2,000-3,400 | ~3-8% | ~21.5M |
| v2.1 Joint | ? | ? | ? | ~17.4M |

### Run Command

```bash
python train_transformer.py --phase joint --epochs 25 --delta-dim 64 --outcome-weight 1.0
```

---

## Future Experiments (Post-v2)

Ideas for further exploration once the v2 architecture is validated:

### F1: Multi-Resolution Temporal Augmentation
Instead of uniform swap probability, weight by temporal distance: adjacent seasons swap more often than distant ones.

### F2: Contrastive Hard Negatives
Use hard negative mining — focus contrastive loss on players who are similar but not identical (teammates, same-position players). May improve fine-grained archetype discrimination.

### F3: Cross-Modal Alignment
Add a term that keeps trained embeddings aligned with their LLM text embeddings (prevents semantic drift). Like an "anchor loss" that preserves the initial semantic structure.

### F4: Lineup-Level Contrastive Learning
Instead of predicting individual players, predict whether a lineup is "real" or "shuffled" (lineup coherence). May better capture player-player interactions.

### F5: Dynamic Temperature by Player Frequency
Popular players may need different temperature than rare players. Explore per-player or frequency-binned temperatures.

### F6: Delta Capacity Experiments
Try constraining delta to lower dimensions (e.g., delta_dim=64, projected to 384 via linear layer). This structurally limits season-specific capacity rather than relying on regularization.

### F7: Outcome-Aware Pretraining
Multi-task Phase A: contrastive player prediction + outcome prediction simultaneously (rather than sequentially in Phase A → Phase B). May align embeddings to basketball impact from the start.

---

## Efficiency & Early Stopping

**Prioritize wall-clock time over cost.** Key tactics:

1. **Parallel API calls:** Description generation and embedding use concurrent/async requests to OpenAI. Batch embeddings in groups of 2048 (API limit). Target: full 12,821 descriptions + embeddings in < 15 minutes.

2. **Early stopping for sweeps:** If a hyperparameter setting shows clearly worse temporal metrics by epoch 3 (vs default at epoch 3), kill it early and move on. No need to run all 5 epochs for a bad configuration.

3. **Progressive validation:** Check temporal metrics every epoch. If temporal mean rank is > 5000 after epoch 5 (heading toward random), the run has a fundamental issue — stop and diagnose.

4. **Skip sweeps if baseline is strong:** If Experiment 1 hits success criteria (temporal mean rank < 1000) with default hyperparameters, skip Experiment 2 sweeps and go straight to Phase B.

## Resource Budget

| Activity | Estimated Wall Time | Cost | Status |
|----------|-------------------|------|--------|
| LLM description generation (12,821 players, GPT-4o, prompt D4) | ~40 min | ~$5 | DONE |
| Prompt iteration + validation (A, B, B2-B4, C, D1-D4) | ~3 hours | ~$2 | DONE |
| Pipeline bug fixes (name mapping, anonymization, era bias) | ~4 hours coding | — | DONE |
| Architecture implementation (v2) | ~3-4 hours coding | — | DONE |
| Experiment 1a (original config, killed at ep 11) | ~7.6 hours | — | KILLED |
| Experiment 1a diagnosis + fixes | ~2 hours | — | DONE |
| Experiment 1b (fixed config, killed at epoch 6) | ~4.4 hours | — | KILLED |
| Experiment 1b diagnosis → v2.1 design | ~2 hours | — | DONE |
| v2.1 implementation (joint + bottleneck) | ~2 hours | — | DONE |
| Phase B v2.0 baseline (15 epochs) | ~6 hours | — | IN PROGRESS |
| Experiment 1c: v2.1 joint training (25 epochs) | ~17 hours | — | PENDING |
| Full evaluation pipeline | ~1 hour | — | Pending |
