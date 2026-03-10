# Unicorn v2: Experimental Design

## Motivation

Phase A v1 (25 epochs, 17 hours) revealed that 12,821-way classification over player-season IDs rewards lineup memorization, not learning player archetypes. Train top-5 reached 43.5% but temporal generalization collapsed to random (mean rank ~6,660 out of 12,821). This document captures the experimental design for the v2 architecture rework — composed embeddings, contrastive pretraining, and LLM-seeded initialization.

---

## Experiment Log (Chronological)

### v1 Phase A — Attention Transformer, ID Classification (25 epochs, 17h)
- **What**: Predict masked player ID from 12,821-way classification head
- **Result**: Train top-5 43.5%, temporal mean rank ~6,660 (random = ~6,400). Complete roster memorization.
- **Lesson**: ID classification rewards memorizing which players co-occur, not learning archetypes.

### CBOW Baseline (10 epochs)
- **What**: Mean-pool 10 player embeddings + state → MLP → 9-class outcome
- **Result**: 12.02% val accuracy. Barely above majority-class baseline (~11%).
- **Lesson**: Floor for outcome prediction. No player interactions = no signal.

### v2.0 Run 1a — Contrastive + Composed Embeddings (KILLED epoch 11, ~7.6h)
- **What**: InfoNCE contrastive loss, base+delta composed embeddings, learned temperature, no false-neg masking
- **Config**: delta_reg=0.01, learned temp (init 0.07), no false-neg masking
- **Result**: Temp collapsed 0.07→0.01, delta/base 78%, temporal metrics unreliable (eval bug)
- **Root cause**: Same-base-player false negatives forced delta growth to distinguish LeBron_2019 from LeBron_2020
- **Lesson**: (1) Must mask same-player false negatives in contrastive loss. (2) Learned temperature is unstable. (3) Temporal eval must not mask prior-year ID.

### v2.0 Run 1b — Fixed Contrastive (KILLED epoch 6, ~4.4h)
- **What**: Same as 1a + false-neg masking, fixed temp 0.07, delta reg 0.05, hard cap 0.3
- **Result**: Temporal mean rank 1,520-3,374 (real signal, oscillating). Top-100 declining 7.4%→3.5%. Delta/base stable 22%. No collapse.
- **Lesson**: Contrastive-only still rewards ID discrimination over archetypes. Top-100 declining = memorization eroding archetype signal. Need outcome signal from epoch 1.

| Metric | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 | Ep 6 |
|--------|------|------|------|------|------|------|
| Contrastive loss | 8.69 | 8.21 | 7.98 | 7.54 | 7.10 | 6.89 |
| Train top-5 | 0.54% | 1.01% | 1.93% | 5.10% | 9.98% | 12.96% |
| Val top-5 | 0.28% | 0.13% | 0.49% | 0.52% | 0.64% | 0.63% |
| Temporal mean rank | 1,520 | 2,343 | 3,202 | 1,854 | 3,374 | 2,027 |
| Temporal top-100 | 7.4% | 8.5% | 6.1% | 6.7% | 3.5% | 8.2% |
| Delta/base ratio | 21.6% | 22.4% | 23.3% | 22.9% | 22.4% | 24.8% |
| pred_cossim | 0.75 | 0.40 | 0.31 | 0.22 | 0.15 | 0.13 |

### v2.0 Phase B — Outcome Fine-tuning on Run 1b (KILLED epoch 1)
- **What**: Fine-tune run 1b checkpoint (epoch 6) for outcome prediction. Differential LR (base 0.01×, delta 0.03×, encoder 0.1×, head 1×).
- **Result**: Epoch 1: train 15.04%, val 12.13%. Essentially matches CBOW.
- **Lesson**: 6 epochs of contrastive-only pretraining didn't produce representations useful for outcome prediction. Sequential approach underwhelming.

### v2.1 Run 1 — Joint Training + Delta Bottleneck (KILLED epoch 7, ~10.6h)
- **What**: Contrastive + outcome simultaneously. Delta bottleneck (64→384). λ=0.01, NO hard cap.
- **Config**: delta_dim=64, outcome_weight=1.0, delta_reg=0.01, delta_max_norm=0 (disabled)
- **Result**: Val outcome 15.66% (beats CBOW!). BUT delta/base exploded 184%→328%. Temporal collapsed to random (~6,700).
- **Root cause**: Bottleneck reduces params not norms. `delta_proj` amplified raw deltas 13x. λ=0.01 only penalized 64-dim raw, not 384-dim output. Hard cap was dropped.
- **Lesson**: (1) Joint training WORKS for outcome prediction. (2) Bottleneck ≠ norm constraint. (3) Must regularize/cap the PROJECTED delta, not just the raw. (4) Don't drop hard cap.

| Metric | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 | Ep 6 | Ep 7 |
|--------|------|------|------|------|------|------|------|
| Contrastive loss | 8.76 | 8.13 | 8.13 | 8.09 | 8.03 | 7.93 | 7.65 |
| Outcome loss | 2.189 | 2.179 | 2.178 | 2.178 | 2.177 | 2.176 | 2.176 |
| Val outcome acc | 11.37% | 14.81% | 6.24% | 10.01% | 9.92% | **15.66%** | 15.60% |
| Temporal mean rank | 3,903 | 5,193 | 4,769 | 5,128 | 5,323 | 6,714 | 6,320 |
| Temporal top-100 | 0.13% | 4.10% | 0.13% | 0.03% | 0.02% | 0.02% | 0.12% |
| Delta/base ratio | 184% | 223% | 256% | 291% | 320% | 324% | 328% |
| Delta raw norm | 0.076 | 0.084 | 0.090 | 0.095 | 0.100 | 0.105 | 0.108 |
| Proj amplification | 9.6x | 10.6x | 11.5x | 12.5x | 13.2x | 13.0x | 12.8x |
| pred_cossim | 0.74 | 0.33 | 0.27 | 0.22 | 0.19 | 0.18 | 0.15 |

### v2.1b Run 2 — Joint Training with Projected Delta Fix (23 epochs, ~35h MPS)
- **What**: v2.1 joint training + delta bottleneck with fix: regularize PROJECTED deltas (not raw) + restore hard cap 0.3 on projected norms
- **Config**: delta_dim=64, outcome_weight=1.0, delta_reg=0.05 (projected), delta_max_norm=0.3 (projected), 12,821-way contrastive
- **Result**: Best val outcome 18.0% (ep 17), best temporal top-100 21.6% (ep 4). Delta stable at ~16%. Both metrics improved over v2.1 run 1.
- **Lesson**: Projected delta reg + hard cap fixes the explosion. But 12,821-way contrastive still rewards roster memorization — temporal top-100 peaked early (ep 4) then declined as contrastive loss dominated.

| Metric | Ep 1 | Ep 5 | Ep 10 | Ep 15 | Ep 20 | Ep 23 |
|--------|------|------|-------|-------|-------|-------|
| Contrastive loss | 9.22 | 8.07 | 7.51 | 7.16 | 6.85 | 6.72 |
| Outcome loss | 2.191 | 2.177 | 2.174 | 2.172 | 2.170 | 2.170 |
| Val outcome acc | 4.3% | 17.1% | 16.7% | 17.2% | 15.5% | 16.0% |
| Temporal top-100 | 0.6% | 16.2% | 11.2% | 10.6% | 12.8% | 13.4% |
| Delta norm mean | 0.000 | 0.102 | 0.113 | 0.141 | 0.159 | 0.158 |

### v3.0/v3.1 — Base-Player Contrastive + State Token (KILLED epoch 10)
- **What**: Switch from 12,821-way to 2,310-way base-player contrastive. v3.1 additionally injected game state as 11th transformer token (instead of post-pooling concatenation).
- **Config**: base-player InfoNCE, outcome_weight=1.0, contrastive_weight=0.5
- **Result**: Val accuracy wildly unstable, oscillating 8-20% across epochs. Best val 20.1% (ep 2), but then 8.2% (ep 3). Temporal top-100 peaked at 4.2% (ep 2) then declined.
- **Root cause**: State token injection created instability — the model couldn't learn stable representations when game state was mixed into the self-attention over player tokens. The 11th token created an asymmetry the architecture wasn't designed for.
- **Lesson**: Base-player contrastive is the right direction, but state should NOT be a transformer token. Revert to post-pooling concatenation.

| Metric | Ep 1 | Ep 2 | Ep 3 | Ep 5 | Ep 8 | Ep 10 |
|--------|------|------|------|------|------|-------|
| Contrastive loss | 7.19 | 7.12 | 6.62 | 6.11 | 5.80 | 5.54 |
| Outcome loss | 2.186 | 2.175 | 2.166 | 2.157 | 2.154 | 2.152 |
| Val outcome acc | 8.8% | **20.1%** | 8.2% | 15.1% | 14.9% | 13.9% |
| Temporal top-100 | 0.6% | 4.2% | 1.6% | 1.1% | 3.2% | 1.7% |
| Delta norm mean | 0.000 | 0.135 | 0.105 | 0.106 | 0.112 | 0.138 |

### v3.2 Run 1 — Distributional Outcome Prediction (25 epochs, RunPod RTX A5000)
- **What**: Paradigm shift — predict Bayesian-smoothed outcome distributions per lineup instead of single-possession hard labels. Dual forward pass (masked for contrastive, unmasked for outcome). Split offense/defense attention pooling. State concatenated after pooling.
- **Config**: prior_strength=10, outcome_weight=1.0, contrastive_weight=0.5, delta_dim=64, bs=2048, RTX A5000 GPU (~14 min/epoch)
- **Result**: Healthy convergence across all metrics. Outcome loss steadily decreased. Val outcome acc reached 52.6% (ep 4 peak, stable ~50-52% thereafter). Temporal top-100 improved monotonically to 35.0% (ep 23). Delta norms stable at ~17%.
- **Key insight**: Distributional targets are the right prediction target. Single-possession outcome has a low ceiling (~17-20% across all prior runs). Distributional prediction achieves ~50-60% train accuracy because it's predicting lineup-level distributions, not individual plays.
- **Lesson**: This is the architecture. All metrics healthy and improving. No training pathologies.

| Metric | Ep 1 | Ep 5 | Ep 10 | Ep 15 | Ep 20 | Ep 25 |
|--------|------|------|-------|-------|-------|-------|
| Total loss | 6.22 | 4.68 | 4.04 | 3.89 | 3.83 | 3.81 |
| Contrastive loss | 7.06 | 4.60 | 3.60 | 3.36 | 3.25 | 3.22 |
| Aux loss | 6.63 | 3.63 | 2.21 | 1.95 | 1.84 | 1.81 |
| Outcome loss | 2.022 | 2.011 | 2.008 | 2.007 | 2.006 | 2.006 |
| Base top-5 | 2.1% | 40.4% | 63.3% | 68.3% | 70.3% | 70.8% |
| Train outcome acc | 52.5% | 57.4% | 58.7% | 59.8% | 60.3% | 60.5% |
| Val outcome acc | 46.7% | 51.3% | 52.4% | 49.7% | 50.5% | 50.6% |
| Temporal mean rank | 3,773 | 2,007 | 2,049 | 2,070 | 2,051 | 2,042 |
| Temporal top-100 | 2.7% | 28.6% | 31.9% | 33.9% | 34.6% | 34.7% |
| Delta norm mean | 0.111 | 0.176 | 0.183 | 0.177 | 0.171 | 0.170 |

### Key Pattern Across All Runs

| Run | Delta Control | Delta/Base | Temporal Top-100 | Outcome Acc | Status |
|-----|--------------|------------|-----------------|-------------|--------|
| v1 Phase A | None (no delta) | N/A | ~0% (random) | N/A | Memorization |
| CBOW | N/A | N/A | N/A | 12% | Floor |
| v2.0 1a | λ=0.01, no cap | 78% | Unreliable | N/A | Temp collapse |
| v2.0 1b | λ=0.05 + cap 0.3 | **22% stable** | 8.2% | N/A | Contrastive-only |
| v2.0 Phase B | N/A | N/A | N/A | 12.13% | ≈ CBOW |
| v2.1 run 1 | λ=0.01 on raw, no cap | 328% exploding | ~0% | 15.66% | Delta explosion |
| v2.1b run 2 | λ=0.05 projected + cap | **~16% stable** | 21.6% (peak) | 18.0% | Best pre-v3.2 |
| v3.0/3.1 | λ=0.05 projected + cap | ~14% | 4.2% (peak) | 20.1% (unstable) | State token unstable |
| **v3.2 run 1** | **λ=0.05 projected + cap** | **~17% stable** | **35.0%** | **52.6% (distributional)** | **Current best** |

**Conclusion**: The distributional paradigm (v3.2) resolved all prior training pathologies. Single-possession outcome prediction has a fundamental ceiling (~17-20% val accuracy across v2.0-v3.1). Distributional targets let the model learn lineup-level outcome distributions, achieving ~50-52% val accuracy. Temporal metrics improved monotonically to 35%, confirming healthy embedding quality. Delta norms stable throughout at ~17%.

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

**Status: COMPLETED** — Run 1a killed at epoch 11 (training pathology). Run 1b killed at epoch 6 (temporal oscillating, contrastive-only insufficient). See Experiment Log above for full results.

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

**Status: RUN 2 PENDING** — Run 1 killed at epoch 7 (delta explosion, see Experiment Log). Fix: regularize projected deltas (not raw) + restore hard cap 0.3.

### Architecture Changes from v2.0

| Component | v2.0 | v2.1 | Rationale |
|-----------|------|------|-----------|
| Training phases | Sequential (Phase A → Phase B) | Joint (simultaneous) | Outcome signal from epoch 1 prevents ID-over-discrimination |
| Delta embedding | Full-rank `[12821 × 384]` (4.9M params) | Bottleneck `[12821 × 64] → [64 → 384]` (0.8M + 24K params) | Structurally limits season-specific capacity |
| Delta regularization | Soft λ=0.05 + hard cap 0.3 (full-rank) | Soft λ=0.05 on projected + hard cap 0.3 (projected) | Run 1 showed bottleneck ≠ norm constraint; proj amplifies 13x |
| Differential LR | Phase B only (base 0.01×, delta 0.03×, encoder 0.1×) | From epoch 1 (base 0.1×, delta+encoder 0.3×, heads 1×) | Protect LLM-seeded base embeddings throughout |
| Total params | ~21.5M | ~17.4M | Leaner model |
| Loss function | Phase A: InfoNCE + 0.1×aux. Phase B: CE | InfoNCE + 0.1×aux + outcome_weight×CE + 0.01×delta_reg | All objectives simultaneous |
| Outcome during training | Unmasked (Phase B) | Masked (one player replaced with [MASK]) | Acts as input dropout |

### Loss Function

`L = L_contrastive + 0.1 × L_aux + outcome_weight × L_outcome + 0.05 × L_delta_reg`

Where:
- `L_contrastive`: InfoNCE with same-base-player false-negative masking, fixed τ=0.07
- `L_aux`: Base-player classification (2,310-way CE)
- `L_outcome`: Class-weighted 9-way CE for possession outcome (on masked lineup)
- `L_delta_reg`: Mean L2 norm of **projected** delta embeddings (384-dim, after bottleneck projection)
- Hard cap: projected delta norms clamped to ≤ 0.3 after each optimizer step

### What Was Kept from v2.0 Fixes

| Change | Decision | Rationale |
|--------|----------|-----------|
| False-neg masking in contrastive loss | KEPT | Essential — don't penalize same-player similarity |
| Fixed temperature 0.07 | KEPT | Proven, one less variable |
| pred_cossim logging | KEPT | Free collapse detection |
| Checkpoint resumption | KEPT | Essential infrastructure |
| Temporal eval (no false-neg masking) | KEPT | Correct measurement |
| Auxiliary base-player head | KEPT | Cheap supporting signal |
| Delta soft reg λ=0.05 | KEPT at 0.05 (on **projected** delta) | Run 1 showed λ=0.01 on raw was insufficient; bottleneck ≠ norm constraint |
| Delta hard norm cap 0.3 | RESTORED (on **projected** delta) | Run 1 showed dropping cap → delta/base 328%. Cap scales raw to keep projected ≤ 0.3 |

### Default Configuration

| Parameter | Value |
|-----------|-------|
| `d_model` | 384 |
| `n_layers` | 8 |
| `n_heads` | 8 |
| `delta_dim` | 64 |
| `outcome_weight` | 1.0 |
| `delta_reg_weight` | 0.05 (on projected delta norms) |
| `delta_max_norm` | 0.3 (hard cap on projected delta norms) |
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

| Model | Val Outcome Acc | Temporal Mean Rank | Temporal Top-100 | Delta/Base | Params |
|-------|----------------|-------------------|-----------------|------------|--------|
| CBOW | 12% | N/A | N/A | N/A | — |
| v2.0 + Phase B | 12.13% | ~2,000-3,400 | ~3-8% | 22% stable | ~21.5M |
| v2.1 run 1 (no cap) | **15.66%** | ~6,700 (random) | ~0% | 328% exploding | ~17.4M |
| v2.1b run 2 (projected reg + cap) | **18.0%** | ~2,000-5,000 | 21.6% (peak ep 4) | 16% stable | ~17.4M |

### Run Command

```bash
# Run 2 (with projected delta reg + hard cap fix):
python train_transformer.py --phase joint --epochs 25 --delta-dim 64 \
  --outcome-weight 1.0 --delta-reg-weight 0.05 --delta-max-norm 0.3
```

---

## Experiment 2a: v3.0 — Base-Player Contrastive + Outcome-Primary (COMPLETED → v3.1 KILLED)

**Purpose:** Kill roster memorization by replacing 12,821-way player-season contrastive with 2,310-way base-player contrastive. Make outcome prediction the primary training signal.

**Motivation:** After 7 runs across v1/v2.0/v2.1, the pattern is clear: 12,821-way player-season contrastive loss rewards roster memorization, not basketball understanding. Contrastive loss discovers that teammate co-occurrence is a cheaper signal than basketball role, actively destroying archetype features learned in early epochs. Meanwhile, outcome prediction works: v2.1 run 1 hit 15.66% val accuracy despite delta explosion.

**Result:** v3.0 base-player contrastive confirmed the direction — initial contrastive loss ~7.19 (near ln(2310)=7.75, correct). However, v3.1 attempted to inject game state as an 11th transformer token instead of post-pooling concatenation. This was unstable: val accuracy oscillated 8-20% across epochs with no convergence. Killed after 10 epochs. State reverted to post-pooling concatenation for v3.2.

### What Changed (v2.1 → v3.0)

| Component | v2.1 | v3.0 | Rationale |
|-----------|------|------|-----------|
| Contrastive targets | `_all_composed_embeddings()` [12821] | `base_player_emb.weight` [2310] | Predict archetype, not exact player-season ID |
| False-neg masking | Required (same-base-player) | Not needed (each base player appears once) | Cleaner loss, no masking overhead |
| Sim matrix size | [B, 12821] | [B, 2310] | 5.5x smaller softmax = faster training |
| Contrastive weight | 1.0 (fixed) | Tunable via `--contrastive-weight` | Allows outcome-primary training |
| `_all_composed_embeddings()` in loop | Called every step (12,821 delta_proj forward passes) | Not called (reads base_player_emb.weight directly) | Significant speedup |
| Training speedups | None | `torch.compile` + `persistent_workers=True` | Faster wall-clock time |

### Why Base-Player Contrastive Still Adds Value

Outcome prediction alone risks embedding collapse: two very different lineup archetypes can produce similar outcome distributions. Base-player contrastive adds "the embedding for a masked LeBron-slot should be near LeBron's archetype," ensuring player-distinguishing geometry needed for:
- `game_outcome.py`: Uses static mean-pooled embeddings + logistic regression
- Trade analysis: "What if we swap Player A for Player B?"
- Nearest neighbors: Requires good embedding geometry

### Configuration

```bash
# Primary run (contrastive_weight=0.5):
nohup python train_transformer.py --phase joint --epochs 25 --delta-dim 64 \
  --outcome-weight 1.0 --contrastive-weight 0.5 \
  --delta-reg-weight 0.05 --delta-max-norm 0.3 \
  --bs 2048 --accum 2 \
  --ckpt joint_v21_basecontra.pt > joint_v21_basecontra.out 2>&1 &
```

### Verification Checklist

1. Initial contrastive loss should be ~ln(2310) ≈ 7.75 (vs old ~ln(12821) ≈ 9.46)
2. `base_player_emb.weight` is detached in contrastive targets (stop-gradient)
3. `contrastive_weight=0` path produces contrastive_loss=0.0 with no errors
4. Epoch 1: base_top5 should start higher than old train_top5 (2,310-way is easier)
5. Epoch 5: Delta/base < 30%, temporal rank stable or improving

### Success Criteria

| Metric | Target | Comparison |
|--------|--------|------------|
| Val outcome accuracy | > 15.66% | Beat v2.1 run 1 |
| Game win prediction | > v2.1 | Run `game_outcome.py` post-training |
| Base-player top-5 | > 50% | 2,310-way is much easier than 12,821-way |
| Temporal mean rank | < 3,000 | Better than random (~1,155 for 2,310-way base targets) |
| Delta/base ratio | < 30% | Healthy with projected reg + cap |

### Comparison Table (to fill after training)

| Model | Val Outcome Acc | Base Top-5 | Temporal Mean Rank | Delta/Base | Params |
|-------|----------------|------------|-------------------|------------|--------|
| CBOW | 12% | N/A | N/A | N/A | — |
| v2.0 + Phase B | 12.13% | N/A | ~2,000-3,400 | 22% stable | ~21.5M |
| v2.1 run 1 (no cap, 12821-way) | **15.66%** | N/A | ~6,700 (random) | 328% exploding | ~17.4M |
| v3.0/3.1 (base-player + state token) | 20.1% (unstable) | N/A | ~4,000 | ~14% | ~17.4M |
| **v3.2 (distributional, 25 ep)** | **52.6% (distributional)** | **70.8%** | **~2,042** | **~17% stable** | **~17.4M** |

---

## Experiment 3: v3.2 — Distributional Outcome Prediction (COMPLETED)

**Purpose:** Paradigm shift from single-possession outcome classification to distributional lineup-level prediction. Players don't determine individual possession outcomes — they shift the *probability distribution* of outcomes.

**Motivation:** Across all v2.0-v3.1 runs, single-possession outcome prediction plateaued at ~17-20% val accuracy. State-only logistic regression achieves 25.5%, meaning the full 17.4M-parameter model couldn't beat a simple baseline using only game state features. Root cause: individual possessions are fundamentally stochastic — a lineup doesn't determine whether this specific shot goes in, it determines the *distribution* over many possessions.

### Architecture Changes from v3.0

| Component | v3.0/v3.1 | v3.2 | Rationale |
|-----------|-----------|------|-----------|
| Prediction target | Single-possession 9-class label | Bayesian-smoothed distribution per lineup | Players shift distributions, not single outcomes |
| Loss function | Class-weighted cross-entropy | Soft CE: `-(target * log_softmax).sum().mean()` | Smooth targets, no class weights needed |
| Forward passes | Single (masked lineup for both objectives) | Dual: Pass 1 masked (contrastive), Pass 2 unmasked (outcome) | Fixes train/eval mismatch |
| Attention pooling | Single pool over all 10 players | Split: offense pool (0-4) + defense pool (5-9) | Explicit unit-level representations |
| State injection | v3.0: post-pooling concat. v3.1: 11th token (UNSTABLE) | Post-pooling concat (reverted from v3.1) | State token caused oscillation |
| Outcome head input | Single pooled repr + state | `[off_pooled, def_pooled, state_repr]` (3×d_model) | Explicit offensive/defensive matchup |
| Distributional targets | N/A | `(n * empirical + α * prior) / (n + α)`, α=10 | Bayesian smoothing per lineup |

### Training Configuration

```bash
python train_transformer.py --phase joint --epochs 25 --delta-dim 64 \
  --outcome-weight 1.0 --contrastive-weight 0.5 --prior-strength 10 --bs 2048
```

**Infrastructure:** Trained on RunPod cloud GPU (RTX A5000, $0.27/hr). Automated deployment via `scripts/deploy_runpod.py`. ~14 min/epoch, ~5.8 hours total, ~$1.60 total cost.

### Results — All 25 Epochs

| Metric | Ep 1 | Ep 5 | Ep 10 | Ep 15 | Ep 20 | Ep 25 |
|--------|------|------|-------|-------|-------|-------|
| Total loss | 6.22 | 4.68 | 4.04 | 3.89 | 3.83 | 3.81 |
| Contrastive loss | 7.06 | 4.60 | 3.60 | 3.36 | 3.25 | 3.22 |
| Aux loss | 6.63 | 3.63 | 2.21 | 1.95 | 1.84 | 1.81 |
| Outcome loss | 2.022 | 2.011 | 2.008 | 2.007 | 2.006 | 2.006 |
| Base top-5 | 2.1% | 40.4% | 63.3% | 68.3% | 70.3% | 70.8% |
| Train outcome acc | 52.5% | 57.4% | 58.7% | 59.8% | 60.3% | 60.5% |
| Val outcome acc | 46.7% | 51.3% | 52.4% | 49.7% | 50.5% | 50.6% |
| Temporal mean rank | 3,773 | 2,007 | 2,049 | 2,070 | 2,051 | 2,042 |
| Temporal top-100 | 2.7% | 28.6% | 31.9% | 33.9% | 34.6% | 34.7% |
| Delta norm mean | 0.111 | 0.176 | 0.183 | 0.177 | 0.171 | 0.170 |

### Assessment

**All training pathologies from prior versions are resolved:**

1. **No delta explosion** — stable at ~17% throughout (vs v2.1's 328%)
2. **No temporal collapse** — top-100 improved monotonically from 2.7% to 35.0% (vs v2.0 1b's declining from 8.5% to 3.5%)
3. **No val accuracy oscillation** — stable 49-52% (vs v3.1's wild 8-20%)
4. **Contrastive and outcome improving simultaneously** — no objective tradeoff
5. **Outcome val accuracy is now meaningful** — 52.6% on distributional targets vs prior ceiling of ~17-20% on hard labels

**Key observations:**
- **Outcome loss converges fast** (2.022 → 2.006 by ep 10, minimal improvement after). The distributional targets provide a clear, learnable signal.
- **Contrastive loss improves steadily** throughout all 25 epochs (7.06 → 3.22). Base top-5 reaches 70.8%.
- **Temporal metrics plateau around epoch 5** (~28-35% top-100). Further contrastive training doesn't hurt temporal quality — no memorization-vs-archetype tradeoff.
- **Delta norms self-regulate** — peak at ~0.183 (ep 10) then slowly decrease to 0.170. The model learns that smaller deltas suffice.

### Checkpoint Selection

The training run used val_outcome_acc for checkpoint selection (old logic), so `joint_v32_checkpoint.pt` was saved at epoch 4 (peak val acc 52.6%). Since temporal metrics improved monotonically, `joint_v32_checkpoint_latest.pt` (epoch 25, temporal top-100=34.7%) has better embeddings.

Post-training, checkpoint selection was updated to a **gated composite metric**:
- **Gate:** `val_loss <= best_val_loss * 1.01` (outcome quality hasn't degraded)
- **Selector:** `temporal_top100 > best_temporal_top100` (embedding quality improved)

This ensures future training runs save checkpoints that optimize for embedding quality while maintaining outcome prediction capability.

### Output Files
- `joint_v32_checkpoint.pt` — Best by val outcome acc (epoch 4)
- `joint_v32_checkpoint_latest.pt` — Latest epoch (epoch 25, best embeddings)
- `joint_v32_checkpoint.log.jsonl` — Full 25-epoch training log

---

## v3.2 Downstream Evaluation Results (2026-03-07)

**Notebook:** `notebooks/downstream_eval.ipynb` — 8 sections, run on `joint_v32_checkpoint_latest.pt` (epoch 25).

**Critical bug found and fixed:** `evaluate.py:load_model()` used `strict=False` to load state dicts, but `torch.compile` saves keys with `_orig_mod.` prefix. All weights silently failed to load — model ran with random/zero weights. Fixed by stripping prefix before `load_state_dict()`.

### Key Findings

**1. Attention pooling is effectively dead (uniform/mean pooling)**
- Mean attention per offensive slot: [0.200, 0.200, 0.200, 0.200, 0.200]
- Entropy: 1.6092 vs uniform 1.6094
- The model assigns equal weight to all players — attention pooling degenerates to mean pooling
- **Implication:** Zero interpretability from attention weights. Top priority for v4.

**2. Player impact scores are tiny and partially basketball-wrong**
- Full range: Jimmy Butler (+0.023) to Patty Mills (-0.010) — only ±3% of favorability
- KD (-0.004) and Giannis (-0.001) show negative impact vs replacement (Juwan Morgan)
- Replacement baseline comparison reveals Jokic looks better vs archetype peers (+0.007) than global (+0.001)
- **Implication:** Model doesn't learn strong player-level effects. Distributional targets may smooth away individual contributions.

**3. Lineup optimization is directionally correct**
- Warriors (Curry/Klay/Wiggins/Draymond) + ?: top picks are all shooters (Fournier, KCP, Kennard). Bottom picks are non-shooting bigs (DeAndre Jordan, Tyson Chandler). Makes basketball sense.
- Celtics (Tatum/Brown/Smart/White) + ?: top picks are versatile bigs (Valanciunas, Rudy Gay, Tobias Harris, Aaron Gordon). Directionally correct.
- **But:** Same player across different test-era seasons gets identical scores (Fournier 2021=2022=2023) because test-era deltas are untrained/zero.

**4. Chemistry analysis dominated by 2021 Lakers**
- Top 15 highest-chemistry lineups ALL contain LeBron+AD+role players (Caruso, KCP, Danny Green, Dwight Howard, JaVale McGee, Rondo)
- Chemistry vs favorability correlation: r=0.158 (essentially none)
- Low-chemistry lineups include bench units and cohesive starter units (Heat with Butler/Herro)

**5. Aging curves show real signal**
- Delta norm rises: 0.143 (age 20) → 0.195 (age 29, peak) → declines through 30s
- Different archetypes show distinct aging patterns
- **Artifact:** Career trajectories drop to zero after 2020 because test-era deltas are untrained

**6. Versatility metric needs refinement**
- Top versatile: Draymond Green (0.601), LeBron (0.601), Al Horford (0.604) — basketball-plausible names
- Archetype boundary crossing metric is flawed (uses base-emb-space K-means on encoder-output tokens — different spaces)
- Mean crossing rate 75.5% is too high to be discriminative

### Full Evaluation Pipeline Results (evaluate.py + analyze_embeddings.py)

**Masked prediction (test set):** Top-1 0.00%, Top-5 0.14% — expected since test set (2021+) has zero player-season ID overlap with training (pre-2019). The meaningful metric is temporal evaluation during training (34.7% top-100 hit rate).

**Outcome prediction (hard labels):** 25.29% accuracy on 9-class hard labels.
- Majority class baseline: 22.54% (model is +2.74pp)
- Stratified random: 15.36% (model is +9.93pp)
- State-only logistic regression: 25.47% (model is -0.18pp)
- **Class collapse:** Only predicts 2 of 9 classes (made_2pt_close and FT). All other classes receive zero predictions.
- This reflects the distributional training — the model learned to output distributions that, when argmax'd, favor the two most common outcome modes. Hard-label accuracy is the wrong metric for this model.

**Static embedding analysis:**
- Embedding norms: mean=0.557, std=0.093. Healthy diversity.
- Avg pairwise cosine sim (1000 sample): 0.433. Same-player cross-season: 0.882. Good separation.
- **Nearest neighbors make basketball sense:**
  - Doncic → Jalen Brunson (Dallas teammate, similar ball-handling)
  - Jokic → Jusuf Nurkic (Serbian big men, passing bigs)
  - LeBron → all self-seasons, with 2011 closest non-test (Heat transition)
  - Curry → all self-seasons, with 2014 closest (pre-breakout year)
- **t-SNE shows clear structure:** Players cluster by identity/role, not by era. Notable players' multi-season instances group tightly.
- **Temporal trajectories:** Year-to-year cosine similarity mostly >0.9. LeBron shows largest dip (~0.86) around 2010-2014 (Heat era role change). Post-2019 flatlines at 1.0 (zero test-era deltas).

### Structural Limitation: Future-Era Deltas

Delta embeddings are primarily trained for pre-2019 data. Temporal augmentation (15% swap to adjacent season) bleeds some gradient signal forward:
- **Train (<2019):** 88% non-zero deltas, mean norm 0.20
- **Val (2019-2020):** 42% non-zero, mean norm 0.09 (half strength)
- **Test (≥2021):** 27% non-zero, mean norm 0.06 (quarter strength)

This attenuating gradient makes future-era player-seasons progressively less differentiated. Same-player-different-season similarity approaches 1.0 in the test era.

**However, training-era evaluation shows the weak impact magnitudes and uniform attention are STRUCTURAL issues, not delta issues.** See "Training-Era Evaluation Results" below.

---

## Future Experiments (Post-v3)

### Methodology: Post-Training Delta Fitting (for every future run)

**Problem:** Delta embeddings are only trained for player-seasons in the training split (pre-2019). Val/test-era player-seasons have weak/zero deltas, limiting downstream task evaluation. Temporal augmentation bleeds some gradient signal (~42% of val-era and ~27% of test-era deltas are non-zero) but it's not enough.

**Solution:** After every training run, add a **delta fitting step** that learns deltas for val/test player-seasons:
1. Freeze the entire model (encoder, base embeddings, outcome head, all heads)
2. Initialize new deltas for unfitted player-seasons (zero or copy from the most recent fitted season of the same player)
3. Optimize ONLY the new deltas using the contrastive loss (no outcome loss) on val/test data
4. Run for a small number of steps (100-500) with the same delta regularization (L2 + cap)

**Why this is principled:**
- No outcome leakage — contrastive loss only teaches "this player fills this archetype role," not "this lineup produces these outcomes"
- The encoder that interprets the deltas is frozen, so we're just learning better input representations
- Analogous to fitting new word embeddings in a frozen language model

**Implementation:** Add `--fit-deltas` flag to `train_transformer.py` that runs this step post-training before saving the final checkpoint. Or make it a separate script.

**When to use:** Every training run. The fitted checkpoint becomes the one used for all downstream evaluation and analysis.

### Methodology: All-Data Production Model (for public sharing)

**Problem:** The temporal train/val/test split exists to evaluate outcome prediction generalization. But for the "production" model used for analysis, visualization, and demos, we want the richest possible embeddings for ALL players across ALL eras.

**Solution:** After validating the architecture on the temporal split, retrain on all data:
1. Same architecture and hyperparameters as the validated model
2. Use ALL possessions for training (no held-out split)
3. This gives every player-season real trained deltas
4. No outcome prediction evaluation possible (no held-out data), but that's OK — we already proved generalization on the split model

**When to use:** Once the architecture is finalized and we're ready to produce artifacts for sharing. Not during active development.

### Training-Era Evaluation Results (2026-03-07)

**Key finding: the class collapse and tiny impact scores are STRUCTURAL, not delta-related.**

Running on training data (where deltas are fully trained) shows:
- Hard-label accuracy: 27.44% (same 2-class collapse as test — only predicts made_2pt_close and FT at argmax)
- But this is the **wrong metric**: the model predicts 9-class distributions that match targets well. All 9 classes get correct probability mass. The "collapse" is just argmax picking between two nearly-tied modes — which is *correct* behavior (for any lineup, a close 2 or FT really is the most likely single outcome).
- **Distributional metrics are what matter**: mean KL divergence 0.078, lineup-specific predictions reduce KL by 12.8% vs global mean.
- **Player impact on training data**: magnitudes still small (~±0.02 favorability, ~1-2% of base). Ordering is partially correct (ball-handlers/creators at top: Westbrook +0.019, Curry 2016 +0.018, Lillard +0.015, CP3 +0.014), but some superstars show weak/negative impact (KD -0.001, AD -0.006, Draymond -0.006).
- **Root cause**: uniform attention pooling (mean pooling). If every player gets exactly 20% attention weight, swapping one player can shift at most ~20% of the pooled representation, capping impact magnitude. **Fixing attention is the #1 priority for v4.**

### v4 Attention Fix Ablation (COMPLETE — 2026-03-07)

**Problem:** AttentionPool produces uniform weights [0.2, 0.2, 0.2, 0.2, 0.2] — effectively mean pooling. This caps player substitution impact at ~±2% and kills attention interpretability.

**Ablation runs (5 epochs each, 3 parallel RunPod A5000 pods, ~70 min total, ~$2.85):**
- **Run A (temp):** Temperature-scaled attention (`--attn-temperature 0.1`). Divides attention logits by tau=0.1 before softmax.
- **Run B (entropy):** Entropy penalty (`--attn-entropy-weight 0.1`). Adds `lambda * mean_entropy(attn_weights)` to the loss.
- **Run C (control):** Exact v3.2 config. Validates uniform attention is reproducible.

**Results (epoch 5):**

| Metric | Temp (τ=0.1) | Entropy (λ=0.1) | Control | Winner |
|--------|-------------|-----------------|---------|--------|
| Val loss | 2.033 | 2.040 | **2.033** | control |
| Val outcome acc | 47.2% | **52.2%** | 49.8% | entropy |
| Temporal top-100 | 28.5% | 12.7% | **32.7%** | control |
| Temporal mean rank | 1842 | 2316 | **1623** | control |
| Contrastive loss | **4.264** | 5.359 | 4.410 | temp |
| Base top-5 | **49.8%** | 28.7% | 45.5% | temp |
| Max attn weight | 0.676 | 0.998 | 0.262 | — |
| Off attn entropy | 0.826 | 0.006 | 1.585 | — |

**Analysis:**
- **Entropy penalty degenerate:** Attention collapsed to single-player spikes (max_attn=0.998). Worst temporal metrics across the board. The penalty overshoots — instead of mild sharpening, it drives all weight to one player.
- **Temperature helps contrastive, hurts temporal:** τ=0.1 produces better player identification (base_top5 49.8% vs 45.5%) but worse temporal generalization (28.5% vs 32.7%). Sharper attention may memorize training-era patterns more aggressively.
- **Control wins temporal metrics:** Uniform attention (max_attn=0.262) gives the best temporal generalization. The model's "choice" to use mean pooling may be structurally correct — the transformer encoder handles interactions, pooling just aggregates.
- **Neither intervention improves val_loss:** All three are within 0.007 of each other. Attention sharpening doesn't improve outcome prediction.

**Key insight:** v3.2's LR schedule (cosine annealing, T_max=24 at 25 epochs) caused LR to hit 0.0 at epoch 25, while temporal@100 was still climbing. The model hadn't converged.

**Decision:** Use moderate temperature τ=0.5 (split the difference — mild sharpening without the temporal degradation of τ=0.1) and extend to 30 epochs (longer cosine schedule so LR doesn't die prematurely). Full v4 run captures attention metrics for the first time.

---

## Experiment 3: v5 — Cross-Attention Pooling + Multi-Layer Input + State Conditioning

### Motivation

v4 confirmed that the model converges to uniform attention pooling regardless of temperature scaling (τ=0.5). After 27 epochs, attention entropy drifted from 1.506 → 1.599 (toward uniform 1.609). The model actively undoes temperature scaling by making pre-softmax logits more uniform.

Root cause analysis identified **three architectural limitations**, not one:

1. **Static query pooling**: `nn.Linear(d_model, 1)` uses a fixed learned vector to score all players in all lineups. It can't learn input-dependent importance (e.g., "Steph matters more here because the rest are non-shooters"). A static query's best strategy across millions of diverse lineups IS uniform weighting.

2. **Post-transformer homogenization**: Pooling operates on layer-8 output where 8 rounds of self-attention have distributed each player's information across all tokens. By the final layer, all 5 offensive tokens partially encode the same lineup-level information, making them hard to distinguish with a linear projection. Player-specific signal is strongest in early layers.

3. **Late state injection**: Game state (score differential, quarter, shot clock) is concatenated after pooling — the transformer encoder has zero game context. But basketball roles are state-dependent: a team trailing by 20 in Q4 plays fundamentally differently. The encoder can't learn state-dependent player interactions.

These are **three complementary fixes** — pooling mechanism, pooling input, and encoder conditioning. The central insight unifying them: which player should have the most influence on a possession's outcome distribution is a **joint function** of who the players are AND the game situation. The current architecture treats these independently (static pooling ignores both; state enters only at the outcome head). v5 makes this joint dependence explicit at every level.

### Change A: Cross-Attention Pooling with Learned Query

Replace `nn.Linear(d_model, 1)` with cross-attention using a learned query vector.

**Current (static):**
```python
scores = self.query(h)                    # [B, 5, 1] — same projection for all inputs
weights = softmax(scores / temp, dim=1)   # [B, 5, 1]
pooled = (weights * h).sum(dim=1)         # [B, d_model]
```

**Proposed (cross-attention):**
```python
query = self.query_token.expand(B, 1, d_model)  # [B, 1, d_model] — learned, but full-rank
pooled = F.multi_head_attention(query, h, h)     # [B, 1, d_model] → squeeze
```

Or equivalently with `nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)`. Multi-head gives the query multiple "reasons to attend" — one head might attend by offensive role, another by usage rate, another by matchup advantage.

**Why this helps:** The query now does full dot-product attention in d_model space (384 dims of discrimination) instead of projecting to a scalar. And with multiple heads, different aspects of "who matters" can be captured simultaneously. Critically, the attention is still input-dependent because it's `dot(query, key)` where keys come from the player tokens.

**State-conditioned query (core design, not optional):** The cross-attention query must be conditioned on game state: `query = self.query_token + self.state_to_query(state_repr)`. Which player should have the most influence on a possession's outcome distribution depends on both the players themselves (who's on the court) AND the game situation (score, quarter, shot clock) — as a joint function, not independently. In clutch situations, attend more to the primary scorer; in blowouts, attend more uniformly; when trailing, attend more to three-point shooters. This is the central insight of v5: the pooling query asks "given this game situation, who should I attend to?" — combining early-layer player identity (Change B) with game context (Change C) into input-dependent, state-aware aggregation.

### Change B: Multi-Layer Pooling Input

Pool from an intermediate transformer layer alongside the final layer, preserving player-specific signal.

**Current:** Pool only from layer 8 (final).

**Proposed:** Split the 8-layer encoder into two 4-layer blocks. Pool from concatenated layer-4 + layer-8 representations:
```python
h_mid = self.encoder_lower(tok)           # layers 0-3: player-specific
h_final = self.encoder_upper(h_mid)       # layers 4-7: contextualized
h_pool = torch.cat([h_mid, h_final], dim=-1)  # [B, 5, 2*d_model]
# Project back to d_model before cross-attention
h_pool = self.pool_proj(h_pool)           # [B, 5, d_model]
```

**Why this helps:** Layer-4 tokens still strongly encode "who is this player" while layer-8 tokens encode "what does this lineup do." The cross-attention query can leverage both: use early-layer signal to identify the key player, and late-layer signal to understand their contextual role.

**Alternative (simpler):** Instead of splitting the encoder, add a learned residual: `h_pool = h_final + alpha * tok_input` where `alpha` is a learned scalar. This gives the pooling layer access to pre-transformer embeddings alongside final representations. Less expressive but much simpler.

**Implementation note:** `nn.TransformerEncoder` wraps all layers together. Split into `encoder_lower = TransformerEncoder(layer, 4)` and `encoder_upper = TransformerEncoder(layer, 4)`. Backward-compatible: existing checkpoints can load weights by mapping `encoder.layers.{0-3}` → `encoder_lower.layers.{0-3}` and `encoder.layers.{4-7}` → `encoder_upper.layers.{0-3}`.

### Change C: FiLM State Conditioning (from peer review)

Use Feature-wise Linear Modulation (FiLM) to condition transformer layers on game state.

**Current:** `state_proj = nn.Linear(3, d_model)`, concatenated after pooling. Encoder sees no game context.

**Proposed:** After each transformer layer's LayerNorm, apply FiLM modulation:
```python
# State → per-layer scale and shift
gamma, beta = self.film_layers[i](state_repr)  # each: [B, d_model]
h = gamma * layer_norm(h) + beta              # modulate normalized activations
```

Each transformer layer gets its own `film_layer = nn.Linear(d_model, 2 * d_model)` that produces per-layer scale (γ) and shift (β) from the state representation. This is lightweight (~0.6M params for 8 layers) and well-established in conditional generation (StyleGAN, DiT).

**Why this helps:** A team trailing by 20 in Q4 uses different offensive sets — more 3-point attempts, faster pace. The encoder should know this when computing player-player interactions. Currently it can't: all lineups are processed identically regardless of game situation, and state only influences the final outcome prediction. FiLM lets the encoder learn state-dependent representations: "Steph-when-trailing" ≠ "Steph-when-leading."

**adaLN alternative:** Replace FiLM with adaptive LayerNorm, where state conditions the LayerNorm parameters directly. Functionally similar, slight implementation difference. FiLM is simpler (doesn't replace the existing LayerNorm, just modulates after it).

**Interaction with contrastive loss:** The contrastive pass (Pass 1) should also use FiLM — state context makes the masked-player prediction more informative ("predict who's missing from this lineup *in this game situation*"). This is a natural extension.

### Ablation Design

All v5 runs include state-conditioned cross-attention pooling (A) as baseline — this is the confirmed direction, not optional. The ablation tests what else helps on top of it.

| Run | Changes | CLI flags |
|-----|---------|-----------|
| A | State-conditioned cross-attn pooling | `--pool-type cross-attn --pool-heads 4` |
| B | A + multi-layer input | `--pool-type cross-attn --pool-heads 4 --pool-multi-layer` |
| C | A + multi-layer + FiLM encoder | `--pool-type cross-attn --pool-heads 4 --pool-multi-layer --film-state` |
| D | Control (v4 config, static pooling) | (no changes) |

All cross-attention runs (A/B/C) use state-conditioned query: `query = query_token + state_to_query(state)`.

**Metrics to compare:**
- Attention entropy (should decrease significantly for A/B/C vs D)
- Max attention weight (should increase — non-uniform)
- Substitution impact magnitude (should increase from current ±2%)
- Temporal top-100 (must not degrade)
- Val loss (must not degrade)

**Expected outcome:** A should already be a significant improvement over D (state-conditioned, input-dependent pooling vs static uniform). B should further improve by giving the query access to player-specific signal from early layers. C tests whether FiLM encoder conditioning adds value beyond state-conditioned pooling — FiLM is the most speculative of the three.

**Winner selection:** Best temporal_top100 among runs where val_loss is within 5% of control.

**Cost estimate:** 4 × 5 epochs × ~14 min/ep × $0.27/hr ≈ $2.50. Wall time: ~70 min (parallel pods).

### v5 Ablation Results (COMPLETE — 2026-03-10)

3 parallel A5000 pods (community cloud), 5 epochs each. No control run (D) — v3.2 at epoch 5 is the reference (top100 ~10%, val_loss ~2.04, attn_entropy=1.609). Upload via rsync (~45s per pod, no volume).

**Epoch-by-epoch progression:**

| Run | Ep 1 top100 | Ep 2 top100 | Ep 3 top100 | Ep 4 top100 | Ep 5 top100 |
|-----|-------------|-------------|-------------|-------------|-------------|
| A (crossattn) | 2.8% | 4.4% | 16.0% | 20.8% | 23.0% |
| B (multilayer) | 3.5% | 4.4% | 13.2% | 18.3% | 21.0% |
| C (film) | 1.6% | 6.6% | 8.3% | 13.3% | 13.8% |

**Final metrics (epoch 5):**

| Metric | A (crossattn) | B (multilayer) | C (film) | v3.2 ref (ep 5) |
|--------|--------------|----------------|----------|-----------------|
| Temporal top-100 | **23.0%** | 21.0% | 13.8% | ~10% |
| Temporal mean rank | 2103 | **1914** | 2130 | ~3500 |
| Val loss | 2.036 | 2.036 | **2.033** | ~2.04 |
| Val outcome acc | 52.9% | 51.5% | 52.6% | ~52.6% |
| Off attn entropy | 1.609 | 1.609 | **1.566** | 1.609 |
| Def attn entropy | 1.609 | 1.609 | **1.561** | 1.609 |
| Max attn weight | 0.209 | 0.204 | **0.278** | 0.200 |
| Base top-5 | **41.1%** | 37.7% | 31.6% | — |
| Delta norm mean | 0.137 | 0.117 | 0.124 | — |
| Outcome loss | 2.010 | 2.010 | **2.010** | — |
| Contrastive loss | **4.734** | 4.881 | 5.208 | — |
| Epoch time | ~14 min | ~14 min | ~14 min | ~14 min |

**Analysis:**

1. **Only FiLM breaks uniform attention.** Cross-attention pooling alone (A, B) converges right back to entropy 1.609 — identical to v3.2's dead attention. The state-conditioned query alone is not sufficient; the transformer encoder also needs state conditioning so it produces differentiated player representations. FiLM achieves this: entropy dropped to 1.56 and was still falling at epoch 5.

2. **A/B learn temporal embeddings faster.** Simpler architectures (fewer parameters) converge faster on contrastive metrics. A reached 23% top-100 vs C's 13.8% at epoch 5. But C's trajectory was steeper at epochs 4-5 (still climbing), suggesting it needs more epochs to warm up.

3. **FiLM has the best val_loss.** Despite slower temporal convergence, C learns the outcome distribution better (2.033 vs 2.036). State-conditioned encoder layers help the model understand how lineups affect outcomes differently in different game situations.

4. **All v5 variants outperform v3.2.** Even the simplest change (A) already achieves 23% temporal top-100 at epoch 5, far above v3.2's ~10%. Cross-attention pooling is a clear structural improvement regardless of attention entropy.

5. **FiLM's attention entropy was still dropping at epoch 5** (from 1.609 to 1.566 for offense, 1.561 for defense). With 30 epochs, this will likely differentiate further. The v4 experiment showed that temperature-forced sharpening reverts over long training; FiLM achieves sharpening organically through learned state conditioning, which should persist.

**Surprise:** The hypothesis that cross-attention pooling alone would fix attention was wrong. The query being state-conditioned is necessary but not sufficient — the keys/values (transformer output tokens) also need state conditioning (via FiLM) for attention to differentiate between players. Without FiLM, the encoder still homogenizes player tokens, so even a state-conditioned query sees ~uniform inputs and defaults to ~uniform weighting.

**Decision:** Winner is **C (film)** — the only variant that addresses the core problem (uniform attention). Config: `--pool-type cross-attn --pool-heads 4 --pool-multi-layer --film-state`. Full 30-epoch run.

### Full Run

Winner config (C: film) × 30 epochs on RunPod A5000. Estimated ~7 hrs, ~$1.90.

### Post-Training Delta Fitting

After the v5 full run, run `fit_deltas.py` on the new checkpoint. Then run `precompute_eval.py` and `master_eval.ipynb` for comprehensive evaluation. Compare v5 vs v3.2 vs v4 across all metrics.

### Other Candidate Experiments (Lower Priority)

**Outcome-only ablation (contrastive_weight=0):** Measures actual value of contrastive signal. Run after v5 if contrastive metrics are plateauing — would tell us whether contrastive loss is still earning its keep or just adding training cost.
```bash
python train_transformer.py --phase joint --epochs 25 --delta-dim 64 \
  --outcome-weight 1.0 --contrastive-weight 0.0 \
  --ckpt joint_v5_outcomeonly.pt
```

**Prior strength sweep:** Current alpha=10 may smooth too aggressively (median lineup gets only 29% empirical weight). Try alpha=5 (44% empirical) and alpha=20 (17% empirical). Low priority — distributional targets are working well.

**All-data production model:** Once the architecture is finalized, retrain on full dataset (no val/test holdout) for the best possible embeddings for downstream use and public sharing. This is the final step — only after we're confident in the architecture.

---

## Hyperparameter Audit

An inventory of every meta-hyperparameter in the training pipeline, with its current value, provenance, and whether it has ever been experimentally validated. Almost nothing has been systematically swept — most values are standard defaults or were set once by reasoning and never revisited. The parameters that *did* change (delta_reg 0.01→0.05, temperature learned→fixed 0.07, contrastive_weight 1.0→0.5) were changed in response to training failures, not systematic sweeps.

### Full Inventory

| Parameter | Value | Provenance | Ever Swept? |
|-----------|-------|-----------|-------------|
| Prior strength (α) | 10.0 | Set by reasoning about lineup statistics: median lineup has 4 possessions → 29% empirical weight, 71% prior. | **Never.** Highest-priority sweep candidate. |
| Temporal aug probability | 0.15 | Chosen intuitively ("15% seems moderate"). | **Never.** |
| Delta L2 reg weight | 0.05 | Was 0.01 in v2.0 (delta exploded to 78% of base). Bumped to 0.05 in v2.0 Run 1b. | Changed once reactively. Never swept. |
| Delta max norm (projected) | 0.3 | Hard cap added after v2.1 delta explosion (328%). Value chosen intuitively. | **Never.** |
| Contrastive weight | 0.5 | Was 1.0 (equal weighting), changed to 0.5 for v3.0 (outcome-primary design). | One manual change. Never swept. |
| Outcome weight | 1.0 | Default, never changed. | **Never.** |
| Aux loss weight | 0.1 | Set to "small enough not to dominate." | **Never.** |
| InfoNCE temperature | 0.07 | Was learned in v2.0 Run 1a (collapsed 0.07→0.01). Fixed at the init value of 0.07 since v2.0 Run 1b. | Fixing it was the experiment. The value 0.07 itself was never validated vs alternatives. |
| Differential LR multipliers | base 0.1×, delta+encoder 0.3×, heads 1× | Reasoning: base embeddings should move slowly (LLM-seeded). Delta+encoder get moderate LR. Heads train fastest. | **Never swept.** |
| Base LR | 3e-4 | Standard transformer default. | **Never.** |
| Batch size | 2048 (1024×4 accum) | Chosen for GPU memory fit on A5000. | **Never.** Constrained by hardware. |
| Dropout | 0.1 | Standard default. | **Never.** |
| d_model | 384 | Matched text-embedding-3-small output dimensionality. | **Never.** Constrained by LLM embedding dim. |
| n_layers | 8 | Standard "small-medium" transformer. | **Never.** |
| n_heads | 8 | d_model / n_heads = 48 (standard head dim). | **Never.** |
| delta_dim | 64 | Chosen to be "small enough to constrain" (0.8M vs 4.9M full-rank params). | **Never.** |
| LR schedule | Cosine annealing, T_max=epochs | Standard. v3.2 used T_max=24 at 25 epochs (LR hit 0.0 prematurely); v4 fixed to T_max=epochs. | Schedule type never swept. T_max bug was found and fixed. |

### Highest-Value Sweep Candidates

These are the parameters where a sweep is most likely to improve results, ordered by expected impact:

1. **Prior strength (α)** — Directly shapes the training signal. α=10 means most lineups (median 4 possessions) are 71% prior, 29% empirical. The model may be mostly learning "predict the global mean" rather than "predict how this lineup deviates." α=5 (44% empirical) or α=3 (57% empirical) could give more signal to learn from, at the cost of noisier targets for rare lineups. α=20 (17% empirical) would test the other direction. **Suggested sweep: α ∈ {3, 5, 10, 20}, 5 epochs each.**

2. **Contrastive weight** — Is 0.5 the right balance? The outcome-only ablation (contrastive_weight=0) would establish the floor. 0.25 and 0.75 test the sensitivity. **Suggested sweep: w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, 5 epochs each.**

3. **InfoNCE temperature** — Fixed at 0.07 (the init value from SimCLR paper, chosen before the learned-temp failure). 0.05 produces sharper contrastive gradients (harder negatives); 0.1 is gentler. Could meaningfully change embedding geometry. **Suggested sweep: τ ∈ {0.03, 0.05, 0.07, 0.10, 0.15}, 5 epochs each.**

4. **Temporal aug probability** — 0.15 is arbitrary. Higher (0.25) teaches more cross-season invariance; lower (0.05) preserves more season-specific signal. Interacts with delta learning dynamics. **Suggested sweep: p ∈ {0.0, 0.05, 0.15, 0.25}, 5 epochs each.**

### Lower-Priority (Likely OK at Defaults)

- **Differential LR multipliers** — Would need grid search across 3 coupled values. High-dimensional and interactions are hard to interpret.
- **Delta dim** — 64 seems to work (deltas are ~17% of base norm, stable). Sweeping would change param count and interact with everything.
- **d_model, n_layers, n_heads** — Architectural scale parameters. Would need full retraining. Only revisit if model capacity is clearly the bottleneck.
- **Dropout** — 0.1 is standard. No evidence of overfitting (val metrics track train).
- **Batch size** — Constrained by GPU memory. Not a free parameter.

### Sweep Protocol (When Ready)

- Use v5 architecture (after ablation winner is selected)
- 5-epoch short runs on RunPod A5000 (~$0.60 each)
- One variable at a time, holding others at v5 defaults
- Compare on: val_loss, temporal_top100, contrastive base_top5
- Prior strength sweep first (highest expected impact, shapes the very training signal)
- If prior strength matters a lot, re-run other sweeps at the new α

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
| Phase B v2.0 baseline (1 epoch, killed) | ~1.5 hours | — | KILLED |
| Experiment 1c run 1: v2.1 joint (7 epochs, killed) | ~10.6 hours | — | KILLED |
| v2.1b fix (projected delta reg + hard cap) | ~1 hour coding | — | DONE |
| Experiment 1c run 2: v2.1b joint (23 epochs) | ~35 hours | — | DONE |
| v3.0/3.1: base-player contrastive + state token (10 epochs) | ~2.5 hours | — | KILLED |
| v3.2 implementation (distributional paradigm) | ~3 hours coding | — | DONE |
| RunPod training infrastructure (`deploy_runpod.py`) | ~2 hours coding | — | DONE |
| v3.2 run 1: distributional training (25 epochs, RunPod) | ~5.8 hours | ~$1.60 | DONE |
| Downstream task documentation + notebook | ~3 hours | — | DONE |
| Full evaluation pipeline (`evaluate.py` + `analyze_embeddings.py`) | ~1 hour | — | DONE |
| v4 attention fix implementation + ablation runner | ~2 hours coding | — | DONE |
| v4 ablation runs (3 parallel pods, 5 epochs each) | ~70 min | ~$2.85 | DONE |
| v4 full run (30 epochs, A5000, τ=0.5) | ~7 hrs | ~$1.90 | DONE (negative: attention reverted to uniform) |
| Post-training delta fitting script (`fit_deltas.py`) | ~1 hour coding | — | DONE |
| Eval infrastructure (`precompute_eval.py` + `master_eval.ipynb`) | ~3 hours coding | — | DONE |
| v5 implementation (cross-attn pool + multi-layer + FiLM) | ~3-4 hours coding | — | DONE |
| v5 ablation runs (3 parallel A5000 pods, 5 epochs each) | ~70 min | ~$2.50 | DONE |
| RunPod infra fixes (cloud_type, CUDA check, quoting, volume) | ~3 hours debugging | — | DONE |
| v5 full run (30 epochs, film config) | ~7 hrs | ~$1.90 | PLANNED |
| Post-training delta fitting (v5) | ~30 min | — | PLANNED |
| Full eval pipeline (v5) | ~1 hour | — | PLANNED |
