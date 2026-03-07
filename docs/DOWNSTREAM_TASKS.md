# Downstream Tasks

Unicorn produces two levels of representation:

- **Static composed embeddings** (12,821 x 384): `base[player] + delta[player_season]`, extractable from checkpoint weights without running the model. Each player-season has a fixed vector.
- **Contextual representations**: Transformer output where each player's representation is conditioned on their 9 teammates and the game state. LeBron-next-to-Kyrie != LeBron-next-to-Westbrook.

Static tasks validate the foundation. Contextual tasks are the unique value proposition — they're why we built a transformer instead of a simpler embedding method.

---

## Tier 1 — Core Value

### 1. Player Similarity & Nearest Neighbors

**Static embeddings**

> "Who plays like young LeBron?"

**Why it matters:** The simplest sanity check for embedding quality. If nearest neighbors don't make basketball sense, nothing else matters. Also the most immediately interpretable and narratively compelling output. Prime candidate for rich visualizations (UMAP/t-SNE colored by position or outcome probability shifts).

**Embedding property tested:** Functional role and play style are encoded in embedding geometry. Similar players should be close in cosine similarity.

**Approach:** Cosine similarity search over composed embeddings. Query any player-season, return ranked neighbors. Cross-era queries (e.g., "who in the modern NBA plays like 2005 Steve Nash?") test whether embeddings capture role independent of era.

**Status:** Implemented in `analyze_embeddings.py:find_nearest_neighbors()` and `evaluate_embeddings.ipynb` (Section 2).

---

### 2. Archetype Clustering

**Static embeddings**

> "What natural player types exist in the league?"

**Why it matters:** Validates that embedding geometry has meaningful structure. Useful for scouting, front office analysis, and storytelling. Reveals whether the model discovers archetypes that match basketball intuition (e.g., heliocentric creator, stretch big, 3-and-D wing, rim-running center).

**Embedding property tested:** Embeddings should naturally cluster by functional role, not by superficial stat similarity or era.

**Approach:** K-means or hierarchical clustering on base-player embeddings (2,310 x 384). UMAP/t-SNE for visualization. Compare learned clusters to text-embedding-prior clusters (NMI) to measure how much training reorganized the space.

**Status:** Implemented in `evaluate_embeddings.ipynb` (Section 8) and `explore_text_embeddings.ipynb` (Section 7).

---

### 3. Player Development Trajectories

**Static embeddings**

> "How did this player's role evolve over time?"

**Why it matters:** Leverages the player-season embedding design — each (player, year) has its own composed vector. Produces compelling career narratives and tests whether the delta embeddings capture meaningful year-to-year change.

**Embedding property tested:** Adjacent seasons should be similar (temporal coherence) but players who reinvented their game should show meaningful drift. The delta component should capture real evolution, not noise.

**Approach:** Track composed embedding trajectories across seasons for long-career players. Measure year-to-year cosine similarity and cumulative drift from rookie season.

**Key stories to validate:**
- Brook Lopez: post center -> stretch rim protector (large drift expected)
- Blake Griffin: athletic finisher -> point forward (gradual drift)
- LeBron: remarkably stable archetype across 20 years (small drift)
- Steph Curry: pre-2015 good shooter -> 2015+ gravity-warping force

**Status:** Implemented in `analyze_embeddings.py:plot_temporal_trajectories()` and `evaluate_embeddings.ipynb` (Section 5).

---

### 4. Substitution Sensitivity

**Contextual representations**

> "How much does swapping player X change this lineup's predicted outcome distribution?"

**Why it matters:** Measures how sensitive the model's predictions are to individual player substitutions. A player with high substitution sensitivity meaningfully shifts the predicted outcome distribution when swapped with a replacement-level player. This validates that the model has learned player-level effects, not just lineup-level noise. Note: this measures model sensitivity, not causal impact — the model's opinion of a player's value, which may or may not align with ground truth.

**Embedding property tested:** Embeddings must encode basketball impact, not just player identity. The outcome head must be sensitive to player substitutions.

**Prerequisite — Replacement Value Definition:** Before measuring impact, we must rigorously define the "replacement level" baseline. Candidate definitions:
- **Global centroid:** Base embedding closest to the centroid of all base embeddings (current approach)
- **Low-minute centroid:** Centroid of players with below-median minutes — captures "end of bench" quality
- **Archetype-specific centroid:** For each player, use the centroid of their archetype cluster — measures impact *relative to role peers*
- **Zero token:** A learned or fixed zero-value embedding

The choice of baseline affects all impact scores downstream. The notebook compares multiple definitions and validates which produces the most basketball-meaningful rankings.

**Approach:** For a given player, find all lineups where they appear on offense. Run `forward_finetune` to get the predicted outcome distribution. Replace the player with the replacement-level token and re-run. The delta distribution is their impact signature. The scalar impact score is the change in lineup favorability (expected points).

**Key insight:** This uses the full model pipeline (embeddings -> transformer -> attention pooling -> outcome head), not just the static embeddings. It captures how a player interacts with specific teammates.

**Status:** `scripts/precompute_eval.py:compute_substitution_sensitivity()`, `notebooks/master_eval.ipynb` Section 5.

---

### 5. Lineup Optimization & Counterfactuals

**Contextual representations**

> "Who best complements this 4-man unit?" / "What if we swap player X for player Y?"

**Why it matters:** Direct test of whether the model captures player interaction effects — spacing, synergy, role complementarity. High real-world relevance for roster construction, trade analysis, and lineup decisions.

**Embedding property tested:** Embeddings + transformer must encode how players interact, not just individual quality. The best 5th player should be the one who complements the unit, not simply the "best" player overall.

**Approach:**
- **Offensive optimization:** Fix 4 offensive players + 5 defenders + game state. Score every eligible player-season for the open slot via `forward_finetune`. Rank by lineup favorability.
- **Defensive matchup optimization:** Fix an elite offensive unit, then optimize the 5 defensive slots. "Which archetype of defender actually minimizes the expected points of this specific opposing unit?" This is the natural defensive complement to offensive lineup optimization.
- **Counterfactuals:** Take a real lineup, swap one player for another, compare predicted outcome distributions. Famous what-ifs: "What if Harden stayed in OKC?" "What if the Celtics kept Kyrie?"

**Status:** `notebooks/downstream_eval.ipynb` Section 3.

---

### 6. Superstar Ecosystem Modeling

**Contextual representations**

> "What types of players maximize a superstar's lineup impact?"

**Why it matters:** Goes beyond "who is the best 5th player" to "what archetype of player fits best." Produces actionable roster-building insights and compelling narratives ("LeBron needs sharpshooters, Jokic needs cutters"). This is front-office gold.

**Embedding property tested:** Embeddings must encode role archetype in a way that generalizes across specific players. The model should be sensitive to archetype-level complementarity, not just player-level.

**Approach:** Fix a superstar in one slot, cluster remaining players by archetype (K-means on base embeddings, K=8). For each archetype in each remaining slot, sample players and compute average lineup favorability. Produces a superstar x archetype -> favorability matrix.

**Status:** `notebooks/downstream_eval.ipynb` Section 4.

---

## Tier 2 — Extended Applications

### 7. Game/Lineup Outcome Prediction

**Benchmark task (contextual)**

> "Who wins when these lineups play?"

**Why it matters:** Classic predictive benchmark. Tests whether embeddings encode real basketball impact on an external task they weren't directly trained on.

**Embedding property tested:** Embeddings must capture enough information about player quality and fit to predict game outcomes.

**Approach:** 4-way comparison:
1. **Home-always-wins baseline** — always predict home team wins
2. **Bag-of-player-IDs** — sparse binary features per player, logistic regression
3. **Static embeddings** — mean-pooled composed embeddings (home, away, diff), logistic regression
4. **Contextual embeddings** — run starting 5s through transformer encoder, use attention-pooled offense/defense representations, logistic regression

The contextual method is the key test — if it doesn't beat static, the transformer interactions aren't adding value for this task.

**Status:** All 4 methods implemented in `scripts/precompute_eval.py:compute_game_outcomes()`, visualized in `notebooks/master_eval.ipynb` Section 6.

---

### 7b. Lineup Completion Retrieval

**Benchmark task (contextual)**

> "Given 4 offensive players, which 5th player produces the best predicted outcome?"

**Why it matters:** A retrieval benchmark that tests whether the model can identify which player completes a lineup. Unlike game outcome prediction (binary classification), this is a ranking task over all candidate players — requiring the model to understand player-lineup fit, not just aggregate quality.

**Embedding property tested:** The model must capture how a specific player complements a specific 4-man unit, not just individual player quality.

**Approach:** Hold out the 5th offensive player (position 4). For each held-out lineup, score a random sample of candidate players by running the completed lineup through the model and computing a favorability score (weighted sum of predicted outcome probabilities). Rank candidates by favorability. Report recall@K and MRR. Run on both train and test splits to measure generalization.

**Key metrics:**
- **Recall@K** — is the true player in the top K candidates?
- **MRR** — mean reciprocal rank of the true player
- **Train vs test gap** — how much does retrieval degrade on unseen player-seasons?

**Status:** Implemented in `scripts/precompute_eval.py:compute_lineup_completion()`, visualized in `notebooks/master_eval.ipynb` Section 6.

---

### 8. League Style Evolution

**Static embeddings**

> "How has the structure of basketball changed?"

**Why it matters:** Beautiful application of the temporal embedding structure. The 3-point revolution, death of the traditional big, rise of positionless basketball — these structural shifts should be visible in how the embedding space changes across eras.

**Embedding property tested:** Embeddings across different eras should reflect real stylistic changes in the NBA.

**Approach:** Track embedding centroids per season. Measure archetype prevalence (K-means cluster membership) over time. Visualize the league's center-of-mass moving in reduced dimensionality (UMAP).

**Key stories to validate:**
- 3-point revolution: "shooter" archetype prevalence increasing after 2015
- Positionless basketball: cluster boundaries blurring in modern era
- Death of the post-up big: "traditional center" archetype shrinking

**Status:** Partially explored in `explore_unicorn.ipynb` (Section 1) for outcome distributions, but not yet in embedding space.

---

### 9. Defensive Matchup Analysis

**Contextual representations**

> "Which defenders minimize this offense's expected points?"

**Why it matters:** All the offensive lineup tasks (4-6) focus on offensive synergy. This flips the script. Fix an elite offensive lineup and optimize the defensive slots. The model takes 5 offensive and 5 defensive players — we should exploit both sides.

**Embedding property tested:** The model must capture how defensive players affect offensive outcome distributions, not just offensive synergy.

**Approach:** Fix a known elite offensive unit (e.g., prime Warriors). Score every candidate for each defensive slot. Identify which archetypes of defenders minimize the expected points of specific opposing offenses. The key question: "What kind of team do you build to stop this specific unit?"

**Status:** `notebooks/downstream_eval.ipynb` Section 3 (extension).

---

### 10. Lineup Chemistry / System Effects

**Contextual representations**

> "Why is this team under/overperforming its talent?"

**Why it matters:** The delta between what you'd expect from individual player quality and what the model predicts from lineup interactions captures emergent properties — spacing, scheme fit, player-player synergy.

**Embedding property tested:** The transformer must learn meaningful player-player interactions, not just individual player representations. The gap between pre-encoder (static) and post-encoder (contextual) representations must be meaningful.

**Approach:** For each lineup, compare two representations:
- **Static baseline:** Mean of the 5 offensive players' static composed embeddings (pre-encoder)
- **Contextual:** Transformer encoder output, mean-pooled (post-encoder)
- **Chemistry score:** How much the encoder changed the offensive representation (1 - cosine_similarity)

**Critical caveat:** A large representational shift might indicate "great chemistry" OR simply an unorthodox/unusual lineup that forces the model to aggressively adjust tokens. High chemistry score != good chemistry. Must validate by correlating chemistry scores against actual real-world lineup over/underperformance (e.g., actual Net Rating minus predicted Net Rating, or actual NetRtg minus sum of individual BPMs).

**Status:** `notebooks/downstream_eval.ipynb` Section 5.

---

### 11. Aging Curves & Skill Evolution

**Static embeddings**

> "When do players change the most, and why?"

**Why it matters:** Aggregating delta magnitudes by player age reveals macro patterns in how player identities evolve. Do players experience larger embedding shifts at age 22 (skill development) or age 33 (athletic decline forcing a role change)? This is a compelling macro-level analysis unique to the player-season embedding design.

**Embedding property tested:** The delta component must capture real age-related role changes, not noise.

**Approach:** For each player-season, compute the delta norm (season-specific deviation from base). Aggregate by player age at that season. Plot mean delta magnitude vs age to reveal the "aging curve" of identity change. Segment by archetype — do different player types evolve differently?

**Status:** `notebooks/downstream_eval.ipynb` Section 8.

---

### 12. Role Versatility Detection

**Contextual representations**

> "Which players adapt their role most depending on teammates?"

**Why it matters:** Identifies "chameleons" — players who shift their functional role based on lineup context. This is a direct showcase of the BERT-level contextual representations (a static embedding method could never detect this).

**Embedding property tested:** The transformer must produce meaningfully different contextual representations for the same player in different lineup contexts.

**Approach:** For each player-season, collect their encoder-output token across many different lineups from the data. Measure how much their contextual representation varies.

**Critical refinement:** Raw variance can be misleading — low-minute players with few lineups will show high variance from noise. Two mitigations:
1. **Weight by possession volume** — only include players above a minimum threshold, and weight variance by sample size
2. **Archetype boundary crossing** — define versatility as how often a player's contextual token crosses into a *different* archetype cluster than their static embedding's cluster. This is more robust than raw variance and more interpretable: "Draymond Green plays like a wing in some lineups and a center in others."

**Status:** `notebooks/downstream_eval.ipynb` Section 7.

---

## Measurement & Checkpoint Selection

The downstream tasks require two embedding properties:

1. **Player identity/archetype** (static embedding quality) — measured by **temporal top-100**: "does the prior-year version of an unseen player rank in the top 100 predictions?"
2. **Basketball impact** (outcome prediction quality) — measured by **val outcome loss**: "can the model predict how unseen lineups shift outcome distributions?"

Checkpoint selection uses a **gated composite metric**:
- **Gate:** `val_loss <= best_val_loss * 1.01` (impact quality hasn't degraded)
- **Selector:** `temporal_top100 > best_temporal_top100` (embedding quality improved)

This ensures we never save a checkpoint that's forgotten basketball impact in exchange for better player identity, while preferring the highest-quality embeddings among impact-preserving models.

Contextual interaction quality is implicitly captured by both metrics but not directly measured during training. It is validated post-hoc by the contextual downstream tasks (4-6, 9-12).

---

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| 1. Player Similarity | Done | `precompute_eval.py`, `master_eval.ipynb` Section 2 |
| 2. Archetype Clustering | Done | `precompute_eval.py`, `master_eval.ipynb` Section 7 |
| 3. Player Trajectories | Done | `master_eval.ipynb` Section 7 (aging curves) |
| 4. Substitution Sensitivity | Done | `precompute_eval.py`, `master_eval.ipynb` Section 5 |
| 5. Lineup Optimization | Exploratory | `downstream_eval.ipynb` Section 3 |
| 6. Superstar Ecosystems | Exploratory | `downstream_eval.ipynb` Section 4 |
| 7. Game Outcome | Done | `precompute_eval.py`, `master_eval.ipynb` Section 6 |
| 7b. Lineup Completion | Done | `precompute_eval.py`, `master_eval.ipynb` Section 6 |
| 8. League Evolution | Partial | Needs embedding-space analysis |
| 9. Defensive Matchups | Exploratory | `downstream_eval.ipynb` Section 3 (extension) |
| 10. Lineup Chemistry | Exploratory | `downstream_eval.ipynb` Section 5 |
| 11. Aging Curves | Done | `precompute_eval.py`, `master_eval.ipynb` Section 7 |
| 12. Role Versatility | Exploratory | `downstream_eval.ipynb` Section 7 |
