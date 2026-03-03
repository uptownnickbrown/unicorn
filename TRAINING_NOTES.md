# Unicorn Training Notes

## Phase A: Masked Player Prediction

### Command
```bash
source .unicorn/bin/activate
python train_transformer.py --phase pretrain --epochs 25 --bs 1024 --accum 4
```

### Model Config
- 24.5M parameters
- 8 layers, 8 heads, d_model=384
- Cosine LR schedule: peak 3e-4, 5% warmup
- Curriculum: sorted by season epoch 1, random shuffle after
- ~40 min/epoch on Apple MPS

### Validation Design — The ID Overlap Problem

**Problem discovered during training:** Player-season IDs are unique per (player, season). Since Phase A's train split is seasons <2019 and the original val split was seasons 2019-2020, there is literally ZERO overlap in player-season IDs. The model cannot predict an ID it has never seen, so val accuracy was permanently 0%.

**Solution — Dual evaluation:**

1. **Random holdout (overfitting detection):** 90/10 random split of pre-2019 data. Both subsets share the same player-season IDs, so masked prediction accuracy tracks training progress and detects overfitting.

2. **Temporal evaluation (cross-season generalization):** Uses the season-based val set (2019-2020) but evaluates differently. For each masked player, we check where their *prior-year* ID ranks in the model's predictions. For example, if we mask LeBron_2019 (unseen during training), does the model predict LeBron_2018 (seen during training) as a likely fill? This measures whether the model learns "a LeBron-like player belongs here" vs just memorizing specific IDs.

   Metrics reported each epoch:
   - Mean/median rank of prior-year ID (lower is better; random baseline ~6,400 out of 12,821)
   - Top-10/50/100 hit rate (what % of time does prior-year ID appear in top-k predictions)

### Why Player-Season IDs (Not Single-Player IDs)?

We considered collapsing to a single ID per player (LeBron = LeBron across all seasons), but decided against it:

- **Players genuinely change**: LeBron 2009 (athletic slasher) ≠ LeBron 2020 (floor general). One embedding can't capture both.
- **Blog value**: Career trajectories, breakout seasons, and aging curves are fascinating outputs that require per-season granularity.
- **Prior-year init provides "momentum"**: LeBron_2019's embedding is initialized from LeBron_2018's learned embedding. This provides a soft temporal prior — new seasons start from their predecessor, preserving continuity without forcing identity collapse.
- **80.3% coverage**: 10,298 of 12,821 player-season IDs have a valid prior-year mapping, so this mechanism affects most of the vocabulary.

### Training Progress (Epoch 1–11)

| Epoch | Train Loss | Train Top-1 | Train Top-5 | Val Top-1 | Val Top-5 | Temporal Mean Rank |
|-------|-----------|-------------|-------------|-----------|-----------|-------------------|
| 1     | 8.794     | 0.03%       | 0.17%       | 0.02%     | 0.17%     | 2,886             |
| 2     | 8.071     | 0.17%       | 0.78%       | 0.32%     | 1.53%     | 6,100             |
| 3     | 6.043     | 0.78%       | 3.52%       | 0.75%     | 3.31%     | 5,023             |
| 5     | 5.583     | 1.90%       | 8.16%       | 1.78%     | 7.70%     | 4,560             |
| 8     | 5.210     | 3.50%       | 14.3%       | 1.04%     | 4.59%     | 5,289             |
| 11    | 4.500     | 6.96%       | 26.6%       | 1.40%     | 5.97%     | 7,116             |

**Observations:**
- Training accuracy improving steadily — model is learning
- Growing train/val gap suggests overfitting on training IDs (expected for 12,821-way classification on a fixed set of IDs)
- Temporal metric is noisy and trending worse (2,886 → 7,116). The model may be specializing its predictions toward training-era player-season IDs rather than learning transferable player archetypes. This is a key area for future investigation.

### Checkpoints & Logs
- Best model: `pretrain_checkpoint.pt` (saved by best val top-5 accuracy)
- Epoch log: `pretrain_checkpoint.log.jsonl` (one JSON object per epoch)
- CBOW baseline: `cbow_checkpoint.pt` (12% val accuracy)

---

## Phase B: Outcome Fine-Tuning (Pending)

### Command
```bash
python train_transformer.py --phase finetune --pretrain-ckpt pretrain_checkpoint.pt --epochs 15 --bs 1024 --accum 4
```

### Key Details
- Loads Phase A checkpoint, adds 9-class outcome prediction head
- Differential learning rates: embeddings 0.03x, encoder 0.1x, head 1x
- Class-weighted cross-entropy for imbalanced 9-class outcomes
- Saves to `finetune_checkpoint.pt`

---

## CBOW Baseline

### Results
- Val accuracy: 12.02% (barely above ~11% random for 9 classes)
- Near-zero cross-season embedding coherence
- This is the "before" picture — the transformer should beat this significantly

### Interpretation
CBOW uses simple mean-pooling over player embeddings with an MLP classifier. The fact that it barely beats random confirms that:
1. The 9-class outcome task is genuinely hard
2. Mean-pooling discards lineup interaction information
3. There's room for the attention mechanism to add value

---

## Post-Training Pipeline

After Phase A + B training completes:

```bash
# Evaluation
python evaluate.py --ckpt pretrain_checkpoint.pt --phase pretrain
python evaluate.py --ckpt finetune_checkpoint.pt --phase finetune
python evaluate.py --ckpt cbow_checkpoint.pt --model-type cbow

# Downstream: game outcome prediction
python game_outcome.py --ckpt pretrain_checkpoint.pt

# Embedding analysis
python analyze_embeddings.py --ckpt pretrain_checkpoint.pt --output-dir plots/
```
