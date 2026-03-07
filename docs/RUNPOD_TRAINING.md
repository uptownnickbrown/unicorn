# RunPod GPU Training Guide for Unicorn v3.2

Dual forward pass makes training ~85 min/epoch on MPS (~35 hrs for 25 epochs). An RTX 4090 on RunPod should do ~10-15 min/epoch, completing 25 epochs in ~4-6 hours for ~$3-5.

## Prerequisites

1. **RunPod account**: Sign up at https://runpod.io and add $25 credit
2. **API key**: Generate at https://runpod.io/console/user/settings (API Keys section)
3. **SSH key**: Add your public key (`~/.ssh/id_ed25519.pub`) in RunPod SSH settings
4. **CLI** (for automation): `pip install runpod`

## Option A: Automated Script

```bash
export RUNPOD_API_KEY=your_key
python scripts/deploy_runpod.py
```

The script handles the full lifecycle: create pod, upload data, train, download results, terminate.

See `python scripts/deploy_runpod.py --help` for options (GPU type, epochs, etc.).

### Useful flags

```bash
# Dry run (show plan without executing)
python scripts/deploy_runpod.py --dry-run

# Cheaper GPU
python scripts/deploy_runpod.py --gpu a4000

# Fewer epochs for testing
python scripts/deploy_runpod.py --epochs 3

# Reattach to an existing pod
python scripts/deploy_runpod.py --pod-id abc123 --skip-upload
```

If interrupted (Ctrl+C), the script will try to download partial results and ask before terminating the pod.

## Option B: Manual Step-by-Step

### 1. Create a Pod

**Web UI** (recommended for first time):
- Go to https://runpod.io/console/pods
- Click "Deploy" on an **RTX 4090** ($0.69/hr) or **RTX A4000** ($0.40/hr)
- Template: `runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204`
- Container disk: 20 GB, Volume: 0 GB
- Click Deploy

**CLI alternative:**
```bash
runpod pod create --name unicorn-v32 \
  --gpu "NVIDIA GeForce RTX 4090" --gpu-count 1 \
  --image "runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204" \
  --container-disk 20 --volume 0
```

### 2. Connect via SSH

RunPod shows SSH details in the pod dashboard:
```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

### 3. Upload Data + Code (~280 MB)

From the unicorn project directory:
```bash
rsync -avz --progress -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
  possessions.parquet \
  base_player_text_embeddings.pt \
  player_season_lookup.csv \
  player_descriptions.jsonl \
  nba_dataset.py \
  train_transformer.py \
  prior_year_init.py \
  evaluate.py \
  analyze_embeddings.py \
  requirements-train.txt \
  root@<ip>:/workspace/unicorn/
```

### 4. Install Dependencies + Train

On the pod:
```bash
cd /workspace/unicorn
pip install -r requirements-train.txt

# Start training in tmux (survives SSH disconnects)
tmux new-session -d -s train \
  'python train_transformer.py \
    --phase joint --epochs 25 --delta-dim 64 \
    --outcome-weight 1.0 --contrastive-weight 0.5 \
    --prior-strength 10 --bs 2048 \
    2>&1 | tee training.log'

# Attach to watch live output
tmux attach -t train
# (Ctrl+B, D to detach without stopping)
```

### 5. Monitor Progress

```bash
# From the pod (if detached from tmux)
tail -f /workspace/unicorn/training.log

# Check if training is still running
tmux has-session -t train && echo "Running" || echo "Done"
```

### 6. Download Results

From your local machine:
```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<ip>:/workspace/unicorn/joint_v32_checkpoint.pt .

scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<ip>:/workspace/unicorn/joint_v32_checkpoint.log.jsonl .

scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<ip>:/workspace/unicorn/joint_v32_checkpoint_latest.pt .
```

### 7. Terminate the Pod

**CRITICAL: Pods bill per second. Terminate immediately after downloading results.** A forgotten A5000 pod costs ~$6.50/day.

Web UI: Click "Terminate" on the pod, or:
```bash
runpod pod remove <pod-id>
```

### 8. Verify No Orphaned Pods

Always verify cleanup after any training run:
```bash
python scripts/runpod_cleanup.py
```

If any pods are listed, investigate and terminate:
```bash
python scripts/runpod_cleanup.py --terminate --yes
```

## Verification

After downloading the checkpoint:

```bash
# 1. Check training log looks healthy
python -c "
import json
for line in open('joint_v32_checkpoint.log.jsonl'):
    d = json.loads(line)
    print(f'Ep {d[\"epoch\"]:2d} | loss={d[\"train_loss\"]:.4f} (c={d[\"contrastive_loss\"]:.4f} a={d[\"aux_loss\"]:.4f} o={d[\"outcome_loss\"]:.4f}) | base_top5={d[\"train_base_top5\"]*100:.2f}% out={d[\"train_outcome_acc\"]*100:.2f}% | val out={d[\"val_outcome_acc\"]*100:.2f}%')
"

# 2. Full evaluation
python evaluate.py --ckpt joint_v32_checkpoint.pt --phase joint

# 3. Embedding analysis
python analyze_embeddings.py --ckpt joint_v32_checkpoint.pt
```

What to look for:
- Epoch 1 outcome loss < 2.197 (better than uniform)
- Loss decreasing over epochs
- Val metrics stable (no oscillation like v3.1)

## Cost Estimate

| GPU | Rate | Est. Epoch | 25 Epochs | Total w/ Setup |
|-----|------|-----------|-----------|----------------|
| RTX 4090 | $0.69/hr | ~10-15 min | ~4-6 hrs | ~$3-5 |
| RTX A5000 | $0.27/hr | ~14 min | ~5.8 hrs | ~$1.60 |
| RTX A4000 | $0.40/hr | ~20-30 min | ~8-12 hrs | ~$4-5 |

The $25 minimum deposit gets you many full training runs.

## Actual Run Data (v3.2 Run 1, 2026-03-06)

Trained on RTX A5000 (4090 was unavailable).

- **Throughput**: ~2.13 it/s at bs=2048 (1,818 batches/epoch)
- **Epoch time**: ~14.3 min (vs ~85 min on MPS — 6x speedup)
- **Data upload**: 283 MB in ~65 seconds via rsync
- **Pod startup**: ~2.5 min from creation to SSH ready
- **Checkpoint size**: ~201 MB each (.pt and _latest.pt)

### 2-Epoch Test Results (validated before full run)
| Metric | Epoch 1 | Epoch 2 |
|--------|---------|---------|
| outcome_loss | 2.021 | 2.013 |
| contrastive_loss | 7.193 | 6.662 |
| val_outcome_acc | 43.6% | 47.9% |
| temporal_top100 | 0.25% | 3.6% |
| delta_norm_mean | 0.101 | 0.098 |

All metrics healthy: loss decreasing, val improving, no oscillation.

## Troubleshooting

**Pod won't start**: Check GPU availability. Try a different GPU type or region. RTX 4090s are often unavailable — the script auto-falls back through: 4090 → A5000 → 3090 → A6000 → L40 → 4080 SUPER → A4000.

**SSH connection refused**: Wait 2-3 minutes after pod shows "Running". Some pods take up to 2.5 min for SSH to become ready. The script waits up to 600s.

**Docker image not found**: RunPod changed their naming scheme. Use `runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204` (not the old `runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04` format).

**numpy read-only array error**: The RunPod container has numpy 2.x which makes parquet-loaded arrays read-only. Already fixed in `nba_dataset.py` (added `.copy()` to `to_numpy()` call).

**Training crashes (OOM)**: Reduce batch size. Default is `--bs 2048`; try `--bs 1024`.

**SSH drops during training**: Training continues in tmux. Reconnect and `tmux attach -t train`.

**Script interrupted**: The deploy scripts have `atexit` handlers that auto-terminate pods on unexpected exit (crash, Ctrl+C, terminal close). If you suspect a pod survived anyway, run `python scripts/runpod_cleanup.py` to check.

**Orphaned pods**: If a pod was left running by mistake, terminate it immediately with `python scripts/runpod_cleanup.py --terminate`. A forgotten A5000 costs $6.50/day. **Always verify no pods are running after a training session.**

**API key**: Store in `.env` file in project root as `RUNPOD_API_KEY=your_key`. The script auto-loads it. Don't commit `.env`.
