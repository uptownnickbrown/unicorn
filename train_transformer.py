# train_transformer.py – v5 (contrastive pretraining, composed embeddings)
"""
Unicorn v2 transformer with:
  - Composed embeddings: base_player_emb + delta_emb (base shared across seasons)
  - Contrastive Phase A: InfoNCE loss with stop-gradient targets
  - Auxiliary base-player classification head
  - Temporal augmentation (swap to adjacent season)
  - LLM-seeded embedding initialization support
  - Phase B: Outcome prediction fine-tuning

Run examples:
```bash
# Phase A: Contrastive pretraining
python train_transformer.py --phase pretrain --epochs 25

# Phase B: Outcome prediction fine-tuning
python train_transformer.py --phase finetune --pretrain-ckpt pretrain_v2_checkpoint.pt --epochs 15
```
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nba_dataset import (
    PossessionDataset,
    get_default_dataloaders,
    get_num_players,
    NUM_OUTCOMES,
    OUTCOME_VOCAB,
)
from prior_year_init import (
    build_prior_year_map,
    build_next_year_map,
    build_ps_to_base_tensor,
    build_temporal_swap_tensor,
)

################################################################################
# Attention Pooling
################################################################################

class AttentionPool(nn.Module):
    """Learned single-head attention pooling over a set of tokens.

    Given H [B, N, d_model], computes attention weights and returns
    a weighted sum [B, d_model] plus the weights for interpretability.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.query(h)                          # [B, N, 1]
        weights = torch.softmax(scores, dim=1)          # [B, N, 1]
        pooled = (weights * h).sum(dim=1)               # [B, d_model]
        return pooled, weights.squeeze(-1)              # [B, d_model], [B, N]

################################################################################
# Model
################################################################################

class LineupTransformer(nn.Module):
    """Transformer encoder over 10 player tokens with attention pooling.

    v2 Architecture:
    1. Composed embeddings: base_player_emb[base_id] + delta_emb[ps_id]
    2. Team-side embeddings + position bias
    3. Transformer encoder (self-attention among players)
    4. Attention pool over player outputs
    5. Task heads:
       - Phase A: contrastive prediction (mask_proj) + auxiliary base classification
       - Phase B: outcome prediction
    """

    def __init__(
        self,
        num_player_seasons: int,
        num_base_players: int,
        ps_to_base: torch.Tensor,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_player_seasons = num_player_seasons
        self.num_base_players = num_base_players

        # Composed embeddings: base (shared across seasons) + delta (season-specific)
        self.base_player_emb = nn.Embedding(num_base_players, d_model)
        self.delta_emb = nn.Embedding(num_player_seasons, d_model)
        nn.init.zeros_(self.delta_emb.weight)  # delta starts at zero

        # Non-trainable mapping: player_season_id → base_player_id
        self.register_buffer("ps_to_base", ps_to_base)

        # Team and position
        self.team_emb = nn.Embedding(2, d_model)  # 0=offense, 1=defense
        self.pos_bias = nn.Parameter(torch.zeros(10, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # State projection
        self.state_proj = nn.Linear(3, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Attention pooling
        self.attn_pool = AttentionPool(d_model)

        # Phase A: contrastive prediction head (projects to embedding space)
        self.mask_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # Phase A: auxiliary base-player classification head (2,310-way)
        self.aux_base_head = nn.Linear(d_model, num_base_players)

        # Phase A: fixed temperature for InfoNCE (0.07 = CLIP/MoCo default)
        self.register_buffer("_temperature", torch.tensor(0.07))

        # Phase B head: outcome prediction (9-way)
        self.outcome_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_OUTCOMES),
        )

    def _compose_embedding(self, player_season_ids: torch.Tensor) -> torch.Tensor:
        """Compose player embedding: base[ps_to_base[id]] + delta[id]."""
        base_ids = self.ps_to_base[player_season_ids]
        return self.base_player_emb(base_ids) + self.delta_emb(player_season_ids)

    def _all_composed_embeddings(self) -> torch.Tensor:
        """Get composed embeddings for the full vocabulary. [V, d_model]"""
        all_base_ids = self.ps_to_base  # [V]
        return self.base_player_emb(all_base_ids) + self.delta_emb.weight

    def _embed_players(self, players: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        """Embed player tokens with team and position information."""
        tok = self._compose_embedding(players) + self.team_emb(team_ids) + self.pos_bias
        return tok  # [B, 10, d_model]

    @property
    def temperature(self) -> torch.Tensor:
        """Fixed temperature for contrastive loss."""
        return self._temperature

    def forward_pretrain(self, players, state, team_ids, mask_idx):
        """Phase A: Contrastive masked player prediction.

        Args:
            players: [B, 10] player-season IDs
            state: [B, 3] game state features
            team_ids: [B, 10] team indicators (0/1)
            mask_idx: [B] index (0-9) of the masked player

        Returns:
            pred_emb: [B, d_model] predicted embedding (L2-normalized)
            aux_logits: [B, num_base_players] auxiliary base-player logits
            attn_weights: [B, 10] attention pooling weights
        """
        tok = self._embed_players(players, team_ids)  # [B, 10, d_model]

        # Apply mask: replace masked player's embedding with [MASK] token
        B = tok.size(0)
        mask_idx_expanded = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        tok.scatter_(1, mask_idx_expanded, self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_model))

        h = self.encoder(tok)  # [B, 10, d_model]

        # Extract the masked position's output
        mask_idx_h = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        masked_repr = h.gather(1, mask_idx_h).squeeze(1)  # [B, d_model]

        # Contrastive: project to embedding space
        pred_emb = self.mask_proj(masked_repr)  # [B, d_model]

        # Auxiliary: base-player classification
        aux_logits = self.aux_base_head(masked_repr)  # [B, num_base_players]

        _, attn_weights = self.attn_pool(h)  # for logging only

        return pred_emb, aux_logits, attn_weights

    def forward_finetune(self, players, state, team_ids):
        """Phase B: Outcome prediction.

        Args:
            players: [B, 10] player-season IDs
            state: [B, 3] game state features
            team_ids: [B, 10] team indicators (0/1)

        Returns:
            logits: [B, NUM_OUTCOMES] outcome prediction
            attn_weights: [B, 10] attention pooling weights
        """
        tok = self._embed_players(players, team_ids)
        h = self.encoder(tok)                          # [B, 10, d_model]
        pooled, attn_weights = self.attn_pool(h)       # [B, d_model], [B, 10]
        state_repr = self.state_proj(state)            # [B, d_model]
        combined = torch.cat([pooled, state_repr], dim=1)  # [B, 2*d_model]
        logits = self.outcome_head(combined)           # [B, NUM_OUTCOMES]
        return logits, attn_weights

    def forward(self, players, state, team_ids, mask_idx=None):
        """Unified forward: routes to pretrain or finetune based on mask_idx."""
        if mask_idx is not None:
            return self.forward_pretrain(players, state, team_ids, mask_idx)
        return self.forward_finetune(players, state, team_ids)

################################################################################
# Training utilities
################################################################################

def class_weights(train_dl: DataLoader, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(NUM_OUTCOMES)
    for *_, y in train_dl:
        counts += torch.bincount(y, minlength=NUM_OUTCOMES).float()
    w = 1.0 / counts.clamp(min=1)
    w *= NUM_OUTCOMES / w.sum()
    return w.to(device)


def pretrain_epoch(
    model, dataloader, optimizer, scheduler, device, accum,
    temporal_swap_tensor, temporal_aug_prob,
    delta_reg_weight, aux_loss_weight, delta_max_norm=0.0,
):
    """Phase A training loop: contrastive masked player prediction."""
    model.train()
    total = 0
    loss_sum, contrastive_loss_sum, aux_loss_sum, delta_reg_sum = 0.0, 0.0, 0.0, 0.0
    correct, top5_correct, aux_correct = 0, 0, 0
    pred_cossim_sum, pred_cossim_n = 0.0, 0

    for step, (pl, st, tm, _) in enumerate(tqdm(dataloader, leave=False, desc="pretrain")):
        pl, st, tm = pl.to(device), st.to(device), tm.to(device)
        B = pl.size(0)

        # Temporal augmentation: swap some players to adjacent season
        if temporal_aug_prob > 0 and temporal_swap_tensor is not None:
            swap_mask = torch.rand(B, 10, device=device) < temporal_aug_prob
            swapped = temporal_swap_tensor[pl]  # [B, 10]
            pl = torch.where(swap_mask, swapped, pl)

        # Randomly select one player to mask per sample
        mask_idx = torch.randint(0, 10, (B,), device=device)
        target_ps_ids = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)  # [B]
        target_base_ids = model.ps_to_base[target_ps_ids]  # [B]

        # Forward
        pred_emb, aux_logits, _ = model.forward_pretrain(pl, st, tm, mask_idx)

        # InfoNCE contrastive loss
        # Target: all composed embeddings with stop-gradient
        all_composed = model._all_composed_embeddings().detach()  # [V, d_model]

        # Cosine similarity scaled by temperature
        pred_norm = F.normalize(pred_emb, dim=1)          # [B, d_model]
        target_norm = F.normalize(all_composed, dim=1)     # [V, d_model]
        sim_matrix = pred_norm @ target_norm.T / model.temperature  # [B, V]

        # Mask same-base-player false negatives: don't penalize similarity to other
        # seasons of the same player (they share archetypes, should be nearby)
        target_base = model.ps_to_base[target_ps_ids]           # [B]
        all_base = model.ps_to_base                              # [V]
        same_player_mask = (target_base.unsqueeze(1) == all_base.unsqueeze(0))  # [B, V]
        # Keep the actual target visible, mask out other same-player seasons
        same_player_mask.scatter_(1, target_ps_ids.unsqueeze(1), False)
        sim_matrix = sim_matrix.masked_fill(same_player_mask, -1e9)

        contrastive_loss = F.cross_entropy(sim_matrix, target_ps_ids)

        # Auxiliary base-player classification loss
        aux_loss = F.cross_entropy(aux_logits, target_base_ids)

        # Delta regularization
        delta_reg = model.delta_emb.weight.norm(dim=1).mean()

        # Total loss
        loss = contrastive_loss + aux_loss_weight * aux_loss + delta_reg_weight * delta_reg
        (loss / accum).backward()

        if (step + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Hard cap on delta norms (complements soft L2 regularization)
            if delta_max_norm > 0:
                with torch.no_grad():
                    norms = model.delta_emb.weight.norm(dim=1, keepdim=True)
                    scale = (delta_max_norm / norms).clamp(max=1.0)
                    model.delta_emb.weight.data.mul_(scale)

        # Metrics
        loss_sum += loss.item() * B
        contrastive_loss_sum += contrastive_loss.item() * B
        aux_loss_sum += aux_loss.item() * B
        delta_reg_sum += delta_reg.item() * B

        # Contrastive accuracy: which player ranked #1
        correct += (sim_matrix.argmax(1) == target_ps_ids).sum().item()
        _, top5_pred = sim_matrix.topk(5, dim=1)
        top5_correct += (top5_pred == target_ps_ids.unsqueeze(1)).any(dim=1).sum().item()

        # Auxiliary accuracy
        aux_correct += (aux_logits.argmax(1) == target_base_ids).sum().item()

        # Embedding collapse detection: mean pairwise cosine similarity of predictions
        # Sample every 10 steps to avoid overhead
        if step % 10 == 0:
            with torch.no_grad():
                cossim = (pred_norm @ pred_norm.T).fill_diagonal_(0)
                pred_cossim_sum += cossim.sum().item() / max(B * (B - 1), 1)
                pred_cossim_n += 1

        total += B

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": loss_sum / total,
        "contrastive_loss": contrastive_loss_sum / total,
        "aux_loss": aux_loss_sum / total,
        "delta_reg": delta_reg_sum / total,
        "top1": correct / total,
        "top5": top5_correct / total,
        "aux_acc": aux_correct / total,
        "pred_cossim": pred_cossim_sum / max(pred_cossim_n, 1),
    }


def finetune_epoch(model, dataloader, criterion, optimizer, scheduler, device, accum):
    """Phase B training loop: outcome prediction."""
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0

    for step, (pl, st, tm, y) in enumerate(tqdm(dataloader, leave=False, desc="finetune" if train else "eval")):
        pl, st, tm, y = pl.to(device), st.to(device), tm.to(device), y.to(device)

        logits, _ = model.forward_finetune(pl, st, tm)
        loss = criterion(logits, y)
        if train:
            loss = loss / accum
            loss.backward()
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * accum * pl.size(0)
        else:
            loss_sum += loss.item() * pl.size(0)

        correct += (logits.argmax(1) == y).sum().item()
        total += pl.size(0)

    if train and scheduler is not None:
        scheduler.step()

    return loss_sum / total, correct / total

################################################################################
# Temporal evaluation for Phase A
################################################################################

def build_prior_id_tensor(prior_map: dict, num_players: int, device: torch.device):
    """Build a lookup tensor: prior_ids[current_id] = prior_year_id, or -1."""
    t = torch.full((num_players,), -1, dtype=torch.long, device=device)
    for curr, prev in prior_map.items():
        if curr < num_players and prev < num_players:
            t[curr] = prev
    return t


def temporal_eval(model, temporal_dl, prior_id_tensor, device, max_batches=200):
    """Evaluate masked prediction on future seasons using prior-year ID proximity.

    For contrastive v2: computes cosine similarity between predicted embedding
    and all composed embeddings, then checks where the prior-year ID ranks.
    """
    model.eval()
    all_ranks = []
    top10, top50, top100 = 0, 0, 0
    total = 0

    with torch.no_grad():
        # Pre-compute all composed embeddings once
        all_composed = model._all_composed_embeddings()  # [V, d_model]
        all_composed_norm = F.normalize(all_composed, dim=1)

        for batch_i, (pl, st, tm, _) in enumerate(tqdm(temporal_dl, leave=False, desc="temporal")):
            if batch_i >= max_batches:
                break
            pl, st, tm = pl.to(device), st.to(device), tm.to(device)
            B = pl.size(0)

            mask_idx = torch.randint(0, 10, (B,), device=device)
            masked_ids = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)  # [B]

            pred_emb, _, _ = model.forward_pretrain(pl, st, tm, mask_idx)

            # Cosine similarity against full vocabulary
            # NOTE: No false-negative masking here — temporal eval measures whether
            # the prior-year ID (a same-base-player entry) ranks highly, so we must
            # NOT mask same-base-player entries.
            pred_norm = F.normalize(pred_emb, dim=1)  # [B, d_model]
            sim = pred_norm @ all_composed_norm.T      # [B, V]

            # Look up prior-year IDs for masked players
            prior_ids = prior_id_tensor[masked_ids]  # [B], -1 if no mapping
            has_prior = prior_ids >= 0

            if not has_prior.any():
                continue

            valid_sim = sim[has_prior]              # [n, V]
            valid_priors = prior_ids[has_prior]     # [n]

            # Rank = number of IDs with higher similarity + 1
            prior_sim_vals = valid_sim.gather(1, valid_priors.unsqueeze(1))  # [n, 1]
            ranks = (valid_sim > prior_sim_vals).sum(dim=1) + 1  # [n], 1-indexed

            n = ranks.size(0)
            all_ranks.extend(ranks.cpu().tolist())
            top10 += (ranks <= 10).sum().item()
            top50 += (ranks <= 50).sum().item()
            top100 += (ranks <= 100).sum().item()
            total += n

    if total == 0:
        return {}

    ranks_arr = np.array(all_ranks)
    return {
        "n_samples": total,
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "top10_pct": top10 / total,
        "top50_pct": top50 / total,
        "top100_pct": top100 / total,
    }


################################################################################
# Phase A: Contrastive Pretraining
################################################################################

def run_pretrain(args, device):
    print("=" * 60, flush=True)
    print("PHASE A: Contrastive Pretraining (v2)", flush=True)
    print("=" * 60, flush=True)

    # Data: random holdout for validation (same as v1)
    full_train_ds = PossessionDataset(args.parquet, split="train", shuffle_players=True)
    n_val = int(0.1 * len(full_train_ds))
    n_train = len(full_train_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Phase A split: {n_train:,} train / {n_val:,} val (random 90/10 of pre-2019 data)", flush=True)

    train_dl = DataLoader(
        train_ds, batch_size=args.bs, shuffle=(args.epochs > 1),
        pin_memory=True, num_workers=4,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        pin_memory=True, num_workers=4,
    )

    # Temporal val: season-based val (2019-2020)
    temporal_ds = PossessionDataset(args.parquet, split="val", shuffle_players=False)
    temporal_dl = DataLoader(
        temporal_ds, batch_size=args.bs, shuffle=False,
        pin_memory=True, num_workers=4,
    )

    # Build composed embedding mappings
    num_player_seasons = get_num_players()
    ps_to_base, num_base = build_ps_to_base_tensor(num_player_seasons, args.lookup_csv)
    ps_to_base = ps_to_base.to(device)
    print(f"Vocab: {num_player_seasons:,} player-seasons, {num_base:,} base players", flush=True)

    # Build temporal swap tensor for augmentation
    temporal_swap = build_temporal_swap_tensor(num_player_seasons, args.lookup_csv).to(device)
    print(f"Temporal augmentation prob: {args.temporal_aug_prob}", flush=True)

    # Model
    model = LineupTransformer(
        num_player_seasons, num_base, ps_to_base,
        args.d_model, args.layers, args.heads, args.dropout,
    ).to(device)
    print(f"Model: d={args.d_model}, L={args.layers}, H={args.heads}, "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    # LLM-seeded embedding initialization
    text_emb_path = Path(args.text_emb) if args.text_emb else Path("base_player_text_embeddings.pt")
    if text_emb_path.exists():
        text_emb = torch.load(text_emb_path, map_location="cpu", weights_only=True)
        if text_emb.shape == model.base_player_emb.weight.shape:
            model.base_player_emb.weight.data.copy_(text_emb)
            print(f"Initialized base_player_emb from {text_emb_path} "
                  f"(std={text_emb.std():.4f})", flush=True)
        else:
            print(f"WARNING: text embedding shape {text_emb.shape} != "
                  f"base_player_emb shape {model.base_player_emb.weight.shape}, skipping", flush=True)
    else:
        print(f"No text embeddings found at {text_emb_path}, using random init", flush=True)

    # Build prior-year map (for temporal eval)
    prior_map = build_prior_year_map(args.lookup_csv)
    prior_id_tensor = build_prior_id_tensor(prior_map, num_player_seasons, device)
    print(f"Prior-year mappings: {len(prior_map):,} / {num_player_seasons:,} "
          f"({len(prior_map)/num_player_seasons:.1%})", flush=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_steps = int(0.05 * args.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_steps, 1),
    )

    ckpt_path = args.ckpt or "pretrain_v2_checkpoint.pt"
    log_path = Path(ckpt_path).with_suffix(".log.jsonl")
    best_val_top5 = 0.0
    start_epoch = 1

    # Resume from checkpoint if requested
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.exists():
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["state_dict"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if "scheduler_state" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        start_epoch = resume_ckpt.get("last_epoch", 0) + 1
        best_val_top5 = resume_ckpt.get("best_val_top5", 0.0)
        print(f"Resumed from {resume_path} at epoch {start_epoch} "
              f"(best val top-5: {best_val_top5*100:.2f}%)", flush=True)
    elif args.resume:
        print(f"WARNING: --resume path {args.resume} not found, starting fresh", flush=True)

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        ep_start = time.time()

        # Linear warmup
        if epoch <= warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * epoch / max(warmup_steps, 1)

        metrics = pretrain_epoch(
            model, train_dl, optimizer,
            scheduler if epoch > warmup_steps else None,
            device, args.accum,
            temporal_swap, args.temporal_aug_prob,
            args.delta_reg_weight, args.aux_loss_weight,
            args.delta_max_norm,
        )

        # Validation: contrastive accuracy on held-out data
        model.eval()
        val_total, val_correct, val_top5, val_aux_correct = 0, 0, 0, 0
        with torch.no_grad():
            all_composed = model._all_composed_embeddings()
            all_composed_norm = F.normalize(all_composed, dim=1)

            for pl, st, tm, _ in tqdm(val_dl, leave=False, desc="val"):
                pl, st, tm = pl.to(device), st.to(device), tm.to(device)
                B = pl.size(0)
                mask_idx = torch.randint(0, 10, (B,), device=device)
                target_ps_ids = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)
                target_base_ids = model.ps_to_base[target_ps_ids]

                pred_emb, aux_logits, _ = model.forward_pretrain(pl, st, tm, mask_idx)

                pred_norm = F.normalize(pred_emb, dim=1)
                sim = pred_norm @ all_composed_norm.T / model.temperature

                # Mask same-base-player false negatives (same as training)
                target_base = model.ps_to_base[target_ps_ids]
                all_base = model.ps_to_base
                same_player_mask = (target_base.unsqueeze(1) == all_base.unsqueeze(0))
                same_player_mask.scatter_(1, target_ps_ids.unsqueeze(1), False)
                sim = sim.masked_fill(same_player_mask, -1e9)

                val_correct += (sim.argmax(1) == target_ps_ids).sum().item()
                _, top5_pred = sim.topk(5, dim=1)
                val_top5 += (top5_pred == target_ps_ids.unsqueeze(1)).any(dim=1).sum().item()
                val_aux_correct += (aux_logits.argmax(1) == target_base_ids).sum().item()
                val_total += B

        val_acc = val_correct / val_total
        val_top5_acc = val_top5 / val_total
        val_aux_acc = val_aux_correct / val_total

        # Temporal eval
        temp = temporal_eval(model, temporal_dl, prior_id_tensor, device, max_batches=200)

        # Embedding stats
        with torch.no_grad():
            delta_norms = model.delta_emb.weight.norm(dim=1)
            base_norms = model.base_player_emb.weight.norm(dim=1)

        lr = optimizer.param_groups[0]["lr"]
        temp_val = model.temperature.item()
        ep_time = time.time() - ep_start
        elapsed = time.time() - t0
        eta = ep_time * (args.epochs - epoch)

        print(
            f"Ep {epoch:02d}/{args.epochs} | "
            f"train top1={metrics['top1']*100:5.2f}% top5={metrics['top5']*100:5.2f}% "
            f"loss={metrics['loss']:.4f} (c={metrics['contrastive_loss']:.4f} "
            f"a={metrics['aux_loss']:.4f} d={metrics['delta_reg']:.4f}) | "
            f"val top1={val_acc*100:5.2f}% top5={val_top5_acc*100:5.2f}% aux={val_aux_acc*100:5.2f}% | "
            f"T={temp_val:.4f} lr={lr:.2e} | {ep_time:.0f}s (ETA {eta/60:.0f}m)",
            flush=True,
        )
        if temp:
            print(
                f"  temporal: rank mean={temp['mean_rank']:.0f} "
                f"median={temp['median_rank']:.0f} | "
                f"top10={temp['top10_pct']*100:.1f}% "
                f"top50={temp['top50_pct']*100:.1f}% "
                f"top100={temp['top100_pct']*100:.1f}% "
                f"(n={temp['n_samples']:,})",
                flush=True,
            )
        print(
            f"  embeddings: base_norm={base_norms.mean():.4f} "
            f"delta_norm={delta_norms.mean():.4f} "
            f"delta/base={delta_norms.mean()/base_norms.mean():.2%} "
            f"delta_max={delta_norms.max():.4f} "
            f"pred_cossim={metrics['pred_cossim']:.4f}",
            flush=True,
        )

        # JSON log
        log_entry = {
            "epoch": epoch,
            "train_loss": round(metrics["loss"], 5),
            "contrastive_loss": round(metrics["contrastive_loss"], 5),
            "aux_loss": round(metrics["aux_loss"], 5),
            "delta_reg": round(metrics["delta_reg"], 5),
            "train_top1": round(metrics["top1"], 5),
            "train_top5": round(metrics["top5"], 5),
            "train_aux_acc": round(metrics["aux_acc"], 5),
            "pred_cossim": round(metrics["pred_cossim"], 5),
            "val_top1": round(val_acc, 5),
            "val_top5": round(val_top5_acc, 5),
            "val_aux_acc": round(val_aux_acc, 5),
            "temperature": round(temp_val, 5),
            "delta_norm_mean": round(delta_norms.mean().item(), 5),
            "delta_norm_max": round(delta_norms.max().item(), 5),
            "base_norm_mean": round(base_norms.mean().item(), 5),
            "lr": lr,
            "epoch_sec": round(ep_time, 1),
            "elapsed_sec": round(elapsed, 1),
        }
        if temp:
            log_entry["temporal_mean_rank"] = round(temp["mean_rank"], 1)
            log_entry["temporal_median_rank"] = round(temp["median_rank"], 1)
            log_entry["temporal_top10"] = round(temp["top10_pct"], 5)
            log_entry["temporal_top50"] = round(temp["top50_pct"], 5)
            log_entry["temporal_top100"] = round(temp["top100_pct"], 5)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Resumable checkpoint (saved every epoch)
        resume_state = {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "phase": "pretrain_v2",
            "vocab": OUTCOME_VOCAB,
            "num_player_seasons": num_player_seasons,
            "num_base_players": num_base,
            "d_model": args.d_model,
            "n_layers": args.layers,
            "n_heads": args.heads,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "last_epoch": epoch,
            "best_val_top5": best_val_top5 if val_top5_acc <= best_val_top5 else val_top5_acc,
            "architecture": "v2_contrastive",
        }

        # Save best model (by val top-5 accuracy)
        if val_top5_acc > best_val_top5:
            best_val_top5 = val_top5_acc
            resume_state["best_val_top5"] = best_val_top5
            resume_state["best_epoch"] = epoch
            torch.save(resume_state, ckpt_path)
            print(f"  -> New best val top-5: {best_val_top5*100:.2f}% (saved {ckpt_path})", flush=True)

        # Always save latest for resumption
        latest_path = Path(ckpt_path).with_stem(Path(ckpt_path).stem + "_latest")
        torch.save(resume_state, latest_path)

    total_time = time.time() - t0
    print(f"\nPhase A complete in {total_time/60:.1f}m | Best val top-5: {best_val_top5*100:.2f}%", flush=True)
    print(f"Checkpoint: {ckpt_path} | Log: {log_path}", flush=True)

################################################################################
# Phase B: Outcome Prediction Fine-tuning
################################################################################

def run_finetune(args, device):
    print("=" * 60, flush=True)
    print("PHASE B: Outcome Prediction Fine-tuning", flush=True)
    print("=" * 60, flush=True)

    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt required for fine-tuning")

    # Load pretrained checkpoint
    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True)

    # Detect architecture version
    is_v2 = ckpt.get("architecture") == "v2_contrastive"

    if is_v2:
        num_player_seasons = ckpt["num_player_seasons"]
        num_base = ckpt["num_base_players"]
        d_model = ckpt.get("d_model", args.d_model)
        n_layers = ckpt.get("n_layers", args.layers)
        n_heads = ckpt.get("n_heads", args.heads)
        dropout = ckpt.get("dropout", args.dropout)

        ps_to_base, _ = build_ps_to_base_tensor(num_player_seasons, args.lookup_csv)
        ps_to_base = ps_to_base.to(device)

        model = LineupTransformer(
            num_player_seasons, num_base, ps_to_base,
            d_model, n_layers, n_heads, dropout,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded v2 pretrained model from {args.pretrain_ckpt}", flush=True)

        # Differential LR: very low for base embeddings, low for delta/encoder, full for head
        base_params = list(model.base_player_emb.parameters())
        delta_params = list(model.delta_emb.parameters())
        encoder_params = (
            list(model.encoder.parameters())
            + list(model.attn_pool.parameters())
            + [model.pos_bias]
            + list(model.team_emb.parameters())
        )
        head_params = list(model.outcome_head.parameters()) + list(model.state_proj.parameters())

        optimizer = optim.AdamW([
            {"params": base_params, "lr": args.lr * 0.01},    # base: very low
            {"params": delta_params, "lr": args.lr * 0.03},   # delta: low
            {"params": encoder_params, "lr": args.lr * 0.1},  # encoder: low
            {"params": head_params, "lr": args.lr},            # new head: full
        ], weight_decay=1e-4)
    else:
        raise ValueError(
            f"Checkpoint {args.pretrain_ckpt} is not v2 architecture. "
            "Cannot fine-tune a v1 checkpoint with v2 code."
        )

    train_dl, val_dl, _ = get_default_dataloaders(
        args.parquet, batch_size=args.bs, shuffle_train=True,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dl, device))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = args.ckpt or "finetune_v2_checkpoint.pt"
    log_path = Path(ckpt_path).with_suffix(".log.jsonl")
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = finetune_epoch(
            model, train_dl, criterion, optimizer, scheduler, device, args.accum,
        )
        val_loss, val_acc = finetune_epoch(
            model, val_dl, criterion, None, None, device, 1,
        )
        lr_head = optimizer.param_groups[-1]["lr"]
        ep_time = time.time() - ep_start
        elapsed = time.time() - t0
        eta = ep_time * (args.epochs - epoch)

        print(
            f"Ep {epoch:02d}/{args.epochs} | "
            f"train {train_acc*100:5.2f}% loss={train_loss:.4f} | "
            f"val {val_acc*100:5.2f}% loss={val_loss:.4f} | "
            f"lr_head {lr_head:.2e} | {ep_time:.0f}s (ETA {eta/60:.0f}m)",
            flush=True,
        )

        # JSON log
        log_entry = {
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5), "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5), "lr_head": lr_head,
            "epoch_sec": round(ep_time, 1), "elapsed_sec": round(elapsed, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "phase": "finetune_v2",
                "vocab": OUTCOME_VOCAB,
                "num_player_seasons": num_player_seasons,
                "num_base_players": num_base,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "dropout": dropout,
                "num_outcomes": NUM_OUTCOMES,
                "epochs": args.epochs,
                "best_epoch": epoch,
                "best_val_acc": best_val_acc,
                "architecture": "v2_contrastive",
            }, ckpt_path)
            print(f"  -> New best val acc: {best_val_acc*100:.2f}% (saved {ckpt_path})", flush=True)

    total_time = time.time() - t0
    print(f"\nPhase B complete in {total_time/60:.1f}m | Best val acc: {best_val_acc*100:.2f}%", flush=True)
    print(f"Checkpoint: {ckpt_path} | Log: {log_path}", flush=True)

################################################################################
# Main
################################################################################

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning, message="enable_nested_tensor")
    warnings.filterwarnings("ignore", category=UserWarning, message="pin_memory")

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}", flush=True)

    if args.phase == "pretrain":
        run_pretrain(args, device)
    elif args.phase == "finetune":
        run_finetune(args, device)
    else:
        raise ValueError("--phase must be 'pretrain' or 'finetune'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unicorn v2 transformer training")
    parser.add_argument("--phase", required=True, choices=["pretrain", "finetune"],
                        help="Training phase: 'pretrain' (contrastive) or 'finetune' (outcome)")
    parser.add_argument("--parquet", default="possessions.parquet")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ckpt", default=None, help="Output checkpoint path")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from (loads model, optimizer, scheduler, epoch)")
    parser.add_argument("--pretrain-ckpt", default=None,
                        help="Phase A checkpoint to load for fine-tuning")
    parser.add_argument("--text-emb", default=None,
                        help="Path to base_player_text_embeddings.pt for LLM init")
    parser.add_argument("--lookup-csv", default="player_season_lookup.csv")
    # v2 hyperparameters
    parser.add_argument("--delta-reg-weight", type=float, default=0.05,
                        help="L2 regularization weight for delta embeddings (soft)")
    parser.add_argument("--delta-max-norm", type=float, default=0.3,
                        help="Hard cap on delta embedding norms (0 to disable)")
    parser.add_argument("--aux-loss-weight", type=float, default=0.1,
                        help="Weight for auxiliary base-player classification loss")
    parser.add_argument("--temporal-aug-prob", type=float, default=0.15,
                        help="Probability of swapping a player to adjacent season")
    args = parser.parse_args()
    main(args)
