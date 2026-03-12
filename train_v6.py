# train_v6.py – Hybrid Relation Network with Stint-Level Prediction
"""
Unicorn v6: Shallow transformer + pairwise Relation Network.

Key changes from v3.2:
  - 2-layer transformer (won't oversmooth 10 tokens)
  - Pairwise Relation MLPs: off-off, def-def, matchup interactions
  - Stint-level training (3-15 possessions per stint, ~461K stints/epoch)
  - Multiple contrastive masks per stint (2-3 players masked)
  - No attention pooling (mean-pool confirmed equivalent in v3.2)

Run:
```bash
# Joint training (v6)
python train_v6.py --phase joint --epochs 30 --delta-dim 64 \
  --outcome-weight 1.0 --contrastive-weight 0.5 --prior-strength 10 \
  --bs 2048 --d-pair 64 --n-layers 2 --ckpt joint_v6_checkpoint.pt
```
"""
from __future__ import annotations

import argparse
import json
import math
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
    get_num_players,
    NUM_OUTCOMES,
    OUTCOME_VOCAB,
)
from stint_dataset import StintDataset, get_stint_dataloaders
from prior_year_init import (
    build_prior_year_map,
    build_ps_to_base_tensor,
    build_temporal_swap_tensor,
)

################################################################################
# Pairwise Relation MLP
################################################################################

class PairwiseMLP(nn.Module):
    """Shared MLP that processes pairs of player tokens.

    Input: concatenation of two player representations [B*K, 2*d_model]
    Output: pair relation vector [B*K, d_pair]
    """
    def __init__(self, d_model: int, d_hidden: int = 256, d_pair: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_pair),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


################################################################################
# Model
################################################################################

class RelationNetwork(nn.Module):
    """Hybrid shallow transformer + pairwise Relation Network.

    Architecture:
    1. Composed embeddings: base_player_emb + delta (same as v3.2)
    2. Team-side embeddings + position bias
    3. 2-layer transformer encoder (shallow, won't oversmooth)
    4. Pairwise Relation MLPs:
       - g_off: C(5,2) = 10 off-off pairs (symmetric)
       - g_def: C(5,2) = 10 def-def pairs (symmetric)
       - g_match: 5×5 = 25 matchup pairs (asymmetric)
    5. Mean-pool each group → [B, d_pair] × 3
    6. Outcome head: cat[off, def, match, state] → 9-class

    Contrastive from transformer output h[mask_idx] (before pairwise).
    """

    def __init__(
        self,
        num_player_seasons: int,
        num_base_players: int,
        ps_to_base: torch.Tensor,
        d_model: int = 384,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        delta_dim: int = 64,
        d_pair: int = 64,
        d_pair_hidden: int = 256,
        n_masks: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_player_seasons = num_player_seasons
        self.num_base_players = num_base_players
        self.delta_dim = delta_dim
        self.n_layers = n_layers
        self.d_pair = d_pair
        self.n_masks = n_masks

        # Composed embeddings
        self.base_player_emb = nn.Embedding(num_base_players, d_model)

        if delta_dim > 0:
            self.delta_raw = nn.Embedding(num_player_seasons, delta_dim)
            nn.init.zeros_(self.delta_raw.weight)
            self.delta_proj = nn.Linear(delta_dim, d_model, bias=False)
            nn.init.orthogonal_(self.delta_proj.weight)
        else:
            self.delta_emb = nn.Embedding(num_player_seasons, d_model)
            nn.init.zeros_(self.delta_emb.weight)

        # Non-trainable mapping
        self.register_buffer("ps_to_base", ps_to_base)

        # Team and position
        self.team_emb = nn.Embedding(2, d_model)  # 0=offense, 1=defense
        self.pos_bias = nn.Parameter(torch.zeros(10, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # State projection
        self.state_proj = nn.Linear(3, d_model)

        # 2-layer transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pairwise Relation MLPs
        self.g_off = PairwiseMLP(d_model, d_pair_hidden, d_pair)
        self.g_def = PairwiseMLP(d_model, d_pair_hidden, d_pair)
        self.g_match = PairwiseMLP(d_model, d_pair_hidden, d_pair)

        # Pre-compute pair indices
        # Off-off: C(5,2) = 10 symmetric pairs
        off_pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                off_pairs.append((i, j))
        self.register_buffer("off_pair_idx", torch.tensor(off_pairs, dtype=torch.long))

        # Def-def: C(5,2) = 10 symmetric pairs
        def_pairs = []
        for i in range(5, 10):
            for j in range(i + 1, 10):
                def_pairs.append((i, j))
        self.register_buffer("def_pair_idx", torch.tensor(def_pairs, dtype=torch.long))

        # Matchup: 5×5 = 25 asymmetric pairs (offense × defense)
        match_pairs = []
        for i in range(5):
            for j in range(5, 10):
                match_pairs.append((i, j))
        self.register_buffer("match_pair_idx", torch.tensor(match_pairs, dtype=torch.long))

        # Contrastive head
        self.mask_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.aux_base_head = nn.Linear(d_model, num_base_players)
        self.register_buffer("_temperature", torch.tensor(0.07))

        # Outcome head: [off_pooled, def_pooled, match_pooled, state_repr]
        outcome_input_dim = 3 * d_pair + d_model
        self.outcome_head = nn.Sequential(
            nn.Linear(outcome_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_OUTCOMES),
        )

    def _compose_embedding(self, player_season_ids: torch.Tensor) -> torch.Tensor:
        base_ids = self.ps_to_base[player_season_ids]
        base = self.base_player_emb(base_ids)
        if self.delta_dim > 0:
            delta = self.delta_proj(self.delta_raw(player_season_ids))
        else:
            delta = self.delta_emb(player_season_ids)
        return base + delta

    def _all_composed_embeddings(self) -> torch.Tensor:
        all_base_ids = self.ps_to_base
        base = self.base_player_emb(all_base_ids)
        if self.delta_dim > 0:
            delta = self.delta_proj(self.delta_raw.weight)
        else:
            delta = self.delta_emb.weight
        return base + delta

    def _embed_players(self, players: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        tok = self._compose_embedding(players) + self.team_emb(team_ids) + self.pos_bias
        return tok

    def _compute_pairwise(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pairwise relation outputs from encoder output h [B, 10, d_model].

        Returns:
            off_pooled:   [B, d_pair]
            def_pooled:   [B, d_pair]
            match_pooled: [B, d_pair]
        """
        B = h.size(0)

        # Off-off pairs: symmetric, sorted by slot index
        # off_pair_idx: [10, 2] — indices already sorted (i < j)
        h_i = h[:, self.off_pair_idx[:, 0]]  # [B, 10, d_model]
        h_j = h[:, self.off_pair_idx[:, 1]]  # [B, 10, d_model]
        off_input = torch.cat([h_i, h_j], dim=-1)  # [B, 10, 2*d_model]
        off_out = self.g_off(off_input.reshape(B * 10, -1)).reshape(B, 10, -1)
        off_pooled = off_out.mean(dim=1)  # [B, d_pair]

        # Def-def pairs: symmetric
        h_i = h[:, self.def_pair_idx[:, 0]]
        h_j = h[:, self.def_pair_idx[:, 1]]
        def_input = torch.cat([h_i, h_j], dim=-1)
        def_out = self.g_def(def_input.reshape(B * 10, -1)).reshape(B, 10, -1)
        def_pooled = def_out.mean(dim=1)

        # Matchup pairs: asymmetric (offense, defense)
        h_o = h[:, self.match_pair_idx[:, 0]]  # [B, 25, d_model]
        h_d = h[:, self.match_pair_idx[:, 1]]  # [B, 25, d_model]
        match_input = torch.cat([h_o, h_d], dim=-1)  # [B, 25, 2*d_model]
        match_out = self.g_match(match_input.reshape(B * 25, -1)).reshape(B, 25, -1)
        match_pooled = match_out.mean(dim=1)  # [B, d_pair]

        return off_pooled, def_pooled, match_pooled

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    def forward_pretrain(self, players, state, team_ids, mask_idx):
        """Contrastive masked player prediction (for temporal eval compatibility)."""
        tok = self._embed_players(players, team_ids)
        B = tok.size(0)

        # Mask
        mask_idx_expanded = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        tok.scatter_(1, mask_idx_expanded,
                     self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_model))

        h = self.encoder(tok)

        # Extract masked position
        mask_idx_h = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        masked_repr = h.gather(1, mask_idx_h).squeeze(1)

        pred_emb = self.mask_proj(masked_repr)
        aux_logits = self.aux_base_head(masked_repr)
        return pred_emb, aux_logits, torch.zeros(B, 10, device=h.device)

    def forward_outcome(self, players, state, team_ids):
        """Outcome prediction (unmasked lineup, pairwise relations)."""
        tok = self._embed_players(players, team_ids)
        state_repr = self.state_proj(state)
        h = self.encoder(tok)

        off_pooled, def_pooled, match_pooled = self._compute_pairwise(h)
        combined = torch.cat([off_pooled, def_pooled, match_pooled, state_repr], dim=1)
        logits = self.outcome_head(combined)
        return logits

    def forward_finetune(self, players, state, team_ids):
        """Outcome prediction (compatibility interface for evaluate.py)."""
        logits = self.forward_outcome(players, state, team_ids)
        # Return dummy attention weights for compatibility
        B = players.size(0)
        return logits, torch.full((B, 5), 0.2, device=logits.device), torch.full((B, 5), 0.2, device=logits.device)

    def forward_joint(self, players, state, team_ids, mask_indices):
        """Joint forward: dual pass — masked for contrastive, unmasked for outcome.

        Args:
            mask_indices: [B, n_masks] — multiple mask positions per stint
        """
        B = players.size(0)
        state_repr = self.state_proj(state)

        # --- Pass 1: Contrastive (masked lineup) ---
        # Apply all masks simultaneously
        tok_masked = self._embed_players(players, team_ids)
        for m in range(mask_indices.size(1)):
            idx = mask_indices[:, m]  # [B]
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
            tok_masked.scatter_(1, idx_exp,
                                self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_model))

        h_masked = self.encoder(tok_masked)

        # Extract all masked positions
        pred_embs = []
        aux_logits_list = []
        for m in range(mask_indices.size(1)):
            idx = mask_indices[:, m]
            idx_h = idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
            masked_repr = h_masked.gather(1, idx_h).squeeze(1)
            pred_embs.append(self.mask_proj(masked_repr))
            aux_logits_list.append(self.aux_base_head(masked_repr))

        # Stack: [B*n_masks, d_model] and [B*n_masks, num_base]
        pred_emb = torch.cat(pred_embs, dim=0)
        aux_logits = torch.cat(aux_logits_list, dim=0)

        # --- Pass 2: Outcome (unmasked lineup) ---
        tok_unmasked = self._embed_players(players, team_ids)
        h = self.encoder(tok_unmasked)

        off_pooled, def_pooled, match_pooled = self._compute_pairwise(h)
        combined = torch.cat([off_pooled, def_pooled, match_pooled, state_repr], dim=1)
        outcome_logits = self.outcome_head(combined)

        return pred_emb, aux_logits, outcome_logits


################################################################################
# Training utilities
################################################################################

def joint_stint_epoch(
    model, dataloader, optimizer, scheduler, device, accum,
    temporal_swap_tensor, temporal_aug_prob,
    delta_reg_weight, aux_loss_weight, outcome_weight,
    delta_max_norm=0.0, contrastive_weight=1.0,
    n_masks=2,
):
    """Joint training epoch on stint-level data."""
    model.train()
    total = 0
    loss_sum, contrastive_loss_sum, aux_loss_sum = 0.0, 0.0, 0.0
    outcome_loss_sum, delta_reg_sum = 0.0, 0.0
    correct_top1, correct_top5, aux_correct, outcome_correct = 0, 0, 0, 0
    pred_cossim_sum, pred_cossim_n = 0.0, 0
    # Pairwise stats
    pair_off_std_sum, pair_def_std_sum, pair_match_std_sum, pair_n = 0.0, 0.0, 0.0, 0
    token_div_sum, token_div_n = 0.0, 0

    for step, (pl, st, tm, target_dist) in enumerate(tqdm(dataloader, leave=False, desc="joint")):
        pl, st, tm, target_dist = pl.to(device), st.to(device), tm.to(device), target_dist.to(device)
        B = pl.size(0)

        # Temporal augmentation
        if temporal_aug_prob > 0 and temporal_swap_tensor is not None:
            swap_mask = torch.rand(B, 10, device=device) < temporal_aug_prob
            swapped = temporal_swap_tensor[pl]
            pl = torch.where(swap_mask, swapped, pl)

        # Generate mask indices: n_masks unique positions per sample
        mask_indices = torch.stack([
            torch.randperm(10, device=device)[:n_masks]
            for _ in range(B)
        ])  # [B, n_masks]

        # Target IDs for all masks
        target_ps_ids = pl.gather(1, mask_indices)  # [B, n_masks]
        target_base_ids = model.ps_to_base[target_ps_ids]  # [B, n_masks]

        # Flatten for loss computation
        target_ps_flat = target_ps_ids.reshape(-1)  # [B*n_masks]
        target_base_flat = target_base_ids.reshape(-1)  # [B*n_masks]

        # Forward
        pred_emb, aux_logits, outcome_logits = model.forward_joint(pl, st, tm, mask_indices)

        # --- Contrastive loss (base-player InfoNCE, 2310-way) ---
        pred_norm = F.normalize(pred_emb, dim=1)  # [B*n_masks, d_model]
        if contrastive_weight > 0:
            all_base_emb = model.base_player_emb.weight.detach()
            base_norm = F.normalize(all_base_emb, dim=1)
            sim_matrix = pred_norm @ base_norm.T / model.temperature
            contrastive_loss = F.cross_entropy(sim_matrix, target_base_flat)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            sim_matrix = None

        # --- Auxiliary loss ---
        aux_loss = F.cross_entropy(aux_logits, target_base_flat)

        # --- Outcome loss ---
        log_probs = F.log_softmax(outcome_logits, dim=1)
        outcome_loss = -(target_dist * log_probs).sum(dim=1).mean()

        # --- Delta regularization ---
        if model.delta_dim > 0:
            delta_reg = model.delta_proj(model.delta_raw.weight).norm(dim=1).mean()
        else:
            delta_reg = model.delta_emb.weight.norm(dim=1).mean()

        # --- Total loss ---
        loss = (contrastive_weight * contrastive_loss
                + aux_loss_weight * aux_loss
                + outcome_weight * outcome_loss
                + delta_reg_weight * delta_reg)
        (loss / accum).backward()

        if (step + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if delta_max_norm > 0 and model.delta_dim > 0:
                with torch.no_grad():
                    projected = model.delta_proj(model.delta_raw.weight)
                    norms = projected.norm(dim=1, keepdim=True)
                    scale = (delta_max_norm / norms).clamp(max=1.0)
                    model.delta_raw.weight.data.mul_(scale)

        # Metrics
        loss_sum += loss.item() * B
        contrastive_loss_sum += contrastive_loss.item() * B
        aux_loss_sum += aux_loss.item() * B
        outcome_loss_sum += outcome_loss.item() * B
        delta_reg_sum += delta_reg.item() * B

        if sim_matrix is not None:
            correct_top1 += (sim_matrix.argmax(1) == target_base_flat).sum().item()
            _, top5_pred = sim_matrix.topk(5, dim=1)
            correct_top5 += (top5_pred == target_base_flat.unsqueeze(1)).any(dim=1).sum().item()
        aux_correct += (aux_logits.argmax(1) == target_base_flat).sum().item()
        outcome_correct += (outcome_logits.argmax(1) == target_dist.argmax(1)).sum().item()

        # Token diversity and pairwise stats (sample every 20 steps)
        if step % 20 == 0:
            with torch.no_grad():
                # Token diversity: mean pairwise cosine of encoder outputs
                tok_unmasked = model._embed_players(pl[:32], tm[:32])
                h_sample = model.encoder(tok_unmasked)
                h_norm = F.normalize(h_sample, dim=-1)
                cos_sim = torch.bmm(h_norm, h_norm.transpose(1, 2))
                mask = ~torch.eye(10, dtype=torch.bool, device=device).unsqueeze(0)
                token_div_sum += cos_sim[mask.expand_as(cos_sim)].mean().item()
                token_div_n += 1

                # Pairwise output stats
                off_p, def_p, match_p = model._compute_pairwise(h_sample)
                pair_off_std_sum += off_p.std().item()
                pair_def_std_sum += def_p.std().item()
                pair_match_std_sum += match_p.std().item()
                pair_n += 1

        # Collapse detection
        if step % 20 == 0:
            with torch.no_grad():
                cossim = (pred_norm[:B] @ pred_norm[:B].T).fill_diagonal_(0)
                pred_cossim_sum += cossim.sum().item() / max(B * (B - 1), 1)
                pred_cossim_n += 1

        total += B

    if scheduler is not None:
        scheduler.step()

    # Contrastive metrics are per-mask, not per-stint
    contrastive_total = total * n_masks

    return {
        "loss": loss_sum / total,
        "contrastive_loss": contrastive_loss_sum / total,
        "aux_loss": aux_loss_sum / total,
        "outcome_loss": outcome_loss_sum / total,
        "delta_reg": delta_reg_sum / total,
        "base_top1": correct_top1 / contrastive_total,
        "base_top5": correct_top5 / contrastive_total,
        "aux_acc": aux_correct / contrastive_total,
        "outcome_acc": outcome_correct / total,
        "pred_cossim": pred_cossim_sum / max(pred_cossim_n, 1),
        "token_diversity": token_div_sum / max(token_div_n, 1),
        "pair_off_std": pair_off_std_sum / max(pair_n, 1),
        "pair_def_std": pair_def_std_sum / max(pair_n, 1),
        "pair_match_std": pair_match_std_sum / max(pair_n, 1),
    }


def val_stint_epoch(model, dataloader, device):
    """Validation epoch on stint-level data."""
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0

    with torch.no_grad():
        for pl, st, tm, target_dist in tqdm(dataloader, leave=False, desc="val"):
            pl, st, tm, target_dist = pl.to(device), st.to(device), tm.to(device), target_dist.to(device)

            logits = model.forward_outcome(pl, st, tm)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(target_dist * log_probs).sum(dim=1).mean()

            loss_sum += loss.item() * pl.size(0)
            correct += (logits.argmax(1) == target_dist.argmax(1)).sum().item()
            total += pl.size(0)

    return loss_sum / total, correct / total


################################################################################
# Temporal evaluation (reuse from train_transformer.py)
################################################################################

def build_prior_id_tensor(prior_map: dict, num_players: int, device: torch.device):
    t = torch.full((num_players,), -1, dtype=torch.long)
    for curr, prev in prior_map.items():
        if curr < num_players and prev < num_players:
            t[curr] = prev
    return t.to(device)


def temporal_eval(model, temporal_dl, prior_id_tensor, device, max_batches=200):
    """Evaluate masked prediction on future seasons using prior-year ID proximity."""
    model.eval()
    all_ranks = []
    top10, top50, top100 = 0, 0, 0
    total = 0

    with torch.no_grad():
        all_composed = model._all_composed_embeddings()
        all_composed_norm = F.normalize(all_composed, dim=1)

        for batch_i, (pl, st, tm, _) in enumerate(tqdm(temporal_dl, leave=False, desc="temporal")):
            if batch_i >= max_batches:
                break
            pl, st, tm = pl.to(device), st.to(device), tm.to(device)
            B = pl.size(0)

            mask_idx = torch.randint(0, 10, (B,), device=device)
            masked_ids = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)

            pred_emb, _, _ = model.forward_pretrain(pl, st, tm, mask_idx)

            pred_norm = F.normalize(pred_emb, dim=1)
            sim = pred_norm @ all_composed_norm.T

            prior_ids = prior_id_tensor[masked_ids]
            has_prior = prior_ids >= 0

            if not has_prior.any():
                continue

            valid_sim = sim[has_prior]
            valid_priors = prior_ids[has_prior]

            prior_sim_vals = valid_sim.gather(1, valid_priors.unsqueeze(1))
            ranks = (valid_sim > prior_sim_vals).sum(dim=1) + 1

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
# Joint Training
################################################################################

def run_joint(args, device):
    print("=" * 60, flush=True)
    print("V6 JOINT TRAINING: Relation Network + Stint-Level", flush=True)
    print("=" * 60, flush=True)

    # Data: stint-level loaders
    train_dl, val_dl, _ = get_stint_dataloaders(
        args.stints, batch_size=args.bs, shuffle_train=True,
        prior_strength=args.prior_strength,
    )
    print(f"Stint data: {len(train_dl.dataset):,} train, {len(val_dl.dataset):,} val", flush=True)
    print(f"Prior strength (alpha): {args.prior_strength}", flush=True)

    # Temporal val: uses possessions (PossessionDataset) for backward comparison
    temporal_ds = PossessionDataset(args.parquet, split="val", shuffle_players=False,
                                    prior_strength=args.prior_strength)
    temporal_dl = DataLoader(
        temporal_ds, batch_size=args.bs, shuffle=False,
        pin_memory=True, num_workers=4, persistent_workers=True,
    )

    # Mappings
    num_player_seasons = get_num_players()
    ps_to_base, num_base = build_ps_to_base_tensor(num_player_seasons, args.lookup_csv)
    ps_to_base = ps_to_base.to(device)
    print(f"Vocab: {num_player_seasons:,} player-seasons, {num_base:,} base players", flush=True)

    temporal_swap = build_temporal_swap_tensor(num_player_seasons, args.lookup_csv).to(device)
    print(f"Temporal augmentation prob: {args.temporal_aug_prob}", flush=True)

    # Model
    model = RelationNetwork(
        num_player_seasons, num_base, ps_to_base,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.heads,
        dropout=args.dropout, delta_dim=args.delta_dim,
        d_pair=args.d_pair, d_pair_hidden=args.d_pair_hidden,
        n_masks=args.n_masks,
    ).to(device)
    print(f"Model: d={args.d_model}, L={args.n_layers}, H={args.heads}, "
          f"delta_dim={args.delta_dim}, d_pair={args.d_pair}, n_masks={args.n_masks}, "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    # LLM-seeded embedding init
    text_emb_path = Path(args.text_emb) if args.text_emb else Path("base_player_text_embeddings.pt")
    if text_emb_path.exists():
        text_emb = torch.load(text_emb_path, map_location="cpu", weights_only=True)
        if text_emb.shape == model.base_player_emb.weight.shape:
            model.base_player_emb.weight.data.copy_(text_emb)
            print(f"Initialized base_player_emb from {text_emb_path} "
                  f"(std={text_emb.std():.4f})", flush=True)
        else:
            print(f"WARNING: text embedding shape mismatch, skipping", flush=True)
    else:
        print(f"No text embeddings at {text_emb_path}, using random init", flush=True)

    # Prior-year map for temporal eval
    prior_map = build_prior_year_map(args.lookup_csv)
    prior_id_tensor = build_prior_id_tensor(prior_map, num_player_seasons, device)
    print(f"Prior-year mappings: {len(prior_map):,} / {num_player_seasons:,} "
          f"({len(prior_map)/num_player_seasons:.1%})", flush=True)

    # Differential LR
    base_params = list(model.base_player_emb.parameters())
    if args.delta_dim > 0:
        delta_params = list(model.delta_raw.parameters()) + list(model.delta_proj.parameters())
    else:
        delta_params = list(model.delta_emb.parameters())

    encoder_param_list = (
        list(model.encoder.parameters())
        + [model.pos_bias]
        + list(model.team_emb.parameters())
    )

    pairwise_head_params = (
        list(model.g_off.parameters())
        + list(model.g_def.parameters())
        + list(model.g_match.parameters())
        + list(model.mask_proj.parameters())
        + list(model.aux_base_head.parameters())
        + list(model.outcome_head.parameters())
        + list(model.state_proj.parameters())
    )

    wd = args.weight_decay
    optimizer = optim.AdamW([
        {"params": base_params, "lr": args.lr * 0.1},
        {"params": delta_params, "lr": args.lr * 0.3},
        {"params": encoder_param_list, "lr": args.lr * 0.3},
        {"params": pairwise_head_params, "lr": args.lr},
    ], weight_decay=wd)

    warmup_steps = int(0.05 * args.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_steps, 1),
    )

    # torch.compile for CUDA
    if hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)
        print("Model compiled with torch.compile", flush=True)

    ckpt_path = args.ckpt or "joint_v6_checkpoint.pt"
    log_path = Path(ckpt_path).with_suffix(".log.jsonl")
    best_temporal_top100 = 0.0
    best_val_loss = float("inf")
    start_epoch = 1

    # Resume
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.exists():
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in resume_ckpt["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
        if "optimizer_state" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            except ValueError:
                print("  WARNING: optimizer state mismatch, using fresh optimizer", flush=True)
        if "scheduler_state" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        start_epoch = resume_ckpt.get("last_epoch", 0) + 1
        best_temporal_top100 = resume_ckpt.get("best_temporal_top100", 0.0)
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {resume_path} at epoch {start_epoch}", flush=True)
    elif args.resume:
        print(f"WARNING: --resume path {args.resume} not found, starting fresh", flush=True)

    print(f"Outcome weight: {args.outcome_weight}, Contrastive weight: {args.contrastive_weight}, "
          f"Delta reg: {args.delta_reg_weight}, Delta max norm: {args.delta_max_norm}, "
          f"Prior strength: {args.prior_strength}", flush=True)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        ep_start = time.time()

        # Linear warmup
        if epoch <= warmup_steps:
            warmup_frac = epoch / max(warmup_steps, 1)
            base_lrs = [args.lr * 0.1, args.lr * 0.3, args.lr * 0.3, args.lr]
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = base_lrs[i] * warmup_frac

        metrics = joint_stint_epoch(
            model, train_dl, optimizer,
            scheduler if epoch > warmup_steps else None,
            device, args.accum,
            temporal_swap, args.temporal_aug_prob,
            args.delta_reg_weight, args.aux_loss_weight, args.outcome_weight,
            delta_max_norm=args.delta_max_norm,
            contrastive_weight=args.contrastive_weight,
            n_masks=args.n_masks,
        )

        # Validation
        val_loss, val_acc = val_stint_epoch(model, val_dl, device)

        # Temporal eval (uses PossessionDataset for backward comparison)
        temp = temporal_eval(model, temporal_dl, prior_id_tensor, device, max_batches=200)

        # Embedding stats
        with torch.no_grad():
            if model.delta_dim > 0:
                delta_norms = model.delta_proj(model.delta_raw.weight).norm(dim=1)
            else:
                delta_norms = model.delta_emb.weight.norm(dim=1)
            base_norms = model.base_player_emb.weight.norm(dim=1)

        lr_head = optimizer.param_groups[-1]["lr"]
        ep_time = time.time() - ep_start
        elapsed = time.time() - t0
        eta = ep_time * (args.epochs - epoch)

        print(
            f"Ep {epoch:02d}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} (c={metrics['contrastive_loss']:.4f} "
            f"a={metrics['aux_loss']:.4f} o={metrics['outcome_loss']:.4f} "
            f"d={metrics['delta_reg']:.4f}) | "
            f"train base_top5={metrics['base_top5']*100:5.2f}% out={metrics['outcome_acc']*100:5.2f}% | "
            f"val out={val_acc*100:5.2f}% loss={val_loss:.4f} | "
            f"lr={lr_head:.2e} | {ep_time:.0f}s (ETA {eta/60:.0f}m)",
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
            f"pred_cossim={metrics['pred_cossim']:.4f}",
            flush=True,
        )
        print(
            f"  pairwise: off_std={metrics['pair_off_std']:.4f} "
            f"def_std={metrics['pair_def_std']:.4f} "
            f"match_std={metrics['pair_match_std']:.4f} | "
            f"token_div={metrics['token_diversity']:.4f}",
            flush=True,
        )

        # JSON log
        log_entry = {
            "epoch": epoch,
            "train_loss": round(metrics["loss"], 5),
            "contrastive_loss": round(metrics["contrastive_loss"], 5),
            "aux_loss": round(metrics["aux_loss"], 5),
            "outcome_loss": round(metrics["outcome_loss"], 5),
            "delta_reg": round(metrics["delta_reg"], 5),
            "train_base_top1": round(metrics["base_top1"], 5),
            "train_base_top5": round(metrics["base_top5"], 5),
            "train_aux_acc": round(metrics["aux_acc"], 5),
            "train_outcome_acc": round(metrics["outcome_acc"], 5),
            "pred_cossim": round(metrics["pred_cossim"], 5),
            "token_diversity": round(metrics["token_diversity"], 5),
            "pair_off_std": round(metrics["pair_off_std"], 5),
            "pair_def_std": round(metrics["pair_def_std"], 5),
            "pair_match_std": round(metrics["pair_match_std"], 5),
            "val_loss": round(val_loss, 5),
            "val_outcome_acc": round(val_acc, 5),
            "delta_norm_mean": round(delta_norms.mean().item(), 5),
            "delta_norm_max": round(delta_norms.max().item(), 5),
            "base_norm_mean": round(base_norms.mean().item(), 5),
            "contrastive_weight": args.contrastive_weight,
            "prior_strength": args.prior_strength,
            "lr_head": lr_head,
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

        # Checkpoint
        resume_state = {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "phase": "joint_v6",
            "vocab": OUTCOME_VOCAB,
            "num_player_seasons": num_player_seasons,
            "num_base_players": num_base,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.heads,
            "dropout": args.dropout,
            "delta_dim": args.delta_dim,
            "d_pair": args.d_pair,
            "d_pair_hidden": args.d_pair_hidden,
            "n_masks": args.n_masks,
            "contrastive_weight": args.contrastive_weight,
            "prior_strength": args.prior_strength,
            "num_outcomes": NUM_OUTCOMES,
            "epochs": args.epochs,
            "last_epoch": epoch,
            "best_temporal_top100": best_temporal_top100,
            "best_val_loss": best_val_loss,
            "architecture": "v6_relation",
        }

        # Gated composite: val_loss gate + temporal_top100 selector
        temporal_top100 = temp["top100_pct"] if temp else 0.0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        loss_gate_ok = val_loss <= best_val_loss * 1.01
        if loss_gate_ok and temporal_top100 > best_temporal_top100:
            best_temporal_top100 = temporal_top100
            resume_state["best_temporal_top100"] = best_temporal_top100
            resume_state["best_val_loss"] = best_val_loss
            resume_state["best_epoch"] = epoch
            torch.save(resume_state, ckpt_path)
            print(f"  -> New best: temporal top-100={best_temporal_top100*100:.2f}%, "
                  f"val_loss={val_loss:.5f} (gate: {best_val_loss*1.01:.5f}) "
                  f"(saved {ckpt_path})", flush=True)
        elif temporal_top100 > best_temporal_top100:
            print(f"  -> temporal top-100={temporal_top100*100:.2f}% is new best BUT "
                  f"val_loss={val_loss:.5f} > gate {best_val_loss*1.01:.5f} -- SKIPPED", flush=True)

        # Always save latest
        latest_path = Path(ckpt_path).with_stem(Path(ckpt_path).stem + "_latest")
        torch.save(resume_state, latest_path)

    total_time = time.time() - t0
    print(f"\nV6 training complete in {total_time/60:.1f}m | "
          f"Best temporal top-100: {best_temporal_top100*100:.2f}% | "
          f"Best val loss: {best_val_loss:.5f}", flush=True)
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

    if args.phase == "joint":
        run_joint(args, device)
    else:
        raise ValueError("--phase must be 'joint' for v6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unicorn v6 Relation Network training")
    parser.add_argument("--phase", required=True, choices=["joint"],
                        help="Training phase (v6 only supports 'joint')")
    parser.add_argument("--parquet", default="possessions.parquet",
                        help="Possessions parquet (for temporal eval)")
    parser.add_argument("--stints", default="stints.parquet",
                        help="Stint-level parquet (for training)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=2048)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Transformer encoder layers (default: 2, shallow)")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ckpt", default=None, help="Output checkpoint path")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--text-emb", default=None, help="Text embeddings path")
    parser.add_argument("--lookup-csv", default="player_season_lookup.csv")
    # Delta
    parser.add_argument("--delta-dim", type=int, default=64)
    parser.add_argument("--delta-reg-weight", type=float, default=0.05)
    parser.add_argument("--delta-max-norm", type=float, default=0.3)
    # Loss weights
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--temporal-aug-prob", type=float, default=0.15)
    parser.add_argument("--outcome-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # v6-specific
    parser.add_argument("--d-pair", type=int, default=64,
                        help="Pairwise relation output dimension")
    parser.add_argument("--d-pair-hidden", type=int, default=256,
                        help="Pairwise relation hidden dimension")
    parser.add_argument("--n-masks", type=int, default=2,
                        help="Number of players to mask per stint for contrastive")
    args = parser.parse_args()
    main(args)
