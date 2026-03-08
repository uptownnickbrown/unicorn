# train_transformer.py – v10 (v5: cross-attention pooling + multi-layer + FiLM)
"""
Unicorn transformer with:
  - Composed embeddings: base_player_emb + delta (base shared across seasons)
  - v3.2: Distributional outcome prediction (Bayesian-smoothed lineup targets)
  - Dual forward pass: masked for contrastive, unmasked for outcome
  - Split offense/defense attention pooling
  - Base-player contrastive (2,310-way InfoNCE)
  - Auxiliary base-player classification head
  - Temporal augmentation (swap to adjacent season)
  - LLM-seeded embedding initialization support
  - v5: Cross-attention pooling with state-conditioned query,
        multi-layer pooling input, FiLM state conditioning

Run examples:
```bash
# Phase A: Contrastive pretraining (v2.0, legacy)
python train_transformer.py --phase pretrain --epochs 25

# Phase B: Outcome prediction fine-tuning (v2.0, legacy)
python train_transformer.py --phase finetune --pretrain-ckpt pretrain_v2_checkpoint.pt --epochs 15

# Joint training (v3.2: distributional)
python train_transformer.py --phase joint --epochs 25 --delta-dim 64 --outcome-weight 1.0 --contrastive-weight 0.5 --prior-strength 10

# Joint training (v5: cross-attention pooling + multi-layer + FiLM)
python train_transformer.py --phase joint --epochs 30 --delta-dim 64 --outcome-weight 1.0 --contrastive-weight 0.5 --prior-strength 10 --pool-type cross-attn --pool-heads 4 --pool-multi-layer --film-state
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

    Args:
        d_model: embedding dimension
        temperature: divide logits by this before softmax (lower = sharper attention).
                     Default 1.0 (no scaling). Try 0.1 for more differentiated weights.
    """
    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)
        self.temperature = temperature

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.query(h) / self.temperature       # [B, N, 1]
        weights = torch.softmax(scores, dim=1)          # [B, N, 1]
        pooled = (weights * h).sum(dim=1)               # [B, d_model]
        return pooled, weights.squeeze(-1)              # [B, d_model], [B, N]

class CrossAttentionPool(nn.Module):
    """Multi-head cross-attention pooling with a state-conditioned learned query.

    Instead of a static linear projection (which converges to uniform weights),
    this uses full dot-product attention with multiple heads. The query is a
    learned vector PLUS a projection of the game state, so "who to attend to"
    depends on the game situation.

    Args:
        d_model: embedding dimension
        n_heads: number of attention heads
    """
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.state_to_query = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0,
        )

    def forward(self, h: torch.Tensor, state_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, N, d_model] player tokens
            state_repr: [B, d_model] projected game state
        Returns:
            pooled: [B, d_model]
            weights: [B, N] attention weights (averaged across heads)
        """
        B = h.size(0)
        query = self.query_token.expand(B, 1, -1) + self.state_to_query(state_repr).unsqueeze(1)
        pooled, weights = self.cross_attn(query, h, h)  # [B,1,d], [B,1,N]
        return pooled.squeeze(1), weights.squeeze(1)     # [B,d], [B,N]


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: produces per-layer scale and shift from state.

    After each transformer layer's output, apply: h = gamma * h + beta
    where gamma, beta are derived from the game state.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * d_model)
        # Initialize gamma near 1, beta near 0 (identity at init)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:d_model] = 1.0  # gamma init = 1

    def forward(self, h: torch.Tensor, state_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, d_model] transformer layer output
            state_repr: [B, d_model] projected game state
        Returns:
            modulated h: [B, N, d_model]
        """
        gamma_beta = self.proj(state_repr)        # [B, 2*d_model]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, d_model]
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


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
        delta_dim: int = 0,
        attn_temperature: float = 1.0,
        pool_type: str = "static",        # "static" or "cross-attn"
        pool_heads: int = 4,              # heads for cross-attention pooling
        pool_multi_layer: bool = False,    # pool from mid+final layers
        film_state: bool = False,          # FiLM state conditioning on encoder
    ):
        super().__init__()
        self.d_model = d_model
        self.num_player_seasons = num_player_seasons
        self.num_base_players = num_base_players
        self.delta_dim = delta_dim
        self.attn_temperature = attn_temperature
        self.pool_type = pool_type
        self.pool_multi_layer = pool_multi_layer
        self.film_state = film_state

        # Composed embeddings: base (shared across seasons) + delta (season-specific)
        self.base_player_emb = nn.Embedding(num_base_players, d_model)

        if delta_dim > 0:
            # v2.1: Low-rank delta bottleneck — delta_raw → delta_proj → d_model
            self.delta_raw = nn.Embedding(num_player_seasons, delta_dim)
            nn.init.zeros_(self.delta_raw.weight)
            self.delta_proj = nn.Linear(delta_dim, d_model, bias=False)
            nn.init.orthogonal_(self.delta_proj.weight)
        else:
            # v2.0: Full-rank delta embedding
            self.delta_emb = nn.Embedding(num_player_seasons, d_model)
            nn.init.zeros_(self.delta_emb.weight)  # delta starts at zero

        # Non-trainable mapping: player_season_id → base_player_id
        self.register_buffer("ps_to_base", ps_to_base)

        # Team and position
        self.team_emb = nn.Embedding(2, d_model)  # 0=offense, 1=defense
        self.pos_bias = nn.Parameter(torch.zeros(10, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # State projection (used for pooling query + outcome head)
        self.state_proj = nn.Linear(3, d_model)
        # Keep state_pos_bias for backward-compatible checkpoint loading (unused in v3.2+)
        self.state_pos_bias = nn.Parameter(torch.zeros(1, d_model))

        # Transformer encoder
        n_lower = n_layers // 2 if pool_multi_layer else n_layers
        n_upper = n_layers - n_lower if pool_multi_layer else 0

        def _make_encoder(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4 * d_model, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        if pool_multi_layer:
            # Split encoder: lower (player identity) + upper (lineup context)
            self.encoder_lower = _make_encoder(n_lower)
            self.encoder_upper = _make_encoder(n_upper)
            # Project concatenated mid+final back to d_model for pooling
            self.pool_proj = nn.Linear(2 * d_model, d_model)
            # Keep a dummy 'encoder' for backward-compat checkpoint loading
            self.encoder = nn.Identity()
        else:
            self.encoder = _make_encoder(n_layers)

        # FiLM state conditioning on encoder layers
        if film_state:
            self.film_layers = nn.ModuleList([FiLMLayer(d_model) for _ in range(n_layers)])

        # Attention pooling: split offense/defense for outcome head
        if pool_type == "cross-attn":
            self.attn_pool_off = CrossAttentionPool(d_model, n_heads=pool_heads)
            self.attn_pool_def = CrossAttentionPool(d_model, n_heads=pool_heads)
            # Legacy static pool for contrastive/logging (cheap, always present)
            self.attn_pool = AttentionPool(d_model, temperature=attn_temperature)
        else:
            self.attn_pool = AttentionPool(d_model, temperature=attn_temperature)
            self.attn_pool_off = AttentionPool(d_model, temperature=attn_temperature)
            self.attn_pool_def = AttentionPool(d_model, temperature=attn_temperature)

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

        # Outcome head: takes [off_pooled, def_pooled, state_repr] → 9-way
        self.outcome_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_OUTCOMES),
        )

    def _compose_embedding(self, player_season_ids: torch.Tensor) -> torch.Tensor:
        """Compose player embedding: base[ps_to_base[id]] + delta[id]."""
        base_ids = self.ps_to_base[player_season_ids]
        base = self.base_player_emb(base_ids)
        if self.delta_dim > 0:
            delta = self.delta_proj(self.delta_raw(player_season_ids))
        else:
            delta = self.delta_emb(player_season_ids)
        return base + delta

    def _all_composed_embeddings(self) -> torch.Tensor:
        """Get composed embeddings for the full vocabulary. [V, d_model]"""
        all_base_ids = self.ps_to_base  # [V]
        base = self.base_player_emb(all_base_ids)
        if self.delta_dim > 0:
            delta = self.delta_proj(self.delta_raw.weight)
        else:
            delta = self.delta_emb.weight
        return base + delta

    def _embed_players(self, players: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        """Embed player tokens with team and position information."""
        tok = self._compose_embedding(players) + self.team_emb(team_ids) + self.pos_bias
        return tok  # [B, 10, d_model]

    def _encode(self, tok: torch.Tensor, state_repr: torch.Tensor | None = None):
        """Run tokens through encoder, returning final output and pooling input.

        With pool_multi_layer: returns (h_final, h_pool) where h_pool is projected
        concatenation of mid-layer and final-layer representations.
        Without: returns (h_final, h_final).

        If film_state is enabled and state_repr is provided, applies FiLM
        conditioning after each encoder layer.
        """
        if self.pool_multi_layer:
            if self.film_state and state_repr is not None:
                h = self._encode_with_film(tok, state_repr, self.encoder_lower, layer_offset=0)
                h_mid = h
                h = self._encode_with_film(h_mid, state_repr, self.encoder_upper,
                                           layer_offset=len(self.encoder_lower.layers))
            else:
                h_mid = self.encoder_lower(tok)
                h = self.encoder_upper(h_mid)
            h_pool = self.pool_proj(torch.cat([h_mid, h], dim=-1))
            return h, h_pool
        else:
            if self.film_state and state_repr is not None:
                h = self._encode_with_film(tok, state_repr, self.encoder, layer_offset=0)
            else:
                h = self.encoder(tok)
            return h, h

    def _encode_with_film(self, tok, state_repr, encoder_block, layer_offset=0):
        """Run tokens through an encoder block with FiLM modulation after each layer."""
        h = tok
        for i, layer in enumerate(encoder_block.layers):
            h = layer(h)
            h = self.film_layers[layer_offset + i](h, state_repr)
        return h

    def _pool_for_outcome(self, h_pool, state_repr):
        """Pool offense/defense from h_pool, return (off_pooled, def_pooled, off_attn, def_attn)."""
        if self.pool_type == "cross-attn":
            off_pooled, off_attn = self.attn_pool_off(h_pool[:, :5, :], state_repr)
            def_pooled, def_attn = self.attn_pool_def(h_pool[:, 5:, :], state_repr)
        else:
            off_pooled, off_attn = self.attn_pool_off(h_pool[:, :5, :])
            def_pooled, def_attn = self.attn_pool_def(h_pool[:, 5:, :])
        return off_pooled, def_pooled, off_attn, def_attn

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

        state_repr = self.state_proj(state)
        h, _ = self._encode(tok, state_repr)  # [B, 10, d_model]

        # Extract the masked position's output
        mask_idx_h = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        masked_repr = h.gather(1, mask_idx_h).squeeze(1)  # [B, d_model]

        # Contrastive: project to embedding space
        pred_emb = self.mask_proj(masked_repr)  # [B, d_model]

        # Auxiliary: base-player classification
        aux_logits = self.aux_base_head(masked_repr)  # [B, num_base_players]

        _, attn_weights = self.attn_pool(h)  # [B, 10]

        return pred_emb, aux_logits, attn_weights

    def forward_finetune(self, players, state, team_ids):
        """Outcome prediction (unmasked lineup, split off/def pooling).

        Args:
            players: [B, 10] player-season IDs
            state: [B, 3] game state features
            team_ids: [B, 10] team indicators (0/1)

        Returns:
            logits: [B, NUM_OUTCOMES] outcome prediction
            off_attn: [B, 5] offensive attention weights
            def_attn: [B, 5] defensive attention weights
        """
        tok = self._embed_players(players, team_ids)               # [B, 10, d_model]
        state_repr = self.state_proj(state)                        # [B, d_model]
        _, h_pool = self._encode(tok, state_repr)                  # [B, 10, d_model]
        off_pooled, def_pooled, off_attn, def_attn = self._pool_for_outcome(h_pool, state_repr)
        combined = torch.cat([off_pooled, def_pooled, state_repr], dim=1)  # [B, 3*d_model]
        logits = self.outcome_head(combined)                       # [B, NUM_OUTCOMES]
        return logits, off_attn, def_attn

    def forward_joint(self, players, state, team_ids, mask_idx):
        """Joint forward: dual pass — masked for contrastive, unmasked for outcome.

        Pass 1 (masked): Mask one player, encode, extract masked position for
        contrastive prediction and auxiliary classification.
        Pass 2 (unmasked): Full lineup, encode, split off/def pooling + state
        for outcome distribution prediction.

        Args:
            players: [B, 10] player-season IDs
            state: [B, 3] game state features
            team_ids: [B, 10] team indicators (0/1)
            mask_idx: [B] index (0-9) of the masked player

        Returns:
            pred_emb: [B, d_model] predicted embedding (for contrastive loss)
            aux_logits: [B, num_base_players] auxiliary base-player logits
            outcome_logits: [B, NUM_OUTCOMES] outcome prediction logits
            off_attn: [B, 5] offensive attention weights
            def_attn: [B, 5] defensive attention weights
        """
        B = players.size(0)
        state_repr = self.state_proj(state)  # [B, d_model]

        # --- Pass 1: Contrastive (masked lineup) ---
        tok_masked = self._embed_players(players, team_ids)  # [B, 10, d_model]
        mask_idx_expanded = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        tok_masked.scatter_(1, mask_idx_expanded,
                            self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_model))
        h_masked, _ = self._encode(tok_masked, state_repr)  # [B, 10, d_model]

        mask_idx_h = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        masked_repr = h_masked.gather(1, mask_idx_h).squeeze(1)  # [B, d_model]
        pred_emb = self.mask_proj(masked_repr)
        aux_logits = self.aux_base_head(masked_repr)

        # --- Pass 2: Outcome (unmasked lineup) ---
        tok_unmasked = self._embed_players(players, team_ids)  # [B, 10, d_model]
        _, h_pool = self._encode(tok_unmasked, state_repr)  # [B, 10, d_model]

        off_pooled, def_pooled, off_attn, def_attn = self._pool_for_outcome(h_pool, state_repr)
        combined = torch.cat([off_pooled, def_pooled, state_repr], dim=1) # [B, 3*d_model]
        outcome_logits = self.outcome_head(combined)

        return pred_emb, aux_logits, outcome_logits, off_attn, def_attn

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


def finetune_epoch(model, dataloader, optimizer, scheduler, device, accum):
    """Outcome prediction epoch (train or eval).

    v3.2: Uses soft cross-entropy against distributional targets.
    Returns (loss, accuracy) where accuracy = argmax match against target distribution.
    """
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0

    for step, (pl, st, tm, target_dist) in enumerate(tqdm(dataloader, leave=False, desc="finetune" if train else "eval")):
        pl, st, tm, target_dist = pl.to(device), st.to(device), tm.to(device), target_dist.to(device)

        logits, _, _ = model.forward_finetune(pl, st, tm)
        # Soft cross-entropy against distributional targets
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target_dist * log_probs).sum(dim=1).mean()
        if train:
            loss_scaled = loss / accum
            loss_scaled.backward()
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.item() * pl.size(0)

        correct += (logits.argmax(1) == target_dist.argmax(1)).sum().item()
        total += pl.size(0)

    if train and scheduler is not None:
        scheduler.step()

    return loss_sum / total, correct / total

def joint_epoch(
    model, dataloader, optimizer, scheduler, device, accum,
    temporal_swap_tensor, temporal_aug_prob,
    delta_reg_weight, aux_loss_weight, outcome_weight,
    delta_max_norm=0.0, contrastive_weight=1.0,
    attn_entropy_weight=0.0,
):
    """Joint training loop: contrastive masked prediction + distributional outcome.

    v3.2: Outcome loss uses soft cross-entropy against Bayesian-smoothed
    distributional targets instead of class-weighted hard CE.
    """
    model.train()
    total = 0
    loss_sum, contrastive_loss_sum, aux_loss_sum = 0.0, 0.0, 0.0
    outcome_loss_sum, delta_reg_sum, attn_entropy_sum = 0.0, 0.0, 0.0
    correct_top1, correct_top5, aux_correct, outcome_correct = 0, 0, 0, 0
    pred_cossim_sum, pred_cossim_n = 0.0, 0
    off_entropy_sum, def_entropy_sum, max_attn_sum, entropy_n = 0.0, 0.0, 0.0, 0

    for step, (pl, st, tm, target_dist) in enumerate(tqdm(dataloader, leave=False, desc="joint")):
        pl, st, tm, target_dist = pl.to(device), st.to(device), tm.to(device), target_dist.to(device)
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

        # Joint forward (dual pass: masked for contrastive, unmasked for outcome)
        pred_emb, aux_logits, outcome_logits, off_attn, def_attn = model.forward_joint(pl, st, tm, mask_idx)

        # --- Contrastive loss (base-player InfoNCE, 2310-way) ---
        pred_norm = F.normalize(pred_emb, dim=1)          # [B, d_model]
        if contrastive_weight > 0:
            all_base_emb = model.base_player_emb.weight.detach()  # [2310, d_model]
            base_norm = F.normalize(all_base_emb, dim=1)          # [2310, d_model]
            sim_matrix = pred_norm @ base_norm.T / model.temperature  # [B, 2310]
            # No false-negative masking needed — each base player appears exactly once
            contrastive_loss = F.cross_entropy(sim_matrix, target_base_ids)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            sim_matrix = None

        # --- Auxiliary base-player classification loss ---
        aux_loss = F.cross_entropy(aux_logits, target_base_ids)

        # --- Outcome loss (soft cross-entropy against distributional targets) ---
        log_probs = F.log_softmax(outcome_logits, dim=1)       # [B, 9]
        outcome_loss = -(target_dist * log_probs).sum(dim=1).mean()

        # --- Delta regularization (on projected 384-dim delta, not raw 64-dim) ---
        if model.delta_dim > 0:
            delta_reg = model.delta_proj(model.delta_raw.weight).norm(dim=1).mean()
        else:
            delta_reg = model.delta_emb.weight.norm(dim=1).mean()

        # --- Attention entropy penalty (minimize entropy = sharpen attention) ---
        if attn_entropy_weight > 0:
            off_ent = -(off_attn * (off_attn + 1e-8).log()).sum(dim=1).mean()
            def_ent = -(def_attn * (def_attn + 1e-8).log()).sum(dim=1).mean()
            attn_entropy_loss = (off_ent + def_ent) / 2
        else:
            attn_entropy_loss = torch.tensor(0.0, device=device)

        # --- Total loss ---
        loss = (contrastive_weight * contrastive_loss
                + aux_loss_weight * aux_loss
                + outcome_weight * outcome_loss
                + delta_reg_weight * delta_reg
                + attn_entropy_weight * attn_entropy_loss)
        (loss / accum).backward()

        if (step + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Hard cap on projected delta norms (scale raw to keep projected ≤ cap)
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
        attn_entropy_sum += attn_entropy_loss.item() * B

        if sim_matrix is not None:
            correct_top1 += (sim_matrix.argmax(1) == target_base_ids).sum().item()
            _, top5_pred = sim_matrix.topk(5, dim=1)
            correct_top5 += (top5_pred == target_base_ids.unsqueeze(1)).any(dim=1).sum().item()
        aux_correct += (aux_logits.argmax(1) == target_base_ids).sum().item()
        # Outcome accuracy: model's top prediction matches lineup's most common outcome
        outcome_correct += (outcome_logits.argmax(1) == target_dist.argmax(1)).sum().item()

        # Attention weight statistics (sample every 10 steps)
        if step % 10 == 0:
            with torch.no_grad():
                oe = -(off_attn * (off_attn + 1e-8).log()).sum(dim=1).mean().item()
                de = -(def_attn * (def_attn + 1e-8).log()).sum(dim=1).mean().item()
                off_entropy_sum += oe
                def_entropy_sum += de
                max_attn_sum += off_attn.max(dim=1).values.mean().item()
                entropy_n += 1

        # Embedding collapse detection
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
        "outcome_loss": outcome_loss_sum / total,
        "delta_reg": delta_reg_sum / total,
        "attn_entropy": attn_entropy_sum / total,
        "base_top1": correct_top1 / total,
        "base_top5": correct_top5 / total,
        "aux_acc": aux_correct / total,
        "outcome_acc": outcome_correct / total,
        "pred_cossim": pred_cossim_sum / max(pred_cossim_n, 1),
        "off_attn_entropy": off_entropy_sum / max(entropy_n, 1),
        "def_attn_entropy": def_entropy_sum / max(entropy_n, 1),
        "max_attn_weight": max_attn_sum / max(entropy_n, 1),
    }


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
        model.load_state_dict(resume_ckpt["state_dict"], strict=False)
        if "optimizer_state" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            except ValueError:
                print("  WARNING: optimizer state mismatch (architecture changed), using fresh optimizer", flush=True)
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
        model.load_state_dict(ckpt["state_dict"], strict=False)
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
        prior_strength=args.prior_strength,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = args.ckpt or "finetune_v2_checkpoint.pt"
    log_path = Path(ckpt_path).with_suffix(".log.jsonl")
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = finetune_epoch(
            model, train_dl, optimizer, scheduler, device, args.accum,
        )
        val_loss, val_acc = finetune_epoch(
            model, val_dl, None, None, device, 1,
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
# Joint Training (v2.1)
################################################################################

def run_joint(args, device):
    print("=" * 60, flush=True)
    print("JOINT TRAINING (v3.2): Distributional Outcome + Contrastive", flush=True)
    print("=" * 60, flush=True)

    # Data: season-based splits with Bayesian-smoothed distributional targets
    train_dl, val_dl, _ = get_default_dataloaders(
        args.parquet, batch_size=args.bs, shuffle_train=True,
        prior_strength=args.prior_strength,
    )
    print(f"Prior strength (alpha): {args.prior_strength}", flush=True)

    # Temporal val: season-based val (2019-2020) for temporal_eval
    temporal_ds = PossessionDataset(args.parquet, split="val", shuffle_players=False,
                                    prior_strength=args.prior_strength)
    temporal_dl = DataLoader(
        temporal_ds, batch_size=args.bs, shuffle=False,
        pin_memory=True, num_workers=4, persistent_workers=True,
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
    pool_type = getattr(args, "pool_type", "static")
    pool_heads = getattr(args, "pool_heads", 4)
    pool_multi_layer = getattr(args, "pool_multi_layer", False)
    film_state = getattr(args, "film_state", False)

    model = LineupTransformer(
        num_player_seasons, num_base, ps_to_base,
        args.d_model, args.layers, args.heads, args.dropout,
        delta_dim=args.delta_dim,
        attn_temperature=args.attn_temperature,
        pool_type=pool_type,
        pool_heads=pool_heads,
        pool_multi_layer=pool_multi_layer,
        film_state=film_state,
    ).to(device)
    print(f"Model: d={args.d_model}, L={args.layers}, H={args.heads}, "
          f"delta_dim={args.delta_dim}, pool={pool_type}, "
          f"multi_layer={pool_multi_layer}, film={film_state}, "
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

    # Differential LR for joint training
    base_params = list(model.base_player_emb.parameters())
    if args.delta_dim > 0:
        delta_params = list(model.delta_raw.parameters()) + list(model.delta_proj.parameters())
    else:
        delta_params = list(model.delta_emb.parameters())

    # Collect encoder parameters (handles split encoder for multi-layer)
    encoder_param_list = []
    if pool_multi_layer:
        encoder_param_list += list(model.encoder_lower.parameters())
        encoder_param_list += list(model.encoder_upper.parameters())
        encoder_param_list += list(model.pool_proj.parameters())
    else:
        encoder_param_list += list(model.encoder.parameters())
    encoder_param_list += (
        list(model.attn_pool.parameters())
        + list(model.attn_pool_off.parameters())
        + list(model.attn_pool_def.parameters())
        + [model.pos_bias]
        + list(model.team_emb.parameters())
    )
    if film_state:
        encoder_param_list += list(model.film_layers.parameters())

    head_params = (
        list(model.mask_proj.parameters())
        + list(model.aux_base_head.parameters())
        + list(model.outcome_head.parameters())
        + list(model.state_proj.parameters())
    )

    optimizer = optim.AdamW([
        {"params": base_params, "lr": args.lr * 0.1},
        {"params": delta_params, "lr": args.lr * 0.3},
        {"params": encoder_param_list, "lr": args.lr * 0.3},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    warmup_steps = int(0.05 * args.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_steps, 1),
    )

    # torch.compile for training speedup (CUDA only — MPS backend not well supported)
    if hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)
        print("Model compiled with torch.compile", flush=True)

    # v3.2: No class weights needed — distributional targets handle class balance
    ckpt_path = args.ckpt or "joint_v32_checkpoint.pt"
    log_path = Path(ckpt_path).with_suffix(".log.jsonl")
    best_temporal_top100 = 0.0
    best_val_loss = float("inf")
    start_epoch = 1

    # Resume from checkpoint if requested
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.exists():
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        # strict=False allows loading old checkpoints missing state_pos_bias
        model.load_state_dict(resume_ckpt["state_dict"], strict=False)
        if "optimizer_state" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            except ValueError:
                print("  WARNING: optimizer state mismatch (architecture changed), using fresh optimizer", flush=True)
        if "scheduler_state" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        start_epoch = resume_ckpt.get("last_epoch", 0) + 1
        best_temporal_top100 = resume_ckpt.get("best_temporal_top100", 0.0)
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {resume_path} at epoch {start_epoch} "
              f"(best temporal top-100: {best_temporal_top100*100:.2f}%, "
              f"best val loss: {best_val_loss:.5f})", flush=True)
    elif args.resume:
        print(f"WARNING: --resume path {args.resume} not found, starting fresh", flush=True)

    print(f"Outcome weight: {args.outcome_weight}, Contrastive weight: {args.contrastive_weight}, "
          f"Delta reg: {args.delta_reg_weight}, Delta max norm: {args.delta_max_norm}, "
          f"Prior strength: {args.prior_strength}, "
          f"Attn temp: {args.attn_temperature}, Attn entropy weight: {args.attn_entropy_weight}", flush=True)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        ep_start = time.time()

        # Linear warmup
        if epoch <= warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] if "initial_lr" in pg else pg["lr"]
                # Scale base LR by warmup fraction
            warmup_frac = epoch / max(warmup_steps, 1)
            for i, pg in enumerate(optimizer.param_groups):
                base_lrs = [args.lr * 0.1, args.lr * 0.3, args.lr * 0.3, args.lr]
                pg["lr"] = base_lrs[i] * warmup_frac

        metrics = joint_epoch(
            model, train_dl, optimizer,
            scheduler if epoch > warmup_steps else None,
            device, args.accum,
            temporal_swap, args.temporal_aug_prob,
            args.delta_reg_weight, args.aux_loss_weight, args.outcome_weight,
            delta_max_norm=args.delta_max_norm,
            contrastive_weight=args.contrastive_weight,
            attn_entropy_weight=args.attn_entropy_weight,
        )

        # Validation: outcome accuracy on val set (UNMASKED — real deployment scenario)
        val_loss, val_acc = finetune_epoch(
            model, val_dl, None, None, device, 1,
        )

        # Temporal eval
        temp = temporal_eval(model, temporal_dl, prior_id_tensor, device, max_batches=200)

        # Embedding stats
        with torch.no_grad():
            if model.delta_dim > 0:
                delta_norms = model.delta_proj(model.delta_raw.weight).norm(dim=1)
                delta_raw_norms = model.delta_raw.weight.norm(dim=1)
            else:
                delta_norms = model.delta_emb.weight.norm(dim=1)
                delta_raw_norms = delta_norms
            base_norms = model.base_player_emb.weight.norm(dim=1)

        lr_head = optimizer.param_groups[-1]["lr"]
        temp_val = model.temperature.item()
        ep_time = time.time() - ep_start
        elapsed = time.time() - t0
        eta = ep_time * (args.epochs - epoch)

        print(
            f"Ep {epoch:02d}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} (c={metrics['contrastive_loss']:.4f} "
            f"a={metrics['aux_loss']:.4f} o={metrics['outcome_loss']:.4f} "
            f"d={metrics['delta_reg']:.4f}) | "
            f"train base_top5={metrics['base_top5']*100:5.2f}% out={metrics['outcome_acc']*100:5.2f}% | "
            f"val out={val_acc*100:5.2f}% | "
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
            f"  attention: off_entropy={metrics['off_attn_entropy']:.4f} "
            f"def_entropy={metrics['def_attn_entropy']:.4f} "
            f"max_attn={metrics['max_attn_weight']:.4f} "
            f"(uniform@5={1.6094:.4f})",
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
            "val_loss": round(val_loss, 5),
            "val_outcome_acc": round(val_acc, 5),
            "temperature": round(temp_val, 5),
            "delta_norm_mean": round(delta_norms.mean().item(), 5),
            "delta_norm_max": round(delta_norms.max().item(), 5),
            "base_norm_mean": round(base_norms.mean().item(), 5),
            "attn_entropy": round(metrics["attn_entropy"], 5),
            "off_attn_entropy": round(metrics["off_attn_entropy"], 5),
            "def_attn_entropy": round(metrics["def_attn_entropy"], 5),
            "max_attn_weight": round(metrics["max_attn_weight"], 5),
            "contrastive_weight": args.contrastive_weight,
            "prior_strength": args.prior_strength,
            "attn_temperature": args.attn_temperature,
            "attn_entropy_weight": args.attn_entropy_weight,
            "lr_head": lr_head,
            "epoch_sec": round(ep_time, 1),
            "elapsed_sec": round(elapsed, 1),
        }
        if model.delta_dim > 0:
            log_entry["delta_raw_norm_mean"] = round(delta_raw_norms.mean().item(), 5)
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
            "phase": "joint_v3.2",
            "vocab": OUTCOME_VOCAB,
            "num_player_seasons": num_player_seasons,
            "num_base_players": num_base,
            "d_model": args.d_model,
            "n_layers": args.layers,
            "n_heads": args.heads,
            "dropout": args.dropout,
            "delta_dim": args.delta_dim,
            "contrastive_weight": args.contrastive_weight,
            "prior_strength": args.prior_strength,
            "attn_temperature": args.attn_temperature,
            "attn_entropy_weight": args.attn_entropy_weight,
            "pool_type": pool_type,
            "pool_heads": pool_heads,
            "pool_multi_layer": pool_multi_layer,
            "film_state": film_state,
            "num_outcomes": NUM_OUTCOMES,
            "epochs": args.epochs,
            "last_epoch": epoch,
            "best_temporal_top100": best_temporal_top100,
            "best_val_loss": best_val_loss,
            "architecture": "v5_crossattn" if pool_type == "cross-attn" else "v4_distributional",
        }

        # Save best model: gated composite metric
        # Gate: val outcome loss hasn't degraded beyond 1% of best
        # Selector: temporal top-100 improved (embedding quality)
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

        # Always save latest for resumption
        latest_path = Path(ckpt_path).with_stem(Path(ckpt_path).stem + "_latest")
        torch.save(resume_state, latest_path)

    total_time = time.time() - t0
    print(f"\nJoint training complete in {total_time/60:.1f}m | "
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

    if args.phase == "pretrain":
        run_pretrain(args, device)
    elif args.phase == "finetune":
        run_finetune(args, device)
    elif args.phase == "joint":
        run_joint(args, device)
    else:
        raise ValueError("--phase must be 'pretrain', 'finetune', or 'joint'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unicorn v2 transformer training")
    parser.add_argument("--phase", required=True, choices=["pretrain", "finetune", "joint"],
                        help="Training phase: 'pretrain' (contrastive), 'finetune' (outcome), or 'joint' (both)")
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
    # v2.1 joint training
    parser.add_argument("--delta-dim", type=int, default=0,
                        help="Delta bottleneck dimension (0 = full-rank, 64 = bottleneck)")
    parser.add_argument("--outcome-weight", type=float, default=1.0,
                        help="Weight for outcome loss in joint training")
    parser.add_argument("--contrastive-weight", type=float, default=1.0,
                        help="Weight for contrastive loss (0 disables contrastive)")
    # v3.2 distributional
    parser.add_argument("--prior-strength", type=float, default=10.0,
                        help="Bayesian smoothing alpha for distributional targets (higher = more prior weight)")
    # v4 attention
    parser.add_argument("--attn-temperature", type=float, default=1.0,
                        help="Attention pooling temperature (lower = sharper, try 0.1)")
    parser.add_argument("--attn-entropy-weight", type=float, default=0.0,
                        help="Weight for attention entropy penalty (higher = sharper attention, try 0.1)")
    # v5 architecture
    parser.add_argument("--pool-type", default="static", choices=["static", "cross-attn"],
                        help="Pooling type: 'static' (v3/v4) or 'cross-attn' (v5)")
    parser.add_argument("--pool-heads", type=int, default=4,
                        help="Number of attention heads for cross-attention pooling")
    parser.add_argument("--pool-multi-layer", action="store_true",
                        help="Pool from concatenated mid-layer + final-layer representations")
    parser.add_argument("--film-state", action="store_true",
                        help="Apply FiLM state conditioning on each encoder layer")
    args = parser.parse_args()
    main(args)
