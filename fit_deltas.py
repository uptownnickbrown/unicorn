#!/usr/bin/env python3
"""
Post-training delta fitting for Unicorn.

Freezes the entire model and optimizes delta embeddings for val/test era
player-seasons using contrastive loss only (no outcome leakage).

This is analogous to fitting new word embeddings in a frozen language model.
The encoder knows how to interpret deltas from training; we just learn the
right delta values for player-seasons it hasn't seen.

Usage:
    python fit_deltas.py --ckpt joint_v32_checkpoint_latest.pt --steps 300
    python fit_deltas.py --ckpt joint_v32_checkpoint_latest.pt --steps 300 --output fitted_checkpoint.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from nba_dataset import PossessionDataset, get_num_players
from prior_year_init import build_prior_year_map, build_ps_to_base_tensor


def load_model_for_fitting(ckpt_path: str, device: torch.device):
    """Load model from checkpoint for delta fitting."""
    from train_transformer import LineupTransformer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"]
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    num_ps = ckpt["num_player_seasons"]
    num_base = ckpt["num_base_players"]
    d_model = ckpt.get("d_model", 384)
    n_layers = ckpt.get("n_layers", 8)
    n_heads = ckpt.get("n_heads", 8)
    dropout = ckpt.get("dropout", 0.1)
    delta_dim = ckpt.get("delta_dim", 0)
    attn_temperature = ckpt.get("attn_temperature", 1.0)
    pool_type = ckpt.get("pool_type", "static")
    pool_heads = ckpt.get("pool_heads", 4)
    pool_multi_layer = ckpt.get("pool_multi_layer", False)
    film_state = ckpt.get("film_state", False)

    ps_to_base, _ = build_ps_to_base_tensor(num_ps)
    model = LineupTransformer(
        num_ps, num_base, ps_to_base, d_model, n_layers, n_heads, dropout,
        delta_dim=delta_dim, attn_temperature=attn_temperature,
        pool_type=pool_type, pool_heads=pool_heads,
        pool_multi_layer=pool_multi_layer, film_state=film_state,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model, ckpt


def init_deltas_from_prior(model, lookup_csv: str = "player_season_lookup.csv"):
    """Initialize unfitted deltas from their most recent prior-year delta.

    For player-seasons with zero deltas that have a prior-year mapping to a
    non-zero delta, copy the prior delta as starting point.
    """
    if model.delta_dim == 0:
        return 0

    prior_map = build_prior_year_map(lookup_csv)
    delta_raw = model.delta_raw.weight.data  # [num_ps, delta_dim]
    norms = delta_raw.norm(dim=1)
    num_inited = 0

    for ps_id in range(delta_raw.size(0)):
        if norms[ps_id] > 1e-6:
            continue  # already has a trained delta
        # Walk the prior-year chain to find a non-zero delta
        current = ps_id
        for _ in range(5):  # max 5 hops back
            prior = prior_map.get(current)
            if prior is None or prior >= delta_raw.size(0):
                break
            if norms[prior] > 1e-6:
                delta_raw[ps_id] = delta_raw[prior]
                num_inited += 1
                break
            current = prior

    return num_inited


def fit_deltas(
    model, dataloader, device,
    fitted_mask: torch.Tensor,
    steps: int = 300,
    lr: float = 1e-3,
    delta_reg_weight: float = 0.05,
    delta_max_norm: float = 0.3,
):
    """Optimize delta_raw using contrastive loss with everything else frozen.

    Only unfitted deltas are updated. Training-era deltas (already fitted during
    training) are protected via a gradient mask and restored after each step.

    Args:
        fitted_mask: [num_ps] bool tensor. True = already fitted (freeze), False = fit now.
    """
    # Freeze everything except delta_raw
    for name, param in model.named_parameters():
        if "delta_raw" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Save training-era deltas so we can restore them after each step
    original_delta = model.delta_raw.weight.data.clone()
    fitted_mask = fitted_mask.to(device)

    n_to_fit = (~fitted_mask).sum().item()
    n_frozen = fitted_mask.sum().item()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Delta entries: {n_to_fit} to fit, {n_frozen} frozen (training-era)", flush=True)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    model.train()
    step = 0
    loss_sum, n_sum = 0.0, 0
    data_iter = iter(dataloader)

    print(f"  Fitting deltas for {steps} steps...", flush=True)
    t0 = time.time()

    while step < steps:
        try:
            pl, st, tm, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            pl, st, tm, _ = next(data_iter)

        pl, st, tm = pl.to(device), st.to(device), tm.to(device)
        B = pl.size(0)

        # Randomly mask one player
        mask_idx = torch.randint(0, 10, (B,), device=device)
        target_ps_ids = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)
        target_base_ids = model.ps_to_base[target_ps_ids]

        # Forward (contrastive only — use pretrain path)
        pred_emb, _, _ = model.forward_pretrain(pl, st, tm, mask_idx)
        pred_norm = F.normalize(pred_emb, dim=1)

        # Base-player contrastive loss (2310-way InfoNCE)
        all_base_emb = model.base_player_emb.weight.detach()
        base_norm = F.normalize(all_base_emb, dim=1)
        sim_matrix = pred_norm @ base_norm.T / model.temperature
        contrastive_loss = F.cross_entropy(sim_matrix, target_base_ids)

        # Delta regularization (only on unfitted deltas)
        unfitted_projected = model.delta_proj(model.delta_raw.weight[~fitted_mask])
        delta_reg = unfitted_projected.norm(dim=1).mean()

        loss = contrastive_loss + delta_reg_weight * delta_reg
        loss.backward()

        # Zero out gradients for already-fitted (training-era) deltas
        with torch.no_grad():
            model.delta_raw.weight.grad[fitted_mask] = 0.0

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Restore training-era deltas (belt + suspenders with gradient masking)
        with torch.no_grad():
            model.delta_raw.weight.data[fitted_mask] = original_delta[fitted_mask]

        # Hard cap on projected delta norms (unfitted only)
        if delta_max_norm > 0 and model.delta_dim > 0:
            with torch.no_grad():
                projected = model.delta_proj(model.delta_raw.weight)
                norms = projected.norm(dim=1, keepdim=True)
                scale = (delta_max_norm / norms).clamp(max=1.0)
                # Only cap unfitted deltas
                needs_cap = ~fitted_mask & (scale.squeeze() < 1.0)
                if needs_cap.any():
                    model.delta_raw.weight.data[needs_cap] *= scale[needs_cap]

        loss_sum += contrastive_loss.item()
        n_sum += 1
        step += 1

        if step % 50 == 0 or step == steps:
            elapsed = time.time() - t0
            with torch.no_grad():
                all_delta_norms = model.delta_proj(model.delta_raw.weight).norm(dim=1)
                unfitted_norms = all_delta_norms[~fitted_mask]
                fitted_norms = all_delta_norms[fitted_mask]
                top1 = (sim_matrix.argmax(1) == target_base_ids).float().mean().item()
            print(f"    Step {step}/{steps} | "
                  f"loss={loss_sum/n_sum:.4f} | "
                  f"base_top1={top1*100:.1f}% | "
                  f"unfitted_norm={unfitted_norms.mean():.4f} | "
                  f"fitted_norm={fitted_norms.mean():.4f} | "
                  f"{elapsed:.0f}s", flush=True)
            loss_sum, n_sum = 0.0, 0


def main():
    parser = argparse.ArgumentParser(description="Fit delta embeddings for val/test era player-seasons")
    parser.add_argument("--ckpt", required=True, help="Input checkpoint path")
    parser.add_argument("--output", default=None, help="Output checkpoint path (default: <ckpt>_fitted.pt)")
    parser.add_argument("--parquet", default="possessions.parquet")
    parser.add_argument("--lookup-csv", default="player_season_lookup.csv")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for delta fitting")
    parser.add_argument("--bs", type=int, default=2048, help="Batch size")
    parser.add_argument("--delta-reg-weight", type=float, default=0.05)
    parser.add_argument("--delta-max-norm", type=float, default=0.3)
    args = parser.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}", flush=True)

    # Load model
    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model, ckpt = load_model_for_fitting(args.ckpt, device)

    # Pre-fitting delta stats — identify which deltas are already fitted (training-era)
    with torch.no_grad():
        if model.delta_dim > 0:
            delta_norms = model.delta_proj(model.delta_raw.weight).norm(dim=1)
        else:
            delta_norms = model.delta_emb.weight.norm(dim=1)
        non_zero = (delta_norms > 1e-6).sum().item()
        # Mark training-era deltas as "fitted" (non-zero before prior-year init)
        fitted_mask = delta_norms > 1e-6  # [num_ps] bool tensor
        print(f"  Pre-fitting: {non_zero}/{delta_norms.size(0)} non-zero deltas "
              f"(mean={delta_norms[fitted_mask].mean():.4f} for fitted, "
              f"{delta_norms.size(0) - non_zero} unfitted)", flush=True)

    # Initialize unfitted deltas from prior-year
    n_init = init_deltas_from_prior(model, args.lookup_csv)
    print(f"  Initialized {n_init} unfitted deltas from prior-year chain", flush=True)

    # Create val+test dataloader (contrastive signal from val/test era lineups)
    val_ds = PossessionDataset(args.parquet, split="val", shuffle_players=False,
                                lookup_csv=args.lookup_csv, prior_strength=10.0)
    test_ds = PossessionDataset(args.parquet, split="test", shuffle_players=False,
                                 lookup_csv=args.lookup_csv, prior_strength=10.0)
    combined_ds = ConcatDataset([val_ds, test_ds])
    dataloader = DataLoader(combined_ds, batch_size=args.bs, shuffle=True,
                            pin_memory=True, num_workers=4, persistent_workers=True)
    print(f"  Val+Test data: {len(combined_ds):,} possessions", flush=True)

    # Fit deltas (only unfitted ones — training-era deltas are frozen)
    fit_deltas(model, dataloader, device,
               fitted_mask=fitted_mask,
               steps=args.steps, lr=args.lr,
               delta_reg_weight=args.delta_reg_weight,
               delta_max_norm=args.delta_max_norm)

    # Post-fitting delta stats
    with torch.no_grad():
        if model.delta_dim > 0:
            delta_norms_post = model.delta_proj(model.delta_raw.weight).norm(dim=1)
        else:
            delta_norms_post = model.delta_emb.weight.norm(dim=1)
        non_zero_post = (delta_norms_post > 1e-6).sum().item()
        print(f"\n  Post-fitting: {non_zero_post}/{delta_norms_post.size(0)} non-zero deltas "
              f"(mean={delta_norms_post.mean():.4f})", flush=True)

    # Save
    output_path = args.output
    if output_path is None:
        stem = Path(args.ckpt).stem
        if stem.endswith("_latest"):
            stem = stem[:-len("_latest")]
        output_path = str(Path(args.ckpt).parent / f"{stem}_fitted.pt")

    # Update checkpoint with fitted deltas
    save_dict = dict(ckpt)  # copy all metadata
    save_dict["state_dict"] = model.state_dict()
    save_dict["delta_fitted"] = True
    save_dict["delta_fit_steps"] = args.steps
    save_dict["delta_fit_lr"] = args.lr
    torch.save(save_dict, output_path)
    print(f"\nSaved fitted checkpoint: {output_path}", flush=True)


if __name__ == "__main__":
    main()
