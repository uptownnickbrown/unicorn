# train_transformer.py – v4 (attention pooling, masked prediction, two-phase)
"""
Unicorn transformer with:
  - Attention pooling over player tokens (replaces CLS token)
  - Two-phase BERT-style training:
    Phase A (pretrain): Masked player prediction
    Phase B (finetune): Possession outcome prediction
  - Prior-year embedding initialization support
  - Team-side embeddings (offense/defense)

Run examples:
```bash
# Phase A: Masked player pretraining
python train_transformer.py --phase pretrain --epochs 25

# Phase B: Outcome prediction fine-tuning
python train_transformer.py --phase finetune --pretrain-ckpt pretrain_checkpoint.pt --epochs 15
```
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nba_dataset import (
    get_default_dataloaders,
    get_num_players,
    NUM_OUTCOMES,
    OUTCOME_VOCAB,
)
from prior_year_init import build_prior_year_map, init_embeddings_from_prior

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

    Architecture:
    1. Embed players: player_emb + team_emb + pos_bias
    2. Transformer encoder (self-attention among players)
    3. Attention pool over player outputs
    4. Game state projected separately, concatenated with pooled repr
    5. Task head (masked prediction or outcome prediction)
    """

    def __init__(
        self,
        num_players: int,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_players = num_players

        # Embeddings
        self.player_emb = nn.Embedding(num_players, d_model)
        self.team_emb = nn.Embedding(2, d_model)  # 0=offense, 1=defense
        self.pos_bias = nn.Parameter(torch.zeros(10, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))  # [MASK] for Phase A
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

        # Phase A head: masked player prediction (12,821-way)
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_players),
        )

        # Phase B head: outcome prediction (9-way)
        self.outcome_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_OUTCOMES),
        )

    def _embed_players(self, players, team_ids):
        """Embed player tokens with team and position information."""
        tok = self.player_emb(players) + self.team_emb(team_ids) + self.pos_bias
        return tok  # [B, 10, d_model]

    def forward_pretrain(self, players, state, team_ids, mask_idx):
        """Phase A: Masked player prediction.

        Args:
            players: [B, 10] player-season IDs
            state: [B, 3] game state features
            team_ids: [B, 10] team indicators (0/1)
            mask_idx: [B] index (0-9) of the masked player

        Returns:
            logits: [B, num_players] prediction over player identities
            attn_weights: [B, 10] attention pooling weights
        """
        tok = self._embed_players(players, team_ids)  # [B, 10, d_model]

        # Apply mask: replace masked player's embedding with [MASK] token
        B = tok.size(0)
        mask_idx_expanded = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        tok.scatter_(1, mask_idx_expanded, self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_model))

        h = self.encoder(tok)  # [B, 10, d_model]

        # Use the masked position's output for prediction
        mask_idx_h = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_model)
        masked_repr = h.gather(1, mask_idx_h).squeeze(1)  # [B, d_model]

        logits = self.mask_head(masked_repr)  # [B, num_players]
        _, attn_weights = self.attn_pool(h)   # for logging only

        return logits, attn_weights

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


def pretrain_epoch(model, dataloader, criterion, optimizer, scheduler, device, accum):
    """Phase A training loop: masked player prediction."""
    model.train()
    total, correct, top5_correct, loss_sum = 0, 0, 0, 0.0

    for step, (pl, st, tm, _) in enumerate(tqdm(dataloader, leave=False, desc="pretrain")):
        pl, st, tm = pl.to(device), st.to(device), tm.to(device)
        B = pl.size(0)

        # Randomly select one player to mask per sample
        mask_idx = torch.randint(0, 10, (B,), device=device)
        target = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)  # [B]

        logits, _ = model.forward_pretrain(pl, st, tm, mask_idx)
        loss = criterion(logits, target) / accum

        loss.backward()
        if (step + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_sum += loss.item() * accum * B
        correct += (logits.argmax(1) == target).sum().item()
        _, top5_pred = logits.topk(5, dim=1)
        top5_correct += (top5_pred == target.unsqueeze(1)).any(dim=1).sum().item()
        total += B

    if scheduler is not None:
        scheduler.step()

    return loss_sum / total, correct / total, top5_correct / total


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
# Phase A: Masked Player Pretraining
################################################################################

def run_pretrain(args, device):
    print("=" * 60)
    print("PHASE A: Masked Player Pretraining")
    print("=" * 60)

    train_dl, val_dl, _ = get_default_dataloaders(
        args.parquet, batch_size=args.bs, shuffle_train=(args.epochs > 1),
    )

    num_players = get_num_players()
    model = LineupTransformer(
        num_players, args.d_model, args.layers, args.heads, args.dropout,
    ).to(device)
    print(f"Model: {num_players} players, d={args.d_model}, "
          f"L={args.layers}, H={args.heads}, params={sum(p.numel() for p in model.parameters()):,}")

    # Prior-year embedding initialization
    if args.prior_ckpt:
        prior_map = build_prior_year_map(args.lookup_csv)
        n_init = init_embeddings_from_prior(model, prior_map, args.prior_ckpt)
        print(f"Initialized {n_init:,} embeddings from prior checkpoint")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_steps = int(0.05 * args.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_steps, 1),
    )

    for epoch in range(1, args.epochs + 1):
        # Linear warmup
        if epoch <= warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * epoch / max(warmup_steps, 1)

        train_loss, train_acc, train_top5 = pretrain_epoch(
            model, train_dl, criterion, optimizer,
            scheduler if epoch > warmup_steps else None,
            device, args.accum,
        )

        # Validation: masked prediction accuracy
        model.eval()
        val_total, val_correct, val_top5 = 0, 0, 0
        with torch.no_grad():
            for pl, st, tm, _ in tqdm(val_dl, leave=False, desc="val"):
                pl, st, tm = pl.to(device), st.to(device), tm.to(device)
                B = pl.size(0)
                mask_idx = torch.randint(0, 10, (B,), device=device)
                target = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)
                logits, _ = model.forward_pretrain(pl, st, tm, mask_idx)
                val_correct += (logits.argmax(1) == target).sum().item()
                _, top5_pred = logits.topk(5, dim=1)
                val_top5 += (top5_pred == target.unsqueeze(1)).any(dim=1).sum().item()
                val_total += B

        val_acc = val_correct / val_total
        val_top5_acc = val_top5 / val_total
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Ep {epoch:02d} | "
            f"train top1={train_acc*100:5.2f}% top5={train_top5*100:5.2f}% loss={train_loss:.4f} | "
            f"val top1={val_acc*100:5.2f}% top5={val_top5_acc*100:5.2f}% | "
            f"lr {lr:.2e}"
        )

    ckpt_path = args.ckpt or "pretrain_checkpoint.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "phase": "pretrain",
        "vocab": OUTCOME_VOCAB,
        "num_players": num_players,
        "d_model": args.d_model,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "dropout": args.dropout,
        "epochs": args.epochs,
    }, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

################################################################################
# Phase B: Outcome Prediction Fine-tuning
################################################################################

def run_finetune(args, device):
    print("=" * 60)
    print("PHASE B: Outcome Prediction Fine-tuning")
    print("=" * 60)

    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt required for fine-tuning")

    # Load pretrained checkpoint
    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True)
    num_players = ckpt["num_players"]
    d_model = ckpt.get("d_model", args.d_model)
    n_layers = ckpt.get("n_layers", args.layers)
    n_heads = ckpt.get("n_heads", args.heads)
    dropout = ckpt.get("dropout", args.dropout)

    model = LineupTransformer(num_players, d_model, n_layers, n_heads, dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded pretrained model from {args.pretrain_ckpt}")

    train_dl, val_dl, _ = get_default_dataloaders(
        args.parquet, batch_size=args.bs, shuffle_train=True,
    )

    # Differential LR: low for embeddings/encoder, higher for new classifier head
    emb_params = list(model.player_emb.parameters()) + list(model.team_emb.parameters())
    encoder_params = list(model.encoder.parameters()) + list(model.attn_pool.parameters()) + [model.pos_bias]
    head_params = list(model.outcome_head.parameters()) + list(model.state_proj.parameters())

    optimizer = optim.AdamW([
        {"params": emb_params, "lr": args.lr * 0.03},     # embeddings: very low lr
        {"params": encoder_params, "lr": args.lr * 0.1},  # encoder: low lr
        {"params": head_params, "lr": args.lr},            # new head: full lr
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dl, device))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = finetune_epoch(
            model, train_dl, criterion, optimizer, scheduler, device, args.accum,
        )
        val_loss, val_acc = finetune_epoch(
            model, val_dl, criterion, None, None, device, 1,
        )
        lr_head = optimizer.param_groups[2]["lr"]
        print(
            f"Ep {epoch:02d} | "
            f"train {train_acc*100:5.2f}% loss={train_loss:.4f} | "
            f"val {val_acc*100:5.2f}% loss={val_loss:.4f} | "
            f"lr_head {lr_head:.2e}"
        )

    ckpt_path = args.ckpt or "finetune_checkpoint.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "phase": "finetune",
        "vocab": OUTCOME_VOCAB,
        "num_players": num_players,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "num_outcomes": NUM_OUTCOMES,
        "epochs": args.epochs,
    }, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

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
    print(f"Device: {device}")

    if args.phase == "pretrain":
        run_pretrain(args, device)
    elif args.phase == "finetune":
        run_finetune(args, device)
    else:
        raise ValueError("--phase must be 'pretrain' or 'finetune'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unicorn transformer training")
    parser.add_argument("--phase", required=True, choices=["pretrain", "finetune"],
                        help="Training phase: 'pretrain' (masked player) or 'finetune' (outcome)")
    parser.add_argument("--parquet", default="possessions.parquet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ckpt", default=None, help="Output checkpoint path")
    parser.add_argument("--pretrain-ckpt", default=None,
                        help="Phase A checkpoint to load for fine-tuning")
    parser.add_argument("--prior-ckpt", default=None,
                        help="Prior training run checkpoint for embedding warm-start")
    parser.add_argument("--lookup-csv", default="player_season_lookup.csv")
    args = parser.parse_args()
    main(args)
