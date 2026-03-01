# train_transformer.py – v3 clean & warning‑free
"""
Transformer lineup‑encoder with same‑team bias, curriculum loader, and clean
console (no deprecation or pin‑memory warnings on M‑series).

Run example:
```bash
python train_transformer.py \
  --parquet possessions.parquet \
  --epochs 20 \
  --bs 1024 \
  --accum 4
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

from nba_dataset import get_default_dataloaders, NUM_OUTCOMES, outcome_vocab

################################################################################
# Model
################################################################################

class LineupTransformer(nn.Module):
    """[CLS] token carries game state; 10 player tokens carry player+team info."""

    def __init__(self, num_players: int, d_model: int = 384, n_layers: int = 8, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.player_emb = nn.Embedding(num_players, d_model)
        self.team_emb   = nn.Embedding(2, d_model)         # 0 offence, 1 defence
        self.state_proj = nn.Linear(3, d_model)
        self.pos_bias   = nn.Parameter(torch.zeros(11, d_model))
        self.cls_drop   = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, NUM_OUTCOMES)

    def forward(self, players, state, team_ids):
        tok = self.player_emb(players) + self.team_emb(team_ids)   # [B,10,d]
        cls = self.cls_drop(self.state_proj(state)).unsqueeze(1)   # [B,1,d]
        h   = self.encoder(torch.cat([cls, tok], 1) + self.pos_bias)
        return self.classifier(h[:, 0])

################################################################################
# Helpers
################################################################################

def class_weights(train_dl, device):
    counts = torch.zeros(NUM_OUTCOMES)
    for *_ , y in train_dl:
        counts += torch.bincount(y, minlength=NUM_OUTCOMES).float()
    w = (1.0 / counts)
    w *= NUM_OUTCOMES / w.sum()
    return w.to(device)

################################################################################
# Train / eval loops
################################################################################

def run_epoch(model, dataloader, criterion, optimizer, device, accum, amp):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    scaler = (torch.amp.GradScaler(enabled=amp) if train else None)

    for step, (pl, st, tm, y) in enumerate(tqdm(dataloader, leave=False)):
        pl, st, tm, y = pl.to(device), st.to(device), tm.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=amp):
            logits = model(pl, st, tm)
            loss   = criterion(logits, y) / accum
        if train:
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.item() * accum * pl.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += pl.size(0)
    return loss_sum / total, correct / total

################################################################################
# Main
################################################################################

def main(args):
    # Filter noisy but harmless warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="enable_nested_tensor")
    warnings.filterwarnings("ignore", category=UserWarning, message="pin_memory")

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else torch.device("cpu")
    )
    amp_enabled = device.type == "cuda"
    print("Device:", device, "| AMP:", amp_enabled)

    train_dl, val_dl, _ = get_default_dataloaders(args.parquet, batch_size=args.bs)
    vocab_size = train_dl.dataset.players.max() + 1

    model = LineupTransformer(vocab_size, args.d_model, args.layers, args.heads, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dl, device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_dl, criterion, optimizer, device, args.accum, amp_enabled)
        val_loss, val_acc     = run_epoch(model, val_dl, criterion, None,      device, 1,          amp_enabled)
        scheduler.step()
        print(f"Ep {epoch:02d} | train {train_acc*100:5.2f}% | val {val_acc*100:5.2f}% | loss {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    torch.save({"state_dict": model.state_dict(), "vocab": outcome_vocab}, args.ckpt)
    print("Checkpoint saved →", args.ckpt)

################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="possessions.parquet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ckpt", default="transformer_checkpoint.pt")
    args = parser.parse_args()
    main(args)
