# train_cbow.py – v2 (9-class, fixed unpacking, metadata)
"""
CBOW baseline for NBA possession outcome prediction.

Mean-pools player embeddings + state projection → MLP classifier.
This serves as the "no player interaction" baseline (Deep Sets / NBA2Vec equivalent).

Features:
* Class-weighted CrossEntropyLoss
* Cosine LR scheduler with warmup
* Saves training metadata in checkpoint
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nba_dataset import (
    PossessionDataset,
    OUTCOME_VOCAB,
    get_default_dataloaders,
    get_num_players,
    NUM_OUTCOMES,
)

################################################################################
# Model
################################################################################

class CBOWModel(nn.Module):
    def __init__(self, num_players: int, d_emb: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_players, d_emb)
        self.state_fc = nn.Linear(3, d_emb)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_emb, d_emb // 2),
            nn.ReLU(),
            nn.Linear(d_emb // 2, NUM_OUTCOMES),
        )

    def forward(self, players, state):
        x = self.emb(players).mean(dim=1) + self.state_fc(state)
        return self.classifier(x)

################################################################################
# Helpers
################################################################################

def compute_class_weights(train_dl: DataLoader, device: torch.device):
    label_counts = torch.zeros(NUM_OUTCOMES, dtype=torch.float64)
    for _, _, _, label in train_dl:  # (players, state, team_ids, label)
        label_counts += torch.bincount(label, minlength=NUM_OUTCOMES).double()
    weights = (1.0 / label_counts.clamp(min=1)).float()
    weights = (weights * (NUM_OUTCOMES / weights.sum())).to(device)
    return weights


def epoch_loop(model, dataloader, criterion, optimizer=None, device="cpu"):
    model.train(optimizer is not None)
    total, correct, loss_sum = 0, 0, 0.0
    for players, state, _, label in tqdm(dataloader, leave=False):
        players, state, label = players.to(device), state.to(device), label.to(device)
        logits = model(players, state)
        loss = criterion(logits, label)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * players.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += players.size(0)
    return loss_sum / total, correct / total

################################################################################
# Main
################################################################################

def main(args):
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Device:", device)

    train_dl, val_dl, _ = get_default_dataloaders(args.parquet, batch_size=args.bs)

    num_players = get_num_players()
    model = CBOWModel(num_players, d_emb=args.emb_dim).to(device)
    print(f"CBOW model: {num_players} players, d_emb={args.emb_dim}, {NUM_OUTCOMES} classes")

    if args.no_weighted_loss:
        criterion = nn.CrossEntropyLoss()
    else:
        class_w = compute_class_weights(train_dl, device)
        print("Class weights:", class_w.cpu().numpy().round(3).tolist())
        criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = epoch_loop(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = epoch_loop(model, val_dl, criterion, None, device)
        scheduler.step()
        print(
            f"Epoch {epoch:02d} | "
            f"train {train_loss:.4f}/{train_acc*100:5.2f}% | "
            f"val {val_loss:.4f}/{val_acc*100:5.2f}% | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )
        best_val_acc = max(best_val_acc, val_acc)

    torch.save({
        "state_dict": model.state_dict(),
        "vocab": OUTCOME_VOCAB,
        "num_players": num_players,
        "d_emb": args.emb_dim,
        "num_outcomes": NUM_OUTCOMES,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
    }, args.checkpoint)
    print("Saved checkpoint ->", args.checkpoint)

################################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train CBOW baseline")
    ap.add_argument("--parquet", default="possessions.parquet")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--emb-dim", dest="emb_dim", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--checkpoint", default="cbow_checkpoint.pt")
    ap.add_argument("--no-weighted-loss", action="store_true")
    args = ap.parse_args()
    main(args)
