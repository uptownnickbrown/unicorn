# evaluate.py
"""
Evaluation script for Unicorn models.

Computes per-class metrics, confusion matrix, and comparisons against baselines.
Supports both Phase A (masked prediction) and Phase B (outcome prediction) evaluation.

Usage:
```bash
# Evaluate outcome prediction (Phase B)
python evaluate.py --ckpt finetune_checkpoint.pt --phase finetune

# Evaluate masked prediction (Phase A)
python evaluate.py --ckpt pretrain_checkpoint.pt --phase pretrain

# Evaluate CBOW baseline
python evaluate.py --ckpt cbow_checkpoint.pt --model-type cbow
```
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nba_dataset import (
    PossessionDataset,
    get_default_dataloaders,
    get_num_players,
    NUM_OUTCOMES,
    OUTCOME_VOCAB,
    OUTCOME_NAMES,
)

################################################################################
# Model loading
################################################################################

def load_model(ckpt_path: str, model_type: str, device: torch.device):
    """Load a model from checkpoint, auto-detecting architecture."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"]

    if model_type == "cbow":
        from train_cbow import CBOWModel
        num_players = ckpt.get("num_players", state_dict["emb.weight"].shape[0])
        d_emb = ckpt.get("d_emb", state_dict["emb.weight"].shape[1])
        model = CBOWModel(num_players, d_emb)
    else:
        from train_transformer import LineupTransformer
        num_players = ckpt.get("num_players", state_dict["player_emb.weight"].shape[0])
        d_model = ckpt.get("d_model", state_dict["player_emb.weight"].shape[1])
        n_layers = ckpt.get("n_layers", 8)
        n_heads = ckpt.get("n_heads", 8)
        dropout = ckpt.get("dropout", 0.1)
        model = LineupTransformer(num_players, d_model, n_layers, n_heads, dropout)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt

################################################################################
# Outcome evaluation (Phase B / CBOW)
################################################################################

def evaluate_outcome(model, test_dl, device, model_type="transformer"):
    """Run outcome prediction on test set, return predictions and labels."""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for pl, st, tm, y in tqdm(test_dl, desc="Evaluating"):
            pl, st, tm, y = pl.to(device), st.to(device), tm.to(device), y.to(device)

            if model_type == "cbow":
                logits = model(pl, st)
            else:
                logits, _ = model.forward_finetune(pl, st, tm)

            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
    )

################################################################################
# Masked prediction evaluation (Phase A)
################################################################################

def evaluate_masked(model, test_dl, device):
    """Evaluate masked player prediction on test set."""
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.no_grad():
        for pl, st, tm, _ in tqdm(test_dl, desc="Eval masked"):
            pl, st, tm = pl.to(device), st.to(device), tm.to(device)
            B = pl.size(0)

            mask_idx = torch.randint(0, 10, (B,), device=device)
            target = pl.gather(1, mask_idx.unsqueeze(1)).squeeze(1)

            logits, _ = model.forward_pretrain(pl, st, tm, mask_idx)

            top1_correct += (logits.argmax(1) == target).sum().item()
            _, top5_pred = logits.topk(5, dim=1)
            top5_correct += (top5_pred == target.unsqueeze(1)).any(dim=1).sum().item()
            total += B

    return {
        "top1_accuracy": top1_correct / total,
        "top5_accuracy": top5_correct / total,
        "total_samples": total,
    }

################################################################################
# Baselines
################################################################################

def compute_baselines(train_dl, test_labels):
    """Compute baseline accuracies for comparison."""
    # 1. Training class distribution
    train_counts = np.zeros(NUM_OUTCOMES)
    for *_, y in train_dl:
        for label in y.numpy():
            train_counts[label] += 1
    train_dist = train_counts / train_counts.sum()

    # 2. Test class distribution
    test_counts = np.bincount(test_labels, minlength=NUM_OUTCOMES)

    n_test = len(test_labels)
    results = {}

    # Majority class
    majority = np.argmax(train_counts)
    results["majority_class"] = {
        "accuracy": (test_labels == majority).mean(),
        "predicted_class": OUTCOME_NAMES[majority],
    }

    # Stratified random (predict according to training distribution)
    np.random.seed(42)
    random_preds = np.random.choice(NUM_OUTCOMES, size=n_test, p=train_dist)
    results["stratified_random"] = {
        "accuracy": (random_preds == test_labels).mean(),
    }

    # State-only logistic regression
    try:
        from sklearn.linear_model import LogisticRegression
        from nba_dataset import PossessionDataset

        # We need the raw state features — extract from dataloaders
        train_states, train_labels_lr = [], []
        for _, st, _, y in train_dl:
            train_states.append(st.numpy())
            train_labels_lr.append(y.numpy())
        train_states = np.concatenate(train_states)
        train_labels_lr = np.concatenate(train_labels_lr)

        test_states = []
        for _, st, _, _ in tqdm(DataLoader(
            test_dl.dataset, batch_size=2048, shuffle=False, num_workers=0,
        ), desc="State LR", leave=False):
            test_states.append(st.numpy())
        test_states = np.concatenate(test_states)

        lr_model = LogisticRegression(max_iter=500, multi_class="multinomial", n_jobs=-1)
        lr_model.fit(train_states, train_labels_lr)
        lr_preds = lr_model.predict(test_states)

        results["state_only_logreg"] = {
            "accuracy": (lr_preds == test_labels).mean(),
        }
    except Exception as e:
        results["state_only_logreg"] = {"error": str(e)}

    return results, train_dist, test_counts

################################################################################
# Reporting
################################################################################

def print_report(preds, labels, baselines, label_names):
    """Print comprehensive evaluation report."""
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    # Overall accuracy
    acc = (preds == labels).mean()
    print(f"\nModel accuracy: {acc*100:.2f}%")

    # Baselines comparison
    print("\nBaseline comparison:")
    for name, info in baselines.items():
        if "accuracy" in info:
            delta = (acc - info["accuracy"]) * 100
            print(f"  {name:25s}: {info['accuracy']*100:.2f}%  (model is {delta:+.2f}pp)")
        elif "error" in info:
            print(f"  {name:25s}: ERROR - {info['error']}")

    # Per-class report
    names = [label_names[i] for i in range(NUM_OUTCOMES)]
    print("\nPer-class metrics:")
    print(classification_report(labels, preds, target_names=names, digits=3, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion matrix:")
    print(f"{'':>18s}", end="")
    for n in names:
        print(f"{n[:8]:>10s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{names[i]:>18s}", end="")
        for val in row:
            print(f"{val:>10d}", end="")
        print()

    # Save confusion matrix as image
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (accuracy: {acc*100:.2f}%)")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("\nConfusion matrix saved -> confusion_matrix.png")
    except Exception:
        pass

    return {"accuracy": acc, "baselines": baselines}

################################################################################
# Main
################################################################################

def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # Determine model type from checkpoint or flag
    model_type = args.model_type
    if model_type == "auto":
        ckpt_peek = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        if "emb.weight" in ckpt_peek.get("state_dict", {}):
            model_type = "cbow"
        else:
            model_type = "transformer"
        print(f"Auto-detected model type: {model_type}")

    model, ckpt = load_model(args.ckpt, model_type, device)

    # Phase A: masked prediction evaluation
    if args.phase == "pretrain":
        _, _, test_dl = get_default_dataloaders(args.parquet, batch_size=args.bs)
        results = evaluate_masked(model, test_dl, device)
        print(f"\nMasked Player Prediction (Phase A):")
        print(f"  Top-1 accuracy: {results['top1_accuracy']*100:.2f}%")
        print(f"  Top-5 accuracy: {results['top5_accuracy']*100:.2f}%")
        print(f"  Total samples:  {results['total_samples']:,}")
        return

    # Phase B / CBOW: outcome prediction evaluation
    train_dl, _, test_dl = get_default_dataloaders(args.parquet, batch_size=args.bs)
    preds, labels, probs = evaluate_outcome(model, test_dl, device, model_type)
    baselines, train_dist, test_counts = compute_baselines(train_dl, labels)

    report = print_report(preds, labels, baselines, OUTCOME_NAMES)

    # Save report
    if args.save_report:
        report["baselines"] = {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                    for kk, vv in v.items()}
                                for k, v in baselines.items()}
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved -> {args.save_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Unicorn models")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--parquet", default="possessions.parquet")
    parser.add_argument("--phase", default="finetune", choices=["pretrain", "finetune"],
                        help="Evaluation mode")
    parser.add_argument("--model-type", default="auto", choices=["auto", "transformer", "cbow"],
                        help="Model type (auto-detected from checkpoint)")
    parser.add_argument("--bs", type=int, default=2048)
    parser.add_argument("--save-report", default=None, help="Path to save JSON report")
    args = parser.parse_args()
    main(args)
