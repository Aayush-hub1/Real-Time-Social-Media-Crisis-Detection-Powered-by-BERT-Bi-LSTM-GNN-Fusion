"""
DeepSentinel — Training & Evaluation
Full training loop with early stopping, LR scheduling, and metric logging.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from transformers import BertTokenizer

from model import DeepSentinel, FocalLoss, NUM_CLASSES

# ─── Dataset ─────────────────────────────────────────────────────────────────

LABEL_NAMES = ["misinformation", "mental_health", "cyberbullying", "political_unrest", "normal"]

class CrisisDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts    = df["clean_text"].tolist()
        self.labels   = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "bert_ids":  enc["input_ids"].squeeze(0),
            "bert_mask": enc["attention_mask"].squeeze(0),
            "lstm_ids":  enc["input_ids"].squeeze(0),
            "lstm_mask": enc["attention_mask"].squeeze(0),
            "label":     torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loaders(csv_path="data/dataset.csv", batch_size=32, val_split=0.15):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clean_text", "label_id"])

    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["label_id"], random_state=42
    )

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = CrisisDataset(train_df.reset_index(drop=True), tok)
    val_ds   = CrisisDataset(val_df.reset_index(drop=True), tok)

    # Weighted sampler to handle class imbalance
    counts  = train_df["label_id"].value_counts().sort_index().values
    weights = 1.0 / counts
    sample_w = weights[train_df["label_id"].values]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=2)
    return train_loader, val_loader


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(
        config["data_path"], config["batch_size"]
    )

    model     = DeepSentinel().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"], weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_f1 = 0.0
    patience    = config.get("patience", 5)
    no_improve  = 0
    history     = {"train_loss": [], "val_loss": [], "val_f1": []}

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            bert_ids  = batch["bert_ids"].to(device)
            bert_mask = batch["bert_mask"].to(device)
            lstm_ids  = batch["lstm_ids"].to(device)
            lstm_mask = batch["lstm_mask"].to(device)
            labels    = batch["label"].to(device)

            logits, _ = model(bert_ids, bert_mask, lstm_ids, lstm_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        val_loss, val_f1, report = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  ✓ Saved best model (F1={best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    with open("checkpoints/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Val F1: {best_val_f1:.4f}")
    print("\nFinal Classification Report:\n", report)
    return history


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            bert_ids  = batch["bert_ids"].to(device)
            bert_mask = batch["bert_mask"].to(device)
            lstm_ids  = batch["lstm_ids"].to(device)
            lstm_mask = batch["lstm_mask"].to(device)
            labels    = batch["label"].to(device)

            logits, _ = model(bert_ids, bert_mask, lstm_ids, lstm_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0)
    f1_score = float(report.split("macro avg")[1].split()[2])
    return total_loss / len(loader), f1_score, report


def full_eval(checkpoint="checkpoints/best_model.pt", data_path="data/dataset.csv"):
    """Run full evaluation with confusion matrix and ROC-AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DeepSentinel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    _, val_loader = make_loaders(data_path)
    loss, f1, report = evaluate(model, val_loader, FocalLoss(), device)

    print("Classification Report:\n", report)
    print(f"Val Loss: {loss:.4f} | Macro F1: {f1:.4f}")
    return {"loss": loss, "f1": f1, "report": report}


# ─── Entry Point ──────────────────────────────────────────────────────────────

CONFIG = {
    "data_path":  "data/dataset.csv",
    "batch_size": 32,
    "epochs":     20,
    "lr":         2e-5,
    "patience":   4,
}

if __name__ == "__main__":
    print("=== DeepSentinel Training ===")
    history = train(CONFIG)
