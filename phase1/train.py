"""
अन्नदाता AI — Phase 1
Training script: two-phase training strategy.

Phase A (Epochs 1–5):   Freeze backbone, train head only  → fast convergence
Phase B (Epochs 6–20):  Unfreeze all layers, fine-tune    → max accuracy

Run:
  python phase1/train.py

Outputs saved to: checkpoints/
  - best_model.pt       ← best val accuracy
  - last_model.pt       ← last epoch
  - training_log.json   ← loss/accuracy per epoch
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase1.model.model import build_model
from phase1.utils.dataset import get_dataloaders, DATA_DIR

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = Path("checkpoints")
PHASE_A_EPOCHS   = 5      # head-only epochs
PHASE_B_EPOCHS   = 15     # full fine-tune epochs
TOTAL_EPOCHS     = PHASE_A_EPOCHS + PHASE_B_EPOCHS
LR_HEAD          = 1e-3   # higher LR for head (phase A)
LR_FINETUNE      = 3e-4   # lower LR for full model (phase B)
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTHING  = 0.1    # prevents overconfidence
DEVICE           = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f} acc={correct/total:.3f}")
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs     = model(images)
        total_loss += criterion(outputs, labels).item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":            val_acc,
    }, path)


def train():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print(f"\n{'='*55}")
    print("  अन्नदाता AI — Crop Disease Detection Training")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Epochs : {TOTAL_EPOCHS}  (Phase A: {PHASE_A_EPOCHS} + Phase B: {PHASE_B_EPOCHS})")
    print(f"{'='*55}\n")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR)
    num_classes = len(class_names)

    # Model
    model = build_model(num_classes=num_classes, pretrained=True).to(DEVICE)
    model.count_params()

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ── Phase A: Freeze backbone, train head ──────────────────────────────────
    model.freeze_backbone()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE_A_EPOCHS, eta_min=1e-5)

    log        = []
    best_acc   = 0.0
    best_path  = CHECKPOINT_DIR / "best_model.pt"

    print("── Phase A: Head-only training ──")
    for epoch in range(1, PHASE_A_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"[Epoch {epoch:02d}/{TOTAL_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.0f}s")
        log.append({"epoch": epoch, "phase": "A",
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss,     "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    # ── Phase B: Unfreeze all, fine-tune ─────────────────────────────────────
    model.unfreeze_backbone()
    optimizer = AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE_B_EPOCHS, eta_min=1e-6)

    print("\n── Phase B: Full fine-tuning ──")
    for epoch in range(PHASE_A_EPOCHS + 1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"[Epoch {epoch:02d}/{TOTAL_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.0f}s")
        log.append({"epoch": epoch, "phase": "B",
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss,     "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    # ── Save last + log ───────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, TOTAL_EPOCHS, val_acc, CHECKPOINT_DIR / "last_model.pt")
    with open(CHECKPOINT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n── Test set evaluation ──")
    best_model = build_model(num_classes=num_classes, pretrained=False).to(DEVICE)
    ckpt = torch.load(best_path, map_location=DEVICE)
    best_model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc = evaluate(best_model, test_loader, criterion, DEVICE)
    print(f"  Best val_acc : {best_acc:.4f}")
    print(f"  Test acc     : {test_acc:.4f}")
    print(f"\n[अन्नदाता AI] Training complete ✓  Model saved to '{best_path}'")


if __name__ == "__main__":
    train()
