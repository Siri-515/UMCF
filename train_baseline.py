import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import MURADataset
from model_baseline import create_resnet50


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\UCMF-Net")
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RESULTS / "mura_train.csv"
VAL_CSV = RESULTS / "mura_val.csv"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASETS
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_dataset = MURADataset(
    TRAIN_CSV,
    image_size=224,
    train=True
)

val_dataset = MURADataset(
    VAL_CSV,
    image_size=224,
    train=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 60)
print("CREATING RESNET50")
print("=" * 60)

model = create_resnet50(num_classes=2)
model = model.to(DEVICE)

print("Device:", DEVICE)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("STARTING BASELINE TRAINING")
print("=" * 60)

best_val_accuracy = 0.0

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / total
    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / val_total

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        checkpoint_path = (
            CHECKPOINTS /
            "resnet50_baseline_best.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy
            },
            checkpoint_path
        )


    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss:       {train_loss:.4f}")
    print(f"Train Accuracy:   {train_accuracy:.4f}")
    print(f"Validation Loss:  {val_loss:.4f}")
    print(f"Validation Acc:   {val_accuracy:.4f}")
    print(f"Time:             {elapsed / 60:.2f} minutes")
    print("=" * 60)


print()
print("Training completed.")
print(
    "Best model:",
    CHECKPOINTS / "resnet50_baseline_best.pth"
)