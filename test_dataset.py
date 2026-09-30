from pathlib import Path
from dataset import MURADataset

ROOT = Path(r"D:\UCMF-Net")
RESULTS = ROOT / "results"

train_csv = RESULTS / "mura_train.csv"

dataset = MURADataset(
    train_csv,
    image_size=224,
    train=True
)

print("=" * 50)
print("MURA DATASET LOADER TEST")
print("=" * 50)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("Image shape:", sample["image"].shape)
print("Label:", sample["label"].item())
print("Patient ID:", sample["patient_id"])
print("Body part:", sample["body_part"])
print("Image path:", sample["image_path"])