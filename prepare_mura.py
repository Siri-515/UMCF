from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MURA_DIR = PROJECT_ROOT / "data" / "MURA"
OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

records = []

# Search all image files
image_files = [
    p for p in MURA_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
]

print("Total MURA image files found:", len(image_files))

for image_path in image_files:

    path_text = str(image_path).replace("\\", "/").lower()

    # MURA labels are usually represented by study folders:
    # study1_positive -> abnormal
    # study1_negative -> normal

    if "_positive" in path_text:
        label = 1
        class_name = "Abnormal"

    elif "_negative" in path_text:
        label = 0
        class_name = "Normal"

    else:
        continue

    records.append({
        "image_path": str(image_path.resolve()),
        "label": label,
        "class_name": class_name,
        "dataset": "MURA"
    })

df = pd.DataFrame(records)

# Remove duplicate image paths
df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

output_file = OUTPUT_DIR / "mura_labels.csv"
df.to_csv(output_file, index=False)

print("\n========================================")
print("MURA CSV created successfully")
print("========================================")
print("Total labeled images:", len(df))
print("Abnormal:", (df["label"] == 1).sum())
print("Normal:", (df["label"] == 0).sum())
print("Duplicate paths:", df["image_path"].duplicated().sum())
print("CSV:", output_file)

print("\nClass distribution:")
print(df["class_name"].value_counts())

print("\nFirst 5 records:")
print(df.head())