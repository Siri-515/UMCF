from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRACATLAS_DIR = PROJECT_ROOT / "data" / "FracAtlas"
IMAGE_DIR = FRACATLAS_DIR / "images"
OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

records = []

classes = {
    "Non_fractured": 0,
    "Fractured": 1
}

for class_name, label in classes.items():

    class_dir = IMAGE_DIR / class_name

    if not class_dir.exists():
        print(f"WARNING: {class_dir} not found")
        continue

    # Get all image files once, regardless of uppercase/lowercase extension
    image_files = [
        p for p in class_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    print(f"{class_name}: {len(image_files)} images")

    for image_path in image_files:
        records.append({
            "image_path": str(image_path.resolve()),
            "label": label,
            "class_name": class_name,
            "dataset": "FracAtlas"
        })

df = pd.DataFrame(records)

# Remove accidental duplicate paths
df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

output_file = OUTPUT_DIR / "fracatlas_labels.csv"
df.to_csv(output_file, index=False)

print("\n========================================")
print("FracAtlas CSV created successfully")
print("========================================")
print("Total images:", len(df))
print("Fractured:", (df["label"] == 1).sum())
print("Non-fractured:", (df["label"] == 0).sum())
print("Duplicate paths:", df["image_path"].duplicated().sum())
print("CSV:", output_file)

print("\nClass distribution:")
print(df["class_name"].value_counts())

print("\nFirst 5 records:")
print(df.head())