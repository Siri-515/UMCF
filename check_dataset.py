from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

print("Project folder:", PROJECT_ROOT)
print("Data folder:", DATA_DIR)

for dataset in ["MURA", "FracAtlas"]:
    dataset_path = DATA_DIR / dataset

    print("\n-----------------------------")
    print(dataset)
    print("-----------------------------")

    if dataset_path.exists():
        print("Dataset folder found!")

        files = list(dataset_path.rglob("*"))

        image_files = [
            f for f in files
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ]

        print("Total files:", len(files))
        print("Image files:", len(image_files))

        if image_files:
            print("Example image:")
            print(image_files[0])

    else:
        print("Dataset folder NOT found!")