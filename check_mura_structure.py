from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MURA_DIR = PROJECT_ROOT / "data" / "MURA"

print("=" * 60)
print("MURA STRUCTURE CHECK")
print("=" * 60)

splits = ["train", "valid", "test"]

for split in splits:

    split_dir = MURA_DIR / "Dataset" / split

    print(f"\n{split.upper()}")

    if not split_dir.exists():
        print("  Folder not found")
        continue

    images = [
        p for p in split_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    print("  Images:", len(images))

    # Count studies
    study_dirs = set()

    for image in images:
        relative_parts = image.relative_to(split_dir).parts

        # Typical MURA structure:
        # body_part / patient / study / image
        if len(relative_parts) >= 3:
            study = "/".join(relative_parts[:3])
            study_dirs.add(study)

    print("  Approx. studies:", len(study_dirs))

    # Body-part distribution
    body_parts = Counter()

    for image in images:
        relative_parts = image.relative_to(split_dir).parts

        if len(relative_parts) >= 1:
            body_parts[relative_parts[0]] += 1

    print("  Body parts:")

    for body_part, count in sorted(body_parts.items()):
        print(f"    {body_part}: {count}")