from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def show_structure(dataset_name, max_depth=3):
    dataset_path = DATA_DIR / dataset_name

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    if not dataset_path.exists():
        print("Dataset not found!")
        return

    for path in sorted(dataset_path.rglob("*")):
        relative = path.relative_to(dataset_path)

        if len(relative.parts) <= max_depth:
            if path.is_dir():
                print("[DIR ]", relative)
            else:
                print("[FILE]", relative)


show_structure("MURA", max_depth=3)
show_structure("FracAtlas", max_depth=3)