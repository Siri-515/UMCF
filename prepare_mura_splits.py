import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(r"D:\UCMF-Net")
MURA_ROOT = ROOT / "data" / "MURA" / "Dataset"
RESULTS = ROOT / "results"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_mura_images(split_folder, split_name):
    records = []

    print(f"Reading MURA {split_name}...")

    for body_part_dir in sorted(split_folder.iterdir()):

        if not body_part_dir.is_dir():
            continue

        body_part = body_part_dir.name

        for patient_dir in sorted(body_part_dir.iterdir()):

            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name

            for study_dir in sorted(patient_dir.iterdir()):

                if not study_dir.is_dir():
                    continue

                study_id = study_dir.name

                # Determine label from study name
                if "_positive" in study_id.lower():
                    label = 1
                    class_name = "Abnormal"

                elif "_negative" in study_id.lower():
                    label = 0
                    class_name = "Normal"

                else:
                    # Skip studies where label cannot be determined
                    continue

                for image_path in study_dir.rglob("*"):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower() in IMAGE_EXTENSIONS
                    ):
                        records.append({
                            "image_path": str(image_path),
                            "label": label,
                            "class_name": class_name,
                            "body_part": body_part,
                            "patient_id": patient_id,
                            "study_id": study_id,
                            "original_split": split_name,
                            "dataset": "MURA"
                        })

    return records


# ============================================================
# COLLECT DATA
# ============================================================

train_valid_folder = MURA_ROOT / "train_valid"
test_folder = MURA_ROOT / "test"

train_valid_records = collect_mura_images(
    train_valid_folder,
    "train_valid"
)

test_records = collect_mura_images(
    test_folder,
    "test"
)

records = train_valid_records + test_records

df = pd.DataFrame(records)

# Remove duplicate image paths
df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

print()
print("=" * 60)
print("MURA COMPLETE DATASET SUMMARY")
print("=" * 60)

print(f"Total images: {len(df)}")

print("\nOriginal split:")
print(df["original_split"].value_counts())

print("\nClass distribution:")
print(df["class_name"].value_counts())

print("\nBody-part distribution:")
print(df["body_part"].value_counts())


# ============================================================
# SEPARATE TRAIN/VALIDATION AND OFFICIAL TEST
# ============================================================

train_valid_df = df[
    df["original_split"] == "train_valid"
].copy()

test_df = df[
    df["original_split"] == "test"
].copy()


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

patients = train_valid_df["patient_id"].unique()

print()
print(f"Total train_valid patients: {len(patients)}")

train_patients, val_patients = train_test_split(
    patients,
    test_size=0.20,
    random_state=42
)

train_df = train_valid_df[
    train_valid_df["patient_id"].isin(train_patients)
].copy()

val_df = train_valid_df[
    train_valid_df["patient_id"].isin(val_patients)
].copy()


# ============================================================
# SAVE CSV FILES
# ============================================================

train_path = RESULTS / "mura_train.csv"
val_path = RESULTS / "mura_val.csv"
test_path = RESULTS / "mura_test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL PATIENT-LEVEL MURA SPLITS")
print("=" * 60)

print("\nTRAIN")
print(f"Images: {len(train_df)}")
print(f"Patients: {train_df['patient_id'].nunique()}")
print(train_df["class_name"].value_counts())

print("\nVALIDATION")
print(f"Images: {len(val_df)}")
print(f"Patients: {val_df['patient_id'].nunique()}")
print(val_df["class_name"].value_counts())

print("\nTEST")
print(f"Images: {len(test_df)}")
print(f"Patients: {test_df['patient_id'].nunique()}")
print(test_df["class_name"].value_counts())


# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

train_patient_set = set(train_df["patient_id"])
val_patient_set = set(val_df["patient_id"])
test_patient_set = set(test_df["patient_id"])

train_val_overlap = train_patient_set & val_patient_set
train_test_overlap = train_patient_set & test_patient_set
val_test_overlap = val_patient_set & test_patient_set

print()
print("=" * 60)
print("PATIENT-LEVEL LEAKAGE CHECK")
print("=" * 60)

print(f"Train ∩ Validation: {len(train_val_overlap)}")
print(f"Train ∩ Test:       {len(train_test_overlap)}")
print(f"Validation ∩ Test:  {len(val_test_overlap)}")

if len(train_val_overlap) == 0:
    print("✓ No Train-Validation patient overlap")
else:
    print("⚠ Train-Validation overlap detected")

if len(train_test_overlap) == 0:
    print("✓ No Train-Test patient overlap")
else:
    print("⚠ Train-Test overlap detected")

if len(val_test_overlap) == 0:
    print("✓ No Validation-Test patient overlap")
else:
    print("⚠ Validation-Test overlap detected")


# ============================================================
# SAVE COMPLETE LABEL CSV
# ============================================================

complete_csv = RESULTS / "mura_labels_complete.csv"
df.to_csv(complete_csv, index=False)

print()
print("=" * 60)
print("FILES CREATED")
print("=" * 60)

print(train_path)
print(val_path)
print(test_path)
print(complete_csv)