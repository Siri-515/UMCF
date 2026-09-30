import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\UCMF-Net")
RESULTS = ROOT / "results"

train = pd.read_csv(RESULTS / "mura_train.csv")
val = pd.read_csv(RESULTS / "mura_val.csv")
test = pd.read_csv(RESULTS / "mura_test.csv")

train_patients = set(train["patient_id"])
val_patients = set(val["patient_id"])
test_patients = set(test["patient_id"])

train_val_overlap = train_patients & val_patients
train_test_overlap = train_patients & test_patients
val_test_overlap = val_patients & test_patients

print("=" * 50)
print("PATIENT-LEVEL LEAKAGE CHECK")
print("=" * 50)

print(f"Train patients:      {len(train_patients)}")
print(f"Validation patients: {len(val_patients)}")
print(f"Test patients:       {len(test_patients)}")

print()
print(f"Train ∩ Validation: {len(train_val_overlap)}")
print(f"Train ∩ Test:       {len(train_test_overlap)}")
print(f"Validation ∩ Test:  {len(val_test_overlap)}")

print()

if len(train_val_overlap) == 0:
    print("✓ No Train-Validation patient overlap")
else:
    print("⚠ WARNING: Train-Validation patient overlap detected")

if len(train_test_overlap) == 0:
    print("✓ No Train-Test patient overlap")
else:
    print("⚠ WARNING: Train-Test patient overlap detected")

if len(val_test_overlap) == 0:
    print("✓ No Validation-Test patient overlap")
else:
    print("⚠ WARNING: Validation-Test patient overlap detected")

print("=" * 50)