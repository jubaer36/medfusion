import os
import csv
import random
from sklearn.model_selection import train_test_split

def make_patient_csv(
    root_dir="Patient Data",
    output_csv="pairs.csv",
    val_ratio=0.2,
    seed=42
):
    random.seed(seed)

    patient_dirs = [d for d in os.listdir(root_dir) if d.startswith("p")]
    patient_dirs.sort(key=lambda x: int(x[1:]))  # sort numerically (p1, p2, ...)

    pairs = []
    for pid in patient_dirs:
        patient_path = os.path.join(root_dir, pid)
        ct_path = os.path.join(patient_path, "ct.jpg")
        mri_path = os.path.join(patient_path, "mri.jpg")

        if os.path.exists(ct_path) and os.path.exists(mri_path):
            pairs.append((ct_path, mri_path, pid))
        else:
            print(f"⚠️ Missing CT or MRI in {pid}, skipping...")

    # train/val split
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=val_ratio, random_state=seed
    )

    # write csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "ct_path", "mri_path", "patient_id"])
        for ct, mri, pid in train_pairs:
            writer.writerow(["train", ct, mri, pid])
        for ct, mri, pid in val_pairs:
            writer.writerow(["val", ct, mri, pid])

    print(f"✅ CSV saved to {output_csv}")
    print(f"   Train: {len(train_pairs)} samples, Val: {len(val_pairs)} samples")


if __name__ == "__main__":
    make_patient_csv(
        root_dir="Patient Data",   # your dataset root
        output_csv="pairs.csv",
        val_ratio=0.2
    )
