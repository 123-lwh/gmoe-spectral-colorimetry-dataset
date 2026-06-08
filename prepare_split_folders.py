import csv
import shutil
from pathlib import Path


ORIGINAL_IMAGE_DIR = Path(r"E:\image\img")
SPLIT_CSV_PATH = Path(r"E:\py_test\project3\my_dataset_split\data\split_info.csv")
OUTPUT_SPLIT_DIR = Path(r"E:\py_test\shangchaunceshi\case34\data_split_by_csv")

OVERWRITE = True
VALID_SPLITS = {"train", "val", "test"}


def build_image_map(image_dir):
    image_map = {}

    for img_path in image_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() == ".bmp":
            key = img_path.name.lower()

            if key in image_map:
                raise ValueError(
                    f"Duplicate filename: {img_path.name}\n"
                    f"path1: {image_map[key]}\n"
                    f"path2: {img_path}\n"
                )

            image_map[key] = img_path

    return image_map


def main():
    if not ORIGINAL_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Original image folder does not exist: {ORIGINAL_IMAGE_DIR}")

    if not SPLIT_CSV_PATH.exists():
        raise FileNotFoundError(f"Split CSV does not exist: {SPLIT_CSV_PATH}")

    for split in VALID_SPLITS:
        (OUTPUT_SPLIT_DIR / split).mkdir(parents=True, exist_ok=True)

    image_map = build_image_map(ORIGINAL_IMAGE_DIR)
    print(f"Original image count: {len(image_map)}")

    copied_count = {"train": 0, "val": 0, "test": 0}
    missing_files = []
    invalid_rows = []

    with open(SPLIT_CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if "name" not in reader.fieldnames or "status" not in reader.fieldnames:
            raise ValueError("split_info.csv must contain two columns: name and status")

        for row in reader:
            name = row["name"].strip()
            status = row["status"].strip().lower()

            if status not in VALID_SPLITS:
                invalid_rows.append(row)
                continue

            key = name.lower()

            if key not in image_map:
                missing_files.append(name)
                continue

            src_path = image_map[key]
            dst_path = OUTPUT_SPLIT_DIR / status / src_path.name

            if dst_path.exists() and not OVERWRITE:
                continue

            shutil.copy2(src_path, dst_path)
            copied_count[status] += 1

    print("\nSplit completed.")
    print(f"Output directory: {OUTPUT_SPLIT_DIR}")

    print("\nCopied files:")
    for split in ["train", "val", "test"]:
        print(f"{split}: {copied_count[split]}")

    if missing_files:
        print("\nMissing files:")
        for name in missing_files[:30]:
            print(name)
        if len(missing_files) > 30:
            print(f"... {len(missing_files) - 30} more files not shown")

    if invalid_rows:
        print("\nInvalid rows:")
        for row in invalid_rows[:10]:
            print(row)


if __name__ == "__main__":
    main()