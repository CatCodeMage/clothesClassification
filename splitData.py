"""
Splits the dataset into three folders: train (70%), validate (20%) and test (10%).

Each folder gets a copy of its images and its own styles.csv with only its rows:

    dataset/train/images/*.jpg     dataset/train/styles.csv
    dataset/validate/images/*.jpg  dataset/validate/styles.csv
    dataset/test/images/*.jpg      dataset/test/styles.csv

The split is stratified by STRATIFY_COLUMN, so every folder keeps the same class
proportions, and it uses a fixed seed, so running it again gives the same split.
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"
SOURCE_IMAGES_DIR = DATASET_DIR / "images"
SOURCE_STYLES_CSV = DATASET_DIR / "styles.csv"

SPLIT_FRACTIONS = {"train": 0.7, "validate": 0.2, "test": 0.1}
TARGET_COLUMNS = ["subCategory", "articleType", "baseColour"]
# articleType is the finest of the targets and almost always fixes subCategory,
# so stratifying by it keeps the proportions of both.
STRATIFY_COLUMN = "articleType"
RANDOM_SEED = 42


def read_styles(csv_path):
    """Returns the header and the rows of styles.csv.

    Some product names contain unquoted commas, so those lines have more fields
    than the header; the extra fields are joined back into the last column.
    """
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        last_column_index = len(header) - 1
        rows = []
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) < len(header):
                raise ValueError(f"{csv_path}:{line_number}: expected {len(header)} fields, found {len(fields)}: {fields}")
            name_parts = fields[last_column_index:]
            rows.append(fields[:last_column_index] + [",".join(name_parts)])
    return header, rows


def image_path_of(row):
    return SOURCE_IMAGES_DIR / f"{row[0]}.jpg"


def group_rows_by_class(rows, class_column_index):
    rows_by_class = defaultdict(list)
    for row in rows:
        rows_by_class[row[class_column_index]].append(row)
    return rows_by_class


def split_one_class(class_rows, randomizer):
    """Shuffles the rows of one class and cuts them by SPLIT_FRACTIONS."""
    shuffled_rows = class_rows[:]
    randomizer.shuffle(shuffled_rows)
    rows_by_split = {}
    start = 0
    split_names = list(SPLIT_FRACTIONS)
    for split_name in split_names:
        is_last_split = split_name == split_names[-1]
        # The last split takes whatever rounding left, so no row is lost.
        end = len(shuffled_rows) if is_last_split else start + round(len(shuffled_rows) * SPLIT_FRACTIONS[split_name])
        rows_by_split[split_name] = shuffled_rows[start:end]
        start = end
    return rows_by_split


def split_stratified(rows, class_column_index):
    randomizer = random.Random(RANDOM_SEED)
    rows_by_split = {split_name: [] for split_name in SPLIT_FRACTIONS}
    rows_by_class = group_rows_by_class(rows, class_column_index)
    for class_name in sorted(rows_by_class):
        for split_name, class_split_rows in split_one_class(rows_by_class[class_name], randomizer).items():
            rows_by_split[split_name].extend(class_split_rows)
    return rows_by_split


def refuse_if_split_exists():
    """Writing over an earlier split would mix both, so it has to be deleted by hand first."""
    existing_folders = [DATASET_DIR / split_name for split_name in SPLIT_FRACTIONS if (DATASET_DIR / split_name).exists()]
    if existing_folders:
        folder_list = ", ".join(str(folder) for folder in existing_folders)
        raise FileExistsError(f"These folders already exist, delete them to split again: {folder_list}")


def write_split(split_name, header, split_rows):
    split_dir = DATASET_DIR / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    for row in split_rows:
        shutil.copy2(image_path_of(row), images_dir)
    with open(split_dir / "styles.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(split_rows)


def print_summary(header, rows_by_split):
    """Prints the size of each split and, per target, how many classes it holds."""
    total_rows = sum(len(split_rows) for split_rows in rows_by_split.values())
    for split_name, split_rows in rows_by_split.items():
        print(f"{split_name}: {len(split_rows)} images ({len(split_rows) / total_rows:.1%})")
    for target_column in TARGET_COLUMNS:
        target_index = header.index(target_column)
        train_classes = {row[target_index] for row in rows_by_split["train"]}
        print(f"{target_column}: {len(train_classes)} classes in train")
        for split_name in ["validate", "test"]:
            split_classes = {row[target_index] for row in rows_by_split[split_name]}
            missing_classes = sorted(train_classes - split_classes)
            print(f"    {split_name}: {len(split_classes)} classes, missing {len(missing_classes)}: {missing_classes}")


def main():
    refuse_if_split_exists()
    header, rows = read_styles(SOURCE_STYLES_CSV)
    rows_with_image = [row for row in rows if image_path_of(row).exists()]
    print(f"{len(rows)} rows in {SOURCE_STYLES_CSV.name}, {len(rows) - len(rows_with_image)} skipped because their image is missing")

    class_column_index = header.index(STRATIFY_COLUMN)
    rows_by_split = split_stratified(rows_with_image, class_column_index)
    for split_name, split_rows in rows_by_split.items():
        print(f"Copying {split_name}...")
        write_split(split_name, header, split_rows)
    print_summary(header, rows_by_split)


if __name__ == "__main__":
    main()
