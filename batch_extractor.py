"""
Batch feature extraction script using ResNet18.

This script reads images from the dataset along with their metadata from styles.csv,
extracts 512-dimensional visual feature vectors using feature_extractor.py,
and saves the resulting feature vectors and dataset features as compressed .npz files
inside the 'features' directory.

By default it processes the train, test and validation splits created by splitData.py.

All metadata features from styles.csv are preserved, with specific emphasis on
the three targets we will predict:
  - subCategory (e.g., Topwear, Bottomwear, Shoes)
  - articleType (e.g., Tshirts, Casual Shoes, Handbags)
  - baseColour  (e.g., Black, White, Blue)

NOTE: File persistence (.npz saving) is handled exclusively by this script.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from feature_extractor import ResNet18FeatureExtractor, preprocess_image

DEFAULT_DATASET_DIR = "dataset"
DEFAULT_OUTPUT_DIR = "features"
DATASET_SPLITS = {"train": "train", "test": "test", "validation": "validate"}

# Metadata columns present in styles.csv
METADATA_COLUMNS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName",
]

# The 3 primary target features we will guess/predict
TARGET_COLUMNS = ["subCategory", "articleType", "baseColour"]

# Handy short names for the target columns inside the .npz files
TARGET_ALIASES = {"color": "baseColour", "type": "subCategory", "subtype": "articleType"}


class FashionDataset(Dataset):
    """PyTorch Dataset for fashion images and all associated metadata features."""

    def __init__(self, df: pd.DataFrame, images_dir: Path):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.columns = [col for col in self.df.columns if col in METADATA_COLUMNS or col == "filename"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_filename = f"{int(row['id'])}.jpg"
        # A corrupt image stops the run: replacing it would train on a wrong picture with a real label.
        item = {
            "image": preprocess_image(self.images_dir / image_filename),
            "filename": image_filename,
        }
        for col in self.columns:
            item[col] = row[col]

        return item


def prepare_dataframe(csv_path: Path, images_dir: Path, only_apparel: bool = False) -> pd.DataFrame:
    """Load metadata from styles.csv, sanitize all features, and verify image existence."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found at: {images_dir}")

    # The CSVs written by splitData.py quote names with commas, so no row has to be skipped.
    df = pd.read_csv(csv_path)

    # Clean and sanitize all metadata columns
    df["id"] = df["id"].astype(int)
    df["gender"] = df["gender"].fillna("Unknown").astype(str)
    df["masterCategory"] = df["masterCategory"].fillna("Unknown").astype(str)
    df["subCategory"] = df["subCategory"].fillna("Unknown").astype(str)
    df["articleType"] = df["articleType"].fillna("Unknown").astype(str)
    df["baseColour"] = df["baseColour"].fillna("Unknown").astype(str)
    df["season"] = df["season"].fillna("Unknown").astype(str)
    df["year"] = df["year"].fillna(-1).astype(int)
    df["usage"] = df["usage"].fillna("Unknown").astype(str)
    df["productDisplayName"] = df["productDisplayName"].fillna("").astype(str)

    if only_apparel:
        df = df[df["masterCategory"].str.lower() == "apparel"].copy()

    # Add image filename column
    df["filename"] = df["id"].apply(lambda img_id: f"{img_id}.jpg")

    # Only keep records where image exists on disk
    existing_images = set(os.listdir(images_dir))
    df["img_exists"] = df["filename"].apply(lambda fname: fname in existing_images)
    total_before = len(df)
    df = df[df["img_exists"]].drop(columns=["img_exists"]).reset_index(drop=True)
    print(f"Loaded {len(df)} items with existing images out of {total_before} entries in {csv_path.name}.")

    return df


def add_target_keys(npz_data: Dict[str, np.ndarray]) -> None:
    """Adds the explicit target_* labels and the short aliases to the arrays to save."""
    npz_data["filenames"] = npz_data["filename"]
    for target_column in TARGET_COLUMNS:
        npz_data[f"target_{target_column}"] = npz_data[target_column]
    for alias, column in TARGET_ALIASES.items():
        npz_data[alias] = npz_data[column]


def metadata_to_arrays(batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Converts the metadata of one DataLoader batch into numpy arrays."""
    arrays = {}
    for col in METADATA_COLUMNS:
        value = batch[col]
        arrays[col] = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.array(value)
    arrays["filename"] = np.array(batch["filename"])
    return arrays


def save_consolidated(
    output_dir: Path,
    all_features: List[np.ndarray],
    all_metadata: Dict[str, List[np.ndarray]],
    output_name: str = "features_all.npz",
) -> None:
    consolidated_path = output_dir / output_name
    print(f"Consolidating all batches into {consolidated_path}...")
    merged_features = np.concatenate(all_features, axis=0)

    consolidated_data: Dict[str, np.ndarray] = {"features": merged_features}
    for col, chunks in all_metadata.items():
        consolidated_data[col] = np.concatenate(chunks, axis=0)
    add_target_keys(consolidated_data)

    np.savez_compressed(consolidated_path, **consolidated_data)
    print(f"Saved {merged_features.shape[0]} feature vectors of dimension {merged_features.shape[1]} to {consolidated_path}")


def extract_features_in_batches(
    csv_path: str = "dataset/train/styles.csv",
    images_dir: str = "dataset/train/images",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    device_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    only_apparel: bool = False,
    consolidate: bool = True,
    num_workers: int = 0,
    output_name: str = "features_all.npz",
) -> Path:
    """
    Extract ResNet-18 features in batches and save them into the output directory as .npz files.

    Includes all metadata features with emphasis on the 3 targets to guess:
    subCategory, articleType, and baseColour.

    Args:
        csv_path: Path to styles.csv
        images_dir: Path to directory containing .jpg images
        output_dir: Directory where .npz feature files will be saved
        batch_size: Number of images per batch
        device_name: 'cuda', 'mps', 'cpu', or None for auto-detection
        max_samples: Optional maximum number of images to process (useful for testing)
        only_apparel: If True, only extracts items where masterCategory == 'Apparel'
        consolidate: If True, saves all batches into one compressed .npz file
        num_workers: Number of workers for DataLoader
        output_name: Name of the consolidated .npz file

    Returns:
        Path: Path to the output directory containing saved .npz files.
    """
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = ResNet18FeatureExtractor(device=device_name)
    print(f"Using compute device: {extractor.device}")

    df = prepare_dataframe(csv_path, images_dir, only_apparel=only_apparel)
    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].copy()
        print(f"Limiting execution to first {len(df)} samples.")

    dataset = FashionDataset(df, images_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    all_features: List[np.ndarray] = []
    all_metadata: Dict[str, List[np.ndarray]] = {col: [] for col in [*METADATA_COLUMNS, "filename"]}

    print(f"Starting feature extraction for {len(dataset)} images in {len(dataloader)} batches...")
    print(f"Target features to guess: {', '.join(TARGET_COLUMNS)}")

    for batch in tqdm(dataloader, desc="Extracting features"):
        batch_features = extractor.extract_batch(batch["image"])
        batch_metadata = metadata_to_arrays(batch)

        if consolidate:
            all_features.append(batch_features)
            for col, values in batch_metadata.items():
                all_metadata[col].append(values)

    if consolidate and all_features:
        save_consolidated(output_dir, all_features, all_metadata, output_name)

    print(f"Batch feature extraction complete! Files saved in: {output_dir.resolve()}")
    return output_dir


def check_dataset_splits(dataset_dir: Path) -> None:
    """Verify that every expected split contains styles.csv and an images folder."""
    for split_dir_name in DATASET_SPLITS.values():
        split_dir = dataset_dir / split_dir_name
        csv_path = split_dir / "styles.csv"
        images_dir = split_dir / "images"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split folder not found at: {split_dir}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metadata file not found at: {csv_path}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images folder not found at: {images_dir}")


def extract_all_splits(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    device_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    only_apparel: bool = False,
    consolidate: bool = True,
    num_workers: int = 0,
) -> Path:
    """Extract all dataset splits and save one archive for each split."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    check_dataset_splits(dataset_dir)

    for output_name, split_dir_name in DATASET_SPLITS.items():
        split_dir = dataset_dir / split_dir_name
        print(f"\nProcessing {split_dir_name} split...")
        extract_features_in_batches(
            csv_path=split_dir / "styles.csv",
            images_dir=split_dir / "images",
            output_dir=output_dir,
            batch_size=batch_size,
            device_name=device_name,
            max_samples=max_samples,
            only_apparel=only_apparel,
            consolidate=consolidate,
            num_workers=num_workers,
            output_name=f"{output_name}.npz",
        )

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet-18 features from all dataset splits and save as .npz"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to the dataset folder containing train, test and validate (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the split .npz files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images per batch (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'mps', 'cpu', or auto-detect if None)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to process",
    )
    parser.add_argument(
        "--only-apparel",
        action="store_true",
        help="Only extract features for apparel items (masterCategory == 'Apparel')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_all_splits(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        max_samples=args.max_samples,
        only_apparel=args.only_apparel,
        consolidate=True,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
