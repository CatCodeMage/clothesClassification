"""
Batch feature extraction script using ResNet18.

This script reads images from the dataset along with their metadata from styles.csv,
extracts 512-dimensional visual feature vectors using a pretrained ResNet-18 model,
and saves the resulting feature vectors and dataset features as compressed .npz files
inside the 'features' directory.

All metadata features from styles.csv are preserved, with specific emphasis on
the three targets we will predict:
  - masterCategory (e.g., Apparel, Accessories, Footwear)
  - subCategory    (e.g., Topwear, Bottomwear, Shoes)
  - baseColour      (e.g., Black, White, Blue)

NOTE: File persistence (.npz saving) is handled exclusively by this script.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    models = None
    transforms = None


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
TARGET_COLUMNS = ["masterCategory", "subCategory", "baseColour"]


class FashionDataset(Dataset):
    """PyTorch Dataset for fashion images and all associated metadata features."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.columns = [col for col in self.df.columns if col in METADATA_COLUMNS or col == "filename"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = int(row["id"])
        image_filename = f"{image_id}.jpg"
        image_path = self.images_dir / image_filename

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback to a blank image if corrupted to prevent pipeline crash
            print(f"Warning: could not open image {image_path}: {e}", file=sys.stderr)
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        item = {
            "image": image_tensor,
            "filename": image_filename,
        }
        for col in self.columns:
            item[col] = row[col]

        return item


def get_default_device(user_device: Optional[str] = None) -> "torch.device":
    """Select the best available compute device (CUDA, Apple Silicon MPS, or CPU)."""
    if user_device:
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet18_model(device: "torch.device") -> "torch.nn.Module":
    """Instantiate ResNet18 and replace its classification head with an Identity layer."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    # Cut off final fully connected layer to produce 512-dim features
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def prepare_dataframe(csv_path: Path, images_dir: Path, only_apparel: bool = False) -> pd.DataFrame:
    """Load metadata from styles.csv, sanitize all features, and verify image existence."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found at: {images_dir}")

    # styles.csv may contain occasional unescaped commas in productDisplayName
    df = pd.read_csv(csv_path, on_bad_lines="skip")

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


def extract_features_in_batches(
    csv_path: str = "dataset/styles.csv",
    images_dir: str = "dataset/images",
    output_dir: str = "features",
    batch_size: int = 64,
    device_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    only_apparel: bool = False,
    consolidate: bool = True,
    num_workers: int = 0,
) -> Path:
    """
    Extract ResNet-18 features in batches and save them into the output directory as .npz files.
    
    Includes all metadata features with emphasis on the 3 targets to guess:
    masterCategory, subCategory, and baseColour.

    Args:
        csv_path: Path to styles.csv
        images_dir: Path to directory containing .jpg images
        output_dir: Directory where .npz feature files will be saved
        batch_size: Number of images per batch
        device_name: 'cuda', 'mps', 'cpu', or None for auto-detection
        max_samples: Optional maximum number of images to process (useful for testing)
        only_apparel: If True, only extracts items where masterCategory == 'Apparel'
        consolidate: If True, also merges all batches into features_all.npz
        num_workers: Number of workers for DataLoader
        
    Returns:
        Path: Path to the output directory containing saved .npz files.
    """
    if torch is None:
        raise ImportError(
            "PyTorch and torchvision are required for batch feature extraction. "
            "Please install them via: pip install torch torchvision"
        )

    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device(device_name)
    print(f"Using compute device: {device}")

    # Standard ImageNet normalization for ResNet-18
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    df = prepare_dataframe(csv_path, images_dir, only_apparel=only_apparel)
    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].copy()
        print(f"Limiting execution to first {len(df)} samples.")

    dataset = FashionDataset(df, images_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    model = build_resnet18_model(device)

    all_features: List[np.ndarray] = []
    all_metadata: Dict[str, List[np.ndarray]] = {col: [] for col in METADATA_COLUMNS}
    all_metadata["filename"] = []

    print(f"Starting feature extraction for {len(dataset)} images in {len(dataloader)} batches...")
    print(f"Target features to guess: {', '.join(TARGET_COLUMNS)}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            images = batch["image"].to(device)
            # Forward pass through ResNet-18 (512-dimensional output)
            outputs = model(images)
            batch_features = outputs.cpu().numpy().astype(np.float32)

            # Collect all metadata columns
            batch_data: Dict[str, np.ndarray] = {
                "features": batch_features,
            }

            for col in METADATA_COLUMNS:
                val = batch[col]
                if isinstance(val, torch.Tensor):
                    arr = val.cpu().numpy()
                else:
                    arr = np.array(val)
                batch_data[col] = arr
                if consolidate:
                    all_metadata[col].append(arr)

            # Add filename
            filenames = np.array(batch["filename"])
            batch_data["filename"] = filenames
            batch_data["filenames"] = filenames
            if consolidate:
                all_metadata["filename"].append(filenames)

            # Explicit target labels to guess and handy aliases
            batch_data["target_masterCategory"] = batch_data["masterCategory"]
            batch_data["target_subCategory"] = batch_data["subCategory"]
            batch_data["target_baseColour"] = batch_data["baseColour"]
            batch_data["color"] = batch_data["baseColour"]
            batch_data["type"] = batch_data["masterCategory"]
            batch_data["subtype"] = batch_data["subCategory"]

            # Save individual batch .npz into features/ directory
            batch_filename = f"batch_{batch_idx:04d}.npz"
            batch_path = output_dir / batch_filename
            np.savez_compressed(batch_path, **batch_data)

            if consolidate:
                all_features.append(batch_features)

    if consolidate and all_features:
        consolidated_path = output_dir / "features_all.npz"
        print(f"Consolidating all batches into {consolidated_path}...")
        merged_features = np.concatenate(all_features, axis=0)

        consolidated_data: Dict[str, np.ndarray] = {
            "features": merged_features,
        }

        for col, chunks in all_metadata.items():
            consolidated_data[col] = np.concatenate(chunks, axis=0)

        # Aliases for convenience
        consolidated_data["filenames"] = consolidated_data["filename"]
        consolidated_data["target_masterCategory"] = consolidated_data["masterCategory"]
        consolidated_data["target_subCategory"] = consolidated_data["subCategory"]
        consolidated_data["target_baseColour"] = consolidated_data["baseColour"]
        consolidated_data["color"] = consolidated_data["baseColour"]
        consolidated_data["type"] = consolidated_data["masterCategory"]
        consolidated_data["subtype"] = consolidated_data["subCategory"]

        np.savez_compressed(consolidated_path, **consolidated_data)
        print(f"Saved {merged_features.shape[0]} feature vectors of dimension {merged_features.shape[1]} to {consolidated_path}")

    print(f"Batch feature extraction complete! Files saved in: {output_dir.resolve()}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet-18 features in batches from the dataset and save as .npz"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="dataset/styles.csv",
        help="Path to styles.csv file (default: dataset/styles.csv)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="dataset/images",
        help="Path to image directory (default: dataset/images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="features",
        help="Directory to save .npz feature files (default: features)",
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
        "--no-consolidate",
        action="store_true",
        help="Do not save the combined features_all.npz file",
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
    extract_features_in_batches(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        max_samples=args.max_samples,
        only_apparel=args.only_apparel,
        consolidate=not args.no_consolidate,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
