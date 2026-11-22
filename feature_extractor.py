"""
Feature extraction module using pretrained ResNet18.

This module provides the ResNet18FeatureExtractor class, which extracts
512-dimensional feature representations for single images (e.g., for real-time
inference) or image batches directly in memory.

NOTE: This module strictly operates in-memory and does NOT save any feature files.
All feature saving and file persistence (.npz) is handled exclusively by batch_extractor.py.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

INPUT_SIDE_PIXELS = 224
# The product photos have a white background, so padding with white adds no edges.
PADDING_COLOR = (255, 255, 255)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ImageInput = Union[str, Path, Image.Image]

_to_normalized_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_input: ImageInput) -> torch.Tensor:
    """Load and preprocess a single image into a normalized PyTorch tensor.

    It is a module-level function, not a method, so DataLoader workers can use it
    without receiving a copy of the model.
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # The images are 60x80. Padding to a square keeps the whole product and its
    # proportions; the usual 224 center crop would cut off its top and bottom.
    square_image = ImageOps.pad(image, (INPUT_SIDE_PIXELS, INPUT_SIDE_PIXELS), color=PADDING_COLOR)
    return _to_normalized_tensor(square_image)


def select_device(requested_device: Optional[str] = None) -> torch.device:
    """Select compute device: the requested one, else CUDA > MPS (Apple Silicon) > CPU."""
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResNet18FeatureExtractor:
    """ResNet-18 based feature extractor without the final classification head."""

    def __init__(self, device: Optional[str] = None, pretrained: bool = True):
        self.device = select_device(device)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        # Replace classification layer with Identity to output the 512-d pooling features
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_single(self, image_input: ImageInput) -> np.ndarray:
        """
        Extract features for a single image in memory.

        Args:
            image_input: File path or PIL Image.

        Returns:
            np.ndarray: 1D feature vector of shape (512,), float32.
            Does NOT save any files to disk.
        """
        return self.extract_batch([image_input])[0]

    def extract_batch(self, image_inputs: Union[List[ImageInput], torch.Tensor]) -> np.ndarray:
        """
        Extract features for a batch of images in memory.

        Args:
            image_inputs: List of paths/PIL images, or a pre-collated torch.Tensor of shape (B, 3, 224, 224).

        Returns:
            np.ndarray: 2D feature matrix of shape (B, 512), float32.
            Does NOT save any files to disk.
        """
        with torch.no_grad():
            if isinstance(image_inputs, torch.Tensor):
                tensors = image_inputs.to(self.device)
            else:
                tensors = torch.stack([preprocess_image(image) for image in image_inputs]).to(self.device)

            features = self.model(tensors)
            return features.cpu().numpy().astype(np.float32)


@lru_cache(maxsize=None)
def get_shared_extractor(device: Optional[str] = None) -> ResNet18FeatureExtractor:
    """Returns one extractor per device, because loading the model on every call is slow."""
    return ResNet18FeatureExtractor(device=device)


def extract_single_feature(image_path: Union[str, Path], device: Optional[str] = None) -> np.ndarray:
    """Convenience function to extract features for a single image in memory (returns np.ndarray, saves no files)."""
    return get_shared_extractor(device).extract_single(image_path)
