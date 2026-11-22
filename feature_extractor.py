"""
Feature extraction module using pretrained ResNet18.

This module provides the ResNet18FeatureExtractor class, which extracts
512-dimensional feature representations for single images (e.g., for real-time
inference) or image batches directly in memory.

NOTE: This module strictly operates in-memory and does NOT save any feature files.
All feature saving and file persistence (.npz) is handled exclusively by batch_extractor.py.
"""

from pathlib import Path
from typing import Union, List, Optional
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ImportError:
    torch = None
    nn = None
    models = None
    transforms = None


class ResNet18FeatureExtractor:
    """ResNet-18 based feature extractor without the final classification head."""

    def __init__(self, device: Optional[str] = None, pretrained: bool = True):
        if torch is None:
            raise ImportError(
                "PyTorch and torchvision are required for feature extraction. "
                "Please install them via: pip install torch torchvision"
            )

        # Select compute device: CUDA > MPS (Apple Silicon) > CPU
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load ResNet-18 with ImageNet weights
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.model = models.resnet18(pretrained=pretrained)

        # Replace classification layer with Identity to output the 512-d pooling features
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> "torch.Tensor":
        """Load and preprocess a single image into a normalized PyTorch tensor."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        return self.transform(image)

    def extract_single(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Extract features for a single image in memory.
        
        Args:
            image_input: File path or PIL Image.
            
        Returns:
            np.ndarray: 1D feature vector of shape (512,), float32.
            Does NOT save any files to disk.
        """
        with torch.no_grad():
            tensor = self.preprocess_image(image_input).unsqueeze(0).to(self.device)
            features = self.model(tensor)
            return features.cpu().numpy().flatten().astype(np.float32)

    def extract_batch(self, image_inputs: Union[List[Union[str, Path, Image.Image]], "torch.Tensor"]) -> np.ndarray:
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
                tensors = torch.stack([self.preprocess_image(img) for img in image_inputs]).to(self.device)

            features = self.model(tensors)
            return features.cpu().numpy().astype(np.float32)


def extract_single_feature(image_path: Union[str, Path], device: Optional[str] = None) -> np.ndarray:
    """Convenience function to extract features for a single image in memory (returns np.ndarray, saves no files)."""
    extractor = ResNet18FeatureExtractor(device=device)
    return extractor.extract_single(image_path)
