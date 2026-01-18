"""
Document Shadow Removal Dataset
===============================
Handles loading and preprocessing for document shadow removal datasets:
- A-BSDD (Synthetic)
- Doc3DS+ (Reflectance-based)
- A-OSR (Augmented Real)

Supports both training and inference modes with proper augmentation.
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF


class DocumentShadowDataset(Dataset):
    """
    Dataset for document shadow removal.
    
    Expected directory structure:
    dataset_root/
        input/          # Shadowed document images
        target/         # Ground truth shadow-free images
        mask/           # (Optional) Shadow masks
        metadata.csv    # (Optional) Image metadata
    
    Args:
        root_dir: Path to dataset root directory
        stage: Training stage (1 for low-res, 2 for high-res)
        mode: 'train' or 'test'
        transform: Optional additional transforms
        use_mask: Whether to load and return shadow masks
        low_res_size: Target size for stage 1 (H, W)
        high_res_size: Target size for stage 2 (H, W)
    """
    
    def __init__(
        self,
        root_dir: str,
        stage: int = 1,
        mode: str = 'train',
        transform: Optional[transforms.Compose] = None,
        use_mask: bool = True,
        low_res_size: Tuple[int, int] = (192, 256),
        high_res_size: Tuple[int, int] = (768, 1024)
    ):
        self.root_dir = Path(root_dir)
        self.stage = stage
        self.mode = mode
        self.transform = transform
        self.use_mask = use_mask
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        
        # Set target size based on stage
        self.target_size = low_res_size if stage == 1 else high_res_size
        
        # Find all image pairs
        self.input_dir = self.root_dir / 'input'
        self.target_dir = self.root_dir / 'target'
        self.mask_dir = self.root_dir / 'mask'
        
        # Get image list
        self.image_list = self._get_image_list()
        
        # Normalization parameters (ImageNet stats for MobileNet)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Denormalization for visualization
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def _get_image_list(self) -> List[str]:
        """Get list of valid image filenames."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Get input images
        if not self.input_dir.exists():
            raise ValueError(f"Input directory not found: {self.input_dir}")
        
        input_files = set()
        for f in self.input_dir.iterdir():
            if f.suffix.lower() in valid_extensions:
                input_files.add(f.stem)
        
        # Get target images
        if not self.target_dir.exists():
            raise ValueError(f"Target directory not found: {self.target_dir}")
        
        target_files = set()
        for f in self.target_dir.iterdir():
            if f.suffix.lower() in valid_extensions:
                target_files.add(f.stem)
        
        # Find matching pairs
        valid_pairs = sorted(input_files & target_files)
        
        if len(valid_pairs) == 0:
            raise ValueError(f"No valid image pairs found in {self.root_dir}")
        
        print(f"Found {len(valid_pairs)} image pairs in {self.root_dir}")
        return valid_pairs
    
    def _find_image_path(self, directory: Path, stem: str) -> Path:
        """Find image path with any valid extension."""
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Image not found: {stem} in {directory}")
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load image and convert to RGB."""
        return Image.open(path).convert('RGB')
    
    def _load_mask(self, stem: str) -> Optional[Image.Image]:
        """Load mask if available."""
        if not self.use_mask or not self.mask_dir.exists():
            return None
        
        try:
            mask_path = self._find_image_path(self.mask_dir, stem)
            return Image.open(mask_path).convert('L')
        except FileNotFoundError:
            return None
    
    def _apply_augmentation(
        self, 
        input_img: Image.Image, 
        target_img: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
        """Apply synchronized augmentation to input, target, and mask."""
        
        # Random horizontal flip
        if random.random() > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)
            if mask is not None:
                mask = TF.hflip(mask)
        
        # Random rotation (-5 to 5 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            input_img = TF.rotate(input_img, angle, fill=255)
            target_img = TF.rotate(target_img, angle, fill=255)
            if mask is not None:
                mask = TF.rotate(mask, angle, fill=0)
        
        # Random brightness/contrast adjustment (input only)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            input_img = TF.adjust_brightness(input_img, brightness_factor)
        
        return input_img, target_img, mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dict containing:
                - 'input': Shadowed image tensor (C, H, W)
                - 'target': Ground truth tensor (C, H, W)
                - 'mask': Shadow mask tensor (1, H, W) if available
                - 'filename': Original filename
        """
        stem = self.image_list[idx]
        
        # Load images
        input_path = self._find_image_path(self.input_dir, stem)
        target_path = self._find_image_path(self.target_dir, stem)
        
        input_img = self._load_image(input_path)
        target_img = self._load_image(target_path)
        mask = self._load_mask(stem)
        
        # Apply augmentation during training
        if self.mode == 'train':
            input_img, target_img, mask = self._apply_augmentation(
                input_img, target_img, mask
            )
        
        # Resize to target size
        input_img = input_img.resize(
            (self.target_size[1], self.target_size[0]), 
            Image.BILINEAR
        )
        target_img = target_img.resize(
            (self.target_size[1], self.target_size[0]), 
            Image.BILINEAR
        )
        if mask is not None:
            mask = mask.resize(
                (self.target_size[1], self.target_size[0]), 
                Image.NEAREST
            )
        
        # Convert to tensors
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)
        
        # Normalize
        input_tensor = self.normalize(input_tensor)
        target_tensor = self.normalize(target_tensor)
        
        # Prepare output dict
        sample = {
            'input': input_tensor,
            'target': target_tensor,
            'filename': stem
        }
        
        if mask is not None:
            mask_tensor = TF.to_tensor(mask)
            sample['mask'] = mask_tensor
        
        return sample


class MultiScaleDocumentDataset(Dataset):
    """
    Multi-scale dataset for LP-IOANet training.
    
    Provides images at multiple resolutions for Laplacian pyramid training.
    Returns both low-res (for IOANet) and high-res (for refiners) versions.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        low_res_size: Tuple[int, int] = (192, 256),
        mid_res_size: Tuple[int, int] = (384, 512),
        high_res_size: Tuple[int, int] = (768, 1024)
    ):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.low_res_size = low_res_size
        self.mid_res_size = mid_res_size
        self.high_res_size = high_res_size
        
        # Create base dataset at high resolution
        self._base_dataset = DocumentShadowDataset(
            root_dir=root_dir,
            stage=2,  # High resolution
            mode=mode,
            high_res_size=high_res_size
        )
        
        self.normalize = self._base_dataset.normalize
        self.denormalize = self._base_dataset.denormalize
    
    def __len__(self) -> int:
        return len(self._base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get multi-scale sample.
        
        Returns:
            Dict containing images at all pyramid levels:
                - 'input_high', 'target_high': Full resolution
                - 'input_mid', 'target_mid': Mid resolution
                - 'input_low', 'target_low': Low resolution
                - 'filename': Original filename
        """
        # Get high-res sample
        sample = self._base_dataset[idx]
        
        input_high = sample['input']
        target_high = sample['target']
        
        # Generate mid-resolution versions
        input_mid = TF.resize(
            input_high, 
            list(self.mid_res_size), 
            interpolation=TF.InterpolationMode.BILINEAR
        )
        target_mid = TF.resize(
            target_high, 
            list(self.mid_res_size), 
            interpolation=TF.InterpolationMode.BILINEAR
        )
        
        # Generate low-resolution versions
        input_low = TF.resize(
            input_high, 
            list(self.low_res_size), 
            interpolation=TF.InterpolationMode.BILINEAR
        )
        target_low = TF.resize(
            target_high, 
            list(self.low_res_size), 
            interpolation=TF.InterpolationMode.BILINEAR
        )
        
        return {
            'input_high': input_high,
            'target_high': target_high,
            'input_mid': input_mid,
            'target_mid': target_mid,
            'input_low': input_low,
            'target_low': target_low,
            'filename': sample['filename'],
            'mask': sample.get('mask', None)
        }


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    stage: int = 1,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Path to training data
        test_dir: Path to test/validation data
        stage: Training stage (1 or 2)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        train_loader, test_loader
    """
    if stage == 1:
        # Stage 1: Low resolution dataset
        train_dataset = DocumentShadowDataset(
            root_dir=train_dir,
            stage=1,
            mode='train'
        )
        test_dataset = DocumentShadowDataset(
            root_dir=test_dir,
            stage=1,
            mode='test'
        )
    else:
        # Stage 2: Multi-scale dataset
        train_dataset = MultiScaleDocumentDataset(
            root_dir=train_dir,
            mode='train'
        )
        test_dataset = MultiScaleDocumentDataset(
            root_dir=test_dir,
            mode='test'
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    
    denorm = tensor * std + mean
    return denorm.clamp(0, 1).squeeze(0) if denorm.shape[0] == 1 else denorm.clamp(0, 1)


# Test the dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test with a sample directory structure
    test_root = "Dataset/train"
    
    if os.path.exists(test_root):
        print("Testing DocumentShadowDataset (Stage 1)...")
        dataset = DocumentShadowDataset(
            root_dir=test_root,
            stage=1,
            mode='train'
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Input shape: {sample['input'].shape}")
            print(f"Target shape: {sample['target'].shape}")
            print(f"Filename: {sample['filename']}")
            
            # Test multi-scale dataset
            print("\nTesting MultiScaleDocumentDataset...")
            ms_dataset = MultiScaleDocumentDataset(
                root_dir=test_root,
                mode='train'
            )
            
            sample = ms_dataset[0]
            print(f"Input high shape: {sample['input_high'].shape}")
            print(f"Input mid shape: {sample['input_mid'].shape}")
            print(f"Input low shape: {sample['input_low'].shape}")
    else:
        print(f"Test directory not found: {test_root}")
        print("Please create the dataset structure to test.")
