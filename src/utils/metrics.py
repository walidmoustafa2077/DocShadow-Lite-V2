"""
Evaluation Metrics for Document Shadow Removal
==============================================
Implements standard image quality metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from torchvision import transforms


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image tensor
        target: Target image tensor
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()


def _gaussian_kernel(
    size: int = 11,
    sigma: float = 1.5,
    channels: int = 3
) -> torch.Tensor:
    """Create Gaussian kernel for SSIM."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    grid = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
    kernel = torch.exp(-grid / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    val_range: float = 1.0,
    size_average: bool = True
) -> float:
    """
    Calculate Structural Similarity Index.
    
    Args:
        pred: Predicted image tensor (B, C, H, W)
        target: Target image tensor (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Gaussian sigma
        val_range: Value range (1.0 for normalized)
        size_average: Average over batch
        
    Returns:
        SSIM value
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    channels = pred.size(1)
    kernel = _gaussian_kernel(window_size, sigma, channels).to(pred.device)
    
    # Constants
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    # Calculate means
    mu_pred = F.conv2d(pred, kernel, padding=window_size//2, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=window_size//2, groups=channels)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    # Calculate variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, kernel, padding=window_size//2, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel, padding=window_size//2, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, padding=window_size//2, groups=channels) - mu_pred_target
    
    # Calculate SSIM
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    ssim_map = numerator / denominator
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


class Evaluator:
    """
    Comprehensive evaluator for shadow removal models.
    
    Computes multiple metrics and aggregates results.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        compute_lpips: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.compute_lpips = compute_lpips
        
        if compute_lpips:
            try:
                from lpips import LPIPS
                self.lpips_fn = LPIPS(net='vgg').to(self.device)
                self.lpips_fn.eval()
            except ImportError:
                print("LPIPS not available. Install with: pip install lpips")
                self.compute_lpips = False
        
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
    
    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update metrics with a new batch.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            Dict of metric values for this batch
        """
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        metrics = {}
        
        # Calculate PSNR for each image
        batch_size = pred.size(0)
        for i in range(batch_size):
            psnr = calculate_psnr(pred[i], target[i])
            self.psnr_values.append(psnr)
            
            ssim = calculate_ssim(pred[i], target[i])
            self.ssim_values.append(ssim)
        
        metrics['psnr'] = np.mean(self.psnr_values[-batch_size:])
        metrics['ssim'] = np.mean(self.ssim_values[-batch_size:])
        
        # Calculate LPIPS
        if self.compute_lpips:
            # LPIPS expects values in [-1, 1]
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            lpips_val = self.lpips_fn(pred_lpips, target_lpips).mean().item()
            self.lpips_values.append(lpips_val)
            metrics['lpips'] = lpips_val
        
        return metrics
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final aggregated metrics.
        
        Returns:
            Dict of averaged metrics
        """
        results = {
            'psnr': np.mean(self.psnr_values) if self.psnr_values else 0,
            'psnr_std': np.std(self.psnr_values) if self.psnr_values else 0,
            'ssim': np.mean(self.ssim_values) if self.ssim_values else 0,
            'ssim_std': np.std(self.ssim_values) if self.ssim_values else 0,
        }
        
        if self.compute_lpips and self.lpips_values:
            results['lpips'] = np.mean(self.lpips_values)
            results['lpips_std'] = np.std(self.lpips_values)
        
        return results
    
    def print_results(self):
        """Print formatted results."""
        results = self.compute()
        
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"PSNR: {results['psnr']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
        
        if 'lpips' in results:
            print(f"LPIPS: {results['lpips']:.4f} ± {results['lpips_std']:.4f}")
        
        print(f"Number of samples: {len(self.psnr_values)}")
        print("=" * 50)


class ResearchTracker:
    """
    Research-grade metric tracker for LP-IOANet training.
    
    Accumulates PSNR, SSIM, MAE, and loss components with proper precision.
    Ensures metrics are calculated consistently across batches with proper
    clamping and normalization for early training stability.
    
    Formatting specification (matching research requirements):
    - MAE: 5 decimals (global illumination correction precision)
    - PSNR: 2 decimals in dB (target: >30dB for successful de-shadowing)
    - SSIM: 4 decimals (OCR health metric, target: >0.85)
    - Loss values: 5 decimals (gradient monitoring)
    - Learning rate: scientific notation
    """
    
    def __init__(self):
        """Initialize metric storage."""
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators."""
        self.mae = []
        self.psnr = []
        self.ssim = []
        self.total_loss = []
        self.l1_loss = []
        self.lpips_loss = []
    
    def update(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        losses: Dict[str, float]
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            pred: Predicted images, shape (B, C, H, W), range [-1, 1] or [0, 1]
            target: Target images, shape (B, C, H, W), range [-1, 1] or [0, 1]
            losses: Dictionary containing 'total', 'l1', 'lpips' loss values
            
        Note:
            Automatically handles normalization and clamping to [0, 1]
            for stable metric calculation during early training.
        """
        # Detach and convert to CPU numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Normalize to [0, 1] range
        # Handle both [-1, 1] and [0, 1] input ranges
        if pred_np.min() < 0:
            # Input is in [-1, 1] range
            pred_np = (pred_np + 1.0) / 2.0
            target_np = (target_np + 1.0) / 2.0
        
        # Clamp to [0, 1] for stable metric calculation
        pred_np = np.clip(pred_np, 0, 1)
        target_np = np.clip(target_np, 0, 1)
        
        # MAE (Mean Absolute Error) - global metric across batch
        batch_mae = np.mean(np.abs(pred_np - target_np))
        self.mae.append(batch_mae)
        
        # PSNR and SSIM per image in batch
        batch_psnr = []
        batch_ssim = []
        
        for i in range(pred_np.shape[0]):
            # PSNR: Peak Signal-to-Noise Ratio (dB)
            mse = np.mean((pred_np[i] - target_np[i]) ** 2)
            if mse < 1e-8:
                psnr_val = 100.0  # Cap at 100dB for numerical stability
            else:
                psnr_val = 20.0 * np.log10(1.0 / np.sqrt(mse))
            batch_psnr.append(psnr_val)
            
            # SSIM: Structural Similarity Index (per channel, then averaged)
            # Convert from (C, H, W) to (H, W, C) for SSIM calculation
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            target_img = np.transpose(target_np[i], (1, 2, 0))
            
            # Calculate SSIM with data range [0, 1]
            ssim_val = _calculate_ssim(target_img, pred_img, data_range=1.0)
            batch_ssim.append(ssim_val)
        
        self.psnr.append(np.mean(batch_psnr))
        self.ssim.append(np.mean(batch_ssim))
        
        # Store loss components
        self.total_loss.append(losses.get('total', 0.0))
        self.l1_loss.append(losses.get('l1', 0.0))
        self.lpips_loss.append(losses.get('lpips', 0.0))
    
    def get_avg(self) -> Dict[str, float]:
        """
        Get averaged metrics across all batches.
        
        Returns:
            Dictionary with averaged metrics:
            - mae: Mean Absolute Error
            - psnr: Peak Signal-to-Noise Ratio (dB)
            - ssim: Structural Similarity Index
            - total: Total loss
            - l1: L1 loss component
            - lpips: LPIPS loss component
        """
        return {
            'mae': np.mean(self.mae) if self.mae else 0.0,
            'psnr': np.mean(self.psnr) if self.psnr else 0.0,
            'ssim': np.mean(self.ssim) if self.ssim else 0.0,
            'total': np.mean(self.total_loss) if self.total_loss else 0.0,
            'l1': np.mean(self.l1_loss) if self.l1_loss else 0.0,
            'lpips': np.mean(self.lpips_loss) if self.lpips_loss else 0.0,
        }
    
    def format_log(self, epoch: int, max_epochs: int, stage: str = "TRAIN", 
                   lr: float = 0.0, time_sec: float = 0.0) -> str:
        """
        Format metrics into research-grade log string.
        
        Args:
            epoch: Current epoch number (0-indexed)
            max_epochs: Total epochs
            stage: Training stage label ("TRAIN", "VAL", "TEST")
            lr: Current learning rate
            time_sec: Epoch time in seconds
            
        Returns:
            Formatted log string matching specification:
            Epoch XXX/XXX [STAGE] | MAE=X.XXXXX | PSNR=XX.XXdB | 
            SSIM=X.XXXX | Total=X.XXXXX | L1=X.XXXXX | LPIPS=X.XXXXX | 
            LR=X.XXe-0X | Time=XXX.Xs
        """
        avg = self.get_avg()
        
        log_str = (
            f"Epoch {epoch+1:4d}/{max_epochs:4d} [{stage:5s}] | "
            f"MAE={avg['mae']:.5f} | "
            f"PSNR={avg['psnr']:.2f}dB | "
            f"SSIM={avg['ssim']:.4f} | "
            f"Total={avg['total']:.5f} | "
            f"L1={avg['l1']:.5f} | "
            f"LPIPS={avg['lpips']:.5f} | "
            f"LR={lr:.2e}"
        )
        
        if time_sec > 0:
            log_str += f" | Time={time_sec:.1f}s"
        
        return log_str


def _calculate_ssim(img1, img2, data_range=1.0, win_size=11):
    """
    Calculate SSIM between two images.
    
    Args:
        img1, img2: Input images, shape (H, W, C)
        data_range: Maximum pixel value range
        win_size: Gaussian window size
        
    Returns:
        SSIM value in range [-1, 1], typically [0, 1] for similar images
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape, got {img1.shape} and {img2.shape}")
    
    # Constants from paper
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    # Compute mean
    mu1 = _gaussian_blur(img1, win_size)
    mu2 = _gaussian_blur(img2, win_size)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    # Compute variance and covariance
    sigma1_sq = _gaussian_blur(img1 * img1, win_size) - mu1_sq
    sigma2_sq = _gaussian_blur(img2 * img2, win_size) - mu2_sq
    sigma12 = _gaussian_blur(img1 * img2, win_size) - mu1_mu2
    
    # SSIM formula
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    
    return np.mean(ssim_map)


def _gaussian_blur(img, win_size=11, sigma=1.5):
    """Apply Gaussian blur to image."""
    from scipy import ndimage
    return ndimage.gaussian_filter(img, sigma=sigma)



    print("Testing evaluation metrics...")
    
    # Create dummy data
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)
    
    # Test individual metrics
    psnr = calculate_psnr(pred, target)
    print(f"PSNR: {psnr:.2f} dB")
    
    ssim = calculate_ssim(pred, target)
    print(f"SSIM: {ssim:.4f}")
    
    # Test evaluator
    print("\nTesting Evaluator...")
    evaluator = Evaluator(compute_lpips=False)
    evaluator.update(pred, target)
    evaluator.print_results()
