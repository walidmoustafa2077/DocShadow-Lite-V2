"""
Loss Functions for Document Shadow Removal
==========================================
Implements the loss functions specified in the LP-IOANet paper:
- L1 Loss (pixel-wise reconstruction)
- LPIPS Loss (perceptual quality)
- Combined Shadow Loss with configurable weights
- Additional losses for improved training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss as an alternative to LPIPS.
    
    Uses pre-trained VGG16 features to compute perceptual similarity.
    More lightweight than LPIPS while still effective.
    
    Args:
        layers: VGG layers to extract features from
        normalize_input: Whether to normalize input to VGG range
    """
    
    def __init__(
        self,
        layers: Tuple[int, ...] = (3, 8, 15, 22),  # relu1_2, relu2_2, relu3_3, relu4_3
        normalize_input: bool = True
    ):
        super().__init__()
        
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.slices = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_layer:layer+1]))
            prev_layer = layer + 1
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
        
        self.normalize_input = normalize_input
        
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Feature weights (importance of each layer)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            
        Returns:
            Perceptual loss scalar
        """
        # Normalize if needed
        if self.normalize_input:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for i, slice_module in enumerate(self.slices):
            x_pred = slice_module(x_pred)
            x_target = slice_module(x_target)
            loss += self.weights[i] * F.l1_loss(x_pred, x_target)
        
        return loss


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Loss.
    
    Wrapper that handles LPIPS installation gracefully.
    Falls back to VGG perceptual loss if LPIPS is not available.
    
    Args:
        net: Network to use ('vgg', 'alex', 'squeeze')
    """
    
    def __init__(self, net: str = 'vgg'):
        super().__init__()
        
        self.use_lpips = False
        
        try:
            from lpips import LPIPS
            self.lpips = LPIPS(net=net)
            # Freeze LPIPS weights
            for param in self.lpips.parameters():
                param.requires_grad = False
            self.use_lpips = True
            print(f"Using LPIPS loss with {net} backbone")
        except ImportError:
            print("LPIPS not installed. Using VGG perceptual loss instead.")
            print("Install LPIPS with: pip install lpips")
            self.vgg_loss = VGGPerceptualLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image (B, 3, H, W) - values in [-1, 1] or [0, 1]
            target: Target image (B, 3, H, W)
            
        Returns:
            Perceptual loss scalar
        """
        if self.use_lpips:
            # LPIPS expects values in [-1, 1]
            if pred.min() >= 0:
                pred = pred * 2 - 1
                target = target * 2 - 1
            return self.lpips(pred, target).mean()
        else:
            return self.vgg_loss(pred, target)


class SobelEdgeLoss(nn.Module):
    """
    Sobel-filtered L1 loss for edge preservation.
    
    Applies Sobel filters to detect edges and computes loss
    specifically on edge regions. Useful for preserving text strokes.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _apply_sobel(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sobel filter to compute edge magnitude."""
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Apply Sobel filters
        edges_x = F.conv2d(gray, self.sobel_x, padding=1)
        edges_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Compute magnitude
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        
        return edges
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute edge-aware loss.
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            
        Returns:
            Edge loss scalar
        """
        pred_edges = self._apply_sobel(pred)
        target_edges = self._apply_sobel(target)
        
        return F.l1_loss(pred_edges, target_edges)


class AttentionSparsityLoss(nn.Module):
    """
    Sparsity loss for attention masks.
    
    Encourages the model to copy more pixels from input by
    making the attention masks sparse (close to 0 or 1).
    """
    
    def __init__(self, target_sparsity: float = 0.7):
        super().__init__()
        self.target_sparsity = target_sparsity
    
    def forward(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss.
        
        Args:
            attention_mask: Attention mask (B, 1, H, W) with values in [0, 1]
            
        Returns:
            Sparsity loss scalar
        """
        # Encourage values close to 0 or 1
        # Binary entropy: -p*log(p) - (1-p)*log(1-p) is minimized at 0 and 1
        eps = 1e-7
        entropy = -(attention_mask * torch.log(attention_mask + eps) + 
                   (1 - attention_mask) * torch.log(1 - attention_mask + eps))
        
        # Also encourage overall sparsity (more 0s than 1s)
        mean_activation = attention_mask.mean()
        sparsity_penalty = F.relu(mean_activation - self.target_sparsity)
        
        return entropy.mean() + sparsity_penalty


class ShadowLoss(nn.Module):
    """
    Combined loss function for shadow removal.
    
    Implements the paper's loss function: weighted sum of L1 and LPIPS.
    Also supports additional losses for improved training.
    
    Stage 1 (IOANet): L1 + LPIPS
    Stage 2 (LP-IOANet): L1 only (per paper section 4.1)
    
    Args:
        w_l1: Weight for L1 loss
        w_lpips: Weight for LPIPS loss
        w_edge: Weight for edge loss (optional)
        w_attention: Weight for attention sparsity loss (optional)
        stage: Training stage (1 or 2)
    """
    
    def __init__(
        self,
        w_l1: float = 10.0,
        w_lpips: float = 5.0,
        w_edge: float = 0.0,
        w_attention: float = 0.0,
        stage: int = 1
    ):
        super().__init__()
        
        self.w_l1 = w_l1
        self.w_lpips = w_lpips if stage == 1 else 0.0  # No LPIPS in stage 2
        self.w_edge = w_edge
        self.w_attention = w_attention
        
        # Core losses
        self.l1_loss = nn.L1Loss()
        
        if self.w_lpips > 0:
            self.perceptual_loss = LPIPSLoss(net='vgg')
        else:
            self.perceptual_loss = None
        
        # Optional losses
        if self.w_edge > 0:
            self.edge_loss = SobelEdgeLoss()
        else:
            self.edge_loss = None
            
        if self.w_attention > 0:
            self.attention_loss = AttentionSparsityLoss()
        else:
            self.attention_loss = None
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        attention_in: Optional[torch.Tensor] = None,
        attention_out: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            attention_in: Input attention mask (optional)
            attention_out: Output attention mask (optional)
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual loss values
        """
        loss_dict = {}
        total_loss = 0.0
        
        # L1 Loss
        loss_l1 = self.l1_loss(pred, target)
        loss_dict['l1'] = loss_l1.item()
        total_loss += self.w_l1 * loss_l1
        
        # Perceptual Loss
        if self.perceptual_loss is not None and self.w_lpips > 0:
            loss_p = self.perceptual_loss(pred, target)
            loss_dict['lpips'] = loss_p.item()
            total_loss += self.w_lpips * loss_p
        
        # Edge Loss
        if self.edge_loss is not None and self.w_edge > 0:
            loss_e = self.edge_loss(pred, target)
            loss_dict['edge'] = loss_e.item()
            total_loss += self.w_edge * loss_e
        
        # Attention Sparsity Loss
        if self.attention_loss is not None and self.w_attention > 0:
            if attention_in is not None:
                loss_attn = self.attention_loss(attention_in)
                if attention_out is not None:
                    loss_attn += self.attention_loss(attention_out)
                    loss_attn /= 2
                loss_dict['attention'] = loss_attn.item()
                total_loss += self.w_attention * loss_attn
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class MultiScaleShadowLoss(ShadowLoss):
    """
    Multi-scale loss for LP-IOANet Stage 2 training.
    
    Computes loss at multiple resolutions to ensure quality
    at all Laplacian pyramid levels.
    """
    
    def __init__(
        self,
        w_l1: float = 1.0,
        w_lpips: float = 0.0,  # Stage 2: L1 only
        w_low: float = 1.0,
        w_mid: float = 1.0,
        w_high: float = 1.0
    ):
        super().__init__(w_l1=w_l1, w_lpips=w_lpips, stage=2)
        
        self.w_low = w_low
        self.w_mid = w_mid
        self.w_high = w_high
    
    def forward(
        self,
        pred_high: torch.Tensor,
        target_high: torch.Tensor,
        pred_mid: Optional[torch.Tensor] = None,
        target_mid: Optional[torch.Tensor] = None,
        pred_low: Optional[torch.Tensor] = None,
        target_low: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-scale loss.
        
        Args:
            pred_high, target_high: High resolution images
            pred_mid, target_mid: Mid resolution images (optional)
            pred_low, target_low: Low resolution images (optional)
            
        Returns:
            total_loss: Combined multi-scale loss
            loss_dict: Individual loss values
        """
        loss_dict = {}
        total_loss = 0.0
        
        # High resolution loss
        loss_high = self.l1_loss(pred_high, target_high)
        loss_dict['l1_high'] = loss_high.item()
        total_loss += self.w_high * loss_high
        
        # Mid resolution loss
        if pred_mid is not None and target_mid is not None:
            loss_mid = self.l1_loss(pred_mid, target_mid)
            loss_dict['l1_mid'] = loss_mid.item()
            total_loss += self.w_mid * loss_mid
        
        # Low resolution loss
        if pred_low is not None and target_low is not None:
            loss_low = self.l1_loss(pred_low, target_low)
            loss_dict['l1_low'] = loss_low.item()
            total_loss += self.w_low * loss_low
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Test the losses
if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create dummy inputs
    pred = torch.randn(2, 3, 192, 256)
    target = torch.randn(2, 3, 192, 256)
    
    # Test ShadowLoss (Stage 1)
    print("\nStage 1 Loss (with LPIPS):")
    loss_fn = ShadowLoss(w_l1=10.0, w_lpips=5.0, stage=1)
    loss, loss_dict = loss_fn(pred, target)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test Stage 2 loss (L1 only)
    print("\nStage 2 Loss (L1 only):")
    loss_fn_s2 = ShadowLoss(w_l1=1.0, w_lpips=0.0, stage=2)
    loss, loss_dict = loss_fn_s2(pred, target)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test edge loss
    print("\nEdge Loss:")
    edge_loss = SobelEdgeLoss()
    loss_e = edge_loss(pred, target)
    print(f"Edge loss: {loss_e.item():.4f}")
    
    # Test VGG perceptual loss
    print("\nVGG Perceptual Loss:")
    vgg_loss = VGGPerceptualLoss()
    loss_v = vgg_loss(pred, target)
    print(f"VGG perceptual loss: {loss_v.item():.4f}")
