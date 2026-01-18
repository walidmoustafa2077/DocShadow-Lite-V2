"""
LP-IOANet Model Components
==========================
Implements the core modules for document shadow removal:
- IOAttention: Input/Output attention mechanism
- IOANet: Stage 1 low-resolution shadow removal network
- LPTN_Lite_Refiner: Lightweight refinement module
- LPIOANet: Full Laplacian Pyramid network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional


class IOAttention(nn.Module):
    """
    Implements Input/Output Attention mechanism.
    
    Input Attention: Focuses the encoder on shadowed areas by generating
                     spatial attention weights.
    Output Attention: Blends the refined output with the original input
                      using learned alpha masks.
    
    Architecture:
        Conv2d(in_channels -> 64) -> ReLU -> Conv2d(64 -> 32) -> ReLU ->
        Conv2d(32 -> 1) -> Sigmoid
    
    Args:
        in_channels (int): Number of input channels (typically 3 for RGB)
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate attention mask.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention mask of shape (B, 1, H, W) with values in [0, 1]
        """
        return self.conv(x)


class FeatureBlendingDecoder(nn.Module):
    """
    Feature Blending Decoder (FB-Decoder) for IOANet.
    
    Implements progressive upsampling with skip connections from encoder.
    Uses transposed convolutions for learnable upsampling.
    
    Architecture follows asymmetric design: lightweight compared to encoder.
    """
    def __init__(self):
        super().__init__()
        
        # Channel dimensions from MobileNetV2 encoder stages
        # bottleneck: 1280, enc4: 96, enc3: 32, enc2: 24, enc1: 16
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(1280, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.blend4 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.blend3 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.blend2 = nn.Sequential(
            nn.Conv2d(24 + 24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.blend1 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final output projection (no Tanh - keep output in same range as input)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(
        self, 
        bottleneck: torch.Tensor,
        e1: torch.Tensor, 
        e2: torch.Tensor, 
        e3: torch.Tensor, 
        e4: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode features with skip connections.
        
        Args:
            bottleneck: Bottleneck features (B, 1280, H/32, W/32)
            e1-e4: Encoder features at different scales
            
        Returns:
            Refined image tensor (B, 3, H, W)
        """
        # Upsample and blend with skip connections
        d4 = self.up4(bottleneck)
        d4 = self._match_and_concat(d4, e4)
        d4 = self.blend4(d4)
        
        d3 = self.up3(d4)
        d3 = self._match_and_concat(d3, e3)
        d3 = self.blend3(d3)
        
        d2 = self.up2(d3)
        d2 = self._match_and_concat(d2, e2)
        d2 = self.blend2(d2)
        
        d1 = self.up1(d2)
        d1 = self._match_and_concat(d1, e1)
        d1 = self.blend1(d1)
        
        return self.final(d1)
    
    def _match_and_concat(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor
    ) -> torch.Tensor:
        """Match spatial dimensions and concatenate."""
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)


class IOANet(nn.Module):
    """
    Input-Output Attention Network (IOANet) - Stage 1.
    
    Core shadow removal network operating at low resolution (192x256).
    Uses MobileNetV2 as encoder backbone with custom FB-Decoder.
    
    Key Features:
    - Input Attention: Localizes shadow regions for focused processing
    - Pre-trained MobileNetV2 encoder for robust feature extraction
    - Feature Blending Decoder with skip connections
    - Output Attention: Alpha-blending for artifact-free results
    - Long residual connection to preserve non-shadow regions
    
    Expected Input: (B, 3, 192, 256) - Low resolution document image
    Expected Output: (B, 3, 192, 256) - Shadow-removed image
    
    Computational Cost: ~0.8 GFLOPs at 192x256
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load MobileNetV2 backbone
        mnet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        ).features
        
        # Encoder stages (channel dims: 16, 24, 32, 96, 1280)
        self.enc1 = mnet[:2]    # Output: 16 channels, stride 2
        self.enc2 = mnet[2:4]   # Output: 24 channels, stride 4
        self.enc3 = mnet[4:7]   # Output: 32 channels, stride 8
        self.enc4 = mnet[7:14]  # Output: 96 channels, stride 16
        self.bottleneck = mnet[14:]  # Output: 1280 channels, stride 32
        
        # Attention modules
        self.input_attn = IOAttention(3)
        self.output_attn = IOAttention(3)
        
        # Decoder
        self.decoder = FeatureBlendingDecoder()
        
        # Initialize new layers
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder and attention weights."""
        for module in [self.decoder, self.input_attn, self.output_attn]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with optional attention map return.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            return_attention: If True, return attention masks for visualization
            
        Returns:
            out: Shadow-removed image (B, 3, H, W)
            (optional) attn_in: Input attention mask
            (optional) attn_out: Output attention mask
        """
        # Store input for long residual connection
        input_res = x
        
        # Input Attention - weight encoder toward shadow regions
        attn_in = self.input_attn(x)
        # Let encoder ONLY see the shadow area (sharp attention)
        x_attended = x * attn_in
        
        # Encoder forward pass
        e1 = self.enc1(x_attended)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        refined = self.decoder(b, e1, e2, e3, e4)
        
        # Output Attention - blend with input via learned mask
        attn_out = self.output_attn(refined)
        out = attn_out * refined + (1 - attn_out) * input_res
        
        if return_attention:
            return out, attn_in, attn_out
        return out


class LPTN_Lite_Refiner(nn.Module):
    """
    LPTN-Lite Refinement Module.
    
    Lightweight module using depthwise separable convolutions for
    efficient high-resolution detail refinement.
    
    Architecture:
        Residual block with:
        - Depthwise Conv (spatial processing)
        - Pointwise Conv (channel mixing)
        - Activation
        - Depthwise Conv
        - Pointwise Conv
        
    The module learns to refine high-frequency residuals in the
    Laplacian pyramid reconstruction.
    
    Args:
        channels: Number of input/output channels (default: 3)
        hidden_channels: Hidden layer channels (default: 16)
        use_relu6: Use ReLU6 for quantization-aware training
    """
    def __init__(
        self, 
        channels: int = 3, 
        hidden_channels: int = 16,
        use_relu6: bool = False
    ):
        super().__init__()
        
        activation = nn.ReLU6(inplace=True) if use_relu6 else nn.LeakyReLU(0.2, inplace=True)
        
        self.refine = nn.Sequential(
            # First depthwise separable block
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            activation,
            
            # Second depthwise separable block
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            activation,
            
            # Third depthwise separable block  
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply refinement with residual connection.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Refined tensor with same shape
        """
        return x + self.refine(x)


class LaplacianPyramid(nn.Module):
    """
    Laplacian Pyramid Decomposition/Reconstruction.
    
    Decomposes image into low-frequency base and high-frequency residuals.
    Allows efficient processing at multiple scales.
    """
    def __init__(self, levels: int = 2):
        super().__init__()
        self.levels = levels
    
    def decompose(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """
        Decompose image into Laplacian pyramid.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            low_freq: Lowest frequency image
            residuals: List of high-frequency residuals (high to low res)
        """
        residuals = []
        current = x
        
        for _ in range(self.levels):
            # Downsample
            downsampled = F.interpolate(
                current, scale_factor=0.5, mode='bilinear', align_corners=False
            )
            # Upsample back for residual computation
            upsampled = F.interpolate(
                downsampled, size=current.shape[2:], mode='bilinear', align_corners=False
            )
            # High-frequency residual
            residual = current - upsampled
            residuals.append(residual)
            current = downsampled
        
        return current, residuals
    
    def reconstruct(
        self, 
        low_freq: torch.Tensor, 
        residuals: list
    ) -> torch.Tensor:
        """
        Reconstruct image from pyramid.
        
        Args:
            low_freq: Low frequency base image
            residuals: List of residuals (high to low res order)
            
        Returns:
            Reconstructed full-resolution image
        """
        current = low_freq
        
        # Reconstruct from low to high resolution
        for residual in reversed(residuals):
            upsampled = F.interpolate(
                current, size=residual.shape[2:], mode='bilinear', align_corners=False
            )
            current = upsampled + residual
        
        return current


class LPIOANet(nn.Module):
    """
    Laplacian Pyramid IO-Attention Network (LP-IOANet).
    
    Full high-resolution document shadow removal network.
    Combines IOANet for low-frequency shadow removal with
    LPTN-Lite refiners for high-frequency detail preservation.
    
    Architecture:
    1. Decompose input into Laplacian pyramid (2 levels)
    2. Process low-freq image (192x256) through IOANet
    3. Recursively upsample and refine with LPTN-Lite modules
    4. Output full-resolution (768x1024) shadow-free image
    
    Key Advantages:
    - Heavy computation only at low resolution
    - Efficient ~1.47 GFLOPs at 768x1024
    - Preserves text sharpness through residual refinement
    - Supports ~20 FPS on mobile hardware
    
    Args:
        ioanet: Pre-trained IOANet model (frozen during stage 2)
        hidden_channels: Hidden channels for refiners
        use_relu6: Use ReLU6 for mobile deployment
    """
    def __init__(
        self, 
        ioanet: Optional[IOANet] = None,
        hidden_channels: int = 16,
        use_relu6: bool = False
    ):
        super().__init__()
        
        # Stage 1: Low-res shadow removal
        self.ioanet = ioanet if ioanet is not None else IOANet()
        
        # Laplacian pyramid handler
        self.pyramid = LaplacianPyramid(levels=2)
        
        # Stage 2: High-res refinement modules
        # Refiner for 384x512 (mid resolution)
        self.refiner_mid = LPTN_Lite_Refiner(
            channels=3, 
            hidden_channels=hidden_channels,
            use_relu6=use_relu6
        )
        # Refiner for 768x1024 (full resolution)
        self.refiner_high = LPTN_Lite_Refiner(
            channels=3, 
            hidden_channels=hidden_channels,
            use_relu6=use_relu6
        )
    
    def freeze_ioanet(self):
        """Freeze IOANet weights for stage 2 training."""
        for param in self.ioanet.parameters():
            param.requires_grad = False
        self.ioanet.eval()
    
    def unfreeze_ioanet(self):
        """Unfreeze IOANet weights."""
        for param in self.ioanet.parameters():
            param.requires_grad = True
        self.ioanet.train()

    def forward(
        self, 
        x_high: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Full resolution shadow removal.
        
        Args:
            x_high: High-resolution input (B, 3, 768, 1024)
            return_intermediates: Return intermediate results for visualization
            
        Returns:
            y_high: Shadow-removed high-res output (B, 3, 768, 1024)
            (optional) intermediates: Dict of intermediate tensors
        """
        # 1. Laplacian Decomposition
        x_low, residuals = self.pyramid.decompose(x_high)
        res_high, res_mid = residuals  # High-freq residuals at each level
        
        # Get mid-resolution for skip connection computation
        x_mid = F.interpolate(x_high, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # 2. Shadow Removal on Low-Resolution (192x256)
        y_low = self.ioanet(x_low)
        
        # 3. Level 2 Refinement (384x512)
        y_mid_base = F.interpolate(y_low, size=x_mid.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute residual from original mid-res
        x_low_up = F.interpolate(x_low, size=x_mid.shape[2:], mode='bilinear', align_corners=False)
        res_mid_computed = x_mid - x_low_up
        
        # Refine: base + residual through refiner
        y_mid = self.refiner_mid(y_mid_base + res_mid_computed)
        
        # 4. Level 1 Refinement (768x1024)
        y_high_base = F.interpolate(y_mid, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute residual from original high-res
        x_mid_up = F.interpolate(x_mid, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        res_high_computed = x_high - x_mid_up
        
        # Refine: base + residual through refiner
        y_high = self.refiner_high(y_high_base + res_high_computed)
        
        if return_intermediates:
            intermediates = {
                'x_low': x_low,
                'x_mid': x_mid,
                'y_low': y_low,
                'y_mid': y_mid,
                'res_mid': res_mid_computed,
                'res_high': res_high_computed,
            }
            return y_high, intermediates
        
        return y_high


def build_model(
    pretrained_ioanet: Optional[str] = None,
    stage: int = 1,
    hidden_channels: int = 16,
    use_relu6: bool = False
) -> nn.Module:
    """
    Factory function to build LP-IOANet model.
    
    Args:
        pretrained_ioanet: Path to pretrained IOANet checkpoint
        stage: Training stage (1 for IOANet only, 2 for full LP-IOANet)
        hidden_channels: Hidden channels for LPTN-Lite refiners
        use_relu6: Use ReLU6 for mobile deployment
        
    Returns:
        Configured model ready for training
    """
    # Build IOANet
    ioanet = IOANet(pretrained=True)
    
    if pretrained_ioanet is not None:
        state_dict = torch.load(pretrained_ioanet, map_location='cpu')
        ioanet.load_state_dict(state_dict)
        print(f"Loaded IOANet weights from {pretrained_ioanet}")
    
    if stage == 1:
        return ioanet
    
    # Build full LP-IOANet for stage 2
    model = LPIOANet(
        ioanet=ioanet,
        hidden_channels=hidden_channels,
        use_relu6=use_relu6
    )
    model.freeze_ioanet()
    
    return model


# Quick test
if __name__ == "__main__":
    # Test IOANet
    print("Testing IOANet...")
    ioanet = IOANet(pretrained=False)
    x_low = torch.randn(1, 3, 192, 256)
    y_low = ioanet(x_low)
    print(f"IOANet: {x_low.shape} -> {y_low.shape}")
    
    # Test LP-IOANet
    print("\nTesting LP-IOANet...")
    model = LPIOANet(ioanet)
    x_high = torch.randn(1, 3, 768, 1024)
    y_high = model(x_high)
    print(f"LP-IOANet: {x_high.shape} -> {y_high.shape}")
    
    # Count parameters
    ioanet_params = sum(p.numel() for p in ioanet.parameters())
    refiner_params = sum(p.numel() for p in model.refiner_mid.parameters()) + \
                     sum(p.numel() for p in model.refiner_high.parameters())
    
    print(f"\nIOANet parameters: {ioanet_params:,}")
    print(f"Refiner parameters: {refiner_params:,}")
    print(f"Total parameters: {ioanet_params + refiner_params:,}")
