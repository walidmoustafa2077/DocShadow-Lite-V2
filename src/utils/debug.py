"""
Debug & Research Verification Utilities for LP-IOANet
=====================================================
Implements Claude's Research Execution Plan for Stage 1 verification:
- Attention mask visualization
- Error map analysis
- Loss equilibrium monitoring
- Stage transition criteria checking
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def print_claude_plan():
    """Print the research execution plan for Stage 1 training."""
    plan = """
================================================================================
CLAUDE'S RESEARCH EXECUTION PLAN: STAGE 1 (IOANET BASELINE)
================================================================================

1. THE ATTENTION HYPOTHESIS VERIFICATION:
   - Monitor the Input/Output Attention masks.
   - EXPECTATION: Input mask should isolate shadow boundaries. Output mask 
     should be ~0 in non-shadow areas (relying on long residual).
   - FAILURE MODE: If masks are uniform (all 1s), the IOA modules are dead.

2. LOSS EQUILIBRIUM MONITORING:
   - Current Ratio: L1 (10.0) vs LPIPS (5.0).
   - RISK: High L1 weight might over-smooth text. LPIPS is the primary driver
     for OCR-ready sharpness. We monitor the gradient magnitude of each.

3. DATA MIXTURE VALIDATION:
   - BSDD is synthetic. Monitor if the model generalizes to the few A-OSR 
     samples in the val set. Artifacts at shadow edges are the primary risk.

4. STAGE 2 PREP:
   - Freeze IOANet only when MAE < 0.05 and Attention Masks show clear
     differentiation between paper and shadow.

5. KEY METRICS TO WATCH:
   - Attention Mask Statistics: mean, std, min, max
   - Shadow/Non-Shadow Activation Ratio
   - Gradient flow through attention modules
   - Error concentration at shadow boundaries

================================================================================
    """
    print(plan)


class AttentionAnalyzer:
    """
    Analyzes attention mask behavior to verify the IOA hypothesis.
    
    The attention masks are the "soul" of this paper—if they aren't learning
    meaningful shadow regions, the model is just acting as a standard CNN.
    """
    
    def __init__(self):
        self.history = {
            'input_mask_mean': [],
            'input_mask_std': [],
            'output_mask_mean': [],
            'output_mask_std': [],
            'mask_differentiation': [],  # How different are shadow vs non-shadow regions
        }
    
    def analyze(
        self, 
        input_mask: torch.Tensor, 
        output_mask: torch.Tensor,
        shadow_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze attention masks for current batch.
        
        Args:
            input_mask: Input attention mask (B, 1, H, W)
            output_mask: Output attention mask (B, 1, H, W)
            shadow_mask: Ground truth shadow mask if available (B, 1, H, W)
            
        Returns:
            Dictionary of analysis metrics
        """
        stats = {}
        
        # Basic statistics
        stats['in_mask_mean'] = input_mask.mean().item()
        stats['in_mask_std'] = input_mask.std().item()
        stats['in_mask_min'] = input_mask.min().item()
        stats['in_mask_max'] = input_mask.max().item()
        
        stats['out_mask_mean'] = output_mask.mean().item()
        stats['out_mask_std'] = output_mask.std().item()
        stats['out_mask_min'] = output_mask.min().item()
        stats['out_mask_max'] = output_mask.max().item()
        
        # Check for dead attention (failure mode)
        stats['in_mask_is_dead'] = stats['in_mask_std'] < 0.01
        stats['out_mask_is_dead'] = stats['out_mask_std'] < 0.01
        
        # Sparsity (how much the mask relies on residual)
        stats['in_mask_sparsity'] = (input_mask < 0.5).float().mean().item()
        stats['out_mask_sparsity'] = (output_mask < 0.5).float().mean().item()
        
        # If ground truth shadow mask available, compute correlation
        if shadow_mask is not None:
            shadow_mask = shadow_mask.float()
            # Resize if needed
            if shadow_mask.shape != input_mask.shape:
                # Handle dimension mismatch - ensure shadow_mask is (B, C, H, W)
                if shadow_mask.dim() == 3:
                    shadow_mask = shadow_mask.unsqueeze(1)
                shadow_mask = F.interpolate(
                    shadow_mask, size=input_mask.shape[2:], mode='nearest'
                )
            
            # Ensure both tensors have the same batch size
            min_batch = min(input_mask.shape[0], shadow_mask.shape[0])
            input_mask_batch = input_mask[:min_batch]
            shadow_mask_batch = shadow_mask[:min_batch]
            
            # Correlation between predicted attention and actual shadow
            in_mask_flat = input_mask_batch.view(-1)
            shadow_flat = shadow_mask_batch.view(-1)
            
            # Only compute if we have enough samples
            if len(in_mask_flat) > 1 and len(shadow_flat) > 1:
                correlation = torch.corrcoef(
                    torch.stack([in_mask_flat, shadow_flat])
                )[0, 1].item()
                stats['shadow_correlation'] = correlation if not np.isnan(correlation) else 0.0
            
            # Shadow vs Non-Shadow activation ratio
            shadow_region = input_mask_batch[shadow_mask_batch > 0.5]
            non_shadow_region = input_mask_batch[shadow_mask_batch <= 0.5]
            
            if len(shadow_region) > 0 and len(non_shadow_region) > 0:
                stats['shadow_activation'] = shadow_region.mean().item()
                stats['non_shadow_activation'] = non_shadow_region.mean().item()
                stats['activation_ratio'] = stats['shadow_activation'] / (stats['non_shadow_activation'] + 1e-8)
        
        # Update history
        self.history['input_mask_mean'].append(stats['in_mask_mean'])
        self.history['input_mask_std'].append(stats['in_mask_std'])
        self.history['output_mask_mean'].append(stats['out_mask_mean'])
        self.history['output_mask_std'].append(stats['out_mask_std'])
        
        return stats
    
    def check_health(self) -> Tuple[bool, str]:
        """
        Check if attention masks are healthy (not dead/uniform).
        
        Returns:
            (is_healthy, message)
        """
        if len(self.history['input_mask_std']) < 5:
            return True, "Not enough data yet"
        
        recent_in_std = np.mean(self.history['input_mask_std'][-10:])
        recent_out_std = np.mean(self.history['output_mask_std'][-10:])
        
        if recent_in_std < 0.02:
            return False, f"⚠️  INPUT ATTENTION IS DYING! std={recent_in_std:.4f}"
        
        if recent_out_std < 0.02:
            return False, f"⚠️  OUTPUT ATTENTION IS DYING! std={recent_out_std:.4f}"
        
        return True, "✅ Attention masks are healthy"


def denormalize_for_vis(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """Denormalize tensor for visualization."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    
    return (tensor * std + mean).clamp(0, 1)


def save_debug_samples(
    epoch: int,
    model: nn.Module,
    val_loader,
    device: torch.device,
    save_dir: str = "debug_samples",
    num_samples: int = 4
):
    """
    Captures the physical state of the model's reasoning.
    
    Visualizes: Input | GT | Pred | Input_Mask | Output_Mask | Error_Map
    
    This is the key debugging tool for verifying the IOA hypothesis.
    
    Args:
        epoch: Current epoch number
        model: The IOANet or LPIOANet model
        val_loader: Validation dataloader
        device: Device to run on
        save_dir: Directory to save debug images
        num_samples: Number of samples to visualize
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Get one batch from validation
        batch = next(iter(val_loader))
        
        # Handle both Stage 1 and Stage 2 data formats
        if 'input_high' in batch:
            inputs = batch['input_high'].to(device)
            targets = batch['target_high'].to(device)
        else:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
        
        # Get shadow mask if available
        shadow_mask = batch.get('mask', None)
        if shadow_mask is not None:
            shadow_mask = shadow_mask.to(device)
        
        # Forward pass with attention masks
        # Check if model supports returning attention masks
        if hasattr(model, 'ioanet'):
            # LPIOANet - get attention from inner IOANet
            # Need to run IOANet separately to get masks
            from src.models import LPIOANet
            x_low = F.interpolate(inputs, scale_factor=0.25, mode='bilinear', align_corners=False)
            outputs_low, in_mask, out_mask = model.ioanet(x_low, return_attention=True)
            outputs = model(inputs)
            # Upsample masks for visualization
            in_mask = F.interpolate(in_mask, size=inputs.shape[2:], mode='bilinear', align_corners=False)
            out_mask = F.interpolate(out_mask, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        else:
            # Direct IOANet
            outputs, in_mask, out_mask = model(inputs, return_attention=True)
        
        # Calculate Error Map (where the model is failing)
        error_map = torch.abs(outputs - targets).mean(dim=1, keepdim=True)
        # Normalize error map to [0, 1] for visualization
        error_map = error_map / (error_map.max() + 1e-8)
        
        # Denormalize images for visualization
        inputs_vis = denormalize_for_vis(inputs)
        targets_vis = denormalize_for_vis(targets)
        outputs_vis = denormalize_for_vis(outputs)
        
        # Limit number of samples
        num_samples = min(num_samples, inputs.size(0))
        
        # Create grid components
        # Each row: Input | GT | Pred | In_Mask | Out_Mask | Error_Map
        grid_rows = []
        
        for i in range(num_samples):
            row = torch.cat([
                inputs_vis[i],                           # Input (3, H, W)
                targets_vis[i],                          # GT (3, H, W)
                outputs_vis[i],                          # Pred (3, H, W)
                in_mask[i].repeat(3, 1, 1),             # In_Mask as RGB (3, H, W)
                out_mask[i].repeat(3, 1, 1),            # Out_Mask as RGB (3, H, W)
                error_map[i].repeat(3, 1, 1),           # Error_Map as RGB (3, H, W)
            ], dim=2)  # Concatenate along width
            grid_rows.append(row)
        
        # Stack all rows
        grid = torch.cat(grid_rows, dim=1)  # Concatenate along height
        
        # Save the grid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"epoch_{epoch:04d}_{timestamp}.png"
        vutils.save_image(grid, save_path, normalize=False)
        
        # Also save individual attention mask statistics
        analyzer = AttentionAnalyzer()
        shadow_mask_sliced = shadow_mask[:num_samples] if shadow_mask is not None else None
        stats = analyzer.analyze(in_mask[:num_samples], out_mask[:num_samples], shadow_mask_sliced)
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"[DEBUG] Epoch {epoch} - Attention Mask Analysis")
        print(f"{'='*60}")
        print(f"Input Mask:  mean={stats['in_mask_mean']:.4f}, std={stats['in_mask_std']:.4f}, "
              f"range=[{stats['in_mask_min']:.4f}, {stats['in_mask_max']:.4f}]")
        print(f"Output Mask: mean={stats['out_mask_mean']:.4f}, std={stats['out_mask_std']:.4f}, "
              f"range=[{stats['out_mask_min']:.4f}, {stats['out_mask_max']:.4f}]")
        print(f"Input Mask Sparsity:  {stats['in_mask_sparsity']*100:.1f}% pixels < 0.5")
        print(f"Output Mask Sparsity: {stats['out_mask_sparsity']*100:.1f}% pixels < 0.5")
        
        if stats.get('in_mask_is_dead') or stats.get('out_mask_is_dead'):
            print(f"⚠️  WARNING: Attention masks may be dying (low variance)!")
        
        if 'shadow_correlation' in stats:
            print(f"Shadow Correlation: {stats['shadow_correlation']:.4f}")
            print(f"Shadow/Non-Shadow Activation Ratio: {stats.get('activation_ratio', 'N/A'):.2f}")
        
        print(f"{'='*60}")
        print(f"[DEBUG] Samples saved: {save_path}")
        print(f"[DEBUG] Grid layout: Input | GT | Pred | In_Mask | Out_Mask | Error_Map")
        print(f"{'='*60}\n")
        
        return stats


class GradientMonitor:
    """
    Monitors gradient flow through different parts of the model.
    
    Critical for detecting:
    - Dead attention modules (zero gradients)
    - Loss imbalance (L1 vs LPIPS gradient magnitudes)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = {
            'encoder': [],
            'decoder': [],
            'input_attn': [],
            'output_attn': [],
        }
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register backward hooks on key modules."""
        
        def make_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_norm = grad_output[0].norm().item()
                    self.gradient_history[name].append(grad_norm)
            return hook
        
        # Find and hook attention modules
        for name, module in self.model.named_modules():
            if 'input_attn' in name:
                module.register_full_backward_hook(make_hook('input_attn'))
            elif 'output_attn' in name:
                module.register_full_backward_hook(make_hook('output_attn'))
            elif 'enc1' in name:
                module.register_full_backward_hook(make_hook('encoder'))
            elif 'decoder' in name or 'up1' in name:
                module.register_full_backward_hook(make_hook('decoder'))
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get recent gradient statistics."""
        stats = {}
        for name, history in self.gradient_history.items():
            if len(history) > 0:
                recent = history[-100:]  # Last 100 updates
                stats[f'{name}_grad_mean'] = np.mean(recent)
                stats[f'{name}_grad_std'] = np.std(recent)
        return stats
    
    def check_gradient_health(self) -> Tuple[bool, str]:
        """Check if gradients are flowing properly."""
        messages = []
        is_healthy = True
        
        for name, history in self.gradient_history.items():
            if len(history) < 10:
                continue
            
            recent_mean = np.mean(history[-50:])
            
            if recent_mean < 1e-7:
                is_healthy = False
                messages.append(f"⚠️  {name}: Vanishing gradients ({recent_mean:.2e})")
            elif recent_mean > 1e3:
                is_healthy = False
                messages.append(f"⚠️  {name}: Exploding gradients ({recent_mean:.2e})")
        
        if is_healthy:
            messages.append("✅ Gradient flow is healthy")
        
        return is_healthy, "\n".join(messages)


class Stage1ReadinessChecker:
    """
    Checks if Stage 1 training is ready to transition to Stage 2.
    
    Criteria:
    1. MAE < 0.05
    2. Attention masks show clear shadow differentiation
    3. Stable training (low loss variance)
    """
    
    def __init__(self, mae_threshold: float = 0.05, min_epochs: int = 100):
        self.mae_threshold = mae_threshold
        self.min_epochs = min_epochs
        self.mae_history = []
        self.attention_analyzer = AttentionAnalyzer()
    
    def update(
        self, 
        epoch: int, 
        mae: float, 
        input_mask: torch.Tensor,
        output_mask: torch.Tensor
    ):
        """Update with current epoch metrics."""
        self.mae_history.append(mae)
        self.attention_analyzer.analyze(input_mask, output_mask)
    
    def is_ready(self, current_epoch: int) -> Tuple[bool, str]:
        """
        Check if ready to transition to Stage 2.
        
        Returns:
            (is_ready, reason)
        """
        reasons = []
        
        # Check minimum epochs
        if current_epoch < self.min_epochs:
            return False, f"Need at least {self.min_epochs} epochs (current: {current_epoch})"
        
        # Check MAE
        if len(self.mae_history) < 10:
            return False, "Not enough history"
        
        recent_mae = np.mean(self.mae_history[-10:])
        
        if recent_mae > self.mae_threshold:
            reasons.append(f"MAE too high: {recent_mae:.4f} > {self.mae_threshold}")
        else:
            reasons.append(f"✅ MAE: {recent_mae:.4f}")
        
        # Check attention health
        attn_healthy, attn_msg = self.attention_analyzer.check_health()
        reasons.append(attn_msg)
        
        # Check training stability
        if len(self.mae_history) >= 20:
            mae_std = np.std(self.mae_history[-20:])
            if mae_std > 0.01:
                reasons.append(f"Training unstable: MAE std = {mae_std:.4f}")
            else:
                reasons.append(f"✅ Training stable: MAE std = {mae_std:.4f}")
        
        is_ready = (
            recent_mae <= self.mae_threshold and 
            attn_healthy and
            current_epoch >= self.min_epochs
        )
        
        status = "🚀 READY FOR STAGE 2!" if is_ready else "⏳ Not ready yet"
        
        return is_ready, f"{status}\n" + "\n".join(reasons)


def create_loss_breakdown_visualization(
    loss_history: Dict[str, List[float]],
    save_path: str
):
    """
    Create visualization of loss component breakdown over training.
    
    Shows the balance between L1 and LPIPS contributions.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    if 'total' in loss_history:
        axes[0, 0].plot(loss_history['total'], label='Total Loss', color='blue')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # L1 vs LPIPS
    if 'l1' in loss_history and 'lpips' in loss_history:
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        
        l1_line = ax1.plot(loss_history['l1'], label='L1', color='green', alpha=0.7)
        lpips_line = ax2.plot(loss_history['lpips'], label='LPIPS', color='red', alpha=0.7)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('L1 Loss', color='green')
        ax2.set_ylabel('LPIPS Loss', color='red')
        ax1.set_title('L1 vs LPIPS Balance')
        
        lines = l1_line + lpips_line
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.grid(True, alpha=0.3)
    
    # Loss ratio (weighted contributions)
    if 'l1' in loss_history and 'lpips' in loss_history:
        l1_contrib = np.array(loss_history['l1']) * 10.0  # w_l1
        lpips_contrib = np.array(loss_history['lpips']) * 5.0  # w_lpips
        
        total_contrib = l1_contrib + lpips_contrib
        l1_ratio = l1_contrib / (total_contrib + 1e-8)
        
        axes[1, 0].plot(l1_ratio, label='L1 Contribution Ratio', color='purple')
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', label='50% threshold')
        axes[1, 0].set_title('Loss Contribution Ratio')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('L1 / (L1 + LPIPS)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Smoothed total loss
    if 'total' in loss_history and len(loss_history['total']) > 100:
        window = 100
        smoothed = np.convolve(
            loss_history['total'], 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[1, 1].plot(smoothed, label=f'Smoothed (window={window})', color='blue')
        axes[1, 1].set_title('Smoothed Loss Trend')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[DEBUG] Loss breakdown saved: {save_path}")


# Test the utilities
if __name__ == "__main__":
    print_claude_plan()
    
    print("\nTesting AttentionAnalyzer...")
    analyzer = AttentionAnalyzer()
    
    # Create dummy masks
    in_mask = torch.rand(2, 1, 64, 64) * 0.8 + 0.1  # Range [0.1, 0.9]
    out_mask = torch.rand(2, 1, 64, 64) * 0.5  # Range [0, 0.5] - sparser
    
    stats = analyzer.analyze(in_mask, out_mask)
    print(f"Stats: {stats}")
    
    healthy, msg = analyzer.check_health()
    print(f"Health check: {msg}")
    
    print("\nTesting Stage1ReadinessChecker...")
    checker = Stage1ReadinessChecker(mae_threshold=0.05, min_epochs=10)
    
    for epoch in range(15):
        mae = 0.1 - epoch * 0.005  # Decreasing MAE
        checker.update(epoch, mae, in_mask, out_mask)
    
    ready, reason = checker.is_ready(14)
    print(f"Ready for Stage 2? {ready}")
    print(reason)
