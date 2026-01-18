"""
Visualization Utilities for LP-IOANet
====================================
Tools for visualizing:
- Input/output comparisons
- Attention maps
- Laplacian pyramid decomposition
- Training progress
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torchvision.transforms.functional as TF


def denormalize(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """Denormalize tensor for visualization."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    
    denorm = tensor * std + mean
    return denorm.clamp(0, 1).squeeze(0) if tensor.size(0) == 1 else denorm.clamp(0, 1)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    array = tensor.cpu().detach().numpy()
    
    if array.shape[0] in [1, 3]:  # CHW format
        array = np.transpose(array, (1, 2, 0))
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    return array


def create_comparison_grid(
    input_img: torch.Tensor,
    output_img: torch.Tensor,
    target_img: Optional[torch.Tensor] = None,
    attention_in: Optional[torch.Tensor] = None,
    attention_out: Optional[torch.Tensor] = None,
    title: str = "Shadow Removal Results"
) -> 'plt.Figure':
    """
    Create a comparison grid visualization.
    
    Args:
        input_img: Input shadowed image
        output_img: Model output
        target_img: Ground truth (optional)
        attention_in: Input attention map (optional)
        attention_out: Output attention map (optional)
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Determine grid layout
    n_cols = 2
    if target_img is not None:
        n_cols = 3
    if attention_in is not None:
        n_cols += 2
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(title, fontsize=14)
    
    idx = 0
    
    # Input
    input_np = tensor_to_numpy(denormalize(input_img))
    axes[idx].imshow(input_np)
    axes[idx].set_title("Input (Shadowed)")
    axes[idx].axis('off')
    idx += 1
    
    # Output
    output_np = tensor_to_numpy(denormalize(output_img))
    axes[idx].imshow(output_np)
    axes[idx].set_title("Output")
    axes[idx].axis('off')
    idx += 1
    
    # Target
    if target_img is not None:
        target_np = tensor_to_numpy(denormalize(target_img))
        axes[idx].imshow(target_np)
        axes[idx].set_title("Ground Truth")
        axes[idx].axis('off')
        idx += 1
    
    # Attention maps
    if attention_in is not None:
        attn_in_np = tensor_to_numpy(attention_in)
        axes[idx].imshow(attn_in_np, cmap='hot')
        axes[idx].set_title("Input Attention")
        axes[idx].axis('off')
        idx += 1
    
    if attention_out is not None:
        attn_out_np = tensor_to_numpy(attention_out)
        axes[idx].imshow(attn_out_np, cmap='hot')
        axes[idx].set_title("Output Attention")
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_laplacian_pyramid(
    image: torch.Tensor,
    low_freq: torch.Tensor,
    residuals: List[torch.Tensor],
    title: str = "Laplacian Pyramid Decomposition"
) -> 'plt.Figure':
    """
    Visualize Laplacian pyramid decomposition.
    
    Args:
        image: Original image
        low_freq: Low frequency component
        residuals: List of high-frequency residuals
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    n_levels = len(residuals) + 2  # Original + low_freq + residuals
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4))
    fig.suptitle(title, fontsize=14)
    
    # Original
    img_np = tensor_to_numpy(denormalize(image))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Residuals (high to low frequency)
    for i, residual in enumerate(residuals):
        # Normalize residual for visualization
        res_np = tensor_to_numpy(residual)
        res_np = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-8)
        axes[i + 1].imshow(res_np)
        axes[i + 1].set_title(f"Residual Level {i + 1}")
        axes[i + 1].axis('off')
    
    # Low frequency
    low_np = tensor_to_numpy(denormalize(low_freq))
    axes[-1].imshow(low_np)
    axes[-1].set_title("Low Frequency")
    axes[-1].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_training_progress(
    loss_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> 'plt.Figure':
    """
    Visualize training loss curves.
    
    Args:
        loss_history: Dictionary of loss names to value lists
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, values in loss_history.items():
        ax.plot(values, label=name)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_comparison_image(
    input_img: torch.Tensor,
    output_img: torch.Tensor,
    target_img: Optional[torch.Tensor],
    save_path: str,
    denorm: bool = True
):
    """
    Save side-by-side comparison image.
    
    Args:
        input_img: Input image tensor
        output_img: Output image tensor
        target_img: Target image tensor (optional)
        save_path: Output file path
        denorm: Whether to denormalize tensors
    """
    if denorm:
        input_img = denormalize(input_img)
        output_img = denormalize(output_img)
        if target_img is not None:
            target_img = denormalize(target_img)
    
    # Convert to PIL
    input_pil = TF.to_pil_image(input_img.cpu())
    output_pil = TF.to_pil_image(output_img.cpu())
    
    # Create comparison
    width = input_pil.width
    height = input_pil.height
    
    if target_img is not None:
        target_pil = TF.to_pil_image(target_img.cpu())
        combined = Image.new('RGB', (width * 3, height))
        combined.paste(input_pil, (0, 0))
        combined.paste(output_pil, (width, 0))
        combined.paste(target_pil, (width * 2, 0))
    else:
        combined = Image.new('RGB', (width * 2, height))
        combined.paste(input_pil, (0, 0))
        combined.paste(output_pil, (width, 0))
    
    combined.save(save_path)


class VisualizationCallback:
    """
    Training callback for periodic visualization.
    """
    
    def __init__(
        self,
        save_dir: str,
        save_interval: int = 100
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.call_count = 0
    
    def __call__(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        target_img: Optional[torch.Tensor] = None,
        epoch: int = 0,
        step: int = 0
    ):
        """Save visualization if at save interval."""
        self.call_count += 1
        
        if self.call_count % self.save_interval == 0:
            filename = f"epoch{epoch}_step{step}.png"
            save_path = self.save_dir / filename
            save_comparison_image(
                input_img[0] if input_img.dim() == 4 else input_img,
                output_img[0] if output_img.dim() == 4 else output_img,
                target_img[0] if target_img is not None and target_img.dim() == 4 else target_img,
                str(save_path)
            )


# Test
if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # Create dummy data
    input_img = torch.rand(3, 256, 256)
    output_img = torch.rand(3, 256, 256)
    target_img = torch.rand(3, 256, 256)
    
    # Test comparison grid
    fig = create_comparison_grid(input_img, output_img, target_img)
    plt.savefig("test_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved test_comparison.png")
    
    # Clean up
    import os
    os.remove("test_comparison.png")
    print("Test completed!")
