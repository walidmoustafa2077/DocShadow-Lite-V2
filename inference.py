"""
Inference Script for LP-IOANet
==============================
Provides utilities for:
- Single image inference
- Batch inference on directories
- Model export (ONNX, TorchScript)
- Performance benchmarking
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import IOANet, LPIOANet, build_model


class ShadowRemover:
    """
    Inference wrapper for LP-IOANet shadow removal.
    
    Supports both Stage 1 (low-res IOANet) and Stage 2 (full LP-IOANet).
    Handles image loading, preprocessing, inference, and postprocessing.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        stage: Model stage (1 or 2)
        use_amp: Use automatic mixed precision
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        stage: int = 2,
        use_amp: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.stage = stage
        self.use_amp = use_amp and torch.cuda.is_available()
        
        print(f"Loading model from {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Stage: {stage}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Target sizes
        self.low_res_size = (192, 256)
        self.high_res_size = (768, 1024)
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine model type from state dict
        has_refiner = any('refiner' in k for k in state_dict.keys())
        
        if has_refiner or self.stage == 2:
            # Full LP-IOANet
            ioanet = IOANet(pretrained=False)
            model = LPIOANet(ioanet=ioanet)
            self.stage = 2
        else:
            # Just IOANet
            model = IOANet(pretrained=False)
            self.stage = 1
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        return model
    
    def _preprocess(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            tensor: Preprocessed image tensor
            original_size: Original image size (H, W)
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = (image.height, image.width)
        
        # Determine target size based on stage
        target_size = self.high_res_size if self.stage == 2 else self.low_res_size
        
        # Resize
        image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # Convert to tensor and normalize
        tensor = transforms.functional.to_tensor(image)
        tensor = self.normalize(tensor)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device), original_size
    
    def _postprocess(
        self, 
        tensor: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Convert output tensor to PIL Image.
        
        Args:
            tensor: Output tensor (B, C, H, W) or (C, H, W)
            original_size: If provided, resize to this size
            
        Returns:
            PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        tensor = tensor.clamp(0, 1)
        
        # Convert to numpy
        array = tensor.cpu().numpy().transpose(1, 2, 0)
        array = (array * 255).astype(np.uint8)
        
        # Convert to PIL
        image = Image.fromarray(array)
        
        # Resize to original if requested
        if original_size is not None:
            image = image.resize((original_size[1], original_size[0]), Image.BILINEAR)
        
        return image
    
    @torch.no_grad()
    def remove_shadow(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_original_size: bool = True
    ) -> Image.Image:
        """
        Remove shadow from a single image.
        
        Args:
            image: Input image
            return_original_size: If True, resize output to match input size
            
        Returns:
            Shadow-removed image
        """
        # Preprocess
        input_tensor, original_size = self._preprocess(image)
        
        # Inference
        if self.use_amp:
            with torch.cuda.amp.autocast():
                output = self.model(input_tensor)
        else:
            output = self.model(input_tensor)
        
        # Postprocess
        target_size = original_size if return_original_size else None
        result = self._postprocess(output, target_size)
        
        return result
    
    @torch.no_grad()
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> Dict[str, float]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            extensions: Supported image extensions
            
        Returns:
            Dict with processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.glob(f'*{ext}'))
            image_paths.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not image_paths:
            print(f"No images found in {input_dir}")
            return {}
        
        print(f"Processing {len(image_paths)} images...")
        
        total_time = 0
        processed = 0
        
        for img_path in tqdm(image_paths):
            try:
                start = time.time()
                result = self.remove_shadow(str(img_path))
                elapsed = time.time() - start
                total_time += elapsed
                
                # Save result
                output_path = output_dir / img_path.name
                result.save(output_path)
                processed += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        avg_time = total_time / processed if processed > 0 else 0
        
        stats = {
            'processed': processed,
            'total_time': total_time,
            'avg_time': avg_time,
            'fps': 1 / avg_time if avg_time > 0 else 0
        }
        
        print(f"\nProcessed {processed} images")
        print(f"Average time: {avg_time*1000:.1f}ms")
        print(f"FPS: {stats['fps']:.1f}")
        
        return stats
    
    def benchmark(
        self, 
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            num_iterations: Number of iterations for timing
            warmup: Number of warmup iterations
            
        Returns:
            Benchmark statistics
        """
        # Create dummy input
        if self.stage == 2:
            size = (1, 3, self.high_res_size[0], self.high_res_size[1])
        else:
            size = (1, 3, self.low_res_size[0], self.low_res_size[1])
        
        dummy_input = torch.randn(size).to(self.device)
        
        print(f"Benchmarking with input size: {size}")
        print(f"Warmup iterations: {warmup}")
        print(f"Timed iterations: {num_iterations}")
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed iterations
        times = []
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        stats = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }
        
        print(f"\nResults:")
        print(f"Mean: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms")
        print(f"Min: {stats['min_ms']:.2f}ms, Max: {stats['max_ms']:.2f}ms")
        print(f"FPS: {stats['fps']:.1f}")
        
        return stats
    
    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 14
    ):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output file path
            opset_version: ONNX opset version
        """
        if self.stage == 2:
            size = (1, 3, self.high_res_size[0], self.high_res_size[1])
        else:
            size = (1, 3, self.low_res_size[0], self.low_res_size[1])
        
        dummy_input = torch.randn(size).to(self.device)
        
        print(f"Exporting to ONNX: {output_path}")
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=opset_version
        )
        
        print("Export completed!")
    
    def export_torchscript(
        self,
        output_path: str,
        method: str = 'trace'
    ):
        """
        Export model to TorchScript format.
        
        Args:
            output_path: Output file path
            method: 'trace' or 'script'
        """
        if self.stage == 2:
            size = (1, 3, self.high_res_size[0], self.high_res_size[1])
        else:
            size = (1, 3, self.low_res_size[0], self.low_res_size[1])
        
        print(f"Exporting to TorchScript ({method}): {output_path}")
        
        if method == 'trace':
            dummy_input = torch.randn(size).to(self.device)
            scripted = torch.jit.trace(self.model, dummy_input)
        else:
            scripted = torch.jit.script(self.model)
        
        scripted.save(output_path)
        print("Export completed!")


def main():
    parser = argparse.ArgumentParser(description="LP-IOANet Inference")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output image or directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--stage',
        type=int,
        default=2,
        choices=[1, 2],
        help='Model stage'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark'
    )
    parser.add_argument(
        '--export-onnx',
        type=str,
        default=None,
        help='Export to ONNX at specified path'
    )
    parser.add_argument(
        '--export-torchscript',
        type=str,
        default=None,
        help='Export to TorchScript at specified path'
    )
    
    args = parser.parse_args()
    
    # Initialize remover
    remover = ShadowRemover(
        checkpoint_path=args.checkpoint,
        device=args.device,
        stage=args.stage
    )
    
    # Export models
    if args.export_onnx:
        remover.export_onnx(args.export_onnx)
    
    if args.export_torchscript:
        remover.export_torchscript(args.export_torchscript)
    
    # Run benchmark
    if args.benchmark:
        remover.benchmark()
    
    # Process images
    if args.input and args.output:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image
            result = remover.remove_shadow(str(input_path))
            result.save(args.output)
            print(f"Saved result to {args.output}")
        elif input_path.is_dir():
            # Directory
            remover.process_directory(str(input_path), args.output)
        else:
            print(f"Input not found: {args.input}")


if __name__ == "__main__":
    main()
