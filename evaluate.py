"""
Evaluation Script for LP-IOANet
==============================
Evaluates trained model on test dataset with comprehensive metrics.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import build_model, IOANet, LPIOANet
from src.data import DocumentShadowDataset, MultiScaleDocumentDataset
from src.utils import Evaluator, save_comparison_image, denormalize


def evaluate(
    checkpoint_path: str,
    test_dir: str,
    output_dir: str = None,
    stage: int = 2,
    batch_size: int = 1,
    device: str = 'cuda',
    save_images: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_dir: Path to test dataset
        output_dir: Path to save output images
        stage: Model stage (1 or 2)
        batch_size: Batch size
        device: Device to run on
        save_images: Whether to save output images
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Determine model type
    has_refiner = any('refiner' in k for k in state_dict.keys())
    
    if has_refiner or stage == 2:
        ioanet = IOANet(pretrained=False)
        model = LPIOANet(ioanet=ioanet)
        stage = 2
    else:
        model = IOANet(pretrained=False)
        stage = 1
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Setup dataset
    if stage == 1:
        dataset = DocumentShadowDataset(
            root_dir=test_dir,
            stage=1,
            mode='test'
        )
    else:
        dataset = MultiScaleDocumentDataset(
            root_dir=test_dir,
            mode='test'
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(dataset)} images...")
    
    # Setup evaluator
    evaluator = Evaluator(device=str(device), compute_lpips=True)
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if stage == 1:
                input_img = batch['input'].to(device)
                target_img = batch['target'].to(device)
            else:
                input_img = batch['input_high'].to(device)
                target_img = batch['target_high'].to(device)
            
            filenames = batch['filename']
            
            # Inference
            with torch.cuda.amp.autocast():
                output = model(input_img)
            
            # Denormalize for metrics
            output_denorm = denormalize(output)
            target_denorm = denormalize(target_img)
            
            # Update metrics
            evaluator.update(output_denorm, target_denorm)
            
            # Save images
            if save_images and output_dir:
                for i, filename in enumerate(filenames):
                    save_path = output_dir / f"{filename}_comparison.png"
                    save_comparison_image(
                        input_img[i],
                        output[i],
                        target_img[i],
                        str(save_path)
                    )
    
    # Print results
    evaluator.print_results()
    results = evaluator.compute()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LP-IOANet")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='Dataset/test',
        help='Path to test dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Path to save output images'
    )
    parser.add_argument(
        '--stage',
        type=int,
        default=2,
        choices=[1, 2],
        help='Model stage'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save output images'
    )
    
    args = parser.parse_args()
    
    results = evaluate(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        stage=args.stage,
        batch_size=args.batch_size,
        device=args.device,
        save_images=not args.no_save
    )
    
    # Save results to file
    if args.output_dir:
        import json
        results_path = Path(args.output_dir) / 'metrics.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
