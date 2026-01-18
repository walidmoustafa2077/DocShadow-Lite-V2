"""
Two-Stage Training Script for LP-IOANet
=======================================
Implements the training strategy from the paper:

Stage 1: Train IOANet at low resolution (192x256)
    - Full model training
    - L1 + LPIPS loss
    - 1000 epochs

Stage 2: Train LP-IOANet at high resolution (768x1024)
    - Freeze IOANet
    - Train only LPTN-Lite refiners
    - L1 loss only
    - 200 epochs

Includes Claude's Research Execution Plan for attention mask verification.
"""

import os
import sys
import argparse
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import IOANet, LPIOANet, build_model
from src.data import create_dataloaders, denormalize_tensor
from src.losses import ShadowLoss, MultiScaleShadowLoss
from src.utils import (
    print_claude_plan,
    save_debug_samples,
    AttentionAnalyzer,
    Stage1ReadinessChecker,
    create_loss_breakdown_visualization,
    ResearchTracker
)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """
    Trainer class for LP-IOANet.
    
    Handles both Stage 1 and Stage 2 training with proper
    checkpointing, logging, and evaluation.
    """
    
    def __init__(
        self,
        config: Dict,
        stage: int = 1,
        resume: Optional[str] = None
    ):
        self.config = config
        self.stage = stage
        self.device = torch.device(
            config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"Using device: {self.device}")
        print(f"Training Stage: {stage}")
        
        # Setup directories
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir']) / f"stage{stage}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['logging']['log_dir']) / f"stage{stage}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Build model
        self.model = self._build_model(resume)
        
        # Setup dataloaders
        self.train_loader, self.val_loader = self._setup_dataloaders()
        
        # Setup loss function
        self.criterion = self._setup_loss()
        self.criterion = self.criterion.to(self.device)  # Move loss to device
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self._setup_optimizer()
        
        # Setup mixed precision training
        self.use_amp = config['hardware'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        # Debug & Research Verification
        self.debug_dir = Path(config['logging'].get('debug_dir', 'debug_samples'))
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        self.attention_analyzer = AttentionAnalyzer()
        self.loss_history: Dict[str, List[float]] = {
            'total': [], 'l1': [], 'lpips': []
        }
        
        # Stage 1 readiness checker (for transitioning to Stage 2)
        if self.stage == 1:
            self.readiness_checker = Stage1ReadinessChecker(
                mae_threshold=0.05,
                min_epochs=100
            )
        
        # Load checkpoint if resuming
        if resume:
            self._load_checkpoint(resume)
    
    def _build_model(self, resume: Optional[str]) -> nn.Module:
        """Build model for the specified stage."""
        
        if self.stage == 1:
            # Stage 1: IOANet only
            model = IOANet(pretrained=True)
        else:
            # Stage 2: Full LP-IOANet with frozen IOANet
            # Load pre-trained IOANet from Stage 1
            ioanet_path = self.config.get('pretrained_ioanet', None)
            if ioanet_path is None:
                # Look for best checkpoint from Stage 1
                stage1_dir = Path(self.config['logging']['checkpoint_dir']) / "stage1"
                ioanet_path = stage1_dir / "best_model.pth"
            
            if not Path(ioanet_path).exists():
                raise FileNotFoundError(
                    f"Stage 2 requires pretrained IOANet. "
                    f"Expected checkpoint at: {ioanet_path}"
                )
            
            model = build_model(
                pretrained_ioanet=str(ioanet_path),
                stage=2,
                hidden_channels=self.config['model']['lptn_lite']['hidden_channels'],
                use_relu6=self.config['model']['lptn_lite'].get('activation') == 'relu6'
            )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation dataloaders."""
        
        stage_config = self.config['training'][f'stage{self.stage}']
        
        train_loader, val_loader = create_dataloaders(
            train_dir=self.config['dataset']['train_dir'],
            test_dir=self.config['dataset']['test_dir'],
            stage=self.stage,
            batch_size=stage_config['batch_size'],
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        return train_loader, val_loader
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function based on stage."""
        
        stage_config = self.config['training'][f'stage{self.stage}']['loss']
        
        if self.stage == 1:
            criterion = ShadowLoss(
                w_l1=stage_config['l1_weight'],
                w_lpips=stage_config['lpips_weight'],
                stage=1
            )
        else:
            criterion = ShadowLoss(
                w_l1=stage_config['l1_weight'],
                w_lpips=stage_config.get('lpips_weight', 0.0),
                stage=2
            )
        
        return criterion
    
    def _setup_optimizer(self) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Setup optimizer and learning rate scheduler."""
        
        stage_config = self.config['training'][f'stage{self.stage}']
        opt_config = self.config['optimizer']
        
        # Filter trainable parameters
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        optimizer = optim.Adam(
            params,
            lr=stage_config['learning_rate'],
            betas=tuple(opt_config['betas']),
            eps=opt_config['eps'],
            weight_decay=stage_config.get('weight_decay', 0)
        )
        
        # Cosine annealing scheduler with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=stage_config['epochs'],
            eta_min=stage_config['learning_rate'] * 0.01
        )
        
        return optimizer, scheduler
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        is_best: bool = False,
        filename: str = "checkpoint.pth"
    ):
        """Save training checkpoint."""
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        torch.save(state, self.checkpoint_dir / filename)
        
        # Save best model separately
        if is_best:
            if self.stage == 1:
                # For Stage 1, save just IOANet state dict
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / "best_model.pth"
                )
            else:
                torch.save(state, self.checkpoint_dir / "best_model.pth")
            print(f"Saved best model at epoch {epoch}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.global_step = checkpoint['global_step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def _train_epoch_stage1(self, epoch: int) -> Dict[str, float]:
        """Train one epoch for Stage 1 (IOANet)."""
        
        self.model.train()
        losses = AverageMeter()
        tracker = ResearchTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", position=0, leave=True, disable=False)
        
        for batch in pbar:
            input_img = batch['input'].to(self.device)
            target_img = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(input_img)
                    loss, loss_dict = self.criterion(output, target_img)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(input_img)
                loss, loss_dict = self.criterion(output, target_img)
                loss.backward()
                self.optimizer.step()
            
            losses.update(loss.item(), input_img.size(0))
            
            # Track research metrics
            tracker.update(output, target_img, loss_dict)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
            
            # Log to tensorboard
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                
                # Track loss history for analysis
                self.loss_history['total'].append(loss_dict.get('total', loss.item()))
                if 'l1' in loss_dict:
                    self.loss_history['l1'].append(loss_dict['l1'])
                if 'lpips' in loss_dict:
                    self.loss_history['lpips'].append(loss_dict['lpips'])
            
            self.global_step += 1
        
        return {'train_loss': losses.avg, 'train_metrics': tracker.get_avg()}
    
    def _train_epoch_stage2(self, epoch: int) -> Dict[str, float]:
        """Train one epoch for Stage 2 (LP-IOANet)."""
        
        self.model.train()
        # Ensure IOANet stays in eval mode
        self.model.ioanet.eval()
        
        losses = AverageMeter()
        tracker = ResearchTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", position=0, leave=True, disable=False)
        
        for batch in pbar:
            input_high = batch['input_high'].to(self.device)
            target_high = batch['target_high'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(input_high)
                    loss, loss_dict = self.criterion(output, target_high)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(input_high)
                loss, loss_dict = self.criterion(output, target_high)
                loss.backward()
                self.optimizer.step()
            
            losses.update(loss.item(), input_high.size(0))
            
            # Track research metrics
            tracker.update(output, target_high, loss_dict)
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
            
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            
            self.global_step += 1
        
        return {'train_loss': losses.avg, 'train_metrics': tracker.get_avg()}
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation with attention mask analysis."""
        
        self.model.eval()
        losses = AverageMeter()
        mae_meter = AverageMeter()
        tracker = ResearchTracker()
        
        # For attention analysis
        all_input_masks = []
        all_output_masks = []
        
        for batch in tqdm(self.val_loader, desc="Validation", position=0, leave=True, disable=False):
            if self.stage == 1:
                input_img = batch['input'].to(self.device)
                target_img = batch['target'].to(self.device)
            else:
                input_img = batch['input_high'].to(self.device)
                target_img = batch['target_high'].to(self.device)
            
            if self.use_amp:
                with autocast():
                    # Get attention masks for Stage 1
                    if self.stage == 1:
                        output, attn_in, attn_out = self.model(input_img, return_attention=True)
                        all_input_masks.append(attn_in.cpu())
                        all_output_masks.append(attn_out.cpu())
                    else:
                        output = self.model(input_img)
                    
                    loss, loss_dict = self.criterion(output, target_img)
            else:
                if self.stage == 1:
                    output, attn_in, attn_out = self.model(input_img, return_attention=True)
                    all_input_masks.append(attn_in.cpu())
                    all_output_masks.append(attn_out.cpu())
                else:
                    output = self.model(input_img)
                
                loss, loss_dict = self.criterion(output, target_img)
            
            losses.update(loss.item(), input_img.size(0))
            
            # Track research metrics
            tracker.update(output, target_img, loss_dict)
            
            # Calculate MAE for readiness checking
            mae = F.l1_loss(output, target_img).item()
            mae_meter.update(mae, input_img.size(0))
        
        # Log validation loss
        self.writer.add_scalar('val/loss', losses.avg, epoch)
        self.writer.add_scalar('val/mae', mae_meter.avg, epoch)
        
        # Log research metrics
        val_metrics = tracker.get_avg()
        self.writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
        self.writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)
        
        # Attention mask analysis (Stage 1 only)
        if self.stage == 1 and len(all_input_masks) > 0:
            combined_in_mask = torch.cat(all_input_masks, dim=0)
            combined_out_mask = torch.cat(all_output_masks, dim=0)
            
            # Analyze attention masks
            attn_stats = self.attention_analyzer.analyze(
                combined_in_mask, combined_out_mask
            )
            
            # Log attention statistics to tensorboard
            self.writer.add_scalar('attention/input_mask_mean', attn_stats['in_mask_mean'], epoch)
            self.writer.add_scalar('attention/input_mask_std', attn_stats['in_mask_std'], epoch)
            self.writer.add_scalar('attention/output_mask_mean', attn_stats['out_mask_mean'], epoch)
            self.writer.add_scalar('attention/output_mask_std', attn_stats['out_mask_std'], epoch)
            
            # Check attention health
            healthy, msg = self.attention_analyzer.check_health()
            if not healthy:
                print(f"\n{msg}")
            
            # Update readiness checker
            self.readiness_checker.update(
                epoch, mae_meter.avg,
                combined_in_mask[:4], combined_out_mask[:4]
            )
        
        # Save debug samples at specific epochs
        if epoch == 0 or (epoch + 1) % 5 == 0:
            save_debug_samples(
                epoch=epoch,
                model=self.model,
                val_loader=self.val_loader,
                device=self.device,
                save_dir=str(self.debug_dir)
            )
        
        # Log sample images
        if epoch % self.config['logging']['save_interval'] == 0:
            self._log_images(input_img, target_img, output, epoch)
        
        # MAE-based alerts (per Claude's plan)
        if mae_meter.avg > 0.10:
            print(f"⚠️  ALERT: High MAE ({mae_meter.avg:.4f}). Check learning rate.")
        elif mae_meter.avg < 0.02:
            print(f"✅  TARGET REACHED: MAE is within Stage 1 research goals.")
        
        return {'val_loss': losses.avg, 'val_mae': mae_meter.avg, 'val_metrics': val_metrics}
    
    def _log_images(
        self, 
        input_img: torch.Tensor, 
        target_img: torch.Tensor,
        output: torch.Tensor,
        epoch: int
    ):
        """Log sample images to tensorboard."""
        
        # Denormalize for visualization
        input_vis = denormalize_tensor(input_img[0].cpu())
        target_vis = denormalize_tensor(target_img[0].cpu())
        output_vis = denormalize_tensor(output[0].cpu())
        
        self.writer.add_image('val/input', input_vis, epoch)
        self.writer.add_image('val/target', target_vis, epoch)
        self.writer.add_image('val/output', output_vis, epoch)
    
    def train(self):
        """Main training loop with Claude's Research Execution Plan."""
        
        # Print the research execution plan at start
        if self.stage == 1:
            print_claude_plan()
        
        stage_config = self.config['training'][f'stage{self.stage}']
        num_epochs = stage_config['epochs']
        
        print(f"\nStarting Stage {self.stage} training for {num_epochs} epochs")
        print("=" * 50)
        
        for epoch in range(self.start_epoch, num_epochs):
            start_time = time.time()
            
            # Train one epoch
            if self.stage == 1:
                train_metrics = self._train_epoch_stage1(epoch)
            else:
                train_metrics = self._train_epoch_stage2(epoch)
            
            # Validate
            val_metrics = self._validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Check for best model
            is_best = val_metrics['val_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['val_loss']
            
            # Save checkpoint
            if epoch % self.config['logging']['save_interval'] == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
            
            # Log epoch results with research-grade formatting
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            # Extract training metrics if available
            if 'train_metrics' in train_metrics:
                train_tracker_avg = train_metrics['train_metrics']
                train_log = (
                    f"Epoch {epoch+1:4d}/{num_epochs:4d} [TRAIN] | "
                    f"MAE={train_tracker_avg['mae']:.5f} | "
                    f"PSNR={train_tracker_avg['psnr']:.2f}dB | "
                    f"SSIM={train_tracker_avg['ssim']:.4f} | "
                    f"Total={train_tracker_avg['total']:.5f} | "
                    f"L1={train_tracker_avg['l1']:.5f} | "
                    f"LPIPS={train_tracker_avg['lpips']:.5f} | "
                    f"LR={lr:.2e} | Time={epoch_time:.1f}s"
                )
                print(train_log)
            else:
                print(
                    f"Epoch {epoch+1:4d}/{num_epochs:4d} [TRAIN] | "
                    f"Loss={train_metrics.get('train_loss', 0):.5f} | "
                    f"LR={lr:.2e} | Time={epoch_time:.1f}s"
                )
            
            # Validation logging
            if 'val_metrics' in val_metrics:
                val_tracker_avg = val_metrics['val_metrics']
                val_log = (
                    f"Epoch {epoch+1:4d}/{num_epochs:4d} [VAL] | "
                    f"MAE={val_tracker_avg['mae']:.5f} | "
                    f"PSNR={val_tracker_avg['psnr']:.2f}dB | "
                    f"SSIM={val_tracker_avg['ssim']:.4f} | "
                    f"Total={val_tracker_avg['total']:.5f} | "
                    f"L1={val_tracker_avg['l1']:.5f} | "
                    f"LPIPS={val_tracker_avg['lpips']:.5f}"
                )
                if is_best:
                    val_log += " *BEST*"
                print(val_log)
            
            self.writer.add_scalar('train/lr', lr, epoch)
            
            # Check Stage 1 readiness (every 50 epochs after epoch 100)
            if self.stage == 1 and epoch >= 100 and epoch % 50 == 0:
                ready, reason = self.readiness_checker.is_ready(epoch)
                print(f"\n{'='*60}")
                print("STAGE 2 READINESS CHECK")
                print(reason)
                print(f"{'='*60}\n")
            
            # Save loss breakdown visualization periodically
            if epoch > 0 and epoch % 50 == 0:
                create_loss_breakdown_visualization(
                    self.loss_history,
                    str(self.debug_dir / f"loss_breakdown_epoch{epoch}.png")
                )
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        
        # Final loss breakdown
        create_loss_breakdown_visualization(
            self.loss_history,
            str(self.debug_dir / "loss_breakdown_final.png")
        )
        
        self.writer.close()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train LP-IOANet")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--stage', 
        type=int, 
        default=1,
        choices=[1, 2],
        help='Training stage (1: IOANet, 2: LP-IOANet)'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--pretrained_ioanet',
        type=str,
        default=None,
        help='Path to pretrained IOANet for Stage 2'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override pretrained IOANet path if provided
    if args.pretrained_ioanet:
        config['pretrained_ioanet'] = args.pretrained_ioanet
    
    # Create trainer and start training
    trainer = Trainer(config, stage=args.stage, resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
