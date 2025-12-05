"""
Distributed Training for 1.58-Bit Quantized LLMs

Multi-GPU and multi-node distributed training support using:
- PyTorch DistributedDataParallel (DDP)
- Gradient accumulation
- Synchronized batch normalization
- Model parallelism support
- Mixed precision distributed training
- Efficient all-reduce operations

Supports single-machine multi-GPU through to multi-node clusters.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
import time
import numpy as np
from functools import wraps


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Distributed setup
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Mixed precision
    use_mixed_precision: bool = False
    mixed_precision_dtype: str = 'float16'
    
    # Synchronization
    sync_gradients_every: int = 1
    bucket_cap_mb: int = 25


def setup_distributed(config: DistributedConfig):
    """
    Setup distributed training environment.
    
    Args:
        config: DistributedConfig
    """
    # Set environment variables
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    
    # Initialize process group
    dist.init_process_group(
        backend=config.backend,
        rank=config.rank,
        world_size=config.world_size
    )
    
    # Set device
    torch.cuda.set_device(config.local_rank)
    
    print(f"Rank {config.rank} initialized with world size {config.world_size}")


def cleanup_distributed():
    """Cleanup distributed training environment."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is main process."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_on_main(msg: str):
    """Print only on main process."""
    if is_main_process():
        print(msg)


class DistributedModel(nn.Module):
    """
    Wrapper for distributed model training.
    
    Handles:
    - DDP wrapping
    - Synchronization
    - Gradient accumulation
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        find_unused_parameters: bool = False
    ):
        """
        Initialize distributed model.
        
        Args:
            model: Model to wrap
            config: DistributedConfig
            find_unused_parameters: Allow unused parameters
        """
        super().__init__()
        
        self.config = config
        self.model = model
        self.local_rank = config.local_rank
        
        # Wrap with DDP
        self.model = DDP(
            model.cuda(self.local_rank),
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=True,
            bucket_cap_mb=config.bucket_cap_mb
        )
        
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.accumulation_counter = 0
    
    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)
    
    def synchronize_gradients(self):
        """Synchronize gradients across processes."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(dist.get_world_size())
    
    def zero_grad(self):
        """Zero out gradients."""
        self.model.zero_grad()
    
    def backward(self, loss):
        """Backward pass with gradient accumulation."""
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        self.accumulation_counter += 1
        
        # Synchronize after accumulation steps
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            self.accumulation_counter = 0
            return True
        return False
    
    def state_dict(self):
        """Get model state dict."""
        return self.model.module.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dict."""
        self.model.module.load_state_dict(state_dict)


class DistributedDataLoaderFactory:
    """Factory for creating distributed data loaders."""
    
    @staticmethod
    def create_dataloader(
        dataset,
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Create distributed data loader.
        
        Args:
            dataset: Dataset
            batch_size: Batch size per process
            num_workers: Number of data loading workers
            shuffle: Shuffle data
            drop_last: Drop last incomplete batch
            pin_memory: Pin memory for GPU
        
        Returns:
            DataLoader with DistributedSampler
        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return dataloader


class DistributedOptimizer:
    """
    Distributed optimizer wrapper.
    
    Features:
    - Gradient synchronization
    - Loss scaling (for mixed precision)
    - Gradient clipping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        optimizer,
        config: DistributedConfig,
        scaler=None
    ):
        """
        Initialize distributed optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            config: DistributedConfig
            scaler: GradScaler for mixed precision
        """
        self.optimizer = optimizer
        self.config = config
        self.scaler = scaler
        self.step_count = 0
    
    def step(self):
        """Optimizer step."""
        if self.config.use_mixed_precision and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.step_count += 1
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class DistributedTrainer:
    """
    Complete distributed training coordinator.
    
    Orchestrates:
    - Multi-GPU training
    - Checkpointing
    - Metrics aggregation
    - Loss scaling
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        config: DistributedConfig,
        device: str = None
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            optimizer: PyTorch optimizer
            config: DistributedConfig
            device: Device to use (auto-detected if None)
        """
        self.config = config
        self.device = device or torch.device(
            f'cuda:{config.local_rank}' if torch.cuda.is_available() else 'cpu'
        )
        
        # Wrap model
        self.model = DistributedModel(model, config)
        
        # Setup optimizer
        self.optimizer = DistributedOptimizer(optimizer, config)
        
        # Loss scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Metrics
        self.train_metrics = {
            'loss': [],
            'gradient_norm': [],
            'throughput': []
        }
        
        self.step = 0
        self.epoch = 0
    
    def train_step(
        self,
        batch: Dict,
        criterion: Callable,
        accumulation_step: bool = False
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch
            criterion: Loss function
            accumulation_step: Whether this is an accumulation step
        
        Returns:
            Metrics dictionary
        """
        self.model.model.train()
        
        # Mixed precision context
        ctx = torch.cuda.amp.autocast() if self.config.use_mixed_precision else nullcontext()
        
        with ctx:
            # Forward pass
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(**inputs)
            
            # Compute loss
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass with gradient scaling
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.step % self.config.sync_gradients_every == 0:
            if self.config.use_mixed_precision:
                self.scaler.unscale_(self.optimizer.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                max_norm=1.0
            )
        
        # Optimizer step
        if accumulation_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1
        
        # Compute metrics
        gradient_norm = sum(
            p.grad.norm().item() ** 2
            for p in self.model.model.parameters()
            if p.grad is not None
        ) ** 0.5
        
        metrics = {
            'loss': float(loss),
            'gradient_norm': gradient_norm,
            'learning_rate': self.optimizer.get_lr()
        }
        
        # Record metrics
        for key, value in metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)
        
        return metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: Callable,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            criterion: Loss function
            log_interval: Print metrics every N steps
        
        Returns:
            Epoch metrics
        """
        self.epoch += 1
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(dataloader):
            # Determine if this is an accumulation step
            accumulation_step = (
                (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
            )
            
            metrics = self.train_step(batch, criterion, accumulation_step)
            
            if (batch_idx + 1) % log_interval == 0 and is_main_process():
                print(f"Epoch {self.epoch}, Batch {batch_idx + 1}: Loss={metrics['loss']:.4f}")
        
        # Aggregate metrics
        for key in self.train_metrics:
            if self.train_metrics[key]:
                epoch_metrics[key] = np.mean(self.train_metrics[key])
        
        return epoch_metrics
    
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            dataloader: Evaluation dataloader
            criterion: Loss function
        
        Returns:
            Evaluation metrics
        """
        self.model.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
        
        # Average loss across all processes
        if dist.is_initialized():
            loss_tensor = torch.tensor(total_loss).to(self.device)
            count_tensor = torch.tensor(total_samples).to(self.device)
            
            dist.all_reduce(loss_tensor)
            dist.all_reduce(count_tensor)
            
            avg_loss = (loss_tensor / count_tensor).item()
        else:
            avg_loss = total_loss / total_samples
        
        return {'eval_loss': avg_loss}
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        if is_main_process():
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.optimizer.state_dict(),
                'step': self.step,
                'epoch': self.epoch,
                'config': self.config
            }
            
            if self.scaler:
                checkpoint['scaler'] = self.scaler.state_dict()
            
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        if self.scaler and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"Checkpoint loaded from {path}")


# Context manager for mixed precision
class nullcontext:
    """Null context manager."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
