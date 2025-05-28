import time
import psutil
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import numpy as np

from utils.Setup import setup
from utils.Optim import build_optimizer
from dataLoader import create_dataloaders
from config import parse_args
from model import SILN


class EfficiencyMetrics:
    def __init__(self):
        self.process = psutil.Process()
        # Track both allocated and reserved memory
        self.gpu_memory = {
            'train': {
                'allocated': [],
                'reserved': [],
                'peak_allocated': 0,
                'peak_reserved': 0
            },
            'inference': {
                'allocated': [],
                'reserved': [],
                'peak_allocated': 0,
                'peak_reserved': 0
            }
        }
        self.cpu_memory = {'train': [], 'inference': []}
        self.epoch_times = {'train': [], 'inference': []}
        
        # Reset CUDA memory stats at initialization
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0}
            
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2
        }
    
    def get_peak_gpu_memory(self) -> Dict[str, float]:
        """Get peak GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0}
            
        return {
            'allocated': torch.cuda.max_memory_allocated() / 1024**2,
            'reserved': torch.cuda.max_memory_reserved() / 1024**2
        }
    
    def get_cpu_memory_usage(self) -> float:
        """Get current CPU memory usage in MB"""
        return self.process.memory_info().rss / 1024**2
    
    def record_epoch_metrics(self, phase: str):
        """Record memory metrics for an epoch"""
        gpu_mem = self.get_gpu_memory_usage()
        self.gpu_memory[phase]['allocated'].append(gpu_mem['allocated'])
        self.gpu_memory[phase]['reserved'].append(gpu_mem['reserved'])
        
        # Update peak memory if current usage is higher
        peak_mem = self.get_peak_gpu_memory()
        self.gpu_memory[phase]['peak_allocated'] = max(
            self.gpu_memory[phase]['peak_allocated'],
            peak_mem['allocated']
        )
        self.gpu_memory[phase]['peak_reserved'] = max(
            self.gpu_memory[phase]['peak_reserved'],
            peak_mem['reserved']
        )
        
        self.cpu_memory[phase].append(self.get_cpu_memory_usage())
    
    def start_epoch_timer(self, phase: str):
        """Start timing for an epoch"""
        torch.cuda.synchronize()
        self.epoch_times[phase].append(time.time())
    
    def end_epoch_timer(self, phase: str):
        """End timing for an epoch"""
        torch.cuda.synchronize()
        if self.epoch_times[phase]:
            start_time = self.epoch_times[phase].pop()
            duration = time.time() - start_time
            self.epoch_times[phase].append(duration)
    
    def get_paper_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics in a format suitable for academic paper reporting"""
        metrics = {
            'GPU Memory (MB)': {
                'Training (Peak)': self.gpu_memory['train']['peak_allocated'],
                'Training (Average)': np.mean(self.gpu_memory['train']['allocated']),
                'Inference (Peak)': self.gpu_memory['inference']['peak_allocated'],
                'Inference (Average)': np.mean(self.gpu_memory['inference']['allocated'])
            },
            'GPU Memory Reserved (MB)': {
                'Training (Peak)': self.gpu_memory['train']['peak_reserved'],
                'Training (Average)': np.mean(self.gpu_memory['train']['reserved']),
                'Inference (Peak)': self.gpu_memory['inference']['peak_reserved'],
                'Inference (Average)': np.mean(self.gpu_memory['inference']['reserved'])
            },
            'CPU Memory (MB)': {
                'Training': np.mean(self.cpu_memory['train']),
                'Inference': np.mean(self.cpu_memory['inference'])
            },
            'Time per Epoch (s)': {
                'Training': np.mean(self.epoch_times['train']),
                'Inference': np.mean(self.epoch_times['inference'])
            }
        }
        return metrics
    
    def print_paper_metrics(self):
        """This method is kept for compatibility but is no longer used"""
        pass


def evaluate_efficiency(args):
    """
    Evaluate model efficiency metrics including memory usage and runtime.
    Runs a single epoch of training and inference for efficiency measurement.
    """
    # Initialize efficiency metrics
    eff_metrics = EfficiencyMetrics()
    
    # Load data
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)

    # Initialize model
    model = SILN(args)
    model = model.to(args.device)
    optimizer, scheduler = build_optimizer(args, model, len(train_dataloader))
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    # Training phase
    eff_metrics.start_epoch_timer('train')
    eff_metrics.record_epoch_metrics('train')
    
    model.train()
    for batch in train_dataloader:
        pred_user, label = model(args, batch)
        loss = loss_function(pred_user, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    
    eff_metrics.end_epoch_timer('train')

    # Inference phase (validation + test)
    eff_metrics.start_epoch_timer('inference')
    eff_metrics.record_epoch_metrics('inference')
    _ = inference(args, model, val_dataloader)
    _ = inference(args, model, test_dataloader)
    eff_metrics.end_epoch_timer('inference')

    # Print efficiency metrics
    metrics = eff_metrics.get_paper_metrics()
    
    print("\nEfficiency Metrics:")
    print("GPU Memory (MB):")
    print(f"  Training - Allocated: {metrics['GPU Memory (MB)']['Training (Peak)']:.1f} (peak), {metrics['GPU Memory (MB)']['Training (Average)']:.1f} (avg)")
    print(f"  Training - Reserved:  {metrics['GPU Memory Reserved (MB)']['Training (Peak)']:.1f} (peak), {metrics['GPU Memory Reserved (MB)']['Training (Average)']:.1f} (avg)")
    print(f"  Inference - Allocated: {metrics['GPU Memory (MB)']['Inference (Peak)']:.1f} (peak), {metrics['GPU Memory (MB)']['Inference (Average)']:.1f} (avg)")
    print(f"  Inference - Reserved:  {metrics['GPU Memory Reserved (MB)']['Inference (Peak)']:.1f} (peak), {metrics['GPU Memory Reserved (MB)']['Inference (Average)']:.1f} (avg)")
    print(f"CPU Memory (MB): {metrics['CPU Memory (MB)']['Training']:.1f} (train), {metrics['CPU Memory (MB)']['Inference']:.1f} (inference)")
    print(f"Time (s): {metrics['Time per Epoch (s)']['Training']:.2f} (train), {metrics['Time per Epoch (s)']['Inference']:.2f} (inference)")


def inference(args, model, dataloader):
    """Run inference without computing metrics"""
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prediction, _ = model(args, batch)
    return None


def main():
    args = parse_args()
    setup(args)
    evaluate_efficiency(args)


if __name__ == '__main__':
    main()
