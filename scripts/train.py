#!/usr/bin/env python3
"""
Training script for the TransformerLM model.

This script expects pre-tokenized data in .npy format.
To tokenize your text data, use the separate tokenization script:
    python tokenize_and_encode.py --mode train --input_file data.txt --output_file tokens.npy
    python tokenize_and_encode.py --mode encode --input_file data.txt --output_file tokens.npy
"""

import argparse
import json
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.transformer import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.lr_cosine_schedule import lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.get_batch import get_batch
from cs336_basics.experiment_tracker import ExperimentTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TransformerLM model')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=1024, help='Context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--eps', type=float, default=1e-5, help='Layer norm epsilon')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum training iterations')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='Warmup iterations')
    parser.add_argument('--cosine_cycle_iters', type=int, default=10000, help='Cosine cycle iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
    
    # Data paths
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    # Logging settings
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of evaluation iterations')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for tracking')
    parser.add_argument('--experiments_dir', type=str, default='experiments', help='Directory for experiment logs')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def load_data_memmap(data_path: str) -> np.ndarray:
    """
    Load pre-tokenized data using memory mapping for efficient memory usage.
    
    This function expects .npy files containing tokenized data.
    Use tokenize_and_encode.py to convert text files to the required format.
    """
    if data_path.endswith('.npy'):
        return np.load(data_path, mmap_mode='r')
    elif data_path.endswith('.txt'):
        raise ValueError(
            "Text files not supported directly. Please tokenize first using:\n"
            "python tokenize_and_encode.py --mode encode --input_file your_file.txt --output_file tokenized_data.npy"
        )
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Expected .npy files.")


def evaluate_model(model: nn.Module, val_data: np.ndarray, eval_iters: int, 
                  batch_size: int, context_length: int, device: str) -> float:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(val_data, batch_size, context_length, device)
            logits = model(x)
            
            # Flatten for cross entropy
            logits = logits.view(-1, logits.size(-1))
            targets = y.view(-1)
            
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()
    
    model.train()
    return total_loss / eval_iters


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize experiment tracker
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = f"run_{int(time.time())}"
    
    tracker = ExperimentTracker(experiment_name, args.experiments_dir)
    tracker.log_config(vars(args))
    
    # Save config (keep existing behavior)
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data with memory mapping
    print(f"Loading training data from {args.train_data}")
    train_data = load_data_memmap(args.train_data)
    print(f"Training data shape: {train_data.shape}")
    
    print(f"Loading validation data from {args.val_data}")
    val_data = load_data_memmap(args.val_data)
    print(f"Validation data shape: {val_data.shape}")
    
    # Initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.eps,
        device=args.device
    )
    
    model.to(args.device)
    model.train()
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resuming from iteration {start_iter}")
    
    # Training loop
    print(f"Starting training from iteration {start_iter}")
    print(f"Device: {args.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for iteration in range(start_iter, args.max_iters):
        # Get batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        # Update learning rate
        lr = lr_cosine_schedule(
            iteration,
            args.learning_rate,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        logits = logits.view(-1, logits.size(-1))
        targets = y.view(-1)
        loss = cross_entropy(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iteration % args.log_interval == 0:
            print(f"Iter {iteration:5d} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
            tracker.log_metric("train_loss", loss.item(), iteration)
            tracker.log_metric("learning_rate", lr, iteration)
        
        # Evaluation
        if iteration % args.eval_interval == 0 and iteration > 0:
            val_loss = evaluate_model(
                model, val_data, args.eval_iters, 
                args.batch_size, args.context_length, args.device
            )
            print(f"Iter {iteration:5d} | Val Loss: {val_loss:.4f}")
            tracker.log_metric("val_loss", val_loss, iteration)
        
        # Checkpointing
        if iteration % args.checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{iteration}.pt')
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_checkpoint.pt')
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"Saved final checkpoint: {final_checkpoint_path}")
    
    # Generate loss curve plots
    tracker.plot_loss_curves(x_axis="steps")
    tracker.plot_loss_curves(x_axis="time")
    
    print("Training completed!")


if __name__ == "__main__":
    main()