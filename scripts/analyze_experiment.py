#!/usr/bin/env python3
"""
Experiment analysis script for visualizing and comparing training runs.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

from cs336_basics.experiment_tracker import ExperimentTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    
    parser.add_argument('--experiments_dir', type=str, default='experiments', 
                       help='Directory containing experiment logs')
    parser.add_argument('--experiment_names', type=str, nargs='+', 
                       help='List of experiment names to analyze')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['train_loss', 'val_loss'],
                       help='Metrics to plot')
    parser.add_argument('--x_axis', type=str, choices=['steps', 'time'], 
                       default='steps', help='X-axis for plots')
    parser.add_argument('--output_dir', type=str, default='analysis_plots',
                       help='Directory to save analysis plots')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison plots across experiments')
    
    return parser.parse_args()


def load_experiment_data(experiment_dir: str) -> Dict[str, Any]:
    """Load experiment data from directory."""
    experiment_name = os.path.basename(experiment_dir)
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load metrics using tracker
    tracker = ExperimentTracker(experiment_name, os.path.dirname(experiment_dir))
    tracker.load_from_logs()
    
    return {
        'name': experiment_name,
        'config': config,
        'tracker': tracker,
        'metrics': tracker.metrics,
        'steps': tracker.get_steps(),
        'times': tracker.get_wallclock_times()
    }


def plot_individual_experiment(experiment_data: Dict[str, Any], 
                              metrics: List[str], x_axis: str, 
                              output_dir: str) -> None:
    """Plot metrics for a single experiment."""
    name = experiment_data['name']
    tracker = experiment_data['tracker']
    
    # Create individual plots
    tracker.plot_loss_curves(metrics=metrics, x_axis=x_axis, 
                           save_path=os.path.join(output_dir, f"{name}_{x_axis}.png"))


def plot_comparison(experiments: List[Dict[str, Any]], 
                   metrics: List[str], x_axis: str, 
                   output_dir: str) -> None:
    """Create comparison plots across experiments."""
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for exp_data in experiments:
            name = exp_data['name']
            
            if metric in exp_data['metrics']:
                if x_axis == 'steps':
                    x_data = exp_data['steps']
                    x_label = 'Gradient Steps'
                else:
                    x_data = exp_data['times']
                    x_label = 'Wallclock Time (seconds)'
                
                y_data = exp_data['metrics'][metric]
                
                # Ensure x and y data have same length
                min_len = min(len(x_data), len(y_data))
                plt.plot(x_data[:min_len], y_data[:min_len], 
                        label=f"{name}", linewidth=2)
        
        plt.xlabel(x_label)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        save_path = os.path.join(output_dir, f"comparison_{metric}_{x_axis}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved: {save_path}")


def generate_summary_table(experiments: List[Dict[str, Any]], 
                          output_dir: str) -> None:
    """Generate a summary table of experiments."""
    
    summary_data = []
    
    for exp_data in experiments:
        name = exp_data['name']
        config = exp_data['config']
        metrics = exp_data['metrics']
        
        row = {
            'Experiment': name,
            'Learning Rate': config.get('learning_rate', 'N/A'),
            'Batch Size': config.get('batch_size', 'N/A'),
            'Model Dimension': config.get('d_model', 'N/A'),
            'Layers': config.get('num_layers', 'N/A'),
            'Total Steps': len(exp_data['steps']),
            'Total Time (s)': f"{exp_data['times'][-1]:.1f}" if exp_data['times'] else 'N/A'
        }
        
        # Add final metric values
        for metric in ['train_loss', 'val_loss']:
            if metric in metrics and metrics[metric]:
                row[f'Final {metric.replace("_", " ").title()}'] = f"{metrics[metric][-1]:.4f}"
            else:
                row[f'Final {metric.replace("_", " ").title()}'] = 'N/A'
        
        summary_data.append(row)
    
    # Create summary table
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary table saved: {summary_path}")
    
    # Print summary to console
    print("\nExperiment Summary:")
    print("-" * 80)
    for row in summary_data:
        print(f"Experiment: {row['Experiment']}")
        print(f"  Config: LR={row['Learning Rate']}, BS={row['Batch Size']}, "
              f"d_model={row['Model Dimension']}, layers={row['Layers']}")
        print(f"  Results: {row['Total Steps']} steps, {row['Total Time (s)']}s, "
              f"final train_loss={row['Final Train Loss']}, "
              f"final val_loss={row['Final Val Loss']}")
        print()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiment data
    experiments = []
    
    if args.experiment_names:
        # Load specific experiments
        for exp_name in args.experiment_names:
            exp_dir = os.path.join(args.experiments_dir, exp_name)
            if os.path.exists(exp_dir):
                experiments.append(load_experiment_data(exp_dir))
            else:
                print(f"Warning: Experiment directory not found: {exp_dir}")
    else:
        # Load all experiments in directory
        if os.path.exists(args.experiments_dir):
            for exp_name in os.listdir(args.experiments_dir):
                exp_dir = os.path.join(args.experiments_dir, exp_name)
                if os.path.isdir(exp_dir):
                    experiments.append(load_experiment_data(exp_dir))
    
    if not experiments:
        print("No experiments found to analyze.")
        return
    
    print(f"Analyzing {len(experiments)} experiments...")
    
    # Generate individual plots
    for exp_data in experiments:
        plot_individual_experiment(exp_data, args.metrics, args.x_axis, args.output_dir)
    
    # Generate comparison plots if requested
    if args.compare and len(experiments) > 1:
        plot_comparison(experiments, args.metrics, args.x_axis, args.output_dir)
    
    # Generate summary table
    generate_summary_table(experiments, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()