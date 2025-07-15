import json
import os
import time
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np


class ExperimentTracker:
    """
    Tracks experiments with metrics, loss curves, and timing information.
    Supports logging gradient steps and wallclock time.
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize tracking data
        self.metrics: Dict[str, List[float]] = {}
        self.gradient_steps: List[int] = []
        self.wallclock_times: List[float] = []
        self.start_time = time.time()
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
        
        # Initialize metrics file
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.jsonl")
        
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.config = config
        config_path = os.path.join(self.experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metric(self, metric_name: str, value: float, step: int) -> None:
        """
        Log a metric value at a specific gradient step.
        
        Args:
            metric_name: Name of the metric (e.g., 'train_loss', 'val_loss')
            value: Metric value
            step: Gradient step number
        """
        # Initialize metric list if first time
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        # Store metric
        self.metrics[metric_name].append(value)
        
        # Store gradient step and wallclock time
        self.gradient_steps.append(step)
        self.wallclock_times.append(time.time() - self.start_time)
        
        # Log to file
        log_entry = {
            "metric": metric_name,
            "value": value,
            "step": step,
            "wallclock_time": self.wallclock_times[-1],
            "timestamp": time.time()
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_multiple_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log multiple metrics at once for the same step.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Gradient step number
        """
        current_time = time.time() - self.start_time
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
        
        # Store step and time once
        self.gradient_steps.append(step)
        self.wallclock_times.append(current_time)
        
        # Log all metrics to file
        log_entry = {
            "metrics": metrics,
            "step": step,
            "wallclock_time": current_time,
            "timestamp": time.time()
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return self.metrics.get(metric_name, [])
    
    def get_steps(self) -> List[int]:
        """Get gradient steps."""
        return self.gradient_steps
    
    def get_wallclock_times(self) -> List[float]:
        """Get wallclock times in seconds."""
        return self.wallclock_times
    
    def plot_loss_curves(self, metrics: Optional[List[str]] = None, 
                        x_axis: str = "steps", save_path: Optional[str] = None) -> None:
        """
        Plot loss curves for specified metrics.
        
        Args:
            metrics: List of metric names to plot. If None, plots all metrics.
            x_axis: X-axis type - "steps" or "time"
            save_path: Path to save plot. If None, saves to experiment directory.
        """
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        # Choose x-axis data
        if x_axis == "steps":
            x_data = self.gradient_steps
            x_label = "Gradient Steps"
        elif x_axis == "time":
            x_data = self.wallclock_times
            x_label = "Wallclock Time (seconds)"
        else:
            raise ValueError("x_axis must be 'steps' or 'time'")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        for metric_name in metrics:
            if metric_name in self.metrics:
                y_data = self.metrics[metric_name]
                # Ensure x and y data have same length
                min_len = min(len(x_data), len(y_data))
                plt.plot(x_data[:min_len], y_data[:min_len], label=metric_name)
        
        plt.xlabel(x_label)
        plt.ylabel("Loss")
        plt.title(f"Loss Curves - {self.experiment_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, f"loss_curves_{x_axis}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curves saved to: {save_path}")
    
    def save_summary(self) -> None:
        """Save experiment summary."""
        summary = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "total_steps": len(self.gradient_steps),
            "total_time": self.wallclock_times[-1] if self.wallclock_times else 0,
            "metrics_summary": {}
        }
        
        # Add metric summaries
        for metric_name, values in self.metrics.items():
            if values:
                summary["metrics_summary"][metric_name] = {
                    "final_value": values[-1],
                    "min_value": min(values),
                    "max_value": max(values),
                    "mean_value": np.mean(values)
                }
        
        summary_path = os.path.join(self.experiment_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_from_logs(self) -> None:
        """Load experiment data from existing log files."""
        if not os.path.exists(self.metrics_file):
            return
        
        # Clear existing data
        self.metrics.clear()
        self.gradient_steps.clear()
        self.wallclock_times.clear()
        
        # Load from jsonl file
        with open(self.metrics_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                
                if "metric" in entry:
                    # Single metric entry
                    metric_name = entry["metric"]
                    if metric_name not in self.metrics:
                        self.metrics[metric_name] = []
                    self.metrics[metric_name].append(entry["value"])
                    self.gradient_steps.append(entry["step"])
                    self.wallclock_times.append(entry["wallclock_time"])
                elif "metrics" in entry:
                    # Multiple metrics entry
                    for metric_name, value in entry["metrics"].items():
                        if metric_name not in self.metrics:
                            self.metrics[metric_name] = []
                        self.metrics[metric_name].append(value)
                    self.gradient_steps.append(entry["step"])
                    self.wallclock_times.append(entry["wallclock_time"])
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save summary."""
        self.save_summary()