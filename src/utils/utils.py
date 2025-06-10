"""
Utility Functions

This module provides various utility functions for the HDP-HMM live data
processing system, including configuration management, logging, and
performance monitoring.
"""
import torch
import time
import json
import os
import logging
from datetime import datetime
import numpy as np
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hdp_hmm.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("hdp_hmm")

class GPUMemoryMonitor:
    """Monitor GPU memory usage during training."""
    
    def __init__(self, device=None):
        """
        Initialize GPU memory monitor.
        
        Args:
            device (torch.device): GPU device to monitor
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gpu = self.device.type == 'cuda'
        self.snapshots = []
    
    def snapshot(self, tag=None):
        """
        Take a snapshot of current GPU memory usage.
        
        Args:
            tag (str): Optional tag for the snapshot
            
        Returns:
            dict: Memory usage data
        """
        if not self.is_gpu:
            logger.info("GPU monitoring requested but not using GPU")
            return None
        
        try:
            # Get memory usage
            allocated = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved(self.device) / (1024**2)    # MB
            
            # Create snapshot
            timestamp = datetime.now()
            snapshot = {
                'timestamp': timestamp,
                'tag': tag,
                'allocated_mb': allocated,
                'reserved_mb': reserved
            }
            
            self.snapshots.append(snapshot)
            
            logger.debug(f"GPU Memory - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error in GPU memory monitoring: {e}")
            return None
    
    def get_peak_memory(self):
        """
        Get peak memory usage.
        
        Returns:
            float: Peak allocated memory in MB
        """
        if not self.snapshots:
            return 0
            
        return max(s['allocated_mb'] for s in self.snapshots)
    
    def clear(self):
        """Clear all snapshots."""
        self.snapshots = []
    
    def plot_usage(self):
        """
        Plot memory usage over time.
        
        Returns:
            matplotlib.figure.Figure: The figure or None if plotting failed
        """
        if not self.snapshots:
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            # Extract data
            times = [(s['timestamp'] - self.snapshots[0]['timestamp']).total_seconds() 
                    for s in self.snapshots]
            allocated = [s['allocated_mb'] for s in self.snapshots]
            reserved = [s['reserved_mb'] for s in self.snapshots]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(times, allocated, 'b-', label='Allocated')
            ax.plot(times, reserved, 'r--', label='Reserved')
            
            # Add tags as vertical lines if available
            for i, s in enumerate(self.snapshots):
                if s['tag']:
                    ax.axvline(x=times[i], color='g', linestyle=':', alpha=0.5)
                    ax.text(times[i], max(allocated) * 0.9, s['tag'], 
                            rotation=90, alpha=0.7)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('GPU Memory Usage')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting GPU memory usage: {e}")
            return None


class PerformanceMonitor:
    """Monitor performance metrics during training."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.training_times = []
        self.inference_times = []
        self.start_times = {}
    
    def start_timer(self, key="default"):
        """
        Start a timer.
        
        Args:
            key (str): Timer identifier
        """
        self.start_times[key] = time.time()
    
    def stop_timer(self, key="default", category=None):
        """
        Stop a timer and record elapsed time.
        
        Args:
            key (str): Timer identifier
            category (str): Optional category for the timing
            
        Returns:
            float: Elapsed time in seconds
        """
        if key not in self.start_times:
            logger.warning(f"Timer '{key}' was not started")
            return None
            
        elapsed = time.time() - self.start_times[key]
        
        if category == "training":
            self.training_times.append(elapsed)
        elif category == "inference":
            self.inference_times.append(elapsed)
            
        return elapsed
    
    def get_average_time(self, category, last_n=None):
        """
        Get average time for a category.
        
        Args:
            category (str): Category name ("training" or "inference")
            last_n (int): Number of recent times to average, or None for all
            
        Returns:
            float: Average time in seconds
        """
        times = self.training_times if category == "training" else self.inference_times
        
        if not times:
            return None
            
        if last_n is not None:
            times = times[-min(last_n, len(times)):]
            
        return sum(times) / len(times)
    
    def get_system_stats(self):
        """
        Get current system resource statistics.
        
        Returns:
            dict: System statistics
        """
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            try:
                stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
                stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**2)    # MB
            except:
                pass
                
        return stats
    
    def reset(self):
        """Reset all timers and stats."""
        self.training_times = []
        self.inference_times = []
        self.start_times = {}


class ConfigManager:
    """Manage configuration for the HDP-HMM system."""
    
    def __init__(self, config_path="config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """
        Load configuration from file or create default.
        
        Returns:
            dict: Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self):
        """
        Create default configuration.
        
        Returns:
            dict: Default configuration
        """
        config = {
            'model': {
                'n_features': 3,
                'max_states': 20,
                'alpha': 1.0,
                'gamma': 1.0,
                'learning_rate': 0.01
            },
            'data': {
                'window_size': 100,
                'sample_interval': 1.0,
                'use_real_metrics': False
            },
            'training': {
                'max_iterations': 1000,
                'save_interval': 10,
                'checkpoint_interval': 100
            },
            'visualization': {
                'interactive': True,
                'save_plots': True,
                'feature_names': ["CPU", "Memory", "Disk"]
            },
            'paths': {
                'model_dir': 'models',
                'plot_dir': 'plots'
            }
        }
        
        # Save default config
        self.save_config(config)
        
        return config
    
    def save_config(self, config=None):
        """
        Save configuration to file.
        
        Args:
            config (dict): Configuration to save, or None to save current
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if config is not None:
            self.config = config
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Dot-separated path to config value
            default: Default value if key not found
            
        Returns:
            The configuration value or default
        """
        parts = key.split('.')
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): Dot-separated path to config value
            value: Value to set
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        parts = key.split('.')
        config = self.config
        
        try:
            # Navigate to the correct nesting level
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
                
            # Set the value
            config[parts[-1]] = value
            
            # Save the updated config
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Error setting config value: {e}")
            return False


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def setup_device():
    """
    Set up and return the best available device.
    
    Returns:
        torch.device: Device to use for computation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Log additional GPU info
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        logger.info("GPU not available, using CPU")
    
    return device
