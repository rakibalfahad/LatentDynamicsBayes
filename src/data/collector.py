"""
Live Data Collector

This module provides tools for collecting and processing live streaming data
from system metrics or other sources. It supports both real system metrics
collection (via psutil) and simulated data for testing.
"""
import torch
import time
import numpy as np
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available, using simulated data only")

class SystemMetricsCollector:
    """Collects real system metrics using psutil."""
    
    def __init__(self, device=None):
        """
        Initialize the system metrics collector.
        
        Args:
            device (torch.device): Device to place tensors on
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil is required for real system metrics collection.")
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_cpu_metrics(self, per_cpu=True):
        """
        Get CPU utilization metrics.
        
        Args:
            per_cpu (bool): Whether to get per-CPU metrics or average
            
        Returns:
            torch.Tensor: CPU utilization metrics
        """
        if per_cpu:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            return torch.tensor(cpu_percent, device=self.device, dtype=torch.float32)
        else:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return torch.tensor([cpu_percent], device=self.device, dtype=torch.float32)
    
    def get_memory_metrics(self):
        """
        Get memory utilization metrics.
        
        Returns:
            torch.Tensor: Memory metrics [percent_used, available_GB, used_GB]
        """
        mem = psutil.virtual_memory()
        metrics = [
            mem.percent,                     # Percentage used
            mem.available / (1024**3),       # Available GB
            mem.used / (1024**3)             # Used GB
        ]
        return torch.tensor(metrics, device=self.device, dtype=torch.float32)
    
    def get_disk_metrics(self):
        """
        Get disk utilization metrics.
        
        Returns:
            torch.Tensor: Disk metrics [percent_used, read_MB/s, write_MB/s]
        """
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Get disk I/O statistics
        disk_io = psutil.disk_io_counters()
        if hasattr(self, 'last_disk_io') and hasattr(self, 'last_disk_time'):
            time_delta = time.time() - self.last_disk_time
            read_delta = (disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024**2) / time_delta  # MB/s
            write_delta = (disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024**2) / time_delta  # MB/s
        else:
            read_delta = 0
            write_delta = 0
        
        self.last_disk_io = disk_io
        self.last_disk_time = time.time()
        
        metrics = [disk_percent, read_delta, write_delta]
        return torch.tensor(metrics, device=self.device, dtype=torch.float32)
    
    def get_network_metrics(self):
        """
        Get network utilization metrics.
        
        Returns:
            torch.Tensor: Network metrics [received_MB/s, sent_MB/s]
        """
        net_io = psutil.net_io_counters()
        if hasattr(self, 'last_net_io') and hasattr(self, 'last_net_time'):
            time_delta = time.time() - self.last_net_time
            recv_delta = (net_io.bytes_recv - self.last_net_io.bytes_recv) / (1024**2) / time_delta  # MB/s
            sent_delta = (net_io.bytes_sent - self.last_net_io.bytes_sent) / (1024**2) / time_delta  # MB/s
        else:
            recv_delta = 0
            sent_delta = 0
        
        self.last_net_io = net_io
        self.last_net_time = time.time()
        
        metrics = [recv_delta, sent_delta]
        return torch.tensor(metrics, device=self.device, dtype=torch.float32)
    
    def get_temperature_metrics(self):
        """
        Get temperature metrics if available.
        
        Returns:
            torch.Tensor: Temperature metrics or None if not available
        """
        try:
            temperatures = []
            for name, entries in psutil.sensors_temperatures().items():
                if name == 'coretemp':  # CPU temperature on Linux
                    temperatures.extend([entry.current for entry in entries])
                elif name == 'acpitz':  # ACPI thermal zone on Linux
                    temperatures.extend([entry.current for entry in entries])
                # Add more platform-specific temperature sources as needed
            
            if temperatures:
                return torch.tensor(temperatures, device=self.device, dtype=torch.float32)
        except (AttributeError, KeyError, IOError):
            pass  # temperatures not available
        
        # Return a dummy temperature if not available
        return torch.tensor([0.0], device=self.device, dtype=torch.float32)
    
    def get_all_metrics(self):
        """
        Get all available system metrics.
        
        Returns:
            torch.Tensor: Combined system metrics
        """
        metrics = []
        
        # Add CPU metrics (just the average to keep dimensions manageable)
        metrics.append(self.get_cpu_metrics(per_cpu=False))
        
        # Add memory metrics
        metrics.append(self.get_memory_metrics()[0:1])  # Just the percentage
        
        # Add disk metrics
        metrics.append(self.get_disk_metrics()[0:1])  # Just the percentage
        
        # Add network metrics (optional)
        # metrics.extend(self.get_network_metrics())
        
        # Add temperature metrics (if available)
        temp = self.get_temperature_metrics()
        if temp is not None and temp.numel() > 0:
            metrics.append(temp[0:1])  # Just the first temperature
        
        return torch.cat(metrics)


class LiveDataCollector:
    def __init__(self, n_features=3, window_size=100, sample_interval=1.0, use_real_metrics=False, device=None):
        """
        Collect live data streams with sliding window functionality.
        
        Args:
            n_features (int): Number of features to collect
            window_size (int): Size of sliding window
            sample_interval (float): Time between samples (seconds)
            use_real_metrics (bool): Use real system metrics via psutil if True
            device (torch.device): Device to place tensors on
        """
        self.n_features = n_features
        self.window_size = window_size
        self.sample_interval = sample_interval
        self.use_real_metrics = use_real_metrics and PSUTIL_AVAILABLE
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize buffer
        self.buffer = []
        
        # For simulated data
        if not self.use_real_metrics:
            # Simulate multi-state system for synthetic data
            self.sim_means = torch.randn(3, n_features, device=self.device) * 2
            self.sim_covs = torch.eye(n_features, device=self.device).unsqueeze(0).repeat(3, 1, 1) * 0.5
            self.sim_trans = torch.softmax(torch.randn(3, 3, device=self.device), dim=1)
            self.current_state = 0
        else:
            # Initialize real metrics collector
            self.metrics_collector = SystemMetricsCollector(device=self.device)
            print("Using real system metrics for data collection")
    
    def get_sample(self):
        """
        Get a single sample of data.
        
        Returns:
            torch.Tensor: Single data sample
        """
        if self.use_real_metrics:
            # Get real system metrics
            return self.metrics_collector.get_all_metrics()
        else:
            # Simulate data from a three-state HMM
            probs = self.sim_trans[self.current_state]
            self.current_state = torch.multinomial(probs, 1).item()
            sample = torch.distributions.MultivariateNormal(
                self.sim_means[self.current_state],
                self.sim_covs[self.current_state]
            ).sample()
            return sample
    
    def collect_window(self):
        """
        Collect a window of data using sliding window approach.
        
        Returns:
            torch.Tensor: Window of data or None if window not yet full
        """
        # Remove oldest sample if buffer is full
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        
        # Get new sample
        sample = self.get_sample()
        self.buffer.append(sample)
        
        # Return window if it's full
        if len(self.buffer) == self.window_size:
            return torch.stack(self.buffer)
        return None
    
    def get_last_n_samples(self, n=1):
        """
        Get the last n samples from the buffer.
        
        Args:
            n (int): Number of samples to retrieve
            
        Returns:
            torch.Tensor: Last n samples or None if not enough samples
        """
        if len(self.buffer) < n:
            return None
        
        samples = self.buffer[-n:]
        return torch.stack(samples)
    
    def reset_buffer(self):
        """Clear the data buffer."""
        self.buffer = []
