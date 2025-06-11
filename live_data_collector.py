import torch
import time
import numpy as np

class LiveDataCollector:
    def __init__(self, n_features=3, window_size=100, sample_interval=1.0):
        """
        Simulate live system metrics collection.
        
        Args:
            n_features (int): Number of features (e.g., CPU, temp, RAM)
            window_size (int): Size of sliding window
            sample_interval (float): Time between samples (seconds)
        """
        self.n_features = n_features
        self.window_size = window_size
        self.sample_interval = sample_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = []
        
        # Simulate state means for synthetic data
        self.sim_means = torch.randn(3, n_features).to(self.device) * 2
        self.sim_covs = torch.eye(n_features).unsqueeze(0).repeat(3, 1, 1).to(self.device) * 0.5
        self.sim_trans = torch.softmax(torch.randn(3, 3), dim=1).to(self.device)
        self.current_state = 0
    
    def get_sample(self):
        """Simulate a single sample of system metrics."""
        # In practice, replace with psutil calls, e.g.:
        # import psutil
        # cpu = psutil.cpu_percent()
        # ram = psutil.virtual_memory().percent
        # temp = psutil.sensors_temperatures()['coretemp'][0].current
        # return torch.tensor([cpu, temp, ram], device=self.device)
        probs = self.sim_trans[self.current_state]
        self.current_state = torch.multinomial(probs, 1).item()
        sample = torch.distributions.MultivariateNormal(
            self.sim_means[self.current_state],
            self.sim_covs[self.current_state]
        ).sample()
        return sample
    
    def collect_window(self):
        """Collect a window of live data."""
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        sample = self.get_sample()
        self.buffer.append(sample.cpu().numpy())  # Store as NumPy for efficiency
        
        if len(self.buffer) == self.window_size:
            # Convert buffer to a single NumPy array, then to tensor
            buffer_array = np.array(self.buffer)
            return torch.from_numpy(buffer_array).to(self.device)
        return None