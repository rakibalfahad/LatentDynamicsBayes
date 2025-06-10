"""
Data Processing Utilities

This module provides utilities for processing time series data before feeding
it to the HDP-HMM model, including normalization, feature extraction, and
sliding window operations.
"""
import torch
import numpy as np
from collections import deque

class TimeSeriesProcessor:
    def __init__(self, n_features, window_size, normalize=True, device=None):
        """
        Process time series data for HDP-HMM modeling.
        
        Args:
            n_features (int): Number of features in the time series
            window_size (int): Size of sliding window
            normalize (bool): Whether to normalize data
            device (torch.device): Device to place tensors on
        """
        self.n_features = n_features
        self.window_size = window_size
        self.normalize = normalize
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Statistics for normalization
        self.means = torch.zeros(n_features, device=self.device)
        self.stds = torch.ones(n_features, device=self.device)
        self.n_samples = 0
        
        # Buffer for online statistics calculation
        self.buffer = deque(maxlen=1000)  # Keep recent samples for adaptive stats
    
    def update_statistics(self, data):
        """
        Update normalization statistics with new data.
        
        Args:
            data (torch.Tensor): New data to update statistics with
        """
        if not self.normalize:
            return
            
        # Add data to buffer
        if isinstance(data, torch.Tensor):
            self.buffer.extend(data.cpu().numpy())
        else:
            self.buffer.extend(data)
        
        # Update statistics using buffer
        if len(self.buffer) > 10:
            buffer_tensor = torch.tensor(list(self.buffer), device=self.device)
            self.means = buffer_tensor.mean(dim=0)
            self.stds = buffer_tensor.std(dim=0)
            # Avoid division by zero
            self.stds[self.stds < 1e-5] = 1.0
            self.n_samples = len(self.buffer)
    
    def normalize_data(self, data):
        """
        Normalize data using current statistics.
        
        Args:
            data (torch.Tensor): Data to normalize
            
        Returns:
            torch.Tensor: Normalized data
        """
        if not self.normalize or self.n_samples < 10:
            return data
            
        return (data - self.means) / self.stds
    
    def process_window(self, window_data):
        """
        Process a window of time series data.
        
        Args:
            window_data (torch.Tensor): Window of time series data
            
        Returns:
            torch.Tensor: Processed data ready for modeling
        """
        # Update statistics with new data
        self.update_statistics(window_data)
        
        # Normalize data
        processed_data = self.normalize_data(window_data)
        
        return processed_data
    
    def extract_features(self, window_data):
        """
        Extract additional features from raw time series.
        Can be extended for domain-specific feature extraction.
        
        Args:
            window_data (torch.Tensor): Window of time series data
            
        Returns:
            torch.Tensor: Data with additional features
        """
        # This is a placeholder for more complex feature extraction
        # For now, we'll just add first-order differences
        if window_data.shape[0] > 1:
            diffs = torch.diff(window_data, dim=0)
            # Pad with a zero to maintain same size
            diffs = torch.cat([torch.zeros(1, window_data.shape[1], device=self.device), diffs])
            # Return original features and differences
            return torch.cat([window_data, diffs], dim=1)
        
        return window_data
