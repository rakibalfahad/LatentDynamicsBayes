import torch
import numpy as np
import pandas as pd
import os
import glob
import logging

logger = logging.getLogger("hdp_hmm")

class CSVDataProcessor:
    """
    Process CSV files for offline training of the HDP-HMM model.
    
    Each CSV file should have columns representing features, and rows representing time steps.
    """
    def __init__(self, data_dir, window_size=100, stride=None):
        """
        Initialize the CSV data processor.
        
        Args:
            data_dir (str): Directory containing CSV files
            window_size (int): Size of sliding window for processing
            stride (int, optional): Stride for sliding window. If None, stride = window_size (non-overlapping windows)
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_files = []
        self.total_samples = 0
        self.current_position = 0
        self.current_file_idx = 0
        self.current_data = None
        self.n_features = None
        
    def load_csv_files(self):
        """
        Load all CSV files from the data directory.
        
        Returns:
            bool: True if files were found and loaded, False otherwise
        """
        # Find all CSV files in the data directory
        csv_pattern = os.path.join(self.data_dir, "*.csv")
        self.csv_files = sorted(glob.glob(csv_pattern))
        
        if not self.csv_files:
            logger.error(f"No CSV files found in {self.data_dir}")
            return False
        
        logger.info(f"Found {len(self.csv_files)} CSV files in {self.data_dir}")
        
        # Load the first file to determine the number of features
        first_df = pd.read_csv(self.csv_files[0])
        self.n_features = first_df.shape[1]
        logger.info(f"Detected {self.n_features} features in CSV files")
        
        # Calculate total number of samples across all files
        self.total_samples = 0
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file)
            self.total_samples += df.shape[0]
        
        logger.info(f"Total samples across all files: {self.total_samples}")
        
        # Load the first file into memory
        self.current_file_idx = 0
        self.current_position = 0
        self.current_data = pd.read_csv(self.csv_files[0]).values
        
        return True
    
    def get_next_window(self):
        """
        Get the next window of data from the CSV files.
        
        Returns:
            torch.Tensor: Window of data with shape (window_size, n_features), or None if no more data
        """
        if not self.csv_files or self.current_data is None:
            logger.error("No CSV files loaded. Call load_csv_files() first.")
            return None
        
        # Check if we need to load a new file
        while self.current_position + self.window_size > len(self.current_data):
            self.current_file_idx += 1
            if self.current_file_idx >= len(self.csv_files):
                # No more files
                logger.info("Reached the end of all CSV files")
                return None
            
            # Load the next file
            logger.info(f"Loading next file: {self.csv_files[self.current_file_idx]}")
            self.current_data = pd.read_csv(self.csv_files[self.current_file_idx]).values
            self.current_position = 0
        
        # Extract the window
        window_data = self.current_data[self.current_position:self.current_position + self.window_size]
        
        # Move to the next position based on stride
        self.current_position += self.stride
        
        # Convert to tensor
        window_tensor = torch.from_numpy(window_data).float().to(self.device)
        return window_tensor
    
    def get_total_windows(self):
        """
        Calculate the total number of windows available.
        
        Returns:
            int: Approximate number of windows
        """
        if self.total_samples <= 0:
            return 0
        
        # Calculate number of windows with stride
        return 1 + (self.total_samples - self.window_size) // self.stride
    
    def reset(self):
        """Reset the processor to start from the beginning."""
        if self.csv_files:
            self.current_file_idx = 0
            self.current_position = 0
            self.current_data = pd.read_csv(self.csv_files[0]).values
