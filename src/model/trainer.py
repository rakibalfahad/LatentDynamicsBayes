"""
Live Training Module

This module provides functionality for training and fine-tuning the HDP-HMM
model on streaming data, including incremental updates and model persistence.
"""
import torch
import torch.optim as optim
import os
import time
from datetime import datetime

from src.model.hdp_hmm import HDPHMM

class LiveTrainer:
    def __init__(self, n_features, max_states=20, lr=0.01, device=None, 
                 model_dir="models", model_name="hdp_hmm"):
        """
        Manage live training and inference for HDP-HMM.
        
        Args:
            n_features (int): Number of features in the data
            max_states (int): Maximum number of states for the HDP-HMM
            lr (float): Learning rate for optimizer
            device (torch.device): Device to use for computation
            model_dir (str): Directory to save/load models
            model_name (str): Base name for saved models
        """
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")
        
        # Initialize model and optimizer
        self.model = HDPHMM(n_features, max_states, device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Model persistence
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, f"{model_name}.pth")
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Training statistics
        self.losses = []
        self.train_count = 0
        self.last_save_time = time.time()
        self.last_checkpoint_count = 0
    
    def update_model(self, window_data, incremental=True):
        """
        Update model with a new window of data.
        
        Args:
            window_data (torch.Tensor): Window of time series data
            incremental (bool): If True, performs incremental update
            
        Returns:
            float: Loss value
        """
        # Ensure data is on the correct device
        if window_data.device != self.device:
            window_data = window_data.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward-backward pass and compute loss
        _, _, log_likelihood = self.model.forward_backward(window_data)
        loss = -log_likelihood  # Negative log-likelihood
        
        # Backward pass and optimization
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        self.train_count += 1
        
        # Auto-save model periodically
        self._check_auto_save()
        
        return loss_value
    
    def infer(self, window_data):
        """
        Perform inference on a window of data.
        
        Args:
            window_data (torch.Tensor): Window of time series data
            
        Returns:
            tuple: (states, trans_probs)
                states: Most likely state sequence
                trans_probs: Transition probability matrix
        """
        # Ensure data is on the correct device
        if window_data.device != self.device:
            window_data = window_data.to(self.device)
        
        # Set model to eval mode for inference
        self.model.eval()
        
        with torch.no_grad():
            states, trans_probs = self.model.infer_states(window_data)
        
        # Set model back to train mode
        self.model.train()
        
        return states, trans_probs
    
    def predict_next(self, window_data):
        """
        Predict the next observation based on current window.
        
        Args:
            window_data (torch.Tensor): Window of time series data
            
        Returns:
            tuple: (mean, covariance) of predicted next observation
        """
        # Ensure data is on the correct device
        if window_data.device != self.device:
            window_data = window_data.to(self.device)
            
        self.model.eval()
        with torch.no_grad():
            pred_mean, pred_cov = self.model.posterior_predictive(window_data)
        self.model.train()
        
        return pred_mean, pred_cov
    
    def save_model(self, custom_path=None):
        """
        Save the current model state.
        
        Args:
            custom_path (str): Optional custom path to save model
            
        Returns:
            str: Path where model was saved
        """
        save_path = custom_path if custom_path else self.model_path
        
        # Save model state
        self.model.save_model(save_path)
        
        # Save optimizer state and training info
        checkpoint_path = os.path.join(self.model_dir, f"{self.model_name}_checkpoint.pth")
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'train_count': self.train_count,
            'last_save_time': time.time()
        }, checkpoint_path)
        
        self.last_save_time = time.time()
        self.last_checkpoint_count = self.train_count
        
        return save_path
    
    def save_checkpoint(self):
        """
        Save a timestamped checkpoint of the model.
        
        Returns:
            str: Path to the saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.model_dir, f"{self.model_name}_{timestamp}.pth")
        return self.save_model(checkpoint_path)
    
    def load_model(self, custom_path=None):
        """
        Load a saved model state.
        
        Args:
            custom_path (str): Optional custom path to load model from
            
        Returns:
            bool: True if model was successfully loaded, False otherwise
        """
        load_path = custom_path if custom_path else self.model_path
        
        # Try to load model
        success = self.model.load_model(load_path)
        
        if success:
            # Try to load optimizer state and training info
            checkpoint_path = os.path.join(self.model_dir, f"{self.model_name}_checkpoint.pth")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.losses = checkpoint.get('losses', [])
                self.train_count = checkpoint.get('train_count', 0)
                self.last_save_time = checkpoint.get('last_save_time', time.time())
                print(f"Loaded training state from {checkpoint_path}")
            except Exception as e:
                print(f"Could not load optimizer state: {e}")
        
        return success
    
    def get_active_states_count(self):
        """
        Get the current count of active states.
        
        Returns:
            int or str: Number of active states or message if not determined yet
        """
        return self.model.get_active_states_count()
    
    def get_latest_loss(self):
        """
        Get the most recent loss value.
        
        Returns:
            float: Latest loss value or None if no training has occurred
        """
        if self.losses:
            return self.losses[-1]
        return None
    
    def get_mean_loss(self, window=100):
        """
        Get the mean loss over the last `window` updates.
        
        Args:
            window (int): Number of recent losses to average
            
        Returns:
            float: Mean loss or None if not enough updates
        """
        if len(self.losses) > 0:
            recent_losses = self.losses[-min(window, len(self.losses)):]
            return sum(recent_losses) / len(recent_losses)
        return None
    
    def get_training_count(self):
        """
        Get the total number of training updates.
        
        Returns:
            int: Number of training updates
        """
        return self.train_count
    
    def _check_auto_save(self, min_updates=10, time_interval=300):
        """
        Check if model should be auto-saved.
        
        Args:
            min_updates (int): Minimum number of updates before saving
            time_interval (int): Minimum time between saves in seconds
        """
        # Check if enough time has passed and we have enough new updates
        time_since_save = time.time() - self.last_save_time
        updates_since_save = self.train_count - self.last_checkpoint_count
        
        if (time_since_save > time_interval and updates_since_save >= min_updates):
            self.save_model()
            print(f"Auto-saved model after {updates_since_save} updates and {time_since_save:.1f} seconds")
    
    def reset_optimizer(self, lr=None):
        """
        Reset the optimizer, optionally with a new learning rate.
        
        Args:
            lr (float): New learning rate, or None to keep current rate
        """
        if lr is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            # Extract current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = optim.Adam(self.model.parameters(), lr=current_lr)
