import torch
import torch.optim as optim
from hdp_hmm import HDPHMM
import os

class LiveHDPHMM:
    def __init__(self, n_features, max_states=20, lr=0.01, model_path="models/hdp_hmm.pth"):
        """
        Manage live training and inference for HDP-HMM.
        
        Args:
            n_features (int): Number of features
            max_states (int): Maximum number of states
            lr (float): Learning rate
            model_path (str): Path to save/load model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HDPHMM(n_features, max_states).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model_path = model_path
        self.losses = []
    
    def update_model(self, window_data):
        """Update model with a new window of data."""
        try:
            with torch.autograd.set_detect_anomaly(True):  # Keep anomaly detection
                self.optimizer.zero_grad()
                _, _, log_likelihood = self.model.forward_backward(window_data)
                loss = -log_likelihood
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())
                return loss.item()
        except RuntimeError as e:
            print(f"Error in update_model: {e}")
            raise
    
    def infer(self, window_data):
        """Perform inference on a window of data."""
        states, trans_probs = self.model.infer_states(window_data)
        return states, trans_probs
    
    def save_model(self):
        """Save the current model state."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses
        }, self.model_path.replace('.pth', '_checkpoint.pth'))
    
    def load_model(self):
        """Load a saved model state."""
        try:
            self.model.load_model(self.model_path)
            checkpoint = torch.load(self.model_path.replace('.pth', '_checkpoint.pth'))
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.losses = checkpoint['losses']
            print(f"Loaded training state from {self.model_path.replace('.pth', '_checkpoint.pth')}")
        except FileNotFoundError:
            print("No saved model found, starting fresh.")