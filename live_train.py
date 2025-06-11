import torch
import torch.optim as optim
from hdp_hmm import HDPHMM

class LiveHDPHMM:
    def __init__(self, n_features, max_states=20, lr=0.01, model_path="hdp_hmm.pth"):
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
        with torch.autograd.set_detect_anomaly(True):  # Enable anomaly detection
            self.optimizer.zero_grad()
            _, _, log_likelihood = self.model.forward_backward(window_data)
            loss = -log_likelihood
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            return loss.item()
    
    def infer(self, window_data):
        """Perform inference on a window of data."""
        states, trans_probs = self.model.infer_states(window_data)
        return states, trans_probs
    
    def save_model(self):
        """Save the current model state."""
        self.model.save_model(self.model_path)
    
    def load_model(self):
        """Load a saved model state."""
        try:
            self.model.load_model(self.model_path)
            self.optimizer = optim.Adam(self.model.parameters())
        except FileNotFoundError:
            print("No saved model found, starting fresh.")