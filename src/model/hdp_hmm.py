"""
HDP-HMM Model with Stick-Breaking Process

This module implements a Hierarchical Dirichlet Process Hidden Markov Model
with stick-breaking construction for unsupervised learning of state sequences
in time series data. The model supports GPU acceleration through PyTorch.
"""
import torch
import torch.nn as nn
import torch.distributions as dist
import os

class HDPHMM(nn.Module):
    def __init__(self, n_features, max_states=20, alpha=1.0, gamma=1.0, device=None):
        """
        HDP-HMM with stick-breaking construction.
        
        Args:
            n_features (int): Number of features in the observation space
            max_states (int): Maximum number of states to consider
            alpha (float): Concentration parameter for transition Dirichlet process
            gamma (float): Concentration parameter for top-level Dirichlet process
            device (torch.device): Device to run computations on (GPU/CPU)
        """
        super(HDPHMM, self).__init__()
        self.n_features = n_features
        self.max_states = max_states
        self.alpha = alpha
        self.gamma = gamma
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Parameters for stick-breaking process
        self.beta_logits = nn.Parameter(torch.randn(max_states, device=self.device))
        self.pi_logits = nn.Parameter(torch.randn(max_states, max_states, device=self.device))
        
        # Emission parameters (Gaussian mixture)
        self.means = nn.Parameter(torch.randn(max_states, n_features, device=self.device))
        self.log_vars = nn.Parameter(torch.zeros(max_states, n_features, device=self.device))
        
        # Track active states
        self.active_states = None
        self.n_active_states = 0
        
    def stick_breaking(self, logits):
        """
        Compute stick-breaking weights from logits.
        
        Args:
            logits (torch.Tensor): Logits for stick-breaking process
            
        Returns:
            torch.Tensor: Stick-breaking weights
        """
        betas = torch.sigmoid(logits)
        beta_cumprod = torch.cumprod(1 - betas, dim=0)
        weights = betas * torch.cat([torch.ones(1, device=self.device), beta_cumprod[:-1]])
        return weights
    
    def compute_emission_probs(self, observations):
        """
        Compute emission probabilities for each state.
        
        Args:
            observations (torch.Tensor): Observation sequence of shape (seq_length, n_features)
            
        Returns:
            torch.Tensor: Log emission probabilities of shape (seq_length, max_states)
        """
        T = observations.shape[0]
        emission_probs = torch.zeros(T, self.max_states, device=self.device)
        
        for k in range(self.max_states):
            mvn = dist.MultivariateNormal(
                self.means[k],
                covariance_matrix=torch.diag(torch.exp(self.log_vars[k]))
            )
            emission_probs[:, k] = mvn.log_prob(observations)
            
        return emission_probs
    
    def forward_backward(self, observations):
        """
        Forward-backward algorithm for inference.
        
        Args:
            observations (torch.Tensor): Observation sequence of shape (seq_length, n_features)
        
        Returns:
            tuple: (alpha, beta, log_likelihood)
                alpha: Forward probabilities
                beta: Backward probabilities
                log_likelihood: Log likelihood of observations
        """
        T = observations.shape[0]
        beta_weights = self.stick_breaking(self.beta_logits)
        trans_probs = torch.softmax(self.pi_logits, dim=1)
        
        # Compute emission probabilities
        emission_probs = self.compute_emission_probs(observations)
        
        # Forward pass
        alpha = torch.zeros(T, self.max_states, device=self.device)
        alpha[0] = beta_weights * torch.exp(emission_probs[0])
        alpha[0] /= alpha[0].sum() + 1e-10
        
        for t in range(1, T):
            alpha[t] = torch.matmul(alpha[t-1], trans_probs) * torch.exp(emission_probs[t])
            alpha[t] /= alpha[t].sum() + 1e-10
        
        # Backward pass
        beta = torch.ones(T, self.max_states, device=self.device)
        for t in range(T-2, -1, -1):
            beta[t] = torch.matmul(trans_probs, (beta[t+1] * torch.exp(emission_probs[t+1])))
            beta[t] /= beta[t].sum() + 1e-10
        
        # Compute posterior state probabilities
        gamma = alpha * beta
        gamma /= gamma.sum(dim=1, keepdim=True) + 1e-10
        
        # Update active states (states with significant posterior probability)
        state_usage = gamma.sum(dim=0)
        self.active_states = (state_usage > 0.01).nonzero().squeeze()
        self.n_active_states = self.active_states.numel()
        
        # Log likelihood
        log_likelihood = torch.log(alpha[-1].sum() + 1e-10)
        
        return alpha, beta, log_likelihood
    
    def infer_states(self, observations):
        """
        Infer most likely states using Viterbi algorithm.
        
        Args:
            observations (torch.Tensor): Observation sequence of shape (seq_length, n_features)
            
        Returns:
            tuple: (states, trans_probs)
                states: Most likely state sequence
                trans_probs: Transition probability matrix
        """
        T = observations.shape[0]
        beta_weights = self.stick_breaking(self.beta_logits)
        trans_probs = torch.softmax(self.pi_logits, dim=1)
        
        # Compute emission probabilities
        emission_probs = self.compute_emission_probs(observations)
        
        # Viterbi algorithm
        viterbi = torch.zeros(T, self.max_states, device=self.device)
        ptr = torch.zeros(T, self.max_states, dtype=torch.long, device=self.device)
        viterbi[0] = torch.log(beta_weights + 1e-10) + emission_probs[0]
        
        for t in range(1, T):
            trans = viterbi[t-1].unsqueeze(1) + torch.log(trans_probs + 1e-10)
            viterbi[t], ptr[t] = torch.max(trans, dim=0)
            viterbi[t] += emission_probs[t]
        
        states = torch.zeros(T, dtype=torch.long, device=self.device)
        states[-1] = torch.argmax(viterbi[-1])
        for t in range(T-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]
        
        return states, trans_probs
    
    def posterior_predictive(self, observations):
        """
        Compute posterior predictive distribution for next observation.
        
        Args:
            observations (torch.Tensor): Observation sequence of shape (seq_length, n_features)
            
        Returns:
            torch.Tensor: Mean and covariance of posterior predictive for next observation
        """
        alpha, _, _ = self.forward_backward(observations)
        state_probs = alpha[-1]
        
        # Compute expected mean and covariance
        pred_mean = torch.zeros(self.n_features, device=self.device)
        pred_cov = torch.zeros(self.n_features, self.n_features, device=self.device)
        
        for k in range(self.max_states):
            if state_probs[k] > 1e-3:
                pred_mean += state_probs[k] * self.means[k]
                cov_k = torch.diag(torch.exp(self.log_vars[k]))
                pred_cov += state_probs[k] * cov_k
        
        return pred_mean, pred_cov
    
    def get_active_states_count(self):
        """Get the current count of active states."""
        return self.n_active_states if self.n_active_states > 0 else "Not yet determined"
    
    def save_model(self, path):
        """
        Save model state.
        
        Args:
            path (str): Path to save model state
        """
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'state_dict': self.state_dict(),
            'n_features': self.n_features,
            'max_states': self.max_states,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'active_states': self.active_states,
            'n_active_states': self.n_active_states
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model state.
        
        Args:
            path (str): Path to load model state from
            
        Returns:
            bool: True if model was successfully loaded, False otherwise
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
            self.active_states = checkpoint.get('active_states')
            self.n_active_states = checkpoint.get('n_active_states', 0)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
