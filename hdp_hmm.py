import torch
import torch.nn as nn
import torch.distributions as dist

class HDPHMM(nn.Module):
    def __init__(self, n_features, max_states=20, alpha=1.0, gamma=1.0):
        """
        HDP-HMM with stick-breaking construction.
        
        Args:
            n_features (int): Number of features
            max_states (int): Maximum number of states to consider
            alpha (float): Concentration parameter for transition Dirichlet process
            gamma (float): Concentration parameter for top-level Dirichlet process
        """
        super(HDPHMM, self).__init__()
        self.n_features = n_features
        self.max_states = max_states
        self.alpha = alpha
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parameters for stick-breaking process
        self.beta_logits = nn.Parameter(torch.randn(max_states))
        self.pi_logits = nn.Parameter(torch.randn(max_states, max_states))
        
        # Emission parameters (Gaussian)
        self.means = nn.Parameter(torch.randn(max_states, n_features))
        self.log_vars = nn.Parameter(torch.zeros(max_states, n_features))
        
    def stick_breaking(self, logits):
        """Compute stick-breaking weights."""
        betas = torch.sigmoid(logits)
        beta_cumprod = torch.cumprod(1 - betas, dim=0)
        weights = betas * torch.cat([torch.ones(1, device=self.device), beta_cumprod[:-1]])
        return weights
    
    def forward_backward(self, observations):
        """
        Forward-backward algorithm for inference.
        
        Args:
            observations: Tensor of shape (seq_length, n_features)
        
        Returns:
            alpha: Forward probabilities
            beta: Backward probabilities
            log_likelihood: Log likelihood of observations
        """
        T = observations.shape[0]
        beta_weights = self.stick_breaking(self.beta_logits).clone()  # Clone to avoid inplace
        trans_probs = torch.softmax(self.pi_logits, dim=1).clone()  # Clone to avoid inplace
        
        # Emission probabilities (Gaussian)
        emission_probs = torch.stack([
            dist.MultivariateNormal(
                self.means[k],
                covariance_matrix=torch.diag(torch.exp(self.log_vars[k]) + 1e-6)
            ).log_prob(observations)
            for k in range(self.max_states)
        ], dim=1).clone()  # Shape: (T, max_states)
        
        # Forward pass
        alpha = torch.zeros(T, self.max_states, device=self.device)
        alpha_t = (beta_weights * torch.exp(emission_probs[0])).clone()
        alpha[0] = alpha_t / (alpha_t.sum() + 1e-10)  # New tensor
        
        for t in range(1, T):
            matmul_result = torch.matmul(alpha[t-1].clone(), trans_probs)  # Clone input
            alpha_t = (matmul_result * torch.exp(emission_probs[t])).clone()
            alpha[t] = alpha_t / (alpha_t.sum() + 1e-10)  # New tensor
        
        # Backward pass
        beta = torch.ones(T, self.max_states, device=self.device)
        for t in range(T-2, -1, -1):
            beta_t = torch.matmul(trans_probs, (beta[t+1].clone() * torch.exp(emission_probs[t+1]))).clone()
            beta[t] = beta_t / (beta_t.sum() + 1e-10)  # New tensor
        
        # Log likelihood
        log_likelihood = torch.log(alpha[-1].sum() + 1e-10)
        
        return alpha, beta, log_likelihood
    
    def infer_states(self, observations):
        """Infer most likely states using Viterbi algorithm."""
        T = observations.shape[0]
        beta_weights = self.stick_breaking(self.beta_logits).clone()
        trans_probs = torch.softmax(self.pi_logits, dim=1).clone()
        
        emission_probs = torch.stack([
            dist.MultivariateNormal(
                self.means[k],
                covariance_matrix=torch.diag(torch.exp(self.log_vars[k]) + 1e-6)
            ).log_prob(observations)
            for k in range(self.max_states)
        ], dim=1).clone()
        
        viterbi = torch.zeros(T, self.max_states, device=self.device)
        ptr = torch.zeros(T, self.max_states, dtype=torch.long, device=self.device)
        viterbi[0] = torch.log(beta_weights + 1e-10) + emission_probs[0]
        
        for t in range(1, T):
            trans = viterbi[t-1].unsqueeze(1) + torch.log(trans_probs + 1e-10)
            viterbi[t], ptr[t] = torch.max(trans, dim=0)
            viterbi[t] = viterbi[t] + emission_probs[t]
        
        states = torch.zeros(T, dtype=torch.long, device=self.device)
        states[-1] = torch.argmax(viterbi[-1])
        for t in range(T-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]
        
        return states, trans_probs
    
    def save_model(self, path):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """Load model state."""
        self.load_state_dict(torch.load(path))