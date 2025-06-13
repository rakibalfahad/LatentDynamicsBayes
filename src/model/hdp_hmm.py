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
        exp_emission_probs = torch.exp(emission_probs)  # Pre-compute exponential once
        
        # Forward pass
        alpha = torch.zeros(T, self.max_states, device=self.device)
        
        # First time step
        alpha_0 = beta_weights * exp_emission_probs[0]
        alpha_0_sum = alpha_0.sum() + 1e-10
        alpha[0] = alpha_0 / alpha_0_sum
        
        # Remaining time steps
        for t in range(1, T):
            # Clone to avoid in-place modification of computational graph
            prev_alpha = alpha[t-1].clone()
            
            # Matrix multiplication and emission probability
            alpha_t = torch.matmul(prev_alpha, trans_probs) * exp_emission_probs[t]
            alpha_t_sum = alpha_t.sum() + 1e-10
            alpha[t] = alpha_t / alpha_t_sum
        
        # Backward pass
        beta = torch.ones(T, self.max_states, device=self.device)
        
        for t in range(T-2, -1, -1):
            # Pre-compute the emission probability part
            next_beta_emission = beta[t+1] * exp_emission_probs[t+1]
            
            # Matrix multiplication
            beta_t = torch.matmul(trans_probs, next_beta_emission)
            beta_t_sum = beta_t.sum() + 1e-10
            beta[t] = beta_t / beta_t_sum
        
        # Compute posterior state probabilities
        gamma = alpha * beta
        gamma_sum = gamma.sum(dim=1, keepdim=True) + 1e-10
        gamma = gamma / gamma_sum
        
        # Update active states (states with significant posterior probability)
        state_usage = gamma.sum(dim=0)
        # Use non-inplace operation for modifying states
        active_states_mask = (state_usage > 0.01)
        self.active_states = active_states_mask.nonzero().squeeze()
        self.n_active_states = self.active_states.numel()
        
        # Log likelihood - avoid potential in-place operation
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
            # Log transition probs for numerical stability
            log_trans_probs = torch.log(trans_probs + 1e-10)
            
            # Calculate transition scores
            trans = viterbi[t-1].unsqueeze(1) + log_trans_probs
            
            # Get max values and indices
            max_vals, max_indices = torch.max(trans, dim=0)
            
            # Store values and pointers
            viterbi[t] = max_vals + emission_probs[t]
            ptr[t] = max_indices
        
        # Backtrack to find most likely state sequence
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
        state_probs = alpha[-1].clone()  # Clone to prevent modifying original tensor
        
        # Compute expected mean and covariance
        pred_mean = torch.zeros(self.n_features, device=self.device)
        pred_cov = torch.zeros(self.n_features, self.n_features, device=self.device)
        
        # Accumulate weighted contributions from each state
        for k in range(self.max_states):
            if state_probs[k] > 1e-3:
                # Add weighted mean
                state_mean_contribution = state_probs[k] * self.means[k]
                pred_mean = pred_mean + state_mean_contribution
                
                # Add weighted covariance
                cov_k = torch.diag(torch.exp(self.log_vars[k]))
                state_cov_contribution = state_probs[k] * cov_k
                pred_cov = pred_cov + state_cov_contribution
        
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
    
    def update_states(self, observations):
        """
        Dynamically adjust the number of states with birth, merge, and delete mechanisms.
        
        Args:
            observations: Tensor of shape (seq_length, n_features)
            
        Returns:
            tuple: (int: New number of states, dict: State change information)
        """
        # Initialize state change tracking dict
        state_changes = {
            'deleted': [],
            'merged': [],
            'birthed': [],
            'initial_states': self.n_active_states
        }
        
        try:
            with torch.no_grad():
                # Get current model state
                beta_weights = self.stick_breaking(self.beta_logits)
                alpha, beta, _ = self.forward_backward(observations)
                states, trans_probs = self.infer_states(observations)
                
                # 1. DELETE: Remove states with probability below threshold
                threshold = 1e-3
                active_indices = []
                inactive_indices = []
                
                for k in range(self.max_states):
                    if beta_weights[k] > threshold:
                        active_indices.append(k)
                    else:
                        inactive_indices.append(k)
                        state_changes['deleted'].append(k)
                
                # 2. MERGE: Combine similar states
                merge_distance = 0.5  # Threshold for merging
                merged_indices = set()
                
                # Create a copy of active_indices to avoid modification during iteration
                active_indices_copy = active_indices.copy()
                
                i = 0
                while i < len(active_indices_copy):
                    if i in merged_indices:
                        i += 1
                        continue
                        
                    i_idx = active_indices_copy[i]
                    
                    j = i + 1
                    while j < len(active_indices_copy):
                        if j in merged_indices:
                            j += 1
                            continue
                            
                        j_idx = active_indices_copy[j]
                        # Calculate distance between state means
                        dist = torch.norm(self.means[i_idx] - self.means[j_idx])
                        
                        if dist < merge_distance:
                            # Merge j into i by weight averaging
                            weight_i = beta_weights[i_idx]
                            weight_j = beta_weights[j_idx]
                            total_weight = weight_i + weight_j
                            
                            # Update parameters of state i (weighted average)
                            self.means.data[i_idx] = (weight_i * self.means[i_idx] + weight_j * self.means[j_idx]) / total_weight
                            self.log_vars.data[i_idx] = (weight_i * self.log_vars[i_idx] + weight_j * self.log_vars[j_idx]) / total_weight
                            
                            # Update beta logits and transition logits
                            self.beta_logits.data[i_idx] = torch.log(total_weight / (1 - total_weight))
                            
                            # Update transition logits (weighted average)
                            pi_i = torch.softmax(self.pi_logits[i_idx], dim=0)
                            pi_j = torch.softmax(self.pi_logits[j_idx], dim=0)
                            self.pi_logits.data[i_idx] = torch.log((weight_i * pi_i + weight_j * pi_j) / total_weight + 1e-10)
                            
                            # Mark j as merged
                            merged_indices.add(j)
                            inactive_indices.append(j_idx)
                            if j_idx in active_indices:
                                active_indices.remove(j_idx)
                            
                            # Record merge
                            state_changes['merged'].append((j_idx, i_idx))
                        
                        j += 1
                    
                    i += 1
                
                # 3. BIRTH: Add new states if needed
                try:
                    # Calculate negative log-likelihood for each observation
                    emission_probs = self.compute_emission_probs(observations)
                    
                    # Maximum emission probability for each observation
                    max_emission_probs, _ = torch.max(emission_probs, dim=1)
                    
                    # Average negative log-likelihood (lower = better fit)
                    avg_nll = -torch.mean(max_emission_probs)
                    
                    # If model fit is poor and we have inactive states, add a new state
                    if avg_nll > 10.0 and len(active_indices) < self.max_states and len(inactive_indices) > 0:
                        # Find observations with poorest fit
                        poor_fit_mask = max_emission_probs < torch.quantile(max_emission_probs, 0.1)
                        poor_fit_obs = observations[poor_fit_mask]
                        
                        if len(poor_fit_obs) > 0:
                            # Use an inactive state for the new state
                            new_state_idx = inactive_indices[0]
                            inactive_indices.pop(0)
                            
                            # Initialize with mean and variance of poorly fit observations
                            if len(poor_fit_obs) > 1:
                                self.means.data[new_state_idx] = torch.mean(poor_fit_obs, dim=0)
                                self.log_vars.data[new_state_idx] = torch.log(torch.var(poor_fit_obs, dim=0) + 1e-6)
                            else:
                                # If only one observation, use it directly and default variance
                                self.means.data[new_state_idx] = poor_fit_obs[0]
                                self.log_vars.data[new_state_idx] = torch.zeros_like(self.log_vars[0])
                            
                            # Add to active indices with small weight
                            active_indices.append(new_state_idx)
                            # Set beta logit to small value
                            self.beta_logits.data[new_state_idx] = torch.log(torch.tensor(0.05 / 0.95))
                            
                            # Initialize transition probabilities uniformly
                            self.pi_logits.data[new_state_idx] = torch.zeros_like(self.pi_logits[0])
                            
                            # Record birth
                            state_changes['birthed'].append(new_state_idx)
                except Exception as e:
                    print(f"Error in birth mechanism: {e}")
                    state_changes['error'] = f"Birth mechanism error: {str(e)}"
                
                # Update active_states and n_active_states
                self.active_states = torch.tensor(active_indices, device=self.device)
                self.n_active_states = len(active_indices)
                
                # Make sure we always have at least one state
                if self.n_active_states < 1:
                    self.n_active_states = 1
                    self.active_states = torch.tensor([0], device=self.device)
                
                # Record final state information
                state_changes['final_states'] = self.n_active_states
                state_changes['active_states'] = active_indices
                state_changes['inactive_states'] = inactive_indices
                
                return self.n_active_states, state_changes
                
        except Exception as e:
            print(f"Error in update_states: {e}")
            # If an error occurs, don't change the number of states
            state_changes['error'] = str(e)
            return self.n_active_states, state_changes
