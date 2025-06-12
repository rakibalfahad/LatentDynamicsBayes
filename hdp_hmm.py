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
        self.current_states = max_states  # Track current number of active states
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
            for k in range(self.current_states)
        ], dim=1).clone()  # Shape: (T, current_states)
        
        # Forward pass
        alpha = torch.zeros(T, self.current_states, device=self.device)
        alpha_t = (beta_weights[:self.current_states] * torch.exp(emission_probs[0])).clone()
        alpha[0] = alpha_t / (alpha_t.sum() + 1e-10)  # New tensor
        
        for t in range(1, T):
            matmul_result = torch.matmul(alpha[t-1].clone(), trans_probs[:self.current_states, :self.current_states])  # Clone input
            alpha_t = (matmul_result * torch.exp(emission_probs[t])).clone()
            alpha[t] = alpha_t / (alpha_t.sum() + 1e-10)  # New tensor
        
        # Backward pass
        beta = torch.ones(T, self.current_states, device=self.device)
        for t in range(T-2, -1, -1):
            beta_t = torch.matmul(trans_probs[:self.current_states, :self.current_states], 
                                 (beta[t+1].clone() * torch.exp(emission_probs[t+1]))).clone()
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
            for k in range(self.current_states)
        ], dim=1).clone()
        
        viterbi = torch.zeros(T, self.current_states, device=self.device)
        ptr = torch.zeros(T, self.current_states, dtype=torch.long, device=self.device)
        viterbi[0] = torch.log(beta_weights[:self.current_states] + 1e-10) + emission_probs[0]
        
        for t in range(1, T):
            trans = viterbi[t-1].unsqueeze(1) + torch.log(trans_probs[:self.current_states, :self.current_states] + 1e-10)
            viterbi[t], ptr[t] = torch.max(trans, dim=0)
            viterbi[t] = viterbi[t] + emission_probs[t]
        
        states = torch.zeros(T, dtype=torch.long, device=self.device)
        states[-1] = torch.argmax(viterbi[-1])
        for t in range(T-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]
        
        return states, trans_probs[:self.current_states, :self.current_states]
    
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
            'initial_states': self.current_states
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
                
                for k in range(self.current_states):
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
                    emission_probs = torch.stack([
                        torch.distributions.MultivariateNormal(
                            self.means[k],
                            covariance_matrix=torch.diag(torch.exp(self.log_vars[k]) + 1e-6)
                        ).log_prob(observations)
                        for k in range(self.current_states)
                    ], dim=1)
                    
                    # Maximum emission probability for each observation
                    max_emission_probs, _ = torch.max(emission_probs, dim=1)
                    
                    # Average negative log-likelihood (lower = better fit)
                    avg_nll = -torch.mean(max_emission_probs)
                    
                    # If model fit is poor and we have inactive states, add a new state
                    if avg_nll > 10.0 and self.current_states < self.max_states and len(inactive_indices) > 0:
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
                
                # Update current_states count based on active states
                self.current_states = len(active_indices)
                
                # Make sure we always have at least one state
                if self.current_states < 1:
                    self.current_states = 1
                
                # Reorder states for efficiency (all active states at beginning)
                if len(inactive_indices) > 0 and len(active_indices) > 0:
                    try:
                        # Get sorted indices
                        sorted_indices = active_indices + inactive_indices
                        
                        # Create reordering tensors
                        reordered_means = self.means.data.clone()
                        reordered_log_vars = self.log_vars.data.clone()
                        reordered_beta_logits = self.beta_logits.data.clone()
                        reordered_pi_logits = self.pi_logits.data.clone()
                        
                        # Reorder parameters
                        for new_idx, old_idx in enumerate(sorted_indices):
                            if new_idx < self.max_states and old_idx < self.max_states:
                                reordered_means[new_idx] = self.means[old_idx]
                                reordered_log_vars[new_idx] = self.log_vars[old_idx]
                                reordered_beta_logits[new_idx] = self.beta_logits[old_idx]
                                reordered_pi_logits[new_idx] = self.pi_logits[old_idx]
                        
                        # Update parameters
                        self.means.data = reordered_means
                        self.log_vars.data = reordered_log_vars
                        self.beta_logits.data = reordered_beta_logits
                        self.pi_logits.data = reordered_pi_logits
                    except Exception as e:
                        print(f"Error in reordering states: {e}")
                        state_changes['error'] = f"Reordering error: {str(e)}"
                
                # Record final state information
                state_changes['final_states'] = self.current_states
                state_changes['active_states'] = active_indices
                state_changes['inactive_states'] = inactive_indices
                
                return self.current_states, state_changes
                
        except Exception as e:
            print(f"Error in update_states: {e}")
            # If an error occurs, don't change the number of states
            state_changes['error'] = str(e)
            return self.current_states, state_changes
    
    def save_model(self, path):
        """Save model state."""
        checkpoint = {
            'state_dict': self.state_dict(),
            'current_states': self.current_states
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path):
        """Load model state."""
        try:
            checkpoint = torch.load(path)
            # Check if it's the new format (dictionary with 'state_dict') or old format (just state_dict)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
                self.current_states = checkpoint.get('current_states', self.max_states)  # Backward compatibility
            else:
                # Old format - direct state_dict
                self.load_state_dict(checkpoint)
                self.current_states = self.max_states  # Default to max_states for old format
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using initial model parameters.")