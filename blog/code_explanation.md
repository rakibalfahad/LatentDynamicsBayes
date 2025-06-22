---
title: "HDP-HMM Implementation: Understanding the Code"
author: "Your Name"
date: "June 21, 2025"
---

# HDP-HMM Implementation: Technical Deep Dive

This document provides a technical deep dive into the implementation details of our Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM). It's designed for developers and researchers who want to understand the code structure, key algorithms, and implementation choices.

## Core Model Implementation

### HDPHMM Class Structure

The core model is implemented in the `HDPHMM` class in `src/model/hdp_hmm.py`. This class inherits from PyTorch's `nn.Module` to leverage automatic differentiation and GPU acceleration:

```python
class HDPHMM(nn.Module):
    def __init__(self, n_features, max_states=20, alpha=1.0, gamma=1.0, device=None):
        super(HDPHMM, self).__init__()
        self.n_features = n_features
        self.max_states = max_states
        self.alpha = alpha  # DP concentration parameter for transitions
        self.gamma = gamma  # Concentration parameter for stick-breaking process
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize stick-breaking parameters (logits of Beta distribution)
        self.v_logits = nn.Parameter(torch.zeros(max_states, device=self.device))
        
        # Initialize Gaussian emission model parameters
        self.means = nn.Parameter(torch.randn(max_states, n_features, device=self.device))
        self.log_vars = nn.Parameter(torch.zeros(max_states, n_features, device=self.device))
        
        # Current number of active states (starts with 1)
        self.current_states = 1
        
        # State management hyperparameters
        self.delete_threshold = 1e-3
        self.merge_distance = 0.5
        self.birth_threshold = 10.0
```

### Stick-Breaking Process

The stick-breaking process is a key component of the HDP-HMM, generating a probability distribution over potentially infinite states:

```python
def stick_breaking(self):
    """Implement the stick-breaking construction for the HDP."""
    # Sigmoid of the logits gives v_k
    v = torch.sigmoid(self.v_logits[:self.current_states])
    
    # Compute beta weights using the stick-breaking process
    beta_weights = torch.zeros(self.current_states, device=self.device)
    remaining_stick = torch.ones(1, device=self.device)
    
    for k in range(self.current_states):
        beta_weights[k] = v[k] * remaining_stick
        remaining_stick = remaining_stick * (1 - v[k])
    
    return beta_weights
```

### Transition Matrix Computation

The HDP-HMM uses a hierarchical structure to generate the transition matrix:

```python
def compute_transition_matrix(self):
    """Compute the transition probability matrix using the HDP structure."""
    # Get global state weights from stick-breaking
    beta_weights = self.stick_breaking()
    
    # Generate state-specific transition distributions using the DP
    trans_matrix = torch.zeros(self.current_states, self.current_states, device=self.device)
    
    for i in range(self.current_states):
        # For each state, mix between global distribution and self-transition
        # Higher alpha makes transitions more like the global distribution
        self_weight = 1.0 / (1.0 + self.alpha)
        beta_weight = self.alpha / (1.0 + self.alpha)
        
        # Create mixture of global distribution and self-transition
        trans_matrix[i] = beta_weight * beta_weights
        trans_matrix[i, i] += self_weight
        
        # Normalize to ensure valid probability distribution
        trans_matrix[i] = trans_matrix[i] / trans_matrix[i].sum()
    
    return trans_matrix, beta_weights
```

### Emission Model

The emission model uses a multivariate Gaussian to model the observations from each state:

```python
def compute_emission_probs(self, data):
    """
    Compute the emission probabilities p(x_t | z_t = k) for all observations and states.
    Returns a matrix of shape [T, K] where T is the number of observations and K is the number of states.
    """
    T, D = data.shape
    K = self.current_states
    
    # Get means and variances for current states
    means = self.means[:K]
    vars = torch.exp(self.log_vars[:K])  # Convert log variance to variance
    
    # Compute log probability for each observation under each state's Gaussian
    log_probs = torch.zeros(T, K, device=self.device)
    
    for k in range(K):
        # Compute negative Mahalanobis distance (diagonal covariance)
        diff = data - means[k]
        mahalanobis = -0.5 * torch.sum(diff**2 / vars[k], dim=1)
        
        # Add log normalization term
        log_norm = -0.5 * D * torch.log(2 * torch.tensor(math.pi, device=self.device)) - 0.5 * torch.sum(torch.log(vars[k]))
        log_probs[:, k] = mahalanobis + log_norm
    
    return log_probs
```

### Forward-Backward Algorithm

The forward-backward algorithm is implemented to infer the state assignments:

```python
def forward_backward(self, emission_log_probs, trans_matrix):
    """
    Implements the forward-backward algorithm to compute:
    1. Alpha (forward) probabilities
    2. Beta (backward) probabilities
    3. Gamma (posterior state probabilities)
    4. Xi (posterior transition probabilities)
    """
    T, K = emission_log_probs.shape
    log_trans_matrix = torch.log(trans_matrix + 1e-10)
    
    # Initialize alpha and beta
    log_alpha = torch.zeros(T, K, device=self.device)
    log_beta = torch.zeros(T, K, device=self.device)
    
    # Initial state probabilities (uniform)
    log_pi = torch.log(torch.ones(K, device=self.device) / K)
    
    # Forward pass
    log_alpha[0] = log_pi + emission_log_probs[0]
    
    for t in range(1, T):
        # log(sum(exp(log_alpha[t-1] + log_trans_matrix))) + emission_log_probs[t]
        max_val, _ = torch.max(log_alpha[t-1].unsqueeze(1) + log_trans_matrix, dim=0)
        log_alpha[t] = max_val + torch.log(torch.sum(
            torch.exp(log_alpha[t-1].unsqueeze(1) + log_trans_matrix - max_val), dim=0)) + emission_log_probs[t]
    
    # Backward pass
    for t in range(T-2, -1, -1):
        max_val, _ = torch.max(log_trans_matrix + emission_log_probs[t+1] + log_beta[t+1], dim=1)
        log_beta[t] = max_val + torch.log(torch.sum(
            torch.exp(log_trans_matrix + emission_log_probs[t+1] + log_beta[t+1] - max_val.unsqueeze(1)), dim=1))
    
    # Compute gamma (posterior state probabilities)
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
    gamma = torch.exp(log_gamma)
    
    # Compute xi (posterior transition probabilities)
    xi = torch.zeros(T-1, K, K, device=self.device)
    for t in range(T-1):
        xi_t = log_alpha[t].unsqueeze(1) + log_trans_matrix + emission_log_probs[t+1].unsqueeze(0) + log_beta[t+1].unsqueeze(0)
        xi_t = xi_t - torch.logsumexp(xi_t.view(-1), dim=0)
        xi[t] = torch.exp(xi_t)
    
    return gamma, xi
```

### Dynamic State Management

The model includes mechanisms for dynamically adjusting the number of states:

#### Birth Mechanism

```python
def birth_mechanism(self, data, gamma, emission_log_probs):
    """
    Create new states when observations are poorly explained by existing states.
    """
    # Compute negative log likelihood for each observation
    weighted_log_probs = gamma * emission_log_probs
    observation_nlls = -torch.sum(weighted_log_probs, dim=1)
    avg_nll = torch.mean(observation_nlls)
    
    # Check if the average NLL exceeds the threshold and we have room for more states
    if avg_nll > self.birth_threshold and self.current_states < self.max_states:
        # Find the data points with highest negative log-likelihood
        n_samples = min(10, len(observation_nlls))
        worst_indices = torch.topk(observation_nlls, k=n_samples)[1]
        
        # Initialize a new state based on these poorly explained observations
        worst_data = data[worst_indices]
        new_state_mean = torch.mean(worst_data, dim=0)
        new_state_var = torch.var(worst_data, dim=0) + 1e-4  # Add small constant for stability
        
        # Add the new state to the model
        self.means.data[self.current_states] = new_state_mean
        self.log_vars.data[self.current_states] = torch.log(new_state_var)
        self.v_logits.data[self.current_states] = torch.logit(torch.tensor(0.1, device=self.device))
        
        self.current_states += 1
        return True, {"state_idx": self.current_states-1, "mean": new_state_mean.detach().cpu().numpy()}
    
    return False, None
```

#### Merge Mechanism

```python
def merge_mechanism(self, beta_weights):
    """
    Merge states that are too similar to each other.
    """
    merged_pairs = []
    
    # Check pairs of states for potential merging
    for i_idx in range(self.current_states):
        for j_idx in range(i_idx + 1, self.current_states):
            # Compute distance between state means
            dist = torch.norm(self.means[i_idx] - self.means[j_idx])
            
            # If states are close enough, merge them
            if dist < self.merge_distance:
                # Combine parameters by weighted averaging
                weight_i = beta_weights[i_idx]
                weight_j = beta_weights[j_idx]
                total_weight = weight_i + weight_j
                
                # Update parameters of state i
                self.means.data[i_idx] = (weight_i * self.means[i_idx] + 
                                         weight_j * self.means[j_idx]) / total_weight
                self.log_vars.data[i_idx] = torch.log(
                    (weight_i * torch.exp(self.log_vars[i_idx]) + 
                     weight_j * torch.exp(self.log_vars[j_idx])) / total_weight
                )
                
                # Record merged pair
                merged_pairs.append({
                    "source": j_idx, 
                    "target": i_idx,
                    "source_weight": weight_j.item(),
                    "target_weight": weight_i.item()
                })
                
                # Mark state j for deletion (will be handled by delete mechanism)
                beta_weights[j_idx] = 0
    
    return merged_pairs
```

#### Delete Mechanism

```python
def delete_mechanism(self, beta_weights):
    """
    Delete states with negligible probability.
    """
    # Find states with negligible probability
    active_indices = []
    inactive_indices = []
    
    for k in range(self.current_states):
        if beta_weights[k] > self.delete_threshold:
            active_indices.append(k)
        else:
            inactive_indices.append(k)
    
    # If no states to delete, return
    if not inactive_indices:
        return []
    
    # Reorder parameters to keep active states at the beginning
    if active_indices:
        active_indices_tensor = torch.tensor(active_indices, device=self.device)
        
        # Create new tensors with reordered parameters
        new_means = self.means.data[active_indices_tensor].clone()
        new_log_vars = self.log_vars.data[active_indices_tensor].clone()
        new_v_logits = self.v_logits.data[active_indices_tensor].clone()
        
        # Update model parameters
        self.means.data[:len(active_indices)] = new_means
        self.log_vars.data[:len(active_indices)] = new_log_vars
        self.v_logits.data[:len(active_indices)] = new_v_logits
    
    # Update current_states
    self.current_states = len(active_indices)
    
    return [{"state_idx": idx, "weight": beta_weights[idx].item()} for idx in inactive_indices]
```

## Data Processing and Training

### Live Data Collection

Live data collection is handled by the `LiveDataCollector` class:

```python
class LiveDataCollector:
    def __init__(self, n_features, window_size, sample_interval=0.1):
        self.n_features = n_features
        self.window_size = window_size
        self.sample_interval = sample_interval
        self.feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
        # Initialize simulation parameters
        self.state_means = [
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([5.0, 2.0, -3.0]),
            torch.tensor([-2.0, 4.0, 1.0])
        ]
        self.state_vars = [
            torch.tensor([0.5, 0.5, 0.5]),
            torch.tensor([1.0, 0.7, 0.8]),
            torch.tensor([0.8, 1.2, 0.6])
        ]
        self.transition_matrix = torch.tensor([
            [0.95, 0.03, 0.02],
            [0.02, 0.93, 0.05],
            [0.05, 0.05, 0.90]
        ])
        
        # Initial state
        self.current_state = random.randint(0, len(self.state_means) - 1)
        self.data_buffer = collections.deque(maxlen=window_size)
        
    def collect_window(self):
        """Collect a window of data points."""
        # Clear the buffer for a fresh window
        self.data_buffer.clear()
        
        # Generate a window of data
        for _ in range(self.window_size):
            # Generate a data point from the current state
            data_point = torch.normal(
                mean=self.state_means[self.current_state],
                std=torch.sqrt(self.state_vars[self.current_state])
            )
            self.data_buffer.append(data_point)
            
            # Transition to next state
            self.current_state = torch.multinomial(
                self.transition_matrix[self.current_state],
                num_samples=1
            ).item()
            
            # In real implementation, add a sleep to simulate real-time data collection
            time.sleep(self.sample_interval)
        
        # Convert buffer to tensor
        return torch.stack(list(self.data_buffer))
```

### CSV Data Processing

The `CSVDataProcessor` class handles offline batch processing:

```python
class CSVDataProcessor:
    def __init__(self, data_dir, window_size, stride=None):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride or window_size
        self.data = []
        self.current_idx = 0
        self.csv_files = []
    
    def load_csv_files(self):
        """Load all CSV files from the data directory."""
        # Get all CSV files in the directory
        self.csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        # Load each CSV file
        all_data = []
        for file_path in self.csv_files:
            try:
                # Load data without headers
                data = np.loadtxt(file_path, delimiter=',')
                all_data.append(data)
                print(f"Loaded {file_path} with shape {data.shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # Concatenate all data
        if all_data:
            self.data = np.vstack(all_data)
            print(f"Total data shape: {self.data.shape}")
        else:
            raise ValueError("No data could be loaded from CSV files")
    
    def get_next_window(self):
        """Get the next window of data using the specified stride."""
        if self.current_idx + self.window_size > len(self.data):
            return None  # End of data
        
        # Extract window
        window_data = self.data[self.current_idx:self.current_idx + self.window_size]
        
        # Move to next position
        self.current_idx += self.stride
        
        return torch.tensor(window_data, dtype=torch.float32)
```

### Model Training

The `ModelTrainer` class handles model training and inference:

```python
class ModelTrainer:
    def __init__(self, n_features, max_states=20, alpha=1.0, gamma=1.0, learning_rate=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HDPHMM(n_features, max_states, alpha, gamma, device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.losses = []
        self.state_changes = []
    
    def update_model(self, window_data):
        """Train the model on a window of data."""
        # Convert data to tensor if needed
        if not isinstance(window_data, torch.Tensor):
            window_data = torch.tensor(window_data, dtype=torch.float32)
        
        # Move data to device
        window_data = window_data.to(self.device)
        
        # Compute emission probabilities
        emission_log_probs = self.model.compute_emission_probs(window_data)
        
        # Compute transition matrix
        trans_matrix, beta_weights = self.model.compute_transition_matrix()
        
        # Run forward-backward
        gamma, xi = self.model.forward_backward(emission_log_probs, trans_matrix)
        
        # Compute loss (negative log likelihood)
        loss = -torch.sum(gamma * emission_log_probs)
        
        # Record initial state count
        state_change = {"initial_states": self.model.current_states}
        
        # Dynamic state management
        birthed = []
        merged = []
        deleted = []
        
        # Birth: Create new states if needed
        birth_happened, birth_info = self.model.birth_mechanism(window_data, gamma, emission_log_probs)
        if birth_happened:
            birthed.append(birth_info)
        
        # Recompute transition matrix if needed
        if birth_happened:
            trans_matrix, beta_weights = self.model.compute_transition_matrix()
        
        # Merge: Combine similar states
        merged = self.model.merge_mechanism(beta_weights)
        
        # Delete: Remove states with negligible probability
        deleted = self.model.delete_mechanism(beta_weights)
        
        # Record state changes
        state_change.update({
            "birthed": birthed,
            "merged": merged,
            "deleted": deleted,
            "final_states": self.model.current_states
        })
        self.state_changes.append(state_change)
        
        # If state count changed, recompute everything
        if birthed or merged or deleted:
            # Recompute emission probabilities and transition matrix
            emission_log_probs = self.model.compute_emission_probs(window_data)
            trans_matrix, _ = self.model.compute_transition_matrix()
            gamma, xi = self.model.forward_backward(emission_log_probs, trans_matrix)
            loss = -torch.sum(gamma * emission_log_probs)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        self.losses.append(loss.item())
        
        return loss.item()
    
    def infer(self, window_data):
        """Perform inference on a window of data."""
        # Convert data to tensor if needed
        if not isinstance(window_data, torch.Tensor):
            window_data = torch.tensor(window_data, dtype=torch.float32)
        
        # Move data to device
        window_data = window_data.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Compute emission probabilities
            emission_log_probs = self.model.compute_emission_probs(window_data)
            
            # Compute transition matrix
            trans_matrix, _ = self.model.compute_transition_matrix()
            
            # Run forward-backward
            gamma, _ = self.model.forward_backward(emission_log_probs, trans_matrix)
            
            # Get most likely state for each time step
            states = torch.argmax(gamma, dim=1)
        
        # Set model back to training mode
        self.model.train()
        
        return states, trans_matrix
```

## Visualization and Analysis

### Live Visualization

The visualization system provides real-time feedback on the model's performance:

```python
class LiveVisualizer:
    def __init__(self, n_features, window_size):
        self.n_features = n_features
        self.window_size = window_size
        self.fig = plt.figure(figsize=(12, 8))
        self.window_count = 0
        self.state_history = []
        
        # Create output directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
    
    def update_plot(self, data, states, trans_probs, loss, losses, state_counts=None, state_changes=None):
        """Update all visualizations with new data."""
        self.window_count += 1
        self.state_history.append(states.cpu())
        
        # Convert data to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(trans_probs, torch.Tensor):
            trans_probs = trans_probs.cpu().numpy()
        
        # Clear figure for new plots
        self.fig.clear()
        
        # Number of plots
        n_plots = self.n_features + 2
        if state_counts:
            n_plots += 1
        
        # Create the plots
        gs = GridSpec(n_plots, 1, figure=self.fig)
        
        # Plot each feature with state coloring
        for i in range(self.n_features):
            ax = self.fig.add_subplot(gs[i, 0])
            ax.plot(data[:, i], alpha=0.7)
            ax.scatter(range(len(states)), data[:, i], c=states, cmap='tab10', s=20)
            ax.set_ylabel(f'Feature {i+1}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(data))
        
        # Plot transition matrix
        ax_trans = self.fig.add_subplot(gs[self.n_features, 0])
        sns.heatmap(trans_probs, ax=ax_trans, cmap='viridis', 
                   vmin=0, vmax=1, annot=True, fmt='.2f', square=True)
        ax_trans.set_title('Transition Probabilities')
        
        # Plot learning curve
        ax_loss = self.fig.add_subplot(gs[self.n_features + 1, 0])
        ax_loss.plot(losses)
        ax_loss.set_xlabel('Window')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Current Loss: {loss:.4f}')
        ax_loss.grid(True, alpha=0.3)
        
        # Plot state counts if provided
        if state_counts:
            ax_states = self.fig.add_subplot(gs[self.n_features + 2, 0])
            ax_states.plot(state_counts)
            ax_states.set_xlabel('Window')
            ax_states.set_ylabel('Number of States')
            ax_states.set_title('State Count Evolution')
            ax_states.grid(True, alpha=0.3)
        
        # Adjust layout and save
        self.fig.tight_layout()
        self.fig.savefig(f'plots/live_plot_window_{self.window_count}.png')
        
        # Create additional visualizations periodically
        if self.window_count % 5 == 0:
            self.create_state_patterns_visualization(data, states)
            if state_changes:
                self.create_state_evolution_plot(state_changes)
```

### State Pattern Visualization

This visualization shows what patterns each state represents:

```python
def create_state_patterns_visualization(self, data, states):
    """Create a visualization showing what patterns each state represents."""
    # Get unique states
    unique_states = np.unique(states)
    n_states = len(unique_states)
    
    if n_states == 0:
        return
    
    # Create figure
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3*n_states), sharex=True)
    if n_states == 1:
        axes = [axes]
    
    # For each state, calculate statistics
    for i, state in enumerate(unique_states):
        mask = states == state
        state_data = data[mask]
        
        if len(state_data) == 0:
            continue
        
        # Calculate statistics
        mean_pattern = np.mean(state_data, axis=0)
        std_pattern = np.std(state_data, axis=0)
        min_pattern = np.min(state_data, axis=0)
        max_pattern = np.max(state_data, axis=0)
        
        # Feature indices for x-axis
        feature_indices = np.arange(self.n_features)
        
        # Plot mean and standard deviation
        axes[i].errorbar(feature_indices, mean_pattern, yerr=std_pattern, 
                        fmt='o-', capsize=5, label='Mean ± Std Dev')
        
        # Plot min/max range
        axes[i].fill_between(feature_indices, min_pattern, max_pattern, 
                            alpha=0.2, label='Min-Max Range')
        
        # Count occurrences and calculate typical duration
        state_count = np.sum(mask)
        runs = self._find_runs(states == state)
        mean_duration = np.mean([end-start for start, end in runs]) if runs else 0
        
        axes[i].set_title(f'State {state}: {state_count} occurrences ({state_count/len(states):.1%}), '
                         f'Avg Duration: {mean_duration:.1f} steps')
        axes[i].set_xticks(feature_indices)
        axes[i].set_xticklabels([f'Feature {i+1}' for i in range(self.n_features)])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'plots/state_patterns_window_{self.window_count}.png')
    plt.close(fig)
```

## Conclusion

This document provides a technical deep dive into the HDP-HMM implementation. The implementation leverages PyTorch for efficient computation and automatic differentiation, while incorporating sophisticated state management mechanisms to dynamically adjust the model complexity.

The code is structured to support both live streaming and offline batch processing, making it versatile for various applications. The visualization system provides insightful views of the model's behavior, helping users understand the discovered patterns.

For any questions or contributions, please visit our GitHub repository.
