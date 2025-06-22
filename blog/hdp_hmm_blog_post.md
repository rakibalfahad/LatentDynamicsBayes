---
title: "Discovering Hidden Patterns: Implementing HDP-HMM for Time Series Analysis"
author: "Rakib AL Fahad"
date: "June 21, 2025"
---

# Discovering Hidden Patterns: Implementing HDP-HMM for Time Series Analysis

Time series data is ubiquitous in fields ranging from finance and healthcare to industrial monitoring and IoT applications. One of the key challenges in time series analysis is discovering hidden patterns and states that drive the observed data without knowing in advance how many such states exist. This is where the Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) comes into play.

In this blog post, I'll introduce my open-source implementation of HDP-HMM for time series analysis, which offers both live streaming capabilities and offline batch processing. I'll cover the theoretical foundations, the mathematical formulation, implementation details, and practical examples of how to use this powerful tool.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
   - [Hidden Markov Models](#hidden-markov-models)
   - [The Dirichlet Process](#the-dirichlet-process)
   - [Hierarchical Dirichlet Process](#hierarchical-dirichlet-process)
   - [The HDP-HMM Model](#the-hdp-hmm-model)
   - [Stick-Breaking Construction](#stick-breaking-construction)

2. [Implementation Details](#implementation-details)
   - [Core Architecture](#core-architecture)
   - [Dynamic State Management](#dynamic-state-management)
   - [PyTorch Acceleration](#pytorch-acceleration)
   - [Dual-Mode Operation](#dual-mode-operation)

3. [Code Examples](#code-examples)
   - [Live Streaming Mode](#live-streaming-mode)
   - [Offline CSV Processing](#offline-csv-processing)
   - [Visualization Suite](#visualization-suite)

4. [Visualizing Results](#visualizing-results)
   - [State Patterns](#state-patterns)
   - [Transition Dynamics](#transition-dynamics)
   - [State Evolution](#state-evolution)

5. [Practical Applications](#practical-applications)
   - [System Monitoring](#system-monitoring)
   - [Behavioral Analysis](#behavioral-analysis)
   - [Anomaly Detection](#anomaly-detection)

6. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Quick Start Guide](#quick-start-guide)
   - [Advanced Configuration](#advanced-configuration)

## Theoretical Background

### Hidden Markov Models

At the core of our approach lies the Hidden Markov Model (HMM), a statistical model where the system being modeled is assumed to be a Markov process with unobservable (hidden) states. In a traditional HMM, we have:

- A set of hidden states: $S = \{s_1, s_2, ..., s_K\}$
- Transition probabilities: $A = \{a_{ij}\}$ where $a_{ij} = P(z_t = j | z_{t-1} = i)$
- Emission probabilities: $B = \{b_i(o_t)\}$ where $b_i(o_t) = P(o_t | z_t = i)$
- Initial state distribution: $\pi = \{\pi_i\}$ where $\pi_i = P(z_1 = i)$

The limitation of traditional HMMs is that they require specifying the number of hidden states $K$ in advance, which is often unknown in real-world applications.

### The Dirichlet Process

The Dirichlet Process (DP) is a distribution over distributions. A draw $G \sim DP(\alpha, H)$ from a DP with concentration parameter $\alpha$ and base distribution $H$ is itself a distribution. Formally, for any finite partition $(A_1, A_2, ..., A_r)$ of the space:

$$(G(A_1), G(A_2), ..., G(A_r)) \sim \text{Dirichlet}(\alpha H(A_1), \alpha H(A_2), ..., \alpha H(A_r))$$

The DP enables us to create models with a potentially infinite number of components, addressing the limitation of fixed-state HMMs.

### Hierarchical Dirichlet Process

The Hierarchical Dirichlet Process (HDP) extends the DP by allowing multiple DPs to share components through a hierarchical structure:

$$G_0 \sim DP(\gamma, H)$$
$$G_j \sim DP(\alpha, G_0) \text{ for } j = 1, 2, ...$$

This hierarchical structure is crucial for HMMs because it allows different states to share the same set of possible next states while maintaining different transition probabilities.

### The HDP-HMM Model

In the HDP-HMM, we use the HDP to define the transition distributions of an HMM with a potentially infinite number of states:

1. Draw a global distribution over states: $\beta \sim GEM(\gamma)$
2. For each state $i$, draw a transition distribution: $\pi_i \sim DP(\alpha, \beta)$
3. For each time step $t$:
   - Draw state: $z_t \sim \pi_{z_{t-1}}$
   - Draw observation: $x_t \sim F(\theta_{z_t})$

Where:
- $GEM(\gamma)$ is the stick-breaking distribution with parameter $\gamma$
- $F(\theta)$ is the emission distribution with parameters $\theta$

This formulation allows the model to determine the appropriate number of states from the data itself.

### Stick-Breaking Construction

The stick-breaking construction provides a concrete representation of draws from a DP. For a DP with concentration parameter $\alpha$, the stick-breaking process generates weights $\beta_k$ as follows:

1. Draw $v_k \sim \text{Beta}(1, \alpha)$ for $k = 1, 2, ...$
2. Set $\beta_k = v_k \prod_{i=1}^{k-1} (1 - v_i)$

This can be visualized as breaking a stick of unit length: $v_k$ represents the proportion of the remaining stick that is broken off at step $k$. In our implementation, we parameterize this process to enable learning through gradient-based optimization.

## Implementation Details

### Core Architecture

My implementation of the HDP-HMM is built around several core components:

1. **HDPHMM Class**: The main model class implementing the HDP-HMM with stick-breaking construction
2. **LiveHDPHMM**: A wrapper for live streaming data processing and incremental training
3. **CSVDataProcessor**: Handles offline processing of CSV files for batch analysis
4. **LiveVisualizer**: Real-time visualization of model outputs and data

Here's a simplified class diagram showing the relationships:

```
┌───────────────┐       ┌───────────────┐
│     HDPHMM    │◄──────┤  LiveHDPHMM   │
└───────────────┘       └───────┬───────┘
                               △
                               │
                  ┌────────────┴────────────┐
     ┌────────────┴────────────┐            │
┌────┴─────┐              ┌────┴─────────┐  │
│LiveData  │              │CSVData       │  │
│Collector │              │Processor     │  │
└──────────┘              └──────────────┘  │
                                            │
                                      ┌─────┴──────┐
                                      │LiveVisualizer│
                                      └─────────────┘
```

### Dynamic State Management

One of the key innovations in my implementation is the dynamic state management system, which includes birth, merge, and delete mechanisms:

**Birth Mechanism**:
```python
def birth_mechanism(self, data, states, neg_log_likelihood):
    """Create new states when observations are poorly explained by existing states."""
    if avg_nll > self.birth_threshold and self.current_states < self.max_states:
        # Find the data points with highest negative log-likelihood
        worst_indices = torch.topk(
            neg_log_likelihood, k=min(10, len(neg_log_likelihood)))[1]
        
        # Initialize a new state based on these poorly explained observations
        worst_data = data[worst_indices]
        new_state_mean = torch.mean(worst_data, dim=0)
        # Add small constant for stability
        new_state_var = torch.var(worst_data, dim=0) + 1e-4
        
        # Add the new state to the model
        self.add_state(new_state_mean, new_state_var)
        return True
    return False
```

**Merge Mechanism**:
```python
def merge_mechanism(self, beta_weights):
    """Merge states that are too similar to each other."""
    merged_pairs = []
    for i_idx in range(self.current_states):
        for j_idx in range(i_idx + 1, self.current_states):
            # Check distance between state means
            dist = torch.norm(self.means[i_idx] - self.means[j_idx])
            if dist < self.merge_distance:
                # Combine parameters by weighted averaging
                weight_i = beta_weights[i_idx]
                weight_j = beta_weights[j_idx]
                total_weight = weight_i + weight_j
                
                # Update parameters of state i
                self.means.data[i_idx] = (weight_i * self.means[i_idx] + 
                                         weight_j * self.means[j_idx]) / total_weight
                merged_pairs.append((j_idx, i_idx))  # (source, destination)
                
                # Mark state j for deletion (will be handled by delete mechanism)
                beta_weights[j_idx] = 0
    return merged_pairs
```

**Delete Mechanism**:
```python
def delete_mechanism(self, beta_weights):
    """Delete states with negligible probability."""
    # Find states with negligible probability
    active_indices = []
    inactive_indices = []
    
    for k in range(self.current_states):
        if beta_weights[k] > self.delete_threshold:
            active_indices.append(k)
        else:
            inactive_indices.append(k)
    
    # Update model to use only active states
    if inactive_indices:
        self.current_states = len(active_indices)
        # Reorder parameters to keep active states at the beginning
        self._reorder_parameters(active_indices)
        
    return inactive_indices
```

These mechanisms work together to maintain an optimal number of states that best explains the data.

### PyTorch Acceleration

The implementation leverages PyTorch for GPU acceleration and automatic differentiation:

```python
class HDPHMM(nn.Module):
    def __init__(self, n_features, max_states=20, device=None):
        super(HDPHMM, self).__init__()
        self.n_features = n_features
        self.max_states = max_states
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Stick-breaking parameters (logits of Beta distribution)
        self.v_logits = nn.Parameter(torch.zeros(max_states, device=self.device))
        
        # Emission model parameters (Gaussian)
        self.means = nn.Parameter(torch.randn(max_states, n_features, device=self.device))
        self.log_vars = nn.Parameter(torch.zeros(max_states, n_features, device=self.device))
        
        # Current number of active states
        self.current_states = 1  # Start with 1 active state
```

All computations, including the forward-backward algorithm and stick-breaking process, are implemented as tensor operations for efficient GPU utilization.

### Dual-Mode Operation

A key feature of my implementation is its ability to operate in both live streaming and offline batch processing modes:

**Live Mode**:
```python
# In main.py
data_source = LiveDataCollector(n_features, window_size, sample_interval)
for i in range(max_iterations):
    window_data = data_source.collect_window()
    loss = trainer.update_model(window_data)
    states, trans_probs = trainer.infer(window_data)
    visualizer.update_plot(window_data, states, trans_probs, loss, trainer.losses)
```

**Offline Mode**:
```python
# In main.py
data_source = CSVDataProcessor(args.data_dir, window_size, args.stride)
data_source.load_csv_files()
for i in range(max_iterations):
    window_data = data_source.get_next_window()
    if window_data is None:  # End of files
        break
    loss = trainer.update_model(window_data)
    states, trans_probs = trainer.infer(window_data)
    visualizer.update_plot(window_data, states, trans_probs, loss, trainer.losses)
```

This dual-mode capability makes the implementation versatile for both real-time monitoring and retrospective analysis.

## Code Examples

### Live Streaming Mode

To run the model in live streaming mode with simulated data:

```python
# Basic usage with default settings
python main.py

# Customize window size and feature count
python main.py --window-size 200 --n-features 5

# Limit the number of windows to process
python main.py --max-windows 500
```

The model will:
1. Generate simulated data or collect real system metrics
2. Process the data in sliding windows
3. Incrementally update the model parameters
4. Dynamically adjust the number of states
5. Visualize the results in real-time

### Offline CSV Processing

For offline processing of historical data stored in CSV files:

```python
# Generate sample data (optional)
python generate_sample_data.py

# Process CSV files with default settings
python main.py --data-dir data

# Customize window size and stride for overlapping windows
python main.py --data-dir data --window-size 100 --stride 25

# Run in headless mode (for servers without display)
python main.py --data-dir data --no-gui
```

The CSV files should be structured with:
- Each column representing a feature
- Each row representing a time step
- No header row (first row is treated as data)

Example CSV content:
```
0.5,1.2,0.8
0.6,1.3,0.7
0.7,1.4,0.6
...
```

### Visualization Suite

The implementation includes a comprehensive visualization suite:

```python
class LiveVisualizer:
    def __init__(self, n_features, window_size):
        self.n_features = n_features
        self.window_size = window_size
        self.fig = plt.figure(figsize=(12, 8))
        self.window_count = 0
        self.state_history = []
        
    def update_plot(self, data, states, trans_probs, loss, losses, state_counts=[], state_changes=None):
        """Update all visualizations with new data."""
        self.window_count += 1
        self.state_history.append(states.cpu())
        
        # Clear figure for new plots
        self.fig.clear()
        
        # Plot time series with state assignments
        for i in range(self.n_features):
            ax = self.fig.add_subplot(self.n_features + 2, 1, i + 1)
            ax.plot(data[:, i], label=f'Feature {i+1}')
            ax.scatter(range(len(states)), data[:, i], c=states, cmap='plasma', marker='x')
            ax.set_ylabel(f'Feature {i+1}')
            ax.grid(True, alpha=0.3)
        
        # Plot transition matrix
        ax_trans = self.fig.add_subplot(self.n_features + 2, 1, self.n_features + 1)
        sns.heatmap(trans_probs.cpu().numpy(), ax=ax_trans, cmap='viridis', 
                   vmin=0, vmax=1, annot=True, fmt='.2f')
        ax_trans.set_title('Transition Probabilities')
        
        # Plot learning curve
        ax_loss = self.fig.add_subplot(self.n_features + 2, 1, self.n_features + 2)
        ax_loss.plot(losses)
        ax_loss.set_xlabel('Window')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Current Loss: {loss:.4f}')
        ax_loss.grid(True, alpha=0.3)
        
        # Adjust layout and save
        self.fig.tight_layout()
        self.fig.savefig(f'plots/live_plot_window_{self.window_count}.png')
        
        # Create additional visualizations periodically
        if self.window_count % 5 == 0:
            self.create_tile_visualization()
            self.create_state_evolution_plot(state_changes)
```

These visualizations help in understanding the latent structure discovered by the model.

## Visualizing Results

### State Patterns

One of the most valuable visualizations is the state pattern analysis, which shows what patterns each state represents:

```python
def visualize_state_patterns(self, data, states, save_path=None):
    """Create a visualization showing what patterns each state represents."""
    # Convert to numpy for easier manipulation
    data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
    
    # Get unique states
    unique_states = np.unique(states_np)
    n_states = len(unique_states)
    n_features = data_np.shape[1]
    
    # Create figure
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3*n_states), sharex=True)
    if n_states == 1:
        axes = [axes]
    
    # For each state, calculate statistics
    for i, state in enumerate(unique_states):
        mask = states_np == state
        state_data = data_np[mask]
        
        # Calculate statistics
        mean_pattern = np.mean(state_data, axis=0)
        std_pattern = np.std(state_data, axis=0)
        min_pattern = np.min(state_data, axis=0)
        max_pattern = np.max(state_data, axis=0)
        
        # Feature indices for x-axis
        feature_indices = np.arange(n_features)
        
        # Plot mean and standard deviation
        axes[i].errorbar(feature_indices, mean_pattern, yerr=std_pattern, 
                        fmt='o-', capsize=5, label='Mean ± Std Dev')
        
        # Plot min/max range
        axes[i].fill_between(feature_indices, min_pattern, max_pattern, 
                            alpha=0.2, label='Min-Max Range')
        
        # Count occurrences and calculate typical duration
        state_count = np.sum(mask)
        runs = self._find_runs(states_np == state)
        mean_duration = np.mean([end-start for start, end in runs]) if runs else 0
        
        axes[i].set_title(f'State {state}: {state_count} occurrences ({state_count/len(states_np):.1%}), '
                         f'Avg Duration: {mean_duration:.1f} steps')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
```

This visualization provides valuable insights into what each state represents in terms of the input features.

### Transition Dynamics

Understanding how states transition to each other is key to interpreting the temporal dynamics:

```python
def visualize_transition_matrix(self, trans_probs, save_path=None):
    """Create a detailed visualization of the transition probability matrix."""
    # Convert to numpy
    trans_np = trans_probs.cpu().numpy() if isinstance(trans_probs, torch.Tensor) else trans_probs
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(trans_np, cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f')
    
    # Add titles and labels
    plt.title('State Transition Probabilities')
    plt.xlabel('To State')
    plt.ylabel('From State')
    
    # Add diagonal highlighting
    for i in range(min(trans_np.shape)):
        plt.axhline(i + 0.5, color='white', linewidth=0.5)
        plt.axvline(i + 0.5, color='white', linewidth=0.5)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
```

Strong diagonal elements indicate states with high self-transition probabilities (persistence), while off-diagonal elements reveal the pathways between different states.

### State Evolution

Tracking how the number of states evolves over time provides insights into model complexity:

```python
def create_state_evolution_plot(self, state_changes, save_path=None):
    """Create a visualization of state birth, merge, and delete events."""
    if not state_changes:
        return
    
    # Extract data for plotting
    windows = list(range(len(state_changes)))
    
    # Get state counts
    initial_states = [sc.get('initial_states', 0) for sc in state_changes]
    final_states = [sc.get('final_states', 0) for sc in state_changes]
    
    # Get birth, merge, delete events
    births = [len(sc.get('birthed', [])) for sc in state_changes]
    merges = [len(sc.get('merged', [])) for sc in state_changes]
    deletes = [len(sc.get('deleted', [])) for sc in state_changes]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot state count
    ax1.plot(windows, initial_states, 'b-', label='State Count')
    
    # Plot birth events
    birth_indices = [i for i, b in enumerate(births) if b > 0]
    if birth_indices:
        ax1.scatter(birth_indices, [initial_states[i] for i in birth_indices], 
                   marker='^', s=100, c='g', label='Birth')
    
    # Plot merge events
    merge_indices = [i for i, m in enumerate(merges) if m > 0]
    if merge_indices:
        ax1.scatter(merge_indices, [initial_states[i] for i in merge_indices], 
                   marker='o', s=100, c='orange', label='Merge')
    
    # Plot delete events
    delete_indices = [i for i, d in enumerate(deletes) if d > 0]
    if delete_indices:
        ax1.scatter(delete_indices, [initial_states[i] for i in delete_indices], 
                   marker='v', s=100, c='r', label='Delete')
    
    # Add labels and legend
    ax1.set_ylabel('Number of States')
    ax1.set_title('State Evolution Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot event counts as stacked bar chart
    ax2.bar(windows, births, label='Births', color='g')
    ax2.bar(windows, merges, bottom=births, label='Merges', color='orange')
    ax2.bar(windows, deletes, bottom=[births[i] + merges[i] for i in range(len(births))], 
           label='Deletes', color='r')
    
    # Add labels and legend
    ax2.set_xlabel('Window')
    ax2.set_ylabel('Event Count')
    ax2.set_title('State Change Events')
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
```

This visualization helps identify when the model adjusts its complexity to fit the data better.

## Practical Applications

### System Monitoring

One of the primary applications of this HDP-HMM implementation is system monitoring. By processing multivariate time series data from system metrics (CPU, memory, disk I/O, network traffic), the model can:

1. **Discover operational states**: Identify normal operational states like idle, under load, batch processing, etc.
2. **Detect state transitions**: Recognize when the system transitions between states
3. **Identify anomalies**: Flag unusual patterns that don't fit into the learned states
4. **Predict resource needs**: Anticipate resource requirements based on state transitions

Example usage for system monitoring:

```python
# Monitoring system metrics with 1-second sampling interval
python main.py --use-real --sample-interval 1.0 --window-size 60
```

### Behavioral Analysis

The HDP-HMM is particularly effective for behavioral analysis across various domains:

1. **User behavior in web/app analytics**: Discover user interaction patterns
2. **Financial transaction sequences**: Identify spending patterns and fraud indicators
3. **Healthcare patient monitoring**: Recognize activity states and health transitions
4. **IoT sensor data**: Identify usage patterns and operational modes

Example application to financial transaction data:

```python
# Process historical transaction data
python main.py --data-dir financial_data --window-size 100 --stride 25
```

### Anomaly Detection

The model excels at anomaly detection by:

1. **Learning normal patterns**: Automatically discovering states that represent normal behavior
2. **Identifying rare states**: Flagging infrequently observed states
3. **Detecting unlikely transitions**: Identifying unusual transitions between states
4. **Measuring likelihood**: Computing the likelihood of observations under the model

Example anomaly detection implementation:

```python
def detect_anomalies(window_data, states, emission_probs, threshold=-10):
    """Detect anomalies based on emission probabilities."""
    # Get emission probabilities for each observation
    emission_log_probs = emission_probs[torch.arange(len(states)), states]
    
    # Identify observations with very low probability
    anomalies = emission_log_probs < threshold
    
    # Return anomalous indices and their scores
    anomaly_indices = torch.where(anomalies)[0]
    anomaly_scores = emission_log_probs[anomalies]
    
    return anomaly_indices, anomaly_scores
```

## Getting Started

### Installation

To get started with the HDP-HMM implementation:

```bash
# Clone the repository
git clone https://github.com/yourusername/LatentDynamicsBayes.git
cd LatentDynamicsBayes

# Install dependencies
pip install -r requirements.txt

# Run a quick demo
python demo.py
```

### Quick Start Guide

For a quick introduction to the capabilities:

```bash
# Generate sample data
python generate_sample_data.py

# Run in live mode with default settings
python main.py

# Run in offline mode with the generated data
python main.py --data-dir data --window-size 50 --stride 25
```

### Advanced Configuration

Fine-tune the model behavior through the config.json file:

```json
{
  "model": {
    "n_features": 3,
    "max_states": 20,
    "alpha": 1.0,
    "gamma": 1.0,
    "learning_rate": 0.01
  },
  "state_management": {
    "delete_threshold": 1e-3,
    "merge_distance": 0.5,
    "birth_threshold": 10.0
  },
  "training": {
    "max_iterations": 1000,
    "checkpoint_interval": 10
  }
}
```

## Conclusion

The HDP-HMM implementation presented here offers a powerful framework for discovering hidden states and patterns in time series data without specifying the number of states in advance. By combining Bayesian nonparametric methods with modern deep learning tools like PyTorch, it provides both theoretical soundness and practical efficiency.

Whether you're analyzing system metrics in real-time or processing historical data from CSV files, this implementation provides the tools to uncover the latent structure in your time series data. The comprehensive visualization suite helps interpret the results and gain insights into the underlying dynamics of your system.

I encourage you to try out the code, experiment with your own datasets, and contribute to the further development of this open-source project. The repository is available at [GitHub](https://github.com/yourusername/LatentDynamicsBayes).

## References

1. Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet Processes. Journal of the American Statistical Association, 101(476), 1566-1581.

2. Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2008). An HDP-HMM for Systems with State Persistence. In Proceedings of the 25th International Conference on Machine Learning (ICML).

3. Hughes, M. C., & Sudderth, E. B. (2013). Memoized Online Variational Inference for Dirichlet Process Mixture Models. In Advances in Neural Information Processing Systems (NIPS).

4. Hughes, M. C., Stephenson, W. T., & Sudderth, E. (2015). Scalable Adaptation of State Complexity for Nonparametric Hidden Markov Models. In Advances in Neural Information Processing Systems.

5. Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008). Beam Sampling for the Infinite Hidden Markov Model. In Proceedings of the 25th International Conference on Machine Learning (ICML).
