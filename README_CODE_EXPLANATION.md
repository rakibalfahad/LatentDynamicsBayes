# HDP-HMM Implementation Code Explanation

This document provides a detailed explanation of the Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) implementation in this repository, including theoretical foundations, implementation details, and advanced features.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
   - [Hidden Markov Models](#hidden-markov-models)
   - [Dirichlet Process](#dirichlet-process)
   - [Hierarchical Dirichlet Process](#hierarchical-dirichlet-process)
   - [HDP-HMM](#hdp-hmm)
   - [Stick-Breaking Construction](#stick-breaking-construction)

2. [Implementation Overview](#implementation-overview)
   - [Core Components](#core-components)
   - [PyTorch Implementation](#pytorch-implementation)
   - [Inference Algorithms](#inference-algorithms)

3. [Birth, Merge, Delete Mechanisms](#birth-merge-delete-mechanisms)
   - [Theoretical Motivation](#theoretical-motivation)
   - [Implementation Details](#implementation-details)
   - [Parameter Tuning](#parameter-tuning)

4. [Live Streaming Architecture](#live-streaming-architecture)
   - [Data Collection](#data-collection)
   - [Incremental Training](#incremental-training)
   - [Real-time Visualization](#real-time-visualization)

5. [Code Structure](#code-structure)
   - [hdp_hmm.py](#hdp_hmmpy)
   - [trainer.py](#trainerpy)
   - [live_visualize.py](#live_visualizepy)
   - [main.py](#mainpy)

6. [Advanced Topics](#advanced-topics)
   - [Computational Efficiency](#computational-efficiency)
   - [Numerical Stability](#numerical-stability)
   - [Extension Points](#extension-points)

7. [References](#references)

---

## Theoretical Background

### Hidden Markov Models

Hidden Markov Models (HMMs) are statistical models that represent systems where state transitions follow the Markov property (the future state depends only on the current state) and states are not directly observable, but emissions from these states are. Formally, an HMM consists of:

- A set of hidden states: $S = \{s_1, s_2, ..., s_K\}$
- Transition probabilities: $A = \{a_{ij}\}$ where $a_{ij} = P(z_t = j | z_{t-1} = i)$
- Emission probabilities: $B = \{b_i(o_t)\}$ where $b_i(o_t) = P(o_t | z_t = i)$
- Initial state distribution: $\pi = \{\pi_i\}$ where $\pi_i = P(z_1 = i)$

In traditional HMMs, the number of states $K$ is fixed and must be specified in advance, which presents a significant limitation when the true number of states is unknown.

### Dirichlet Process

The Dirichlet Process (DP) is a distribution over distributions. A draw $G \sim DP(\alpha, H)$ from a DP with concentration parameter $\alpha$ and base distribution $H$ is itself a distribution. The DP provides a foundation for nonparametric Bayesian models by allowing an infinite number of components.

### Hierarchical Dirichlet Process

The Hierarchical Dirichlet Process (HDP) extends the DP by allowing multiple DPs to share components. In an HDP, a global DP generates a base distribution that is then used as the base distribution for multiple local DPs. Formally:

$G_0 \sim DP(\gamma, H)$
$G_j \sim DP(\alpha, G_0)$ for $j = 1, 2, ...$

This hierarchical structure enables sharing statistical strength across groups while allowing each group to have its own distribution.

### HDP-HMM

The HDP-HMM, introduced by Teh et al. (2006) and further developed by Fox et al. (2008), combines the HDP with the HMM framework to create a nonparametric HMM that can learn the appropriate number of states from data. In the HDP-HMM:

- Each row of the transition matrix is drawn from a DP
- These DPs share components through the hierarchical structure
- The number of states can grow with the complexity of the data

This allows for a theoretically infinite state space, although in practice, the number of active states remains finite and adapts to the data.

### Stick-Breaking Construction

The stick-breaking construction provides a concrete representation of draws from a DP. For a DP with concentration parameter $\alpha$, the stick-breaking process generates weights $\beta_k$ as follows:

1. Draw $v_k \sim Beta(1, \alpha)$ for $k = 1, 2, ...$
2. Set $\beta_k = v_k \prod_{i=1}^{k-1} (1 - v_i)$

This can be interpreted as breaking a stick of unit length: $v_k$ represents the proportion of the remaining stick that is broken off at step $k$. The resulting weights $\beta_k$ sum to 1 and are typically concentrated on a finite number of values, which aligns with the sparse nature of real-world systems.

In our implementation, we use a parameterized version of the stick-breaking process where the beta distribution parameters are represented as logits that are learned during training.

---

## Implementation Overview

### Core Components

Our HDP-HMM implementation consists of several core components:

1. **HDPHMM Class**: The main model class implementing the HDP-HMM with stick-breaking construction
2. **LiveHDPHMM**: A wrapper for live streaming data processing and incremental training
3. **LiveVisualizer**: Real-time visualization of model outputs and data
4. **LiveDataCollector**: Data collection and preprocessing for streaming data

### PyTorch Implementation

The implementation leverages PyTorch's automatic differentiation and GPU acceleration capabilities:

- **Tensor Operations**: All model parameters and computations use PyTorch tensors
- **Automatic Differentiation**: Gradients are computed automatically for optimization
- **GPU Support**: All operations can be performed on GPU for faster computation
- **Numerical Stability**: Log-space computations and other techniques ensure stability

### Inference Algorithms

The implementation includes several key inference algorithms:

1. **Forward-Backward Algorithm**: Computes posterior state probabilities and model likelihood
2. **Viterbi Algorithm**: Finds the most likely state sequence given observations
3. **Dynamic State Management**: Adjusts the number of states based on data evidence

---

## Birth, Merge, Delete Mechanisms

### Theoretical Motivation

While the HDP-HMM theoretically allows for an infinite number of states, practical implementations require approximations. The birth, merge, and delete mechanisms draw inspiration from split-merge MCMC algorithms (Jain and Neal, 2004) and reversible jump MCMC (Green, 1995), adapting them to the variational inference setting.

These mechanisms allow the model to:
- Add states when the current state space is insufficient (birth)
- Combine redundant states to reduce complexity (merge)
- Remove unused states to improve computational efficiency (delete)

This approach is related to the "memoized variational inference" framework introduced by Hughes and Sudderth (2013), which maintains a summary of past assignments and can revisit and refine previous decisions.

### Implementation Details

#### Delete Mechanism

States with negligible probability (beta weights below a threshold) are marked inactive:

```python
# States with beta_weights below threshold are considered inactive
for k in range(self.current_states):
    if beta_weights[k] > threshold:
        active_indices.append(k)
    else:
        inactive_indices.append(k)
        state_changes['deleted'].append(k)
```

#### Merge Mechanism

States with similar emission distributions are combined through weighted averaging:

```python
# Check distance between state means
dist = torch.norm(self.means[i_idx] - self.means[j_idx])
if dist < merge_distance:
    # Combine parameters by weighted averaging
    weight_i = beta_weights[i_idx]
    weight_j = beta_weights[j_idx]
    total_weight = weight_i + weight_j
    
    # Update parameters of state i
    self.means.data[i_idx] = (weight_i * self.means[i_idx] + weight_j * self.means[j_idx]) / total_weight
    # ...additional parameter updates
    
    # Record merge
    state_changes['merged'].append((j_idx, i_idx))
```

#### Birth Mechanism

New states are created when observations are poorly explained by existing states:

```python
# If model fit is poor and we have inactive states, add a new state
if avg_nll > birth_threshold and self.current_states < self.max_states:
    # Create new state from poorly fit observations
    # ...implementation details
    
    # Record birth
    state_changes['birthed'].append(new_state_idx)
```

### State Change Tracking and Visualization

The implementation includes robust mechanisms for tracking and visualizing state changes:

#### State Change Tracking

The `update_states` method returns detailed information about state changes:

```python
state_changes = {
    'deleted': deleted_states,         # List of deleted state IDs
    'merged': merged_pairs,            # List of (source, destination) merge pairs
    'birthed': birthed_states,         # List of new state IDs
    'initial_states': initial_states,  # Number of states before update
    'final_states': current_states,    # Number of states after update
    'active_states': active_indices,   # List of active state IDs
    'inactive_states': inactive_indices # List of inactive state IDs
}
```

#### Formatted State Update Display

The `format_state_update_stats` method in `LiveHDPHMM` converts state change information into a concise, readable format:

```python
# Example output: "States: 9 → 10 | Changes: +1 birth, ~2 merges | Details: Birth: state(s) 12; Merge: 5→3, 8→4"
stats_str = self.format_state_update_stats(state_change_info)
print(f"\n[Window {self.window_count}] State Update: {stats_str}")
```

#### State Evolution Timeline

The `print_state_evolution_summary` method provides a text-based visualization of state evolution:

```
State Evolution Timeline:
     0   1   2   3   4
   ────────────────────
 0│ ●   ●   ●   ●   ● 
 1│ ●   ●   ●   ⊗     
 2│ ●   ●   ⊙         
 3│ ●   ●   ●   ●   ● 
 4│         ⊕   ●   ● 

Legend: ● Active  ⊕ Birth  ⊗ Delete  ⊙ Merge
```

This visualization helps identify patterns in how states are created, merged, and deleted over time.

### Parameter Tuning

The behavior of these mechanisms can be adjusted through several key parameters:

- **Delete Threshold** (default: 1e-3): Controls how aggressively states are pruned
- **Merge Distance** (default: 0.5): Determines the similarity threshold for merging states
- **Birth Threshold** (default: 10.0): Sets the poor-fit threshold for creating new states

These parameters can be tuned based on data characteristics and application requirements:

| Data Characteristic | Recommended Parameter Adjustments |
|---------------------|-----------------------------------|
| High noise | Increase delete threshold, increase merge distance |
| Complex patterns | Decrease birth threshold, decrease merge distance |
| Limited computational resources | Increase delete threshold, increase merge distance |
| Need for detailed state resolution | Decrease merge distance, decrease delete threshold |

---

## Live Streaming Architecture

### Data Collection

The `LiveDataCollector` class handles data acquisition in a streaming setting:

- Maintains a sliding window of observations
- Can collect real system metrics or simulate data
- Preprocesses data for model consumption

The sliding window approach allows the model to focus on recent data while maintaining enough context for state inference.

### Incremental Training

The `LiveHDPHMM` class manages incremental model updates:

- Updates model parameters with each new window of data
- Periodically adjusts the state space via birth/merge/delete
- Tracks loss and state counts over time
- Handles checkpointing for model persistence

This incremental approach allows the model to adapt to changing data distributions over time.

### Real-time Visualization

The `LiveVisualizer` class provides comprehensive visualization capabilities:

#### Basic Visualizations
- Time series plots with state assignments
- Transition probability heatmaps
- State occupation histograms

#### Advanced Visualizations
- **Tile Visualization**: Shows state assignments across time windows as a color-coded grid
- **State Sequence Visualization**: Displays state transitions with data in the style of bnpy
- **Learning Curve Visualization**: Tracks loss with smoothing, EMA, and correlation to state counts

#### State Evolution Visualization
- Text-based state evolution timeline with symbols for different events
- Detailed state change statistics with birth, merge, and delete events
- Visual correlation between state changes and model performance

The `create_learning_curve` method implements a sophisticated loss visualization:

```python
def create_learning_curve(self, losses, state_counts=None, save_path=None):
    # Plot raw loss values
    ax1.plot(window_indices, losses_np, 'b-', alpha=0.5, label='Raw Loss')
    
    # Plot smoothed loss trend with adaptive window size
    window_size = min(25, max(5, len(losses_np) // 10))
    if len(losses_np) > window_size:
        smoothed = np.convolve(losses_np, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(smoothed_x, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
    
    # Add exponential moving average
    if len(losses_np) > 10:
        alpha = 0.1  # Smoothing factor
        ema = np.zeros_like(losses_np)
        ema[0] = losses_np[0]
        for i in range(1, len(losses_np)):
            ema[i] = alpha * losses_np[i] + (1 - alpha) * ema[i-1]
        ax1.plot(window_indices, ema, 'g-', linewidth=2, label='Exp. Moving Avg')
    
    # Plot state counts and calculate correlation with loss
    if state_counts and len(state_counts) > 0:
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(state_indices, state_counts_np, 'g-', linewidth=2, label='Active States')
        
        # Add correlation analysis
        if len(losses_np) == len(state_counts_np):
            correlation = np.corrcoef(losses_np, state_counts_np)[0, 1]
            ax2.text(0.02, 0.02, f'Correlation with loss: {correlation:.2f}',
                    transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
```

These visualizations help in understanding model behavior and diagnosing issues during training.

---

## Code Structure

### hdp_hmm.py

The core model implementation, including:

- `HDPHMM` class with stick-breaking construction
- Forward-backward and Viterbi algorithms
- State space management via birth/merge/delete
- Model persistence (save/load)

Key methods:
- `stick_breaking`: Implements the stick-breaking process
- `forward_backward`: Computes posterior probabilities and likelihood
- `infer_states`: Finds the most likely state sequence
- `update_states`: Dynamically adjusts the state space

### trainer.py

Manages training, inference, and state tracking:

- `LiveHDPHMM` class for incremental training
- Optimization and parameter updates
- Loss and state count tracking
- Comprehensive state change monitoring
- Checkpointing and model persistence

Key methods:
- `update_model`: Performs incremental parameter updates
- `update_states_safe`: Safely manages dynamic state adjustments
- `format_state_update_stats`: Formats state changes for clear display
- `print_state_evolution_summary`: Creates text-based state evolution visualization
- `infer`: Performs inference on new data
- `save_model`/`load_model`: Handles model persistence
- `save_transition_matrix`: Exports transition probabilities in multiple formats

The trainer now tracks detailed information about state changes, including birth, merge, and delete events, to help monitor and debug the model's dynamic state management.

### live_visualize.py

Provides comprehensive visualization capabilities:

- `LiveVisualizer` class for real-time plots and saved visualizations
- Time series visualization with state coloring
- Transition probability heatmaps and exports
- State count tracking and evolution visualization
- Advanced visualization types: tile plots, state sequences, learning curves

Key methods:
- `update_plot`: Updates all visualizations with new data
- `create_tile_visualization`: Creates tile plots of state assignments
- `show_state_sequence`: Generates state sequence visualizations (bnpy-style)
- `create_learning_curve`: Creates detailed loss and state count visualizations
- `create_state_evolution_plot`: Visualizes state birth, merge, and delete events

The visualizations support both interactive mode and headless operation with automatic saving of all plots.

### main.py

The entry point for running the system:

- Parameter configuration
- Component initialization
- Main processing loop
- Error handling and logging

---

## Advanced Topics

### Computational Efficiency

Several techniques improve computational efficiency:

- **Sparse State Representation**: Focus computation on active states
- **Parameter Reordering**: Keep active states at the beginning of parameter tensors
- **GPU Acceleration**: Leverage GPU for matrix operations
- **Dynamic State Management**: Reduce computation by pruning unused states

### Numerical Stability

Numerical stability is ensured through:

- **Log-space Computations**: Prevent underflow in probability calculations
- **Cloning**: Avoid in-place operations that can disrupt gradient flow
- **Small Constants**: Add small constants to denominators to prevent division by zero
- **Robust Error Handling**: Catch and handle numerical errors gracefully

### Extension Points

The implementation can be extended in several ways:

- **Alternative Emission Models**: Replace Gaussian emissions with other distributions
- **Hierarchical Priors**: Add priors on hyperparameters for more flexible modeling
- **Online Hyperparameter Adaptation**: Adjust hyperparameters during training
- **Parallelized Window Processing**: Process multiple windows in parallel for higher throughput

---

## References

- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet Processes. Journal of the American Statistical Association, 101(476), 1566-1581.

- Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2008). An HDP-HMM for Systems with State Persistence. In Proceedings of the 25th International Conference on Machine Learning (ICML).

- Hughes, M. C., & Sudderth, E. B. (2013). Memoized Online Variational Inference for Dirichlet Process Mixture Models. In Advances in Neural Information Processing Systems (NIPS).

- Jain, S., & Neal, R. M. (2004). A Split-Merge Markov Chain Monte Carlo Procedure for the Dirichlet Process Mixture Model. Journal of Computational and Graphical Statistics, 13(1), 158-182.

- Green, P. J. (1995). Reversible Jump Markov Chain Monte Carlo Computation and Bayesian Model Determination. Biometrika, 82(4), 711-732.

- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic Variational Inference. The Journal of Machine Learning Research, 14(1), 1303-1347.

- Johnson, M. J., & Willsky, A. S. (2013). Bayesian Nonparametric Hidden Semi-Markov Models. Journal of Machine Learning Research, 14(Feb), 673-701.

- Kingma, D. P., & Welling, M. (2013). Auto-encoding Variational Bayes. arXiv preprint arXiv:1312.6114.

- Hughes, M. C., Stephenson, W. T., & Sudderth, E. (2015). Scalable Adaptation of State Complexity for Nonparametric Hidden Markov Models. In Advances in Neural Information Processing Systems.

- Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008). Beam Sampling for the Infinite Hidden Markov Model. In Proceedings of the 25th International Conference on Machine Learning (ICML).

- Blei, D. M., & Jordan, M. I. (2006). Variational Inference for Dirichlet Process Mixtures. Bayesian Analysis, 1(1), 121-143.
