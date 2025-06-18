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
   - [Offline Data Processing](#offline-data-processing)
   - [Incremental Training](#incremental-training)
   - [Real-time Visualization](#real-time-visualization)
   - [State Evolution Tracking](#state-evolution-tracking)

5. [Offline CSV Processing](#offline-csv-processing)
   - [CSVDataProcessor Implementation](#csvdataprocessor-implementation)
   - [Integration with Main Pipeline](#integration-with-main-pipeline)
   - [Windowing and Sequencing](#windowing-and-sequencing)
   - [Usage and Configuration](#usage-and-configuration)

6. [Visualization Suite](#visualization-suite)
   - [Basic Time Series Visualization](#basic-time-series-visualization)
   - [State Pattern Analysis](#state-pattern-analysis)
   - [Composite Visualizations](#composite-visualizations)
   - [Learning Curves and Performance Metrics](#learning-curves-and-performance-metrics)
   - [Headless Operation](#headless-operation)

7. [Code Structure](#code-structure)
   - [hdp_hmm.py](#hdp_hmmpy)
   - [trainer.py](#trainerpy)
   - [live_visualize.py](#live_visualizepy)
   - [main.py](#mainpy)
   - [csv_processor.py](#csv_processorpy)

8. [Advanced Topics](#advanced-topics)
   - [Computational Efficiency](#computational-efficiency)
   - [Numerical Stability](#numerical-stability)
   - [Extension Points](#extension-points)

9. [References](#references)

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
```

#### Birth Mechanism

New states are created when observations are poorly explained by existing states:

```python
# If model fit is poor and we have inactive states, add a new state
if avg_nll > birth_threshold and self.current_states < self.max_states:
    # Create new state from poorly fit observations
    # ...implementation details
```

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

The `LiveVisualizer` class provides real-time visualization capabilities:

- Time series plots with state assignments
- State count tracking over time
- Transition probability heatmaps
- Tile visualization of state assignments over time
- Learning curve visualization with advanced smoothing techniques
- State evolution plots showing birth, merge, and delete events
- State sequence visualization inspired by bnpy

These visualizations help in understanding model behavior and diagnosing issues during training. The implementation is designed to be robust across different environments, including headless servers and various operating systems.

```python
# Example of robust visualization with cross-platform compatibility
try:
    # Use subplots_adjust instead of tight_layout for better compatibility
    self.fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1, left=0.1, right=0.95)
except Exception as e:
    print(f"Warning: Figure layout adjustment failed: {e}")

# Always save the figure, regardless of display mode
try:
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    self.fig.savefig(f'plots/live_plot_window_{self.window_count}.png', dpi=300, bbox_inches='tight')
    if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
        plt.pause(0.01)  # Only pause for display if in GUI mode
except Exception as e:
    print(f"Warning: Failed to save live plot: {e}")
```

---

## Robust Visualization Implementation

The visualization system has been enhanced to ensure reliability across different environments and platforms:

### Cross-Platform Layout Management

Instead of using `tight_layout()` which can fail on some systems resulting in blank plots, the code now uses explicit `subplots_adjust()` with carefully chosen parameters:

```python
# Instead of this (can fail on some systems):
self.fig.tight_layout()

# We now use this (more robust):
try:
    # Use subplots_adjust instead of tight_layout for better compatibility
    self.fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1, left=0.1, right=0.95)
except Exception as e:
    print(f"Warning: Figure layout adjustment failed: {e}")
```

### Directory Management

All plot saving operations now include automatic directory creation to prevent failures:

```python
# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Save with bbox_inches to ensure all content is captured
plt.savefig(f'plots/state_tiles_window_{self.window_count}.png', dpi=300, bbox_inches='tight')
```

### Error Handling

All visualization operations are wrapped in try-except blocks with appropriate error messages:

```python
try:
    # Visualization code
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
except Exception as e:
    print(f"Warning: Failed to save plot: {e}")
finally:
    # Always close the figure to prevent memory leaks
    plt.close()
```

### Headless Environment Detection

The code automatically detects if it's running in a headless environment and adjusts accordingly:

```python
# Use non-interactive backend if running in headless mode
import matplotlib
if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
    matplotlib.use('Agg')  # Use non-GUI backend

# ...

# Always save the figure, regardless of display mode
self.fig.savefig(f'plots/live_plot_window_{self.window_count}.png', dpi=300)
if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
    plt.pause(0.01)  # Only pause for display if in GUI mode
```

These improvements ensure that the visualization system works reliably across different platforms, including headless servers and various operating systems, and that generated plots are always properly saved and displayed.

---

## State Visualization and Monitoring

The visualization system provides comprehensive monitoring of model dynamics during training, with a particular focus on state evolution.

### State Evolution Tracking

The system tracks state changes in detail through several mechanisms:

```python
# In update_states method
state_changes = {
    'deleted': deleted_states,    # States removed due to low probability
    'merged': merged_pairs,       # States merged due to similarity
    'birthed': birthed_states,    # New states created for poorly fit data
    'initial_states': initial_states,  # Number of states before update
    'final_states': current_states,    # Number of states after update
    'active_states': active_indices    # Currently active state indices
}
```

This information is used to generate both textual and graphical representations of state dynamics.

### Visualization Components

#### Format State Update Stats

Converts raw state change information into a human-readable format:

```python
def format_state_update_stats(self, state_change_info):
    """Format state update statistics in a clear, concise way"""
    # Generates output like:
    # States: 9 → 10 | Changes: +1 birth, ~2 merges | Details: Birth: state(s) 12; Merge: 5→3, 8→4
```

#### State Evolution Plot

Creates a comprehensive visualization of state changes over time:

```python
def create_state_evolution_plot(self, state_changes, save_path=None):
    """Create a visualization of state birth, merge, and delete events"""
    # Shows state count over time with markers for birth, merge, delete events
    # Includes a bar chart showing the number of each event type per update
```

#### Text-based State Timeline

Generates a text-based visualization showing state evolution:

```python
def print_state_evolution_summary(self):
    """
    Print a text-based visualization of state evolution through time.
    Shows births, merges, and deletes of states across training windows.
    """
    # Creates a timeline with symbols:
    # ● Active  ⊕ Birth  ⊗ Delete  ⊙ Merge
```

### Tile and State Sequence Visualization

The tile and state sequence visualizations provide complementary views of state assignments:

```python
def create_tile_visualization(self):
    """
    Create a tile visualization showing state assignments over time,
    similar to the visualization in bnpy.
    
    The visualization shows:
    - Each row represents a time point
    - Each column represents a window
    - Colors represent different states
    - This helps visualize state persistence and transitions over time
    """
    # Create a 2D array from state history
    state_matrix = np.zeros((time_points, windows))
    
    # Fill the matrix with state assignments
    for w in range(windows):
        window_states = self.state_history[w]
        for t in range(min(time_points, len(window_states))):
            state_matrix[t, w] = window_states[t]
    
    # First subplot: State tile visualization
    plt.subplot(2, 1, 1)
    im = plt.imshow(state_matrix, aspect='auto', interpolation='none', 
                   cmap='plasma', vmin=0, vmax=max_state)
```

The state sequence visualization displays data alongside state assignments:

```python
def show_state_sequence(self, data, states, max_states=None, save_path=None):
    """
    Create a visualization showing data sequence with state assignments below,
    inspired by the bnpy visualization style.
    """
    # Plot time series data in top subplot
    for dim in range(data_np.shape[1]):
        axes[0].plot(data_np[:, dim], '.-', label=f'Feature {dim+1}')
    
    # Create state image in bottom subplot
    img_height = 100  # Height of the state image
    state_img = np.tile(states_np, (img_height, 1))
    
    # Display the state image
    img = axes[1].imshow(state_img, interpolation='nearest', aspect='auto',
                       vmin=-0.5, vmax=max_states-0.5, cmap=cmap)
```

Key features of these visualizations:

1. **Multi-window History**: Tracks state assignments across multiple windows
2. **Transition Analysis**: Computes and displays state transition frequencies
3. **Color Consistency**: Uses consistent color mapping for states across visualizations
4. **Combined Data/State View**: Shows both raw data and state assignments together
5. **Cross-platform Layout**: Uses explicit subplot adjustment instead of tight_layout
6. **Robust Error Handling**: Prevents crashes from visualization issues

These visualizations are particularly valuable for identifying patterns in state usage and understanding how the model segments the time series data.

### State Pattern and Composite Visualizations

Two advanced visualization methods have been added to provide deeper insights into state characteristics:

```python
def visualize_state_patterns(self, data=None, states=None, save_path=None):
    """
    Create a comprehensive visualization showing what patterns each state represents.
    Shows average pattern, standard deviation, and min/max ranges for each state.
    """
    # Calculate statistics for each state
    for i, state in enumerate(unique_states):
        mask = states_np == state
        state_data = data_np[mask]
        
        # Calculate statistics
        mean_pattern = np.mean(state_data, axis=0)
        std_pattern = np.std(state_data, axis=0)
        min_pattern = np.min(state_data, axis=0)
        max_pattern = np.max(state_data, axis=0)
        
        # Plot statistics to understand state characteristics
        # ...
```

The composite visualization provides a unified view combining multiple aspects:

```python
def create_composite_state_visualization(self, data=None, states=None, save_path=None):
    """
    Create a comprehensive visualization that combines state sequence and patterns.
    
    This visualization combines:
    1. Time series data with state coloring
    2. State sequence visualization (similar to bnpy)
    3. State pattern summaries
    4. Transition probabilities between states
    5. State duration histogram
    """
    # Creates several subplots showing different aspects of state behavior
    # ...
```

These visualizations are automatically generated periodically during training:

```python
# In update_plot method
# Create state pattern visualization every 10 windows
if self.window_count % 10 == 0:
    save_path = f'plots/state_patterns_window_{self.window_count}.png'
    self.visualize_state_patterns(data, states, save_path)
    
    # Also create a composite visualization
    save_path = f'plots/composite_viz_window_{self.window_count}.png'
    self.create_composite_state_visualization(data, states, save_path)
```

The visualizations help users interpret what each state represents in the data, how states transition between each other, and how long the model typically stays in each state. This is particularly valuable for understanding the latent structure discovered by the model.

### State-Specific Time Series Visualization

The system includes a specialized visualization that shows time series data for each discovered state separately, allowing detailed examination of the temporal patterns captured by each state:

```python
def visualize_state_time_series(self, data=None, states=None, save_path=None):
    """
    Create visualizations showing time series data for each state separately.
    
    This method:
    1. Groups data points by state
    2. For each state, shows the time series of all features
    3. Highlights where each state appears in the full sequence
    """
```

This visualization creates:

1. **Individual State Files**: One file per state showing:
   - State occurrence mask (where in the sequence this state appears)
   - Time series for each feature with this state's data points highlighted
   - Statistical indicators (mean, standard deviation) for each feature in this state

2. **Summary Visualization**: A comprehensive overview showing:
   - Complete time series with color-coded state assignments
   - Timeline of state occurrences across the sequence

Example output:

For each state `X`, the system generates:
- `plots/state_time_series/state_X_window_Y.png`: Detailed view of state X's time series pattern
- `plots/state_time_series/all_states_summary_window_Y.png`: Overview of all states

These visualizations make it easy to:
- Identify when each state is active in the sequence
- Compare feature patterns across different states
- Understand the temporal characteristics of each state
- Detect potential state confusion or redundancy

### Implementation Details

The state visualization system uses several advanced techniques:

1. **Error Handling**: All visualizations are wrapped in try-except blocks to prevent crashes during rendering or saving
2. **Compatible Layout**: Uses `subplots_adjust` instead of `tight_layout` for better cross-platform compatibility
3. **Headless Support**: Detects display availability and adjusts accordingly for server environments
4. **File Output**: All visualizations are saved to disk, with both timestamped and "latest" versions
5. **Plot Directory Management**: Automatic creation of plots directory to prevent save failures
6. **Adaptive Margins**: Uses `bbox_inches='tight'` to ensure all plot content is visible
7. **Fallback Mechanisms**: Creates simplified error plots if advanced visualization fails

```python
# Example of robust error handling with fallback mechanism
try:
    # Attempt to create advanced visualization
    self.create_state_evolution_plot(state_changes, save_path)
except Exception as e:
    print(f"Error creating state evolution plot: {e}")
    try:
        # Create a new simple figure with just text explaining the error
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating state evolution plot:\n{e}",
                ha='center', va='center', fontsize=12, 
                transform=plt.gca().transAxes)
                
        plt.savefig(save_path, dpi=300)
        plt.savefig('plots/latest_state_evolution.png', dpi=300)
    except:
        print("Could not create error message plot")
finally:
    # Always close the figure to prevent memory leaks
    plt.close()
```

### Learning Curve Visualization

The learning curve visualization connects state changes to model performance:

```python
def create_learning_curve(self, losses, state_counts=None, save_path=None):
    """Create a detailed learning curve visualization for model performance debugging"""
    # Shows raw loss, smoothed trends, and exponential moving average
    # When state_counts provided, shows correlation between states and loss
    
    # Convert to numpy array
    losses_np = np.array(losses)
    window_indices = np.arange(len(losses_np))
    
    # Plot raw loss values
    ax1.plot(window_indices, losses_np, 'b-', alpha=0.5, label='Raw Loss')
    
    # Plot smoothed loss trend
    window_size = min(25, max(5, len(losses_np) // 10))  # Adaptive window size
    if len(losses_np) > window_size:
        smoothed = np.convolve(losses_np, np.ones(window_size)/window_size, mode='valid')
        smoothed_x = window_indices[window_size-1:]
        ax1.plot(smoothed_x, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
    
    # Add exponential moving average
    if len(losses_np) > 10:
        alpha = 0.1  # Smoothing factor
        ema = np.zeros_like(losses_np)
        ema[0] = losses_np[0]
        for i in range(1, len(losses_np)):
            ema[i] = alpha * losses_np[i] + (1 - alpha) * ema[i-1]
        ax1.plot(window_indices, ema, 'g-', linewidth=2, label='Exp. Moving Avg')
```

Features of the learning curve visualization:

1. **Multiple Smoothing Methods**: Raw values, window smoothing, and exponential moving average
2. **Adaptive Window Size**: Automatically adjusts smoothing window based on data length
3. **Progress Indicators**: Displays loss reduction percentage and min/max points
4. **State Correlation**: Shows correlation between loss and number of states
5. **Log Scale Option**: Automatically adds log scale inset for wide-range loss values
6. **Dynamic Saving**: Creates both timestamped and latest versions for easy reference

This comprehensive visualization system provides valuable insights into how the model's state space evolves during training, helping users understand and debug the dynamic state management process.

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

Manages training and inference:

- `LiveHDPHMM` class for incremental training
- Optimization and parameter updates
- Loss tracking and convergence monitoring
- Checkpointing and model persistence

Key methods:
- `update_model`: Performs incremental parameter updates
- `infer`: Performs inference on new data
- `save_model`/`load_model`: Handles model persistence

### live_visualize.py

Provides visualization capabilities:

- `LiveVisualizer` class for real-time plots
- Time series visualization with state coloring
- Transition probability heatmaps
- State count tracking
- Tile visualization of state evolution

Key methods:
- `update_plot`: Updates all visualizations with new data
- `create_tile_visualization`: Creates tile plots of state assignments

### main.py

The entry point for running the system:

- Parameter configuration
- Component initialization
- Main processing loop
- Error handling and logging

### csv_processor.py

Handles offline processing of CSV files:

- `CSVDataProcessor` class for batch processing
- Loading and parsing multiple CSV files
- Implementing sliding window functionality
- Managing data across file boundaries
- Integration with the main HDP-HMM pipeline

Key methods:
- `load_csv_files`: Discovers and loads CSV files from a directory
- `get_next_window`: Retrieves the next window of data with proper handling of file transitions
- `get_total_windows`: Calculates the total number of windows for progress tracking
- `reset`: Resets the processor to start from the beginning

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

---

## Visualization Suite

The project includes a comprehensive visualization suite for understanding and interpreting model behavior. The visualization components are primarily implemented in `live_visualize.py` and focus on providing insights into state assignments, patterns, and model performance.

### Basic Time Series Visualization

The most fundamental visualization is the time series plot with state assignments:

```python
def update_plot(self, data, states, trans_probs, loss, losses, state_counts=[], state_changes=None):
    """Update live plot with new data."""
    # Plot time series and states
    for i in range(self.n_features):
        ax = self.fig.add_subplot(self.n_features + 2, 1, i + 1)
        ax.plot(data[:, i], label=f'Feature {i+1}')
        ax.scatter(range(len(states)), data[:, i], c=states, cmap='plasma', marker='x', label='Inferred States')
```

This visualization shows:
- Raw time series data for each feature
- State assignments overlaid as colored markers
- Current loss value and trend
- Number of active states

### State Pattern Analysis

The `visualize_state_patterns` method provides deep insights into what patterns each state represents:

```python
def visualize_state_patterns(self, data=None, states=None, save_path=None):
    """Create a comprehensive visualization showing what patterns each state represents."""
    # For each state, calculate and visualize:
    # - Mean pattern across features
    # - Standard deviation as shaded area
    # - Min/max range as error bars
    # - State frequency and median duration
```

This visualization helps answer key questions:
- What data pattern does each state represent?
- How consistent is each state (variance)?
- How frequently does each state occur?
- How long does each state typically last?

### Composite Visualizations

The `create_composite_state_visualization` method combines multiple views into a comprehensive dashboard:

```python
def create_composite_state_visualization(self, data=None, states=None, save_path=None):
    """
    Create a comprehensive visualization that combines:
    1. Time series data with state coloring
    2. State sequence visualization
    3. State pattern summaries
    4. Transition probabilities between states
    5. State duration histogram
    """
```

This composite view provides a holistic understanding of model behavior in a single visualization, particularly useful for presentations and reports.

### State Evolution Tracking

The system tracks how states evolve over time, including birth, merge, and delete operations:

```python
def create_state_evolution_plot(self, state_changes, save_path=None):
    """Create a visualization of state birth, merge, and delete events."""
    # Plot state count over time
    # Highlight births, merges, and deletions
    # Show detailed change counts in stacked bar chart
```

This visualization helps monitor model complexity:
- When and why new states are created
- When similar states are merged
- When unused states are deleted
- Overall trend in model complexity

### Learning Curves and Performance Metrics

The `create_learning_curve` method provides insights into model training progress:

```python
def create_learning_curve(self, losses, state_counts=None, save_path=None):
    """Create a detailed learning curve visualization for model performance debugging."""
    # Plot raw and smoothed loss values
    # Add exponential moving average trend
    # Mark minimum loss point
    # Calculate loss reduction percentage
    # Show correlation between loss and state count
```

Features of the learning curve visualization:

1. **Multiple Smoothing Methods**: Raw values, window smoothing, and exponential moving average
2. **Adaptive Window Size**: Automatically adjusts smoothing window based on data length
3. **Progress Indicators**: Displays loss reduction percentage and min/max points
4. **State Correlation**: Shows correlation between loss and number of states
5. **Log Scale Option**: Automatically adds log scale inset for wide-range loss values
6. **Dynamic Saving**: Creates both timestamped and latest versions for easy reference

This comprehensive visualization system provides valuable insights into how the model's state space evolves during training, helping users understand and debug the dynamic state management process.

---

## Offline CSV Processing

In addition to live streaming, the system supports offline processing of CSV files:

```python
class CSVDataProcessor:
    """
    Process CSV files for offline training of the HDP-HMM model.
    
    Each CSV file should have columns representing features, and rows representing time steps.
    """
```

The `CSVDataProcessor` class provides the following capabilities:

- Processes multiple CSV files in sequence
- Extracts features and creates sliding windows for model training
- Handles different window sizes and stride lengths
- Provides a uniform interface with the live data collector

This allows the system to work with historical data stored in CSV format, which is useful for:

1. **Retrospective Analysis**: Analyzing past data to discover patterns
2. **Model Development**: Testing and refining models on benchmark datasets
3. **Batch Processing**: Running the model on large datasets overnight or in a scheduled job

The implementation automatically detects the number of features from the CSV files and processes them in alphabetical order:

```python
def load_csv_files(self):
    """Load all CSV files from the data directory."""
    csv_pattern = os.path.join(self.data_dir, "*.csv")
    self.csv_files = sorted(glob.glob(csv_pattern))
    
    # Load the first file to determine the number of features
    first_df = pd.read_csv(self.csv_files[0])
    self.n_features = first_df.shape[1]
```

The sliding window approach allows for flexible processing with configurable window sizes and strides:

```python
def get_next_window(self):
    """Get the next window of data from the CSV files."""
    # Extract the window
    window_data = self.current_data[self.current_position:self.current_position + self.window_size]

    # Move to the next position based on stride
    self.current_position += self.stride
```

The processor also handles the transition between files automatically, ensuring a continuous stream of windows even when the data spans multiple CSV files:

```python
# Check if we need to load a new file
while self.current_position + self.window_size > len(self.current_data):
    self.current_file_idx += 1
    if self.current_file_idx >= len(self.csv_files):
        # No more files
        logger.info("Reached the end of all CSV files")
        return None
    
    # Load the next file
    logger.info(f"Loading next file: {self.csv_files[self.current_file_idx]}")
    self.current_data = pd.read_csv(self.csv_files[self.current_file_idx]).values
    self.current_position = 0
```

### Usage and Configuration

The offline CSV processing mode can be invoked with the following command:

```bash
python main.py --data-dir data --window-size 50 --stride 25 --n-features 3
```

Each CSV file should have the following format:
- Each column represents a feature
- Each row represents a time step
- The number of columns must be consistent across all files
- No header row is required (first row is treated as data)

For example:

```
0.5,1.2,0.8
0.6,1.3,0.7
0.7,1.4,0.6
...
```

The system will process all CSV files in alphabetical order, treating them as a continuous sequence of time series data.

#### Benefits of Offline Processing

The offline processing mode offers several advantages:

1. **Historical Analysis**: Process previously collected data to discover patterns and states
2. **Benchmarking**: Compare model performance across different datasets
3. **Parameter Tuning**: Test different window sizes, stride values, and model parameters
4. **Batch Processing**: Process large datasets overnight or as scheduled jobs
5. **Reproducibility**: Run the same analysis multiple times on fixed data

#### Comparison between Live and Offline Modes

| Aspect | Live Mode | Offline Mode |
|--------|-----------|--------------|
| Data Source | Real-time streams or simulated data | CSV files in a specified directory |
| Processing Speed | Limited by data collection rate | As fast as the processor can handle |
| Window Management | Fixed window size | Configurable window size and stride |
| Feature Detection | Fixed number of features | Automatically detected from CSV files |
| Initialization | Requires pre-filling the window | Directly loads from CSV files |
| End Condition | Manual stop or max windows | End of all CSV files or max windows |
| Applicable Use Cases | Real-time monitoring, online learning | Historical analysis, model development |
| Visualization | Real-time updates | Generated at intervals and end of processing |

#### Example Use Cases

1. **System Monitoring with Live Mode**:
   ```bash
   python main.py --window-size 100 --n-features 3
   ```
   This setup monitors system metrics in real-time, continuously updating the model as new data arrives.

2. **Historical Data Analysis with Offline Mode**:
   ```bash
   python main.py --data-dir historical_data --window-size 50 --stride 25
   ```
   This processes historical data stored in CSV files to discover patterns and states.

3. **Benchmark Testing with Offline Mode**:
   ```bash
   python main.py --data-dir benchmark_dataset --window-size 100 --no-gui
   ```
   This runs the model on a standard benchmark dataset without GUI updates for faster processing.

4. **Model Parameter Tuning with Offline Mode**:
   ```bash
   for ws in 50 100 200; do
     for stride in 25 50 100; do
       python main.py --data-dir data --window-size $ws --stride $stride --max-windows 20
     done
   done
   ```
   This script tests different window sizes and strides on the same dataset for parameter optimization.

The dual-mode capability makes the HDP-HMM implementation flexible for both real-time and batch processing applications while maintaining consistent visualization and analysis capabilities.

---

## Visualization Suite

The project includes a comprehensive visualization suite for understanding and interpreting model behavior. The visualization components are primarily implemented in `live_visualize.py` and focus on providing insights into state assignments, patterns, and model performance.

### Basic Time Series Visualization

The most fundamental visualization is the time series plot with state assignments:

```python
def update_plot(self, data, states, trans_probs, loss, losses, state_counts=[], state_changes=None):
    """Update live plot with new data."""
    # Plot time series and states
    for i in range(self.n_features):
        ax = self.fig.add_subplot(self.n_features + 2, 1, i + 1)
        ax.plot(data[:, i], label=f'Feature {i+1}')
        ax.scatter(range(len(states)), data[:, i], c=states, cmap='plasma', marker='x', label='Inferred States')
```

This visualization shows:
- Raw time series data for each feature
- State assignments overlaid as colored markers
- Current loss value and trend
- Number of active states

### State Pattern Analysis

The `visualize_state_patterns` method provides deep insights into what patterns each state represents:

```python
def visualize_state_patterns(self, data=None, states=None, save_path=None):
    """Create a comprehensive visualization showing what patterns each state represents."""
    # For each state, calculate and visualize:
    # - Mean pattern across features
    # - Standard deviation as shaded area
    # - Min/max range as error bars
    # - State frequency and median duration
```

This visualization helps answer key questions:
- What data pattern does each state represent?
- How consistent is each state (variance)?
- How frequently does each state occur?
- How long does each state typically last?

### Composite Visualizations

The `create_composite_state_visualization` method combines multiple views into a comprehensive dashboard:

```python
def create_composite_state_visualization(self, data=None, states=None, save_path=None):
    """
    Create a comprehensive visualization that combines:
    1. Time series data with state coloring
    2. State sequence visualization
    3. State pattern summaries
    4. Transition probabilities between states
    5. State duration histogram
    """
```

This composite view provides a holistic understanding of model behavior in a single visualization, particularly useful for presentations and reports.

### State Evolution Tracking

The system tracks how states evolve over time, including birth, merge, and delete operations:

```python
def create_state_evolution_plot(self, state_changes, save_path=None):
    """Create a visualization of state birth, merge, and delete events."""
    # Plot state count over time
    # Highlight births, merges, and deletions
    # Show detailed change counts in stacked bar chart
```

This visualization helps monitor model complexity:
- When and why new states are created
- When similar states are merged
- When unused states are deleted
- Overall trend in model complexity

### Learning Curves and Performance Metrics

The `create_learning_curve` method provides insights into model training progress:

```python
def create_learning_curve(self, losses, state_counts=None, save_path=None):
    """Create a detailed learning curve visualization for model performance debugging."""
    # Plot raw and smoothed loss values
    # Add exponential moving average trend
    # Mark minimum loss point
    # Calculate loss reduction percentage
    # Show correlation between loss and state count
```

Features of the learning curve visualization:

1. **Multiple Smoothing Methods**: Raw values, window smoothing, and exponential moving average
2. **Adaptive Window Size**: Automatically adjusts smoothing window based on data length
3. **Progress Indicators**: Displays loss reduction percentage and min/max points
4. **State Correlation**: Shows correlation between loss and number of states
5. **Log Scale Option**: Automatically adds log scale inset for wide-range loss values
6. **Dynamic Saving**: Creates both timestamped and latest versions for easy reference

This comprehensive visualization system provides valuable insights into how the model's state space evolves during training, helping users understand and debug the dynamic state management process.

---
