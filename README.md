# Bayesian Non-Parametric Modeling with HDP-HMM for Live Streaming Data

This project implements a Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) with stick-breaking construction for unsupervised learning of state sequences in multidimensional time series data. The model is specifically designed to work with live streaming data such as system metrics (CPU utilization, temperature, RAM usage, etc.) and supports incremental training, real-time inference, and visualization.

The implementation is inspired by and builds upon the theoretical foundations of the [bnpy library](https://github.com/bnpy/bnpy), with a focus on GPU acceleration and real-time streaming data processing using PyTorch.

## Features

- **Bayesian non-parametric modeling** using HDP-HMM to automatically determine the number of states
- **PyTorch implementation** with GPU acceleration for faster training and inference
- **Live data streaming** support with sliding window processing
- **Incremental model updates** for continuous learning
- **Real-time visualization** of time series data, state assignments, and transition probabilities
- **Model persistence** with checkpointing for resuming training
- **Performance monitoring** for tracking training and inference times

## Project Structure

```
bnpy_gpu/
├── config.json              # Configuration file
├── main.py                  # Main entry point
├── models/                  # Saved models directory
├── plots/                   # Output plots directory
└── src/                     # Source code
    ├── data/                # Data collection and processing
    │   ├── collector.py     # Live data collection module
    │   └── processor.py     # Data preprocessing utilities
    ├── model/               # Model implementation
    │   ├── hdp_hmm.py       # HDP-HMM model implementation
    │   └── trainer.py       # Training and inference module
    ├── utils/               # Utility functions
    │   └── utils.py         # Helper utilities and performance monitoring
    └── visualization/       # Visualization tools
        └── visualizer.py    # Real-time visualization module
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Seaborn
- psutil (optional, for real system metrics)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/bnpy_gpu.git
cd bnpy_gpu
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage

Run the model with default settings:

```bash
python main.py
```

### Quick Demo

For a quicker demonstration with simulated data:

```bash
python demo.py
```

### Using Real System Metrics

To use real system metrics instead of simulated data:

```bash
python main.py --use-real
```

### Running Without GUI

For headless operation (e.g., on a server):

```bash
python main.py --no-gui
```

### Custom Configuration

Specify a custom configuration file:

```bash
python main.py --config my_config.json
```

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 1.8+ (with CUDA support recommended)
- NumPy
- Matplotlib
- Seaborn
- psutil (for real system metrics)

Ensure a CUDA-compatible GPU for acceleration (the code falls back to CPU if unavailable).

## Running Notes

- **Real Data**: When using real system metrics, ensure psutil can access all required sensors. Some metrics (like temperature) may not be available on all systems.
- **Incremental Training**: The model updates with each new window, allowing adaptation to changing system dynamics.
- **Visualization**: The interactive plot updates every `sample_interval` seconds. Adjust `window_size` and `sample_interval` based on your system's data rate.
- **Model Persistence**: The model is saved periodically and on exit, enabling resumption or fine-tuning.
- **Scalability**: For high-frequency data, consider increasing the `sample_interval` or reducing `max_states` to improve performance.

## Configuration

The `config.json` file contains all configurable parameters:

- **model**: Model hyperparameters (n_features, max_states, alpha, gamma, learning_rate)
- **data**: Data collection settings (window_size, sample_interval, use_real_metrics)
- **training**: Training parameters (max_iterations, save_interval, checkpoint_interval)
- **visualization**: Visualization options (interactive, save_plots, feature_names)
- **paths**: Directory paths for saving models and plots

## Theoretical Background

### Hierarchical Dirichlet Process Hidden Markov Models (HDP-HMM)

The Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) is a Bayesian nonparametric extension of the traditional Hidden Markov Model that automatically infers the appropriate number of hidden states from data. This section provides the theoretical foundations of HDP-HMM and its inference algorithms.

#### The Dirichlet Process and Stick-Breaking Construction

At the core of HDP-HMM is the Dirichlet Process (DP), a distribution over distributions. The DP is parameterized by a concentration parameter α and a base distribution H. A draw G ~ DP(α, H) is itself a distribution. The stick-breaking construction (Sethuraman, 1994) provides a concrete representation of the DP:

G = ∑_{k=1}^∞ β_k δ_{θ_k}

where:
- β_k are mixing weights determined by a stick-breaking process
- δ_{θ_k} is a point mass at θ_k
- θ_k ~ H are parameters drawn from the base distribution

The stick-breaking weights β_k are constructed as:

β_k = v_k ∏_{l=1}^{k-1} (1 - v_l)

where v_k ~ Beta(1, α).

#### Hierarchical Dirichlet Process (HDP)

For the HDP-HMM, we need a hierarchical extension of the DP. The HDP (Teh et al., 2006) defines a set of distributions {G_j} that share atoms, where each G_j ~ DP(α, G_0) and G_0 ~ DP(γ, H). This hierarchical structure enables state sharing across different time steps in the HDP-HMM.

#### HDP-HMM Formulation

In the HDP-HMM, each state i has a transition distribution π_i over next states, with all transition distributions sharing the same set of states. The generative process is:

1. Draw G_0 ~ DP(γ, H)
2. For each state i, draw transition distribution G_i ~ DP(α, G_0)
3. For each time step t:
   - Draw state z_t ~ π_{z_{t-1}}
   - Draw observation x_t ~ F(θ_{z_t})

where F(θ) is the emission distribution with parameters θ.

#### Sticky HDP-HMM

The standard HDP-HMM can exhibit unrealistic rapid switching between states. The Sticky HDP-HMM (Fox et al., 2008) addresses this by adding self-transition bias:

G_i ~ DP(α + κ, (αG_0 + κδ_i)/(α + κ))

where κ > 0 increases the probability of self-transitions.

### Memoized Online Variational Inference

The paper "Memoized Online Variational Inference for Dirichlet Process Mixture Models" (Hughes & Sudderth, 2013) introduces an efficient inference algorithm applicable to HDP-HMM and other nonparametric models. This approach combines the advantages of stochastic online methods with the ability to revisit past decisions.

#### Key Components of Memoized Variational Inference

1. **Variational Approximation**: Approximates the true posterior p(z|x) with a simpler distribution q(z) by minimizing the KL divergence.

2. **Memoization**: Maintains a summary of past assignments, allowing the algorithm to revisit and refine previous decisions.

3. **Birth and Merge Moves**: Enables the algorithm to create new states ("birth") or combine similar states ("merge"), allowing flexible model complexity.

4. **Scalable Learning**: Can process data in small batches, making it suitable for large datasets and streaming applications.

The inference algorithm proceeds by:
1. Processing a batch of data
2. Updating local parameters for the current batch
3. Updating global parameters using sufficient statistics
4. Performing birth/merge moves to adjust model complexity
5. Memoizing the updated state

This approach is particularly well-suited for live streaming applications, as it can incrementally update the model as new data arrives while maintaining a coherent global state representation.

### Relationship to bnpy Implementation

The [bnpy library](https://github.com/bnpy/bnpy) provides implementations of these algorithms and models, including:

- Full-dataset variational inference (VB)
- Memoized variational inference (moVB)
- Stochastic online variational inference (soVB)

Our GPU-accelerated implementation builds on these theoretical foundations while focusing on real-time performance for live data streams. The primary innovations in our implementation include:

1. PyTorch-based GPU acceleration
2. Specialized data structures for sliding window processing
3. Enhanced visualization capabilities for real-time monitoring
4. Incremental updates optimized for streaming data

For more details on the theoretical aspects and original implementations, see the following papers:
- "Memoized online variational inference for Dirichlet process mixture models." Hughes & Sudderth (NIPS 2013)
- "Scalable adaptation of state complexity for nonparametric hidden Markov models." Hughes, Stephenson & Sudderth (NIPS 2015)

For real system metrics, you can replace the simulation with psutil calls:

```python
def get_sample(self):
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = psutil.sensors_temperatures()['coretemp'][0].current
    return torch.tensor([cpu, temp, ram], device=self.device)
```

### 2. HDP-HMM Model (`src/model/hdp_hmm.py`)

The core model implementation using Hierarchical Dirichlet Process Hidden Markov Model with stick-breaking construction.

Key features:
- PyTorch implementation with GPU acceleration
- Stick-breaking process for automatic state determination
- Gaussian emission model for continuous observations
- Forward-backward algorithm for inference
- Viterbi algorithm for most likely state sequence
- Model persistence (save/load) functionality
- Posterior predictive distribution for forecasting

### 3. Live Training Module (`src/model/trainer.py`)

This module handles incremental training and live inference on streaming data.

Key features:
- Incremental model updates with new data windows
- Automatic learning rate adjustment
- Periodic model checkpointing
- GPU-accelerated training and inference
- Loss tracking and performance statistics
- Model persistence with versioned checkpoints

### 4. Live Visualization (`src/visualization/visualizer.py`)

This module provides real-time visualization of model outputs and data.

Key features:
- Interactive real-time plots of time series with state assignments
- Transition probability heatmaps
- State distribution histograms
- Training loss curves
- Support for both interactive and headless operation
- Automatic plot saving at configurable intervals

### 5. Data Processor (`src/data/processor.py`)

This module handles data preprocessing before feeding to the model.

Key features:
- Online normalization of streaming data
- Feature extraction from raw time series
- Sliding window management
- Adaptive statistics calculation

## How It Works

### Live Data Processing Pipeline

1. **Data Collection**: 
   - The `LiveDataCollector` class collects real-time system metrics or simulates them
   - It maintains a sliding window of `window_size` samples

2. **Data Processing**: 
   - The `TimeSeriesProcessor` normalizes and extracts features from the raw data
   - Adaptive statistics ensure proper scaling as data distribution changes

3. **Incremental Training**: 
   - The `LiveTrainer` class trains the model incrementally on each new window of data
   - The model adapts to changing patterns in the data stream
   - Training occurs on GPU when available for faster updates

4. **State Inference**: 
   - After training, the model infers the most likely state sequence using the Viterbi algorithm
   - It also computes the transition probability matrix between states
   - The number of active states is automatically determined by the HDP prior

5. **Visualization**: 
   - The `LiveVisualizer` class displays real-time plots of:
     - Time series data with colored state assignments
     - State distribution histograms
     - Transition probability heatmaps
     - Training loss curves

6. **Model Persistence**: 
   - The model is periodically saved to disk (in the `models/` directory)
   - Checkpoints allow resuming training or deploying the trained model
   - Optimizer state is also saved for consistent training resumption

### HDP-HMM Model

The Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) is a Bayesian non-parametric extension of the standard HMM that automatically determines the appropriate number of states. Key features of our implementation:

- **Stick-breaking construction**: Instead of fixing the number of states in advance, the model uses a stick-breaking process to determine the appropriate number of states from data
- **Gaussian emission model**: Each state is associated with a multivariate Gaussian distribution for modeling continuous observations
- **Forward-backward algorithm**: Used for computing posterior state probabilities and model likelihood
- **Viterbi algorithm**: Efficiently finds the most likely state sequence given observations
- **Posterior predictive**: Provides forecasting capability for future observations
- **GPU acceleration**: All tensor operations are GPU-compatible for faster training and inference

## Dynamic State Management with Birth, Merge, and Delete Mechanisms

The HDP-HMM implementation includes dynamic state management mechanisms that allow the model to adaptively adjust the number of states based on data evidence. These mechanisms help maintain model complexity that best fits the data without unnecessary computational overhead.

### Theoretical Background

While traditional HDP-HMM relies solely on Bayesian nonparametric priors to determine the state space, explicit birth, merge, and delete mechanisms provide additional control over model complexity during inference:

1. **Birth**: Creates new states when the model detects regions of observations that are poorly explained by existing states
2. **Merge**: Combines states with similar emission distributions to reduce redundancy
3. **Delete**: Removes states with negligible probability mass to improve computational efficiency

These mechanisms are inspired by the "split-merge" and "birth-death" MCMC algorithms in Bayesian nonparametrics, but adapted for online variational inference settings.

### Implementation Details

The dynamic state management is implemented in the `update_states` method of the `HDPHMM` class:

- **Delete**: States with beta weights below a threshold (default: 1e-3) are marked as inactive
- **Merge**: States with means closer than a threshold (default: 0.5 in normalized space) are combined by weighted averaging
- **Birth**: When model fit is poor (high negative log-likelihood), a new state is initialized using observations that are poorly explained

### State Evolution Visualization

The implementation includes comprehensive state tracking and visualization tools to monitor how these mechanisms affect model complexity:

1. **Real-time Console Output**: 
   - Detailed information about state updates during training
   - Clear display of birth, merge, and delete events with affected state IDs
   - Statistics on state counts and changes

2. **State Evolution Timeline**:
   - Text-based visualization showing state evolution across training windows
   - Symbols indicate active states (●), births (⊕), merges (⊙), and deletions (⊗)
   - Helps identify patterns in how states are created, merged, and deleted

3. **State Evolution Summary**:
   - Detailed final report on all state changes during training
   - Counts of birth, merge, and delete events by state ID
   - Identifies which states were most frequently created, merged, or deleted

This enhanced state tracking helps debug model behavior and fine-tune hyperparameters for optimal state management.

#### Birth Mechanism

The birth mechanism monitors the average negative log-likelihood of observations and creates new states when the model fit is poor:

- **When**: When the average negative log-likelihood exceeds a threshold and the current number of states is below the maximum
- **How**: A new state is initialized using the mean and variance of poorly fit observations
- **Why**: Improves model flexibility by adding states where needed

#### Merge Mechanism

The merge mechanism identifies and combines states with similar emission distributions:

- **When**: When two states have emission means closer than a specified distance threshold
- **How**: Parameters are combined through weighted averaging based on state probabilities
- **Why**: Reduces redundancy and prevents unnecessary state proliferation

#### Delete Mechanism

The delete mechanism removes states with negligible probability:

- **When**: When a state's beta weight falls below a specified threshold
- **How**: The state is marked inactive and its parameters are no longer updated
- **Why**: Improves computational efficiency by focusing on meaningful states

### Tuning Parameters

The behavior of these mechanisms can be adjusted through the following parameters:

| Parameter | Default | Description | Effect of Increasing | Effect of Decreasing |
|-----------|---------|-------------|---------------------|----------------------|
| `delete_threshold` | 1e-3 | Minimum beta weight for a state to remain active | More aggressive state pruning, fewer active states | Less pruning, more states preserved |
| `merge_distance` | 0.5 | Maximum Euclidean distance between means for state merging | More aggressive merging, fewer distinct states | Less merging, more distinct states |
| `birth_threshold` | 10.0 | Negative log-likelihood threshold for creating new states | Fewer new states created | More new states created |

### Monitoring State Dynamics

The implementation includes visualization of state counts over time, allowing you to monitor how these mechanisms affect model complexity:

- A dedicated plot shows the number of active states over time
- Logs report the current number of active states periodically
- The transition probability heatmap adjusts to show only active states

### Tuning Recommendations

- **High Noise Data**: Increase `delete_threshold` (e.g., 5e-3) and `merge_distance` (e.g., 1.0) to prevent noise from creating spurious states
- **Complex Systems**: Decrease `birth_threshold` (e.g., 5.0) to allow more states to capture complex patterns
- **Computational Efficiency**: Increase `delete_threshold` and `birth_threshold` to maintain fewer states
- **High Precision**: Decrease `merge_distance` (e.g., 0.3) to prevent merging of potentially distinct states

These mechanisms work together to maintain an optimal number of states that balances model complexity with computational efficiency. By properly tuning these parameters, you can ensure the model adapts appropriately to your specific data characteristics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Extending the Solution

### Adding New Data Sources

You can extend the `LiveDataCollector` class to support additional data sources:

```python
class CustomDataCollector(LiveDataCollector):
    def __init__(self, api_key, endpoint, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.endpoint = endpoint
        
    def get_sample(self):
        # Implement custom data collection logic
        # For example, fetch data from an API
        response = requests.get(self.endpoint, headers={"Authorization": self.api_key})
        data = response.json()
        return torch.tensor([data["metric1"], data["metric2"], data["metric3"]], 
                           device=self.device)
```

### Customizing the Model

You can modify the emission model in `hdp_hmm.py` to better suit your data:

```python
# Example: Change from Gaussian to Student's t-distribution for robustness to outliers
def compute_emission_probs(self, observations):
    T = observations.shape[0]
    emission_probs = torch.zeros(T, self.max_states, device=self.device)
    
    for k in range(self.max_states):
        # Student's t-distribution with df=5 for heavier tails
        t_dist = dist.StudentT(df=5, loc=self.means[k], 
                             scale=torch.exp(self.log_vars[k]/2))
        emission_probs[:, k] = torch.sum(t_dist.log_prob(observations), dim=1)
        
    return emission_probs
```

### Adding Anomaly Detection

You can extend the system to detect anomalies based on state transitions or unlikely observations:

```python
def detect_anomalies(self, window_data, states, trans_probs):
    # 1. Rare state transitions
    state_counts = torch.bincount(states)
    rare_states = torch.where(state_counts < 3)[0]
    rare_state_indices = torch.where(torch.isin(states, rare_states))[0]
    
    # 2. Unlikely observations (low emission probability)
    emission_probs = self.model.compute_emission_probs(window_data)
    state_emission_probs = emission_probs[torch.arange(len(states)), states]
    unlikely_obs = torch.where(state_emission_probs < -10)[0]
    
    return {
        "rare_state_indices": rare_state_indices,
        "unlikely_observation_indices": unlikely_obs
    }
```

### Tile Visualization

In addition to the real-time plots of time series data with state assignments, the system also generates state tile visualizations inspired by the bnpy visualization module:

- **State Tiles**: Shows state assignments over time as a color-coded tile grid
  - Each row represents a time point within a window
  - Each column represents a different window
  - Colors represent different state assignments
  - Helps visualize state persistence and transitions over time

- **Transition Matrix**: Below the tiles, a heatmap shows the normalized transition probabilities between states
  - Darker colors indicate more frequent transitions
  - Annotations show the probability values
  - Only counts actual transitions (when state changes)

These visualizations are automatically saved to the `plots/` directory every 5 windows as `state_tiles_window_X.png` where X is the window number.

The tile visualization is particularly useful for:

- Identifying state persistence patterns
- Detecting unusual state transitions
- Visualizing the effects of the birth, merge, and delete mechanisms on state assignments
- Tracking how state assignments evolve over longer time periods

![Example Tile Visualization](plots/state_tiles_window_example.png)

### State Sequence Visualization

In addition to the tile visualization, the system also generates state sequence visualizations inspired by the bnpy visualization style:

- **Top Panel**: Shows the time series data for each feature
- **Bottom Panel**: Displays state assignments as a color-coded band
- **Colorbar**: Maps colors to state indices

This visualization provides a clear view of how state assignments relate to the underlying data features and helps identify patterns such as:

- State transitions triggered by specific data patterns
- Periods of state persistence
- The relationship between feature values and state assignments

The state sequence visualizations are automatically saved to the `plots/` directory as `state_sequence_window_X.png` where X is the window number.

![Example State Sequence Visualization](plots/state_sequence_window_example.png)

### Transition Matrix Exports

The system automatically saves the transition probability matrix in multiple formats:

- **PNG Image**: Heatmap visualization of the transition probabilities
- **NumPy File (.npy)**: Raw numerical data for programmatic analysis
- **CSV File (.csv)**: Tab-delimited format for easy import into spreadsheets or other tools

These files are saved at several points:
- **During visualization**: Updated with each window and saved to `plots/transition_probs.png`
- **Periodically during training**: Saved along with model checkpoints to `plots/latest_transition_matrix.*`
- **At the end of processing**: Final matrix saved to `plots/final_transition_matrix.*`

The transition matrix is a key output of the model, showing:
- The probability of transitioning from one state to another
- State persistence (diagonal values)
- Rare transitions (off-diagonal low-probability transitions)
- Potential state clusters (groups of states with high transition probabilities between them)

These exports facilitate further analysis of the discovered state dynamics, enabling integration with other analysis tools and visualization software.

### Learning Curve Visualization

The system provides comprehensive learning curve visualizations to track model performance and help with debugging:

- **Training Loss Curve**: Shows the negative log-likelihood evolution over time
- **Smoothed Trends**: Includes both windowed smoothing and exponential moving average
- **State Count Correlation**: Visualizes the relationship between state count changes and loss values
- **Optimization Metrics**: Displays percentage reduction in loss and highlights minimum loss points

Key features of the learning curve visualization:

1. **Multiple Loss Representations**:
   - Raw loss values for detailed inspection
   - Smoothed trends to identify overall patterns
   - Exponential moving average for stable trend visualization
   - Log-scale inset for handling large dynamic ranges

2. **State Count Integration**:
   - State count plot aligned with loss curve
   - Correlation coefficient between loss and state count
   - Annotations for significant state count changes

3. **Performance Metrics**:
   - Percentage reduction in loss from initial to current value
   - Identification of minimum loss points with timing information
   - Visual indicators of model convergence

The learning curve visualizations are automatically saved to the `plots/` directory as `learning_curve_window_X.png` and `latest_learning_curve.png`.

![Example Learning Curve Visualization](plots/learning_curve_example.png)

## References

1. Hughes, M. C., & Sudderth, E. B. (2013). Memoized Online Variational Inference for Dirichlet Process Mixture Models. In Advances in Neural Information Processing Systems (NIPS).

2. Hughes, M. C., Stephenson, W. T., & Sudderth, E. B. (2015). Scalable Adaptation of State Complexity for Nonparametric Hidden Markov Models. In Advances in Neural Information Processing Systems (NIPS).

3. Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2008). An HDP-HMM for Systems with State Persistence. In Proceedings of the 25th International Conference on Machine Learning (ICML).

4. Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet Processes. Journal of the American Statistical Association, 101(476), 1566-1581.

5. Sethuraman, J. (1994). A Constructive Definition of Dirichlet Priors. Statistica Sinica, 4, 639-650.

6. bnpy: Bayesian Nonparametric Machine Learning for Python. (n.d.). GitHub repository: https://github.com/bnpy/bnpy
