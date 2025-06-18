# Bayesian Non-Parametric Modeling with HDP-HMM for Live Streaming Data

This project implements a Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) with stick-breaking construction for unsupervised learning of state sequences in multidimensional time series data. The model is specifically designed to work with live streaming data such as system metrics (CPU utilization, temperature, RAM usage, etc.) and supports incremental training, real-time inference, and comprehensive visualization.

The implementation is inspired by and builds upon the theoretical foundations of the [bnpy library](https://github.com/bnpy/bnpy), with a focus on GPU acceleration and real-time streaming data processing using PyTorch. It features dynamic state management (birth, merge, delete) for automatic discovery of the optimal number of states.

## Features

- **Bayesian non-parametric modeling** using HDP-HMM to automatically determine the number of states
- **Dynamic state management** with birth, merge, and delete operations for adaptive model complexity
- **PyTorch implementation** with GPU acceleration for faster training and inference
- **Dual-mode operation**:
  - Live streaming data processing for real-time analysis
  - Offline batch processing from CSV files for historical data analysis
- **Incremental model updates** for continuous learning
- **Comprehensive visualization suite**:
  - Time series data with state assignments
  - State pattern analysis showing what each state represents
  - State-specific time series visualizations for each discovered state
  - State evolution tracking with birth/merge/delete events
  - Transition probability heatmaps
  - Learning curves and model performance metrics
  - Composite visualizations combining multiple views
- **Model persistence** with checkpointing for resuming training
- **Performance monitoring** for tracking training and inference times
- **Robust error handling** with headless operation support for deployment in production environments

## Project Structure

```
LatentDynamicsBayes/
├── config.json              # Configuration file
├── main.py                  # Main entry point
├── hdp_hmm.py               # Core HDP-HMM implementation
├── live_train.py            # Live training module
├── live_data_collector.py   # Data collection for live streaming
├── csv_processor.py         # Offline CSV data processing
├── generate_sample_data.py  # Generate sample CSV data for testing
├── live_visualize.py        # Comprehensive visualization suite
├── demo.py                  # Quick demonstration script
├── data/                    # Directory for CSV files (created by generate_sample_data.py)
│   ├── sample_data_1.csv    # Sample CSV data file
│   ├── sample_data_2.csv    # Sample CSV data file
│   └── sample_data_3.csv    # Sample CSV data file
├── models/                  # Saved models directory
│   ├── hdp_hmm.pth          # Trained model
│   └── hdp_hmm_checkpoint.pth # Training checkpoint
├── plots/                   # Generated visualizations
│   ├── live_plot/           # Live time series plots subdirectory
│   ├── state_patterns/      # State pattern visualizations subdirectory
│   ├── state_time_series/   # State-specific time series analysis subdirectory
│   ├── state_evolution/     # State change tracking subdirectory
│   ├── learning_curve/      # Training progress subdirectory
│   ├── composite_viz/       # Combined visualizations subdirectory
│   ├── state_sequence/      # State sequence visualizations subdirectory
│   ├── state_tiles/         # State tile visualizations subdirectory
│   ├── transition_matrix/   # Transition probability matrices subdirectory
│   └── *_latest.png         # Latest plots kept in the main directory
└── src/                     # Source code modules
    ├── data/                # Data processing
    │   ├── collector.py     # Data collection utilities
    │   └── processor.py     # Data preprocessing utilities
    ├── model/               # Model components
    │   ├── hdp_hmm.py       # Alternative HDP-HMM implementation
    │   └── trainer.py       # Training and inference module
    ├── utils/               # Utility functions
    │   └── utils.py         # Helper utilities
    └── visualization/       # Visualization components
        └── visualizer.py    # Visualization utilities
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
git clone https://github.com/yourusername/LatentDynamicsBayes.git
cd LatentDynamicsBayes
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage (Live Mode)

Run the model with default settings using synthetic data:

```bash
python main.py
```

### Offline Mode with CSV Files

Process data from CSV files stored in a directory:

```bash
python main.py --data-dir data --window-size 50 --stride 25 --n-features 3
```

Each CSV file should have columns representing features and rows representing time steps. The files will be processed in alphabetical order. Multiple CSV files can be used and will be processed sequentially.

Parameters for offline processing:
- `--data-dir`: Directory containing CSV files (required for offline mode)
- `--window-size`: Size of the sliding window in time steps
- `--stride`: Number of time steps to advance between windows (defaults to window_size)
- `--n-features`: Number of features in the data (automatically detected from CSV files)

#### Generating Sample Data

To generate sample CSV files for testing the offline mode:

```bash
python generate_sample_data.py
```

This will create CSV files in the `data/` directory with synthetic time series data that contains known state patterns.

### Quick Demo

For a quicker demonstration with simulated data:

```bash
python demo.py
```

### Using Real System Metrics

To use real system metrics instead of simulated data (live mode):

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

### Command Line Arguments

The following command-line arguments are available:

```
--no-gui           Run without GUI visualization
--config CONFIG    Path to config file
--max-windows N    Maximum number of windows to process (default: 1000)
--data-dir DIR     Directory containing CSV files for offline processing
--window-size N    Size of sliding window (default: 100)
--stride N         Stride for sliding window in offline mode (default: window_size)
--n-features N     Number of features in the data (default: 3)
```

#### Offline Mode Parameters

When running in offline mode (with `--data-dir`), the following parameters control the CSV processing:

- `--data-dir`: Path to directory containing CSV files. Setting this parameter enables offline mode.
- `--window-size`: Number of time steps to include in each processing window.
- `--stride`: Number of time steps to advance between consecutive windows:
  - When `stride = window_size`: Windows are non-overlapping
  - When `stride < window_size`: Windows overlap by (window_size - stride) time steps
  - When `stride > window_size`: Some time steps are skipped between windows
- `--n-features`: Number of features in the data. In offline mode, this is automatically detected from the CSV files.

#### CSV File Format

Each CSV file should have the following format:
- Each column represents a feature
- Each row represents a time step
- No header row is required (first row is treated as data)
- Files are processed in alphabetical order

Example CSV content:
```
0.5,1.2,0.8
0.6,1.3,0.7
0.7,1.4,0.6
...
```

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

## Plot Organization

The visualizations are organized to maintain a clean and efficient directory structure:

1. **Main Directory Plots** - Only the most important plots are kept in the main `plots/` directory:
   - **Latest plots** (e.g., `learning_curve_latest.png`) - Always show the most recent state
   - **Final plots** (e.g., `final_state_evolution.png`) - Created at the end of a run

2. **Subdirectories** - All intermediate plots are organized into type-specific subdirectories:
   - `plots/live_plot/` - Time series with state assignments for each window
   - `plots/state_patterns/` - Patterns representing each state
   - `plots/state_time_series/` - Detailed time series for each state
   - `plots/state_evolution/` - Tracking of state changes (birth, merge, delete)
   - `plots/learning_curve/` - Model learning progress
   - `plots/composite_viz/` - Combined multi-panel visualizations
   - `plots/state_sequence/` - State sequence visualizations
   - `plots/state_tiles/` - Tile visualizations of state assignments
   - `plots/transition_matrix/` - Transition probability matrices

This organization ensures that the main directory remains uncluttered while maintaining a complete history of visualizations for analysis. You can review the latest state at a glance in the main directory, or explore the full history in the subdirectories.

## Code Organization

This repository is organized with a dual structure for flexibility:

1. **Root-level implementation** (hdp_hmm.py, live_train.py, live_visualize.py):
   - Provides a simple, integrated interface for quick experimentation
   - Imports core functionality from the src modules
   - Main entry point for most use cases

2. **Modular src structure** (src/model, src/data, etc.):
   - Contains the core implementation details
   - More maintainable and testable code structure
   - Designed for integration into larger applications

This approach allows for both rapid prototyping (using the root-level files) and integration into larger systems (using the src modules directly).

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

## Enhanced State Visualization Features

The system now includes several advanced visualization features for better understanding and debugging of state dynamics:

### State Evolution Visualization

A dedicated visualization shows the birth, merge, and delete events across training windows:

- **State Count Plot**: Tracks the number of active states over time, with markers for state change events
- **Change Event Counts**: Shows the number of birth, merge, and delete events per update window
- **State Timeline**: A textual representation of state evolution across training windows

### Detailed State Change Reporting

The system provides comprehensive reporting of state changes during and after training:

- **Real-time Updates**: Prints detailed information about state changes during training
- **Change Summary Format**: `States: 9 → 10 | Changes: +1 birth, ~2 merges | Details: Birth: state(s) 12; Merge: 5→3, 8→4`
- **Final Summary**: At the end of training, shows total birth, merge, and delete events by state

### Learning Curve Visualization

An enhanced learning curve visualization for model performance debugging:

- **Raw and Smoothed Loss**: Shows both raw loss values and smoothed trends (with moving averages)
- **State Count Correlation**: Visualizes the relationship between state count changes and loss value
- **Performance Metrics**: Displays loss reduction percentage and minimum loss value

### Robust Visualization System

All visualizations are now more robust across different systems:

- **Headless Support**: Full visualization support in headless/no-GUI mode
- **Error Handling**: Graceful handling of visualization errors without crashing the main training loop
- **File Outputs**: All visualizations are saved to the `plots/` directory with consistent naming

### Interpreting Visualizations

The visualizations provide key insights into model behavior:

- **Optimal State Count**: The state evolution plot helps identify when the model reaches a stable number of states
- **Convergence**: The learning curve shows when the model has converged to a stable solution
- **State Dynamics**: The state change reports reveal which states are most active, merged, or deleted

These enhanced visualization features make it much easier to debug, understand, and optimize the model's behavior, particularly how it determines the optimal number of states.

## Visualization Features

The project includes comprehensive visualization capabilities for real-time monitoring and analysis:

- **Time Series Visualization**: Display of multidimensional time series data with state assignments
- **State Evolution Tracking**: Visualization of state birth, merge, and delete events over time
- **Transition Probability Heatmaps**: Visual representation of state transition dynamics
- **Learning Curve Analysis**: Detailed loss tracking with smoothing and state correlation
- **Tile Plots**: Visualization of state assignments across multiple windows
- **State Sequence Visualization**: Detailed view of state transitions within a window
- **State Pattern Analysis**: Visual representation of what patterns each state represents
- **Comprehensive State Reports**: Composite visualizations showing state statistics, durations, and transitions

All visualizations are designed with robustness in mind:
- Compatible with both GUI and headless environments
- Error handling to prevent crashes during visualization
- Automatic plot directory creation and management
- Cross-platform layout compatibility using `subplots_adjust`
- Both interactive display and file saving capabilities

For detailed interpretation guidance, see the [Visualization Interpretation](#visualization-interpretation) section.

## Visualization Interpretation

### State Evolution Plot

The state evolution plot shows how the number of states changes over time:
- **Green triangles (▲)**: Birth of new states
- **Orange circles (●)**: Merge of similar states
- **Red triangles (▼)**: Deletion of inactive states

This plot helps identify when the model adjusts its complexity to fit the data better.

### Learning Curve Plot

The learning curve plot shows the model's loss over time:
- **Blue line**: Raw loss values
- **Red line**: Smoothed trend
- **Green line**: Exponential moving average

A decreasing trend indicates the model is learning effectively. Plateaus may suggest convergence or local minima.

### Tile Plot

The tile plot shows state assignments across multiple windows:
- Each row represents a time point
- Each column represents a window
- Colors represent different states

This visualization helps identify patterns of state persistence and transitions over time.

### Transition Probability Heatmap

The transition probability heatmap shows the likelihood of transitioning between states:
- Rows represent "from" states
- Columns represent "to" states
- Darker colors indicate higher probabilities

Strong diagonal elements indicate states with high self-transition probabilities (persistence).

### State Pattern Visualization

The state pattern visualization shows what patterns each state represents:
- **Mean Value Profile**: The average pattern of each feature for each state
- **Variation Display**: Standard deviation and min/max ranges for each feature
- **State Duration**: Shows how long the model typically stays in each state
- **State Frequency**: Indicates how often each state occurs in the data

This visualization is essential for interpreting what each state represents in terms of the underlying data patterns.

### Composite State Visualization

The composite visualization provides a comprehensive view combining:
- **Time Series with State Coloring**: Shows which parts of the data are assigned to each state
- **State Sequence Timeline**: Clear visualization of the state sequence
- **Feature Distribution by State**: Statistical summary of feature values for each state
- **Transition Probability Matrix**: Shows how states transition to each other
- **State Duration Histogram**: Distribution of how long the model stays in each state

This visualization is particularly useful for understanding the relationship between states and the underlying data patterns.

## Headless and Remote Server Usage

The system is designed to work in both GUI and headless environments. When running on a remote server or in an environment without a display:

```bash
# Option 1: Set environment variable (recommended)
export DISPLAY=""
python main.py

# Option 2: Use the no-gui flag
python main.py --no-gui
```

In headless mode:
- All plots are automatically saved to the `plots/` directory
- The matplotlib backend automatically switches to a non-GUI version
- Interactive display is disabled while file saving remains active
- Special error handling ensures failed visualizations don't crash the program

This makes the system ideal for long-running experiments on remote servers where you can periodically check the generated plots for analysis.

## Live vs. Offline Mode

The system supports two modes of operation, each with its own advantages:

### Live Mode

- **Data Source**: Real-time streams or simulated data
- **Processing Style**: Continuous, incremental updates
- **Use Cases**: 
  - Real-time monitoring of systems
  - Online learning from streaming data
  - Continuous adaptation to changing dynamics
- **Command**: `python main.py`

### Offline Mode (CSV Processing)

- **Data Source**: CSV files in a specified directory
- **Processing Style**: Batch processing of historical data
- **Use Cases**:
  - Analysis of previously collected datasets
  - Model development and testing
  - Benchmark comparisons
  - Parameter tuning
- **Command**: `python main.py --data-dir data --window-size 50 --stride 25`

### Comparison

| Aspect | Live Mode | Offline Mode |
|--------|-----------|--------------|
| Data Source | Real-time streams or simulated data | CSV files in a directory |
| Processing Speed | Limited by data collection rate | As fast as the processor can handle |
| Window Management | Fixed window size | Configurable window size and stride |
| Feature Detection | Fixed number of features | Automatically detected from CSV files |
| End Condition | Manual stop or max windows | End of all CSV files or max windows |
| Applicable Use Cases | Real-time monitoring, online learning | Historical analysis, model development |
| Visualization | Real-time updates | Generated at intervals and end of processing |

Both modes produce the same outputs (model files, visualizations, etc.) and use the same underlying HDP-HMM algorithm. The choice between modes depends on your specific use case and data availability.

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

### State-Specific Time Series Analysis

To analyze the characteristics of each discovered state, the system generates detailed state-specific time series visualizations:

```bash
# Run the model with at least 10 windows to trigger state-specific visualization
python main.py --max-windows 10

# For headless environments
python main.py --no-gui --max-windows 10
```

This generates:

1. **Individual state files** in `plots/state_time_series/`:
   - `state_X_window_Y.png`: Shows all time series data with points belonging to state X highlighted
   - Includes statistics (mean, std) and temporal distribution for each state

2. **Summary visualization**:
   - `all_states_summary_window_Y.png`: Overview of all states with color-coding

To interpret these visualizations:

- **State Occurrences (top panel)**: Shows when each state is active in the sequence
- **Feature Patterns (per-feature panels)**: Reveals the characteristic pattern of each feature for this state
- **Statistical Indicators**: Mean and standard deviation lines help understand the state's distribution
- **Full Series Context**: Gray background shows the full time series for context

These visualizations help answer questions like:
- "What does state 3 actually represent in my data?"
- "When does state 5 typically occur?"
- "Which features are most distinctive for state 2?"

## References

1. Hughes, M. C., & Sudderth, E. B. (2013). Memoized Online Variational Inference for Dirichlet Process Mixture Models. In Advances in Neural Information Processing Systems (NIPS).

2. Hughes, M. C., Stephenson, W. T., & Sudderth, E. B. (2015). Scalable Adaptation of State Complexity for Nonparametric Hidden Markov Models. In Advances in Neural Information Processing Systems (NIPS).

3. Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2008). An HDP-HMM for Systems with State Persistence. In Proceedings of the 25th International Conference on Machine Learning (ICML).

4. Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet Processes. Journal of the American Statistical Association, 101(476), 1566-1581.

5. Sethuraman, J. (1994). A Constructive Definition of Dirichlet Priors. Statistica Sinica, 4, 639-650.

6. bnpy: Bayesian Nonparametric Machine Learning for Python. (n.d.). GitHub repository: https://github.com/bnpy/bnpy
