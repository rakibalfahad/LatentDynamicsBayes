# LatentDynamicsBayes: HDP-HMM for Time Series Analysis

**A comprehensive tutorial on using the Bayesian Non-Parametric Modeling framework with HDP-HMM for time series data analysis**

![LatentDynamicsBayes Banner](https://example.com/banner.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Features Overview](#features-overview)
4. [Quick Start Guide](#quick-start-guide)
5. [Live Mode Tutorial](#live-mode-tutorial)
6. [Offline CSV Processing Tutorial](#offline-csv-processing-tutorial)
7. [Visualization Guide](#visualization-guide)
8. [Advanced Configuration](#advanced-configuration)
9. [Understanding the Output](#understanding-the-output)
10. [How to Cite](#how-to-cite)
11. [References](#references)

## Introduction

[LatentDynamicsBayes](https://github.com/yourusername/LatentDynamicsBayes) is a powerful implementation of the Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) with stick-breaking construction for unsupervised learning of state sequences in multidimensional time series data. This Bayesian non-parametric approach automatically determines the appropriate number of hidden states from the data, making it ideal for discovering latent patterns and structure in complex time series.

The implementation is specifically designed to work with both live streaming data (such as system metrics) and offline historical data (via CSV files). It supports incremental training, real-time inference, and comprehensive visualization, all accelerated with PyTorch and GPU computation when available.

### What is HDP-HMM?

The HDP-HMM extends traditional Hidden Markov Models by using a Hierarchical Dirichlet Process prior, allowing the model to automatically determine the number of hidden states that best explains the data. This Bayesian non-parametric approach is particularly valuable when:

- The true number of states is unknown
- The complexity of the data may change over time
- You need to discover natural groupings in temporal data
- You want to avoid manual parameter tuning

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Seaborn
- pandas
- psutil (optional, for real system metrics)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/LatentDynamicsBayes.git
cd LatentDynamicsBayes
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Verify the installation:

```bash
python demo.py
```

This will run a quick demonstration using simulated data.

## Features Overview

LatentDynamicsBayes provides a comprehensive toolkit for time series analysis:

### Core Features

- **Bayesian non-parametric modeling** with HDP-HMM to automatically determine the optimal number of states
- **Dynamic state management** with birth, merge, and delete operations
- **Dual-mode operation**: live streaming and offline batch processing
- **PyTorch implementation** with GPU acceleration
- **Incremental model updates** for continuous learning
- **Model persistence** with checkpointing

### Visualization Suite

- **Time series visualization** with state assignments
- **State pattern analysis** showing what each state represents
- **State evolution tracking** with birth/merge/delete events
- **Transition probability heatmaps**
- **Learning curves** and model performance metrics
- **State-specific time series analysis**
- **Composite visualizations** combining multiple views

### Practical Features

- **Robust error handling** with headless operation support
- **Performance monitoring** for tracking training and inference times
- **Organized plot management** with automatic directory structure
- **Comprehensive logging** with detailed state updates
- **Sample data generation** for testing and demonstration

## Quick Start Guide

### Basic Live Mode (with simulated data)

```bash
python main.py
```

### Offline Mode with CSV Files

```bash
python main.py --data-dir data --window-size 50 --stride 25
```

### Headless Operation (for servers without display)

```bash
python main.py --no-gui
```

### Generate Sample Data for Testing

```bash
python generate_sample_data.py
```

## Live Mode Tutorial

Live mode processes data in real-time, either from actual system metrics or simulated data. This is ideal for monitoring systems, continuous learning, and real-time pattern detection.

### Step 1: Start with a simple run

```bash
python main.py
```

This will:
- Initialize the model with default parameters
- Generate simulated data
- Incrementally train the model on sliding windows of data
- Visualize the results in real-time

### Step 2: Experiment with different parameters

```bash
python main.py --window-size 200 --max-windows 500
```

- `window-size`: Controls how many time steps are included in each sliding window
- `max-windows`: Limits the total number of windows to process

### Step 3: Save and analyze the results

The model automatically saves:
- The trained model to `models/hdp_hmm.pth`
- Visualization plots to the `plots/` directory
- Transition matrices to `plots/transition_matrix/`

### Step 4: Using real system metrics

To use real system metrics instead of simulated data:

```bash
python main.py --use-real
```

This will collect metrics such as CPU usage, memory utilization, and temperature readings (when available).

## Offline CSV Processing Tutorial

Offline mode processes historical data from CSV files, which is ideal for retrospective analysis, batch processing, and model development on benchmark datasets.

### Step 1: Prepare your CSV files

Each CSV file should:
- Have columns representing features
- Have rows representing time steps
- No header row is required (first row is treated as data)

Example CSV format:
```
0.5,1.2,0.8
0.6,1.3,0.7
0.7,1.4,0.6
...
```

### Step 2: Generate sample data (optional)

```bash
python generate_sample_data.py
```

This will create sample CSV files in the `data/` directory with known state patterns.

### Step 3: Process the CSV files

```bash
python main.py --data-dir data --window-size 50 --stride 25
```

Parameters:
- `data-dir`: Directory containing CSV files
- `window-size`: Number of time steps in each window
- `stride`: Number of time steps to advance between windows (use smaller values for overlapping windows)

### Step 4: Experiment with different window configurations

For non-overlapping windows:
```bash
python main.py --data-dir data --window-size 100 --stride 100
```

For heavily overlapping windows (75% overlap):
```bash
python main.py --data-dir data --window-size 100 --stride 25
```

### Step 5: Analyze the results

After processing completes:
- Check the `plots/` directory for visualizations
- Examine `final_state_patterns.png` to understand what each state represents
- View `final_transition_matrix.png` to see state transition patterns
- Explore state-specific time series in `plots/state_time_series/`

## Visualization Guide

The visualization system provides comprehensive insights into the model's behavior and the discovered patterns in your data.

### Key Visualizations

1. **Time Series with State Assignments**

   ![Time Series Example](https://example.com/time_series.png)

   This plot shows your raw data with color-coded state assignments, helping you see how the model segments your time series.

2. **State Pattern Analysis**

   ![State Patterns Example](https://example.com/state_patterns.png)

   This visualization shows what pattern each state represents, including:
   - Mean value for each feature
   - Standard deviation (shaded area)
   - Min/max range
   - State frequency and typical duration

3. **State Evolution Plot**

   ![State Evolution Example](https://example.com/state_evolution.png)

   This plot shows how the number of states changes over time, with markers for:
   - Birth of new states (green triangles)
   - Merge of similar states (orange circles)
   - Deletion of inactive states (red triangles)

4. **Transition Probability Heatmap**

   ![Transition Matrix Example](https://example.com/transition_matrix.png)

   This heatmap shows the probability of transitioning from one state to another:
   - Rows represent "from" states
   - Columns represent "to" states
   - Darker colors indicate higher probabilities
   - Strong diagonal elements indicate persistent states

5. **Learning Curve**

   ![Learning Curve Example](https://example.com/learning_curve.png)

   This plot shows the model's loss over time, helping you identify:
   - Overall learning trend
   - Convergence patterns
   - Correlation between state changes and model performance

### Interpreting Visualizations

When analyzing the visualizations, look for:

1. **In State Patterns**:
   - Distinct patterns for each state
   - Clear separation between states
   - Consistent patterns with low variance

2. **In Transition Matrix**:
   - Strong self-transitions (diagonal)
   - Clear transition pathways between states
   - Absence of uniform transition probabilities

3. **In State Evolution**:
   - Stabilization of state count over time
   - Reduction in birth/merge/delete events as training progresses
   - Correlation between state changes and learning curve improvements

## Advanced Configuration

The behavior of the HDP-HMM model can be fine-tuned through several key parameters.

### Model Parameters

These parameters can be adjusted in `config.json`:

```json
{
  "model": {
    "n_features": 3,
    "max_states": 20,
    "alpha": 1.0,
    "gamma": 1.0,
    "learning_rate": 0.01
  }
}
```

- `n_features`: Number of input features
- `max_states`: Maximum number of states to consider
- `alpha`: Concentration parameter for the HDP
- `gamma`: Top-level concentration parameter
- `learning_rate`: Learning rate for optimization

### Dynamic State Management

Fine-tune the birth, merge, and delete mechanisms:

```json
{
  "state_management": {
    "delete_threshold": 1e-3,
    "merge_distance": 0.5,
    "birth_threshold": 10.0
  }
}
```

- `delete_threshold`: Minimum beta weight for a state to remain active
- `merge_distance`: Maximum distance between means for state merging
- `birth_threshold`: Negative log-likelihood threshold for creating new states

### Tuning Recommendations

- **High Noise Data**: Increase `delete_threshold` (e.g., 5e-3) and `merge_distance` (e.g., 1.0)
- **Complex Systems**: Decrease `birth_threshold` (e.g., 5.0) to allow more states
- **Computational Efficiency**: Increase `delete_threshold` and `birth_threshold`
- **High Precision**: Decrease `merge_distance` (e.g., 0.3) to prevent merging distinct states

## Understanding the Output

### State Interpretation

Each discovered state represents a distinctive pattern in your time series data. To interpret a state:

1. Look at its mean pattern across features
2. Check its variance (standard deviation) to assess consistency
3. Examine when and how often it occurs in the sequence
4. Analyze its incoming and outgoing transitions

### Transition Dynamics

The transition matrix reveals the temporal dynamics of your system:

- High self-transition probabilities indicate persistent states
- Strong transitions between specific states suggest common patterns
- Absence of transitions between states indicates separate regimes or modes

### State Count Evolution

The evolution of the state count provides insights into model complexity:

- Initial rapid increase in states as the model discovers patterns
- Merging of similar states as the model refines its understanding
- Eventual stabilization around an optimal number of states

## How to Cite

If you use LatentDynamicsBayes in your research, please cite it as:

```bibtex
@software{LatentDynamicsBayes2025,
  author = {Your Name},
  title = {LatentDynamicsBayes: HDP-HMM for Time Series Analysis},
  url = {https://github.com/yourusername/LatentDynamicsBayes},
  version = {1.0.0},
  year = {2025},
}
```

For the underlying methodologies, please also cite:

```bibtex
@article{Teh2006,
  title={Hierarchical {D}irichlet Processes},
  author={Teh, Yee Whye and Jordan, Michael I and Beal, Matthew J and Blei, David M},
  journal={Journal of the American Statistical Association},
  volume={101},
  number={476},
  pages={1566--1581},
  year={2006}
}

@inproceedings{Fox2008,
  title={An {HDP-HMM} for Systems with State Persistence},
  author={Fox, Emily B and Sudderth, Erik B and Jordan, Michael I and Willsky, Alan S},
  booktitle={Proceedings of the 25th International Conference on Machine Learning},
  year={2008}
}

@inproceedings{Hughes2013,
  title={Memoized Online Variational Inference for {D}irichlet Process Mixture Models},
  author={Hughes, Michael C and Sudderth, Erik B},
  booktitle={Advances in Neural Information Processing Systems},
  year={2013}
}
```

## References

1. Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet Processes. Journal of the American Statistical Association, 101(476), 1566-1581.

2. Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2008). An HDP-HMM for Systems with State Persistence. In Proceedings of the 25th International Conference on Machine Learning (ICML).

3. Hughes, M. C., & Sudderth, E. B. (2013). Memoized Online Variational Inference for Dirichlet Process Mixture Models. In Advances in Neural Information Processing Systems (NIPS).

4. Hughes, M. C., Stephenson, W. T., & Sudderth, E. (2015). Scalable Adaptation of State Complexity for Nonparametric Hidden Markov Models. In Advances in Neural Information Processing Systems.

5. [bnpy: Bayesian Nonparametric Machine Learning for Python](https://github.com/bnpy/bnpy)

---

*This tutorial was created on June 17, 2025*

*[LatentDynamicsBayes](https://github.com/yourusername/LatentDynamicsBayes) - A PyTorch implementation of HDP-HMM for time series analysis*
