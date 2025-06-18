import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Helper function to find contiguous regions
def find_contiguous_regions(mask):
    """Find contiguous regions in a boolean array."""
    in_region = False
    start = 0
    regions = []
    
    for i, val in enumerate(mask):
        if val and not in_region:
            in_region = True
            start = i
        elif not val and in_region:
            in_region = False
            regions.append((start, i))
    
    if in_region:
        regions.append((start, len(mask)))
    
    return regions

# Generate sample data
def generate_sample_data(n_samples=1000, n_features=3, n_states=3, noise_level=0.2):
    """Generate sample time series data with state transitions."""
    # Define state means
    state_means = np.random.randn(n_states, n_features) * 2
    
    # Generate state sequence
    # Create transition matrix with strong self-transitions
    trans_mat = np.eye(n_states) * 0.95
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                trans_mat[i,j] = (1 - trans_mat[i,i]) / (n_states - 1)
    
    # Generate state sequence
    states = np.zeros(n_samples, dtype=int)
    states[0] = np.random.randint(0, n_states)
    
    for t in range(1, n_samples):
        states[t] = np.random.choice(n_states, p=trans_mat[states[t-1]])
    
    # Generate data
    data = np.zeros((n_samples, n_features))
    for t in range(n_samples):
        data[t] = state_means[states[t]] + np.random.randn(n_features) * noise_level
    
    return data, states

# Generate multiple datasets
np.random.seed(42)  # For reproducibility
n_datasets = 3
n_samples_per_dataset = 500
n_features = 3
n_states = 3

for i in range(n_datasets):
    data, states = generate_sample_data(
        n_samples=n_samples_per_dataset, 
        n_features=n_features,
        n_states=n_states
    )
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{j+1}' for j in range(n_features)])
    
    # Save to CSV
    file_path = f'data/sample_data_{i+1}.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated {file_path} with {n_samples_per_dataset} samples and {n_features} features")
    
    # Visualize the data (optional)
    plt.figure(figsize=(12, 6))
    for j in range(n_features):
        plt.subplot(n_features, 1, j+1)
        plt.plot(data[:, j])
        plt.ylabel(f'Feature {j+1}')
        plt.grid(True, alpha=0.3)
        
        # Color the background by state
        for s in range(n_states):
            state_mask = states == s
            for start, end in find_contiguous_regions(state_mask):
                plt.axvspan(start, end, color=f'C{s}', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'data/sample_data_{i+1}_viz.png', dpi=100)
    plt.close()

print(f"Generated {n_datasets} CSV files in the 'data' directory")
