import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class LiveVisualizer:
    def __init__(self, n_features, window_size):
        """
        Visualize live inference results.
        
        Args:
            n_features (int): Number of features
            window_size (int): Size of sliding window
        """
        self.n_features = n_features
        self.window_size = window_size
        self.fig, self.axes = plt.subplots(n_features + 1, 1, figsize=(12, 8))
        plt.ion()  # Enable interactive mode
    
    def update_plot(self, data, states, trans_probs, loss, losses):
        """Update live plot with new data.
        
        Args:
            data: Tensor of shape (window_size, n_features)
            states: Inferred states
            trans_probs: Transition probabilities
            loss: Current loss value
            losses: List of all loss values
        """
        self.fig.clear()
        data = data.cpu().numpy()
        states = states.cpu().numpy()
        
        # Plot time series and states
        for i in range(self.n_features):
            ax = self.fig.add_subplot(self.n_features + 1, 1, i + 1)
            ax.plot(data[:, i], label=f'Feature {i+1}')
            ax.scatter(range(len(states)), data[:, i], c=states, cmap='plasma', marker='x', label='Inferred States')
            ax.legend()
            ax.set_title(f'Feature {i+1} Time Series')
        
        # Plot loss
        ax = self.fig.add_subplot(self.n_features + 1, 1, self.n_features + 1)
        ax.plot(np.arange(len(losses)), losses, label='Loss')
        ax.set_title(f'Loss: {loss:.4f}')
        ax.set_xlabel('Window')
        ax.set_ylabel('Loss')
        ax.legend()
        
        self.fig.tight_layout()
        plt.pause(0.01)
        
        # Save transition probabilities heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(trans_probs.cpu().detach().numpy(), annot=True, cmap='Blues')
        plt.title('Transition Probabilities')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.savefig('transition_probs.png')
        plt.close()
    
    def close(self):
        """Close the plot."""
        plt.ioff()
        plt.close(self.fig)