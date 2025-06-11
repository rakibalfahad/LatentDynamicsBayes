"""
Live Visualization Module

This module provides real-time visualization of the HDP-HMM model output,
including time series plots, state assignments, transition probabilities,
and training metrics.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime
import os

class LiveVisualizer:
    def __init__(self, n_features, window_size, feature_names=None, max_states=20,
                 output_dir="plots", interactive=True):
        """
        Visualize live inference results from HDP-HMM.
        
        Args:
            n_features (int): Number of features in the data
            window_size (int): Size of sliding window
            feature_names (list): Names of features for plotting
            max_states (int): Maximum number of states in the model
            output_dir (str): Directory to save plot images
            interactive (bool): Whether to show interactive plots
        """
        self.n_features = n_features
        self.window_size = window_size
        self.max_states = max_states
        self.output_dir = output_dir
        self.interactive = interactive
        
        # Feature names for plot labels
        self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize figures
        if interactive:
            plt.ion()  # Enable interactive mode
            self._setup_plots()
        
        # Training metrics
        self.losses = []
        self.active_states_history = []
        self.time_steps = []
        self.start_time = datetime.now()
    
    def _setup_plots(self):
        """Set up the plot figures and axes."""
        # Time series and state assignment plot
        self.ts_fig, self.ts_axes = plt.subplots(self.n_features, 1, figsize=(12, 2 * self.n_features))
        if self.n_features == 1:
            self.ts_axes = [self.ts_axes]
        self.ts_fig.tight_layout()
        
        # State distribution plot
        self.state_fig, self.state_ax = plt.subplots(figsize=(8, 4))
        
        # Transition matrix plot
        self.trans_fig, self.trans_ax = plt.subplots(figsize=(8, 6))
        
        # Loss plot
        self.loss_fig, self.loss_ax = plt.subplots(figsize=(8, 4))
    
    def update_plot(self, data, states, trans_probs, loss, active_states=None, save=False):
        """
        Update all visualizations with new data.
        
        Args:
            data (torch.Tensor): Window of time series data
            states (torch.Tensor): Inferred state sequence
            trans_probs (torch.Tensor): Transition probability matrix
            loss (float): Current loss value
            active_states (int): Number of active states
            save (bool): Whether to save plots to disk
        """
        # Convert to numpy for plotting
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(trans_probs, torch.Tensor):
            trans_probs = trans_probs.cpu().detach().numpy()
        
        # Update metrics
        self.losses.append(loss)
        if active_states is not None:
            self.active_states_history.append(active_states)
        self.time_steps.append((datetime.now() - self.start_time).total_seconds())
        
        if self.interactive:
            # Update time series plots
            self._update_time_series_plot(data, states)
            
            # Update state distribution plot
            self._update_state_distribution_plot(states)
            
            # Update transition matrix plot
            self._update_transition_matrix_plot(trans_probs)
            
            # Update loss plot
            self._update_loss_plot()
            
            # Refresh
            plt.pause(0.01)
        
        # Save plots if requested
        if save:
            self._save_plots()
    
    def _update_time_series_plot(self, data, states):
        """Update time series plot with state assignments."""
        # Clear previous plots
        for ax in self.ts_axes:
            ax.clear()
        
        # Get unique states and assign colors
        unique_states = np.unique(states)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        # Plot each feature with state coloring
        time_points = np.arange(data.shape[0])
        
        for i in range(self.n_features):
            ax = self.ts_axes[i]
            
            # Plot data line
            ax.plot(time_points, data[:, i], 'k-', alpha=0.5, label=self.feature_names[i])
            
            # Plot colored segments for each state
            for j, state in enumerate(unique_states):
                mask = states == state
                if np.any(mask):
                    ax.plot(time_points[mask], data[mask, i], 'o-', 
                            color=colors[j], markersize=4, alpha=0.7, 
                            label=f"State {state}")
            
            # Set labels and title
            ax.set_title(f"{self.feature_names[i]} with State Assignments")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            
            # Add legend on the first plot only
            if i == 0:
                handles = [mpatches.Patch(color=colors[j], label=f"State {state}") 
                           for j, state in enumerate(unique_states)]
                ax.legend(handles=handles, loc='upper right')
        
        self.ts_fig.tight_layout()
    
    def _update_state_distribution_plot(self, states):
        """Update state distribution histogram."""
        self.state_ax.clear()
        
        # Plot state histogram
        unique_states, counts = np.unique(states, return_counts=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        # Calculate percentages
        percentages = counts / np.sum(counts) * 100
        
        # Create bar chart
        bars = self.state_ax.bar(unique_states, percentages, color=colors, alpha=0.7)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            self.state_ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', rotation=0)
        
        # Set labels and title
        self.state_ax.set_xlabel("State")
        self.state_ax.set_ylabel("Percentage")
        self.state_ax.set_title("State Distribution")
        self.state_ax.set_xticks(unique_states)
        
        # Set reasonable y limit
        self.state_ax.set_ylim(0, max(percentages) * 1.2)
        
        self.state_fig.tight_layout()
    
    def _update_transition_matrix_plot(self, trans_probs):
        """Update transition probability matrix heatmap."""
        self.trans_ax.clear()
        
        # Create heatmap
        sns.heatmap(trans_probs, annot=True, cmap='Blues', vmin=0, vmax=1,
                   square=True, cbar=True, ax=self.trans_ax)
        
        # Set labels and title
        self.trans_ax.set_title("Transition Probabilities")
        self.trans_ax.set_xlabel("To State")
        self.trans_ax.set_ylabel("From State")
        
        self.trans_fig.tight_layout()
    
    def _update_loss_plot(self):
        """Update loss plot."""
        self.loss_ax.clear()
        
        # Plot loss curve
        self.loss_ax.plot(self.time_steps, self.losses, 'b-', label="Training Loss")
        
        # Plot moving average if we have enough points
        if len(self.losses) > 10:
            window_size = min(10, len(self.losses)//2)
            moving_avg = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            ma_time = self.time_steps[window_size-1:]
            self.loss_ax.plot(ma_time, moving_avg, 'r-', label=f"{window_size}-point Moving Avg")
        
        # Plot active states if available
        if self.active_states_history:
            ax2 = self.loss_ax.twinx()
            ax2.plot(self.time_steps[:len(self.active_states_history)], 
                    self.active_states_history, 'g-', label="Active States")
            ax2.set_ylabel("Active States", color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_ylim(0, self.max_states)
        
        # Set labels and title
        self.loss_ax.set_xlabel("Time (seconds)")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_title("Training Loss Over Time")
        self.loss_ax.legend(loc='upper right')
        
        # Set y-axis to log scale if range is large
        if len(self.losses) > 1 and max(self.losses) / (min(self.losses) + 1e-10) > 100:
            self.loss_ax.set_yscale('log')
        
        self.loss_fig.tight_layout()
    
    def _save_plots(self):
        """Save all plots to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save time series plot
        self.ts_fig.savefig(os.path.join(self.output_dir, f"time_series_{timestamp}.png"))
        
        # Save state distribution plot
        self.state_fig.savefig(os.path.join(self.output_dir, f"state_distribution_{timestamp}.png"))
        
        # Save transition matrix plot
        self.trans_fig.savefig(os.path.join(self.output_dir, f"transition_matrix_{timestamp}.png"))
        
        # Save loss plot
        self.loss_fig.savefig(os.path.join(self.output_dir, f"loss_{timestamp}.png"))
    
    def create_summary_plot(self, window_data, states, feature_names=None):
        """
        Create a static summary plot combining all visualizations.
        
        Args:
            window_data (torch.Tensor): Window of time series data
            states (torch.Tensor): Inferred state sequences
            feature_names (list): Optional list of feature names
            
        Returns:
            matplotlib.figure.Figure: The summary figure
        """
        # Convert to numpy for plotting
        if isinstance(window_data, torch.Tensor):
            window_data = window_data.cpu().numpy()
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        
        # Use provided feature names or default ones
        if feature_names is None:
            feature_names = self.feature_names
        
        # Create a larger figure for the summary
        fig = plt.figure(figsize=(12, 10))
        
        # 1. Time series with state assignments
        unique_states = np.unique(states)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        time_points = np.arange(window_data.shape[0])
        
        for i in range(min(3, self.n_features)):  # Limit to first 3 features for clarity
            ax = fig.add_subplot(4, 1, i+1)
            
            # Plot data line
            ax.plot(time_points, window_data[:, i], 'k-', alpha=0.5)
            
            # Plot colored segments for each state
            for j, state in enumerate(unique_states):
                mask = states == state
                if np.any(mask):
                    ax.plot(time_points[mask], window_data[mask, i], 'o-', 
                            color=colors[j], markersize=4, alpha=0.7)
            
            # Set labels and title
            ax.set_title(f"{feature_names[i]} with State Assignments")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
        
        # Add legend
        handles = [mpatches.Patch(color=colors[j], label=f"State {state}") 
                  for j, state in enumerate(unique_states)]
        fig.legend(handles=handles, loc='lower center', ncol=min(5, len(unique_states)))
        
        # 4. State distribution
        ax = fig.add_subplot(4, 1, 4)
        unique_states, counts = np.unique(states, return_counts=True)
        ax.bar(unique_states, counts, color=colors, alpha=0.7)
        ax.set_xlabel("State")
        ax.set_ylabel("Count")
        ax.set_title("State Distribution")
        ax.set_xticks(unique_states)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the legend
        fig.suptitle("HDP-HMM Analysis Summary", fontsize=16, y=0.98)
        
        return fig
    
    def close(self):
        """Close all plots."""
        if self.interactive:
            plt.ioff()
            plt.close('all')
