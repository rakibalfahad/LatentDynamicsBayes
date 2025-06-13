import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Use non-interactive backend if running in headless mode
        import matplotlib
        if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
            matplotlib.use('Agg')  # Use non-GUI backend
        
        self.fig, self.axes = plt.subplots(n_features + 2, 1, figsize=(12, 10))  # Added one more subplot
        plt.ion()  # Enable interactive mode if available
        
        # State history tracking for tile visualization
        self.state_history = []
        self.window_count = 0
        self.max_history_length = 50  # Maximum number of windows to track
        self.current_data = None  # Store current data for visualization
    
    def update_plot(self, data, states, trans_probs, loss, losses, state_counts=[], state_changes=None):
        """Update live plot with new data.
        
        Args:
            data: Tensor of shape (window_size, n_features)
            states: Inferred states
            trans_probs: Transition probabilities
            loss: Current loss value
            losses: List of all loss values
            state_counts: List of state counts over time
            state_changes: List of state change dictionaries from the trainer
        """
        self.fig.clear()
        data = data.cpu().numpy()
        states = states.cpu().numpy()
        
        # Store states for tile visualization
        self.window_count += 1
        self.state_history.append(states)
        # Also store the current data for state sequence visualization
        self.current_data = data.copy()  # Store a copy of the current data
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
        
        # Plot time series and states
        for i in range(self.n_features):
            ax = self.fig.add_subplot(self.n_features + 2, 1, i + 1)
            ax.plot(data[:, i], label=f'Feature {i+1}')
            ax.scatter(range(len(states)), data[:, i], c=states, cmap='plasma', marker='x', label='Inferred States')
            ax.legend()
            ax.set_title(f'Feature {i+1} Time Series')
        
        # Plot loss
        ax = self.fig.add_subplot(self.n_features + 2, 1, self.n_features + 1)
        ax.plot(np.arange(len(losses)), losses, label='Loss')
        ax.set_title(f'Loss: {loss:.4f}')
        ax.set_xlabel('Window')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot state counts if available
        if state_counts:
            ax = self.fig.add_subplot(self.n_features + 2, 1, self.n_features + 2)
            ax.plot(np.arange(len(state_counts)), state_counts, 'g-', label='Active States')
            ax.set_title(f'Number of Active States: {state_counts[-1]}')
            ax.set_xlabel('Update')
            ax.set_ylabel('State Count')
            ax.legend()
            # Set y-axis limits with some padding
            max_states = max(state_counts) + 1
            min_states = max(0, min(state_counts) - 1)
            ax.set_ylim(min_states, max_states)
        
        try:
            # Use subplots_adjust instead of tight_layout for better compatibility
            self.fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1, left=0.1, right=0.95)
        except Exception as e:
            print(f"Warning: Figure layout adjustment failed: {e}")
        
        # Always save the figure, regardless of display mode
        try:
            self.fig.savefig(f'plots/live_plot_window_{self.window_count}.png', dpi=300)
            if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
                plt.pause(0.01)  # Only pause for display if in GUI mode
        except Exception as e:
            print(f"Warning: Failed to save live plot: {e}")
        
        # Save transition probabilities heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(trans_probs.cpu().detach().numpy(), annot=True, cmap='Blues')
        plt.title('Transition Probabilities')
        plt.xlabel('To State')
        plt.ylabel('From State')
        
        # Make sure plots directory exists
        os.makedirs('plots', exist_ok=True)
        
        # Save as image with bbox_inches to ensure everything is captured
        plt.savefig('plots/transition_probs.png', dpi=300, bbox_inches='tight')
        
        # Also save as numpy file for later analysis
        trans_probs_np = trans_probs.cpu().detach().numpy()
        np.save('plots/transition_matrix', trans_probs_np)
        
        # Save the latest transition matrix with timestamp
        np.save(f'plots/transition_matrix_window_{self.window_count}', trans_probs_np)
        
        plt.close()
        
        # Save learning curve visualization if we have enough data
        if len(losses) > 1:
            self.create_learning_curve(losses, state_counts, f'plots/learning_curve_window_{self.window_count}.png')
        
        # Create and save tile visualization every 5 windows
        if self.window_count % 5 == 0 and len(self.state_history) > 0:
            self.create_tile_visualization()
        
        # Create learning curve visualization every 5 windows
        if self.window_count % 5 == 0 and len(losses) > 0:
            save_path = f'plots/learning_curve_window_{self.window_count}.png'
            self.create_learning_curve(losses, state_counts, save_path)
            
        # Create state evolution plot if state changes are provided
        if state_changes and len(state_changes) > 0:
            save_path = f'plots/state_evolution_window_{self.window_count}.png'
            self.create_state_evolution_plot(state_changes, save_path)
            
        # Create state pattern visualization every 10 windows
        if self.window_count % 10 == 0:
            save_path = f'plots/state_patterns_window_{self.window_count}.png'
            self.visualize_state_patterns(data, states, save_path)
            
            # Also create a composite visualization
            save_path = f'plots/composite_viz_window_{self.window_count}.png'
            self.create_composite_state_visualization(data, states, save_path)
            
            # Create state-specific time series visualizations
            self.visualize_state_time_series(data, states)
            
            # Create state time series visualization
            save_path = f'plots/state_time_series_window_{self.window_count}.png'
            self.visualize_state_time_series(data, states, save_path)
    
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
        if not self.state_history:
            return
        
        # Create a 2D array from state history
        # Each column is a window, each row is a time point
        # Determine the number of time points in each window
        time_points = min(self.window_size, len(self.state_history[0]))
        windows = len(self.state_history)
        
        # Create matrix for visualization (time_points x windows)
        state_matrix = np.zeros((time_points, windows))
        
        # Fill the matrix with state assignments
        for w in range(windows):
            window_states = self.state_history[w]
            for t in range(min(time_points, len(window_states))):
                state_matrix[t, w] = window_states[t]
        
        # Create the tile visualization
        plt.figure(figsize=(12, 8))
        
        # First subplot: State tile visualization
        plt.subplot(2, 1, 1)
        max_state = int(np.max(state_matrix)) + 1
        im = plt.imshow(state_matrix, aspect='auto', interpolation='none', 
                       cmap='plasma', vmin=0, vmax=max_state)
        
        plt.colorbar(im, label='State')
        plt.title(f'State Assignments Over Time (Windows {self.window_count-windows+1} to {self.window_count})')
        plt.ylabel('Time within Window')
        plt.xlabel('Window Number')
        
        # Adjust x-axis ticks to show window numbers
        window_indices = np.arange(windows)
        window_numbers = np.arange(self.window_count-windows+1, self.window_count+1)
        plt.xticks(window_indices[::max(1, windows//10)], window_numbers[::max(1, windows//10)])
        
        # Second subplot: State transition frequency
        plt.subplot(2, 1, 2)
        
        # Count state transitions within each window
        transitions = {}
        for w in range(windows):
            window_states = self.state_history[w]
            for t in range(1, len(window_states)):
                from_state = int(window_states[t-1])
                to_state = int(window_states[t])
                if from_state != to_state:  # Only count actual transitions
                    key = (from_state, to_state)
                    transitions[key] = transitions.get(key, 0) + 1
        
        # Create transition matrix for visualization
        if transitions:
            max_observed_state = max(max(k[0], k[1]) for k in transitions.keys())
            trans_matrix = np.zeros((max_observed_state+1, max_observed_state+1))
            for (from_state, to_state), count in transitions.items():
                trans_matrix[from_state, to_state] = count
            
            # Normalize by row sums
            row_sums = trans_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            trans_matrix = trans_matrix / row_sums
            
            # Plot transition matrix
            sns.heatmap(trans_matrix, annot=True, cmap='Blues', 
                       xticklabels=range(max_observed_state+1),
                       yticklabels=range(max_observed_state+1))
            plt.title('State Transition Probabilities')
            plt.xlabel('To State')
            plt.ylabel('From State')
        else:
            plt.text(0.5, 0.5, 'No transitions observed yet', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
        
        # Use subplots_adjust for better compatibility
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15)
        
        # Make sure plots directory exists
        os.makedirs('plots', exist_ok=True)
        
        # Save with bbox_inches to ensure everything is captured
        plt.savefig(f'plots/state_tiles_window_{self.window_count}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create the state sequence visualization for the most recent window
        if self.state_history:
            recent_states = self.state_history[-1]
            
            if self.current_data is not None and len(self.current_data) == len(recent_states):
                self.show_state_sequence(
                    self.current_data, 
                    recent_states,
                    max_states=max_state,
                    save_path=f'plots/state_sequence_window_{self.window_count}.png'
                )
    
    def show_state_sequence(self, data, states, max_states=None, save_path=None):
        """
        Create a visualization showing data sequence with state assignments below,
        inspired by the bnpy visualization style.
        
        Args:
            data: Tensor of shape (window_size, n_features)
            states: Tensor of state assignments (window_size,)
            max_states: Maximum number of states to consider for colormap
            save_path: Path to save the visualization, if None, display only
        
        Returns:
            fig, axes: The figure and axes objects
        """
        import matplotlib.cm as cm
        
        # Convert tensors to numpy if needed
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
        
        # Determine the number of states for colormap
        if max_states is None:
            max_states = int(np.max(states_np)) + 1
        
        # Create colormap
        cmap = cm.get_cmap('tab10', max_states)
        
        # Create figure with two subplots stacked vertically
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), 
                                sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot time series data in top subplot
        for dim in range(data_np.shape[1]):
            axes[0].plot(data_np[:, dim], '.-', label=f'Feature {dim+1}')
        
        axes[0].set_ylabel('Feature Value')
        axes[0].set_title('Time Series Data with State Assignments')
        axes[0].legend(loc='upper right')
        
        # Create state image in bottom subplot
        # Create an image by repeating the state assignments vertically
        img_height = 100  # Height of the state image
        state_img = np.tile(states_np, (img_height, 1))
        
        # Display the state image
        img = axes[1].imshow(state_img, interpolation='nearest', aspect='auto',
                           vmin=-0.5, vmax=max_states-0.5, cmap=cmap)
        axes[1].set_yticks([])  # Hide y-axis ticks
        axes[1].set_xlabel('Time')
        
        # Add colorbar
        bbox = axes[1].get_position()
        cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_ticks(np.arange(max_states))
        cbar.set_label('State')
        
        # Don't use tight_layout when we have a custom colorbar axis
        # Adjust spacing manually instead
        fig.subplots_adjust(right=0.85, hspace=0.15)
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
            
        return fig, axes
    
    def create_learning_curve(self, losses, state_counts=None, save_path=None):
        """
        Create a detailed learning curve visualization for model performance debugging.
        
        Args:
            losses: List of loss values over time
            state_counts: Optional list of state counts over time
            save_path: Path to save the visualization, if None uses default path
        """
        if not losses:
            return
            
        # Create figure
        plt.figure(figsize=(12, 10))
        
        if state_counts:
            # Create two subplots if we have state counts
            ax1 = plt.subplot(2, 1, 1)
        else:
            # Just one plot if we only have losses
            ax1 = plt.subplot(1, 1, 1)
        
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
        
        # Add min/max markers
        min_loss = np.min(losses_np)
        min_idx = np.argmin(losses_np)
        ax1.scatter(min_idx, min_loss, color='green', s=100, marker='*', 
                    label=f'Min Loss: {min_loss:.4f} at window {min_idx}')
        
        # Calculate loss reduction
        if len(losses_np) > 1:
            first_loss = losses_np[0]
            last_loss = losses_np[-1]
            percent_reduction = ((first_loss - last_loss) / first_loss) * 100 if first_loss != 0 else 0
            ax1.text(0.02, 0.02, f'Loss reduction: {percent_reduction:.2f}%\nInitial: {first_loss:.4f} → Current: {last_loss:.4f}',
                    transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add grid and labels
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel('Training Window')
        ax1.set_ylabel('Loss Value (-log likelihood)')
        ax1.set_title('Training Loss Curve (Learning Curve)')
        ax1.legend(loc='upper right')
        
        # Add log scale version as inset if range is large enough
        loss_range = np.max(losses_np) - np.min(losses_np)
        if loss_range > 0 and np.max(losses_np) / (np.min(losses_np) + 1e-10) > 10:
            # Create an inset for log scale
            axins = ax1.inset_axes([0.65, 0.5, 0.3, 0.3])
            axins.semilogy(window_indices, losses_np, 'b-', alpha=0.5)
            if len(losses_np) > window_size:
                axins.semilogy(smoothed_x, smoothed, 'r-', linewidth=2)
            axins.set_title('Log Scale')
            axins.grid(True, linestyle='--', alpha=0.7)
        
        # Plot state counts if available
        if state_counts and len(state_counts) > 0:
            ax2 = plt.subplot(2, 1, 2)
            
            # Make sure state_counts doesn't contain None values
            valid_state_counts = [s for s in state_counts if s is not None]
            if not valid_state_counts:
                print("Warning: No valid state counts to display")
                return
                
            state_counts_np = np.array(valid_state_counts)
            state_indices = np.arange(len(state_counts_np))
            
            ax2.plot(state_indices, state_counts_np, 'g-', linewidth=2, label='Active States')
            ax2.scatter(state_indices, state_counts_np, color='green', s=30)
            
            # Add annotations for significant changes
            if len(state_counts_np) > 1:
                try:
                    changes = np.diff(state_counts_np)
                    significant_changes = np.where(np.abs(changes) > 1)[0]
                    for idx in significant_changes:
                        ax2.annotate(f'{changes[idx]:+.0f}', 
                                    (idx+1, state_counts_np[idx+1]),
                                    xytext=(0, 10), textcoords='offset points',
                                    ha='center', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not compute state changes due to {e}")
            
            # Add grid and labels
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Number of Active States')
            ax2.set_title('State Count Evolution')
            
            # Add correlation analysis between loss and state count
            if len(losses_np) == len(state_counts_np):
                correlation = np.corrcoef(losses_np, state_counts_np)[0, 1]
                ax2.text(0.02, 0.02, f'Correlation with loss: {correlation:.2f}',
                        transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
            
            # Set y-axis limits with some padding
            max_states = max(state_counts_np) + 1
            min_states = max(0, min(state_counts_np) - 1)
            ax2.set_ylim(min_states, max_states)
        
        # Use subplots_adjust for better compatibility
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
        
        # Save figure
        if save_path is None:
            save_path = f'plots/learning_curve_window_{self.window_count}.png'
            
        try:
            # Make sure plots directory exists
            os.makedirs('plots', exist_ok=True)
            
            # Save figures with bbox_inches='tight' to ensure all content is included
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Also save a general learning curve that gets updated
            plt.savefig('plots/latest_learning_curve.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save learning curve plot: {e}")
        finally:
            # Always close the figure to prevent memory leaks
            plt.close()
    
    def create_state_evolution_plot(self, state_changes, save_path=None):
        """
        Create a visualization of state birth, merge, and delete events.
        
        Args:
            state_changes: List of state change dictionaries from the trainer
            save_path: Path to save the visualization, if None uses default path
        """
        if not state_changes:
            print("Warning: No state changes to visualize")
            return
            
        try:
            plt.figure(figsize=(14, 10))
            
            # Create a timeline of state counts
            window_indices = list(range(len(state_changes)))
            initial_states = [sc.get('initial_states', 0) if sc else 0 for sc in state_changes]
            final_states = [sc.get('final_states', 0) if sc else 0 for sc in state_changes]
            
            # Create main plot showing state count
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(window_indices, initial_states, 'b--', label='Initial States', alpha=0.7)
            ax1.plot(window_indices, final_states, 'g-', label='Final States', linewidth=2)
            
            # Highlight state changes with markers
            for i, sc in enumerate(state_changes):
                if not sc:
                    continue
                    
                # Mark births
                if sc.get('birthed'):
                    ax1.scatter(i, final_states[i], color='green', s=100, marker='^', 
                               label='Birth' if i == 0 else "")
                
                # Mark merges
                if sc.get('merged'):
                    ax1.scatter(i, final_states[i], color='orange', s=100, marker='o',
                               label='Merge' if i == 0 else "")
                
                # Mark deletions
                if sc.get('deleted'):
                    ax1.scatter(i, final_states[i], color='red', s=100, marker='v',
                               label='Delete' if i == 0 else "")
            
            # Add grid and labels
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_xlabel('Update Window')
            ax1.set_ylabel('Number of States')
            ax1.set_title('State Evolution Over Time')
            ax1.legend(loc='upper right')
            
            # Create second plot with detailed change counts
            ax2 = plt.subplot(2, 1, 2)
            
            # Process state changes safely
            births = [len(sc.get('birthed', [])) if sc else 0 for sc in state_changes]
            merges = [len(sc.get('merged', [])) if sc else 0 for sc in state_changes]
            deletes = [len(sc.get('deleted', [])) if sc else 0 for sc in state_changes]
            
            # Create stacked bar chart
            width = 0.7
            ax2.bar(window_indices, births, width, label='Births', color='green', alpha=0.7)
            ax2.bar(window_indices, merges, width, bottom=births, label='Merges', color='orange', alpha=0.7)
            
            # Calculate bottom for deletes safely
            bottoms = []
            for i in range(len(births)):
                bottoms.append(births[i] + merges[i])
            
            ax2.bar(window_indices, deletes, width, bottom=bottoms, 
                   label='Deletes', color='red', alpha=0.7)
            
            # Add labels for significant changes
            for i in range(len(state_changes)):
                total_changes = births[i] + merges[i] + deletes[i]
                if total_changes > 0:
                    change_str = ""
                    if births[i] > 0:
                        change_str += f"+{births[i]} "
                    if merges[i] > 0:
                        change_str += f"M{merges[i]} "
                    if deletes[i] > 0:
                        change_str += f"-{deletes[i]}"
                    
                    ax2.text(i, total_changes + 0.1, change_str.strip(), 
                            ha='center', va='bottom',
                            fontsize=9)
            
            # Add grid and labels
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_xlabel('Update Window')
            ax2.set_ylabel('Number of State Changes')
            ax2.set_title('Detailed State Changes Per Update')
            ax2.legend(loc='upper right')
            
            # Set y-axis limits with some padding
            max_changes = max([births[i] + merges[i] + deletes[i] for i in range(len(births))], default=1)
            ax2.set_ylim(0, max_changes + 1)
            
            # Use subplots_adjust for better compatibility
            plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
            
            # Save figure
            if save_path is None:
                save_path = f'plots/state_evolution_window_{self.window_count}.png'
                
            try:
                # Make sure plots directory exists
                os.makedirs('plots', exist_ok=True)
                
                # Save figures with bbox_inches='tight' to ensure all content is included
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                # Also save a general state evolution plot that gets updated
                plt.savefig('plots/latest_state_evolution.png', dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Warning: Failed to save state evolution plot: {e}")
            
        except Exception as e:
            print(f"Error creating state evolution plot: {e}")
            # If we get an error, try to create a simplified version
            try:
                # Create a new simple figure with just text explaining the error
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Error creating state evolution plot:\n{e}",
                        ha='center', va='center', fontsize=12, 
                        transform=plt.gca().transAxes)
                
                if save_path is None:
                    save_path = f'plots/state_evolution_window_{self.window_count}.png'
                
                plt.savefig(save_path, dpi=300)
                plt.savefig('plots/latest_state_evolution.png', dpi=300)
            except:
                print("Could not create error message plot")
        finally:
            # Make sure to close the figure even if there's an error
            plt.close()
    
    def visualize_state_patterns(self, data=None, states=None, save_path=None):
        """
        Create a comprehensive visualization showing what patterns each state represents.
        
        Args:
            data: Optional tensor of shape (window_size, n_features). Uses current_data if None.
            states: Optional tensor of state assignments. Uses most recent states if None.
            save_path: Path to save the visualization. If None, generates a default path.
            
        This visualization shows:
        1. Average pattern for each state across all features
        2. Distribution of data within each state (min/max/std)
        3. State transition diagram with most common transitions
        4. State sequence timeline with data overlay
        """
        # Use current data if none provided
        if data is None and self.current_data is not None:
            data = self.current_data
        
        if states is None and self.state_history and len(self.state_history) > 0:
            states = self.state_history[-1]
            
        if data is None or states is None:
            print("No data or states available for pattern visualization")
            return
            
        # Convert to numpy if needed
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
        
        # Identify unique states
        unique_states = np.unique(states_np)
        n_states = len(unique_states)
        n_features = data_np.shape[1]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 4 * (n_states + 2)))
        
        # 1. First subplot: Full time series with state coloring
        ax_full = plt.subplot2grid((n_states + 2, 1), (0, 0))
        
        # Plot each feature with different line style
        for f in range(n_features):
            ax_full.plot(data_np[:, f], '.-', alpha=0.7, label=f'Feature {f+1}')
            
        # Add state background coloring
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10', max(10, n_states))
        
        # Create colored background for states
        for i, state in enumerate(unique_states):
            mask = states_np == state
            indices = np.where(mask)[0]
            if len(indices) > 0:
                for idx in indices:
                    ax_full.axvspan(idx-0.5, idx+0.5, alpha=0.2, color=cmap(i))
                    
        ax_full.set_title('Time Series with State Assignments')
        ax_full.legend(loc='upper right')
        ax_full.set_xlabel('Time')
        ax_full.set_ylabel('Value')
        
        # 2. For each state, show average pattern and variation
        state_patterns = {}
        for i, state in enumerate(unique_states):
            # Extract data segments for this state
            mask = states_np == state
            state_data = data_np[mask]
            
            # Store patterns for later use
            state_patterns[state] = state_data
            
            # Create subplot for this state
            ax_state = plt.subplot2grid((n_states + 2, 1), (i + 1, 0))
            
            # Skip if no data for this state
            if len(state_data) == 0:
                ax_state.text(0.5, 0.5, f'No data for State {state}', 
                              ha='center', va='center', transform=ax_state.transAxes)
                continue
                
            # Calculate statistics
            mean_pattern = np.mean(state_data, axis=0)
            std_pattern = np.std(state_data, axis=0)
            min_pattern = np.min(state_data, axis=0)
            max_pattern = np.max(state_data, axis=0)
            
            # Plot mean pattern for each feature
            x = np.arange(n_features)
            ax_state.plot(x, mean_pattern, 'o-', linewidth=2, label='Mean')
            
            # Plot variance as shaded area
            ax_state.fill_between(x, mean_pattern - std_pattern, mean_pattern + std_pattern, 
                                 alpha=0.3, label='±1 Std Dev')
            
            # Plot min/max as error bars
            ax_state.errorbar(x, mean_pattern, yerr=[mean_pattern - min_pattern, max_pattern - mean_pattern],
                             fmt='none', alpha=0.5, label='Min/Max')
            
            # Calculate frequency and median duration
            state_freq = np.sum(mask) / len(states_np)
            
            # Find durations of continuous segments
            durations = []
            current_run = 0
            for j in range(len(states_np)):
                if states_np[j] == state:
                    current_run += 1
                elif current_run > 0:
                    durations.append(current_run)
                    current_run = 0
            if current_run > 0:
                durations.append(current_run)
                
            median_duration = np.median(durations) if durations else 0
            
            # Add state statistics to title
            ax_state.set_title(f'State {state} Pattern (Freq: {state_freq:.2f}, Med. Duration: {median_duration:.1f})')
            ax_state.set_xlabel('Feature')
            ax_state.set_ylabel('Value')
            ax_state.set_xticks(x)
            ax_state.set_xticklabels([f'F{i+1}' for i in range(n_features)])
            ax_state.legend(loc='upper right')
            ax_state.grid(True, alpha=0.3)
            
            # Color the background with the state color
            ax_state.set_facecolor(cmap(i, alpha=0.1))
        
        # Last subplot: State transition visualization
        ax_trans = plt.subplot2grid((n_states + 2, 1), (n_states + 1, 0))
        
        # Create state transition image
        transitions = np.zeros((2, len(states_np)-1))
        for t in range(len(states_np)-1):
            transitions[0, t] = states_np[t]
            transitions[1, t] = states_np[t+1]
            
        # Display the transitions
        img = ax_trans.imshow(transitions, aspect='auto', interpolation='none',
                              cmap=cmap, vmin=0, vmax=max(unique_states))
        
        # Add labels
        ax_trans.set_title('State Transitions Over Time')
        ax_trans.set_yticks([0, 1])
        ax_trans.set_yticklabels(['From', 'To'])
        ax_trans.set_xlabel('Time')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax_trans, orientation='horizontal', pad=0.2)
        cbar.set_label('State')
        cbar.set_ticks(unique_states)
        
        # Use subplots_adjust for better compatibility
        plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.1, right=0.95)
        
        # Generate save path if not provided
        if save_path is None:
            # Make sure plots directory exists
            os.makedirs('plots', exist_ok=True)
            save_path = f'plots/state_patterns_window_{self.window_count}.png'
        
        # Calculate state durations
        durations = []
        state_labels = []
        current_state = states_np[0]
        current_run = 1
        
        for t in range(1, len(states_np)):
            if states_np[t] == current_state:
                current_run += 1
            else:
                durations.append(current_run)
                state_labels.append(current_state)
                current_state = states_np[t]
                current_run = 1
                
        # Add the last run
        if current_run > 0:
            durations.append(current_run)
            state_labels.append(current_state)
            
        # Convert to numpy arrays
        durations = np.array(durations)
        state_labels = np.array(state_labels)
        
        # Generate save path if not provided
        if save_path is None:
            # Make sure plots directory exists
            os.makedirs('plots', exist_ok=True)
            save_path = f'plots/state_patterns_window_{self.window_count}.png'
        
        # Generate save path if not provided
        if save_path is None:
            # Make sure plots directory exists
            os.makedirs('plots', exist_ok=True)
            save_path = f'plots/composite_viz_window_{self.window_count}.png'
            
        try:
            # Save with bbox_inches to ensure all content is captured
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig('plots/latest_composite_viz.png', dpi=300, bbox_inches='tight')
            print(f"Composite visualization saved to {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save composite visualization: {e}")
        finally:
            plt.close(fig)
    
    def close(self):
        """Close the plot."""
        try:
            plt.ioff()
            plt.close(self.fig)
        except Exception as e:
            print(f"Warning when closing visualizer: {e}")
            
    def create_composite_state_visualization(self, data=None, states=None, save_path=None):
        """
        Create a comprehensive visualization that combines state sequence and patterns.
        
        This visualization combines:
        1. Time series data with state coloring
        2. State sequence visualization (similar to bnpy)
        3. State pattern summaries
        4. Transition probabilities between states
        
        Args:
            data: Optional tensor of shape (window_size, n_features). Uses current_data if None.
            states: Optional tensor of state assignments. Uses most recent states if None.
            save_path: Path to save the visualization. If None, generates a default path.
        """
        # Use current data if none provided
        if data is None and self.current_data is not None:
            data = self.current_data
        
        if states is None and self.state_history and len(self.state_history) > 0:
            states = self.state_history[-1]
            
        if data is None or states is None:
            print("No data or states available for composite visualization")
            return
            
        # Convert to numpy if needed
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
        
        # Identify unique states
        unique_states = np.unique(states_np)
        n_states = len(unique_states)
        n_features = data_np.shape[1]
        
        # Create color map
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10', max(10, n_states))
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Set up subplots manually
        ax_time = plt.subplot2grid((7, 3), (0, 0), colspan=3, rowspan=2)
        ax_seq = plt.subplot2grid((7, 3), (2, 0), colspan=3, rowspan=1)
        
        # Pattern plots
        ax_patterns = []
        for f in range(min(n_features, 3)):  # Limit to 3 features to avoid crowding
            ax = plt.subplot2grid((7, 3), (3, f), rowspan=2)
            ax_patterns.append(ax)
            
        # Transition and duration plots
        ax_trans = plt.subplot2grid((7, 3), (5, 0), colspan=2, rowspan=2)
        ax_dur = plt.subplot2grid((7, 3), (5, 2), rowspan=2)
        
        # 1. Top panel: Time series with state coloring
        # Plot each feature
        for f in range(n_features):
            ax_time.plot(data_np[:, f], '.-', alpha=0.7, label=f'Feature {f+1}')
            
        # Add state background coloring
        for t in range(len(states_np)):
            state_idx = np.where(unique_states == states_np[t])[0][0]
            ax_time.axvspan(t-0.5, t+0.5, alpha=0.2, color=cmap(state_idx))
        
        ax_time.set_title('Time Series with State Assignments')
        ax_time.legend(loc='upper right')
        ax_time.set_ylabel('Value')
        ax_time.grid(True, alpha=0.3)
        
        # 2. State sequence visualization (similar to bnpy)
        # Create state image by repeating states
        img_height = 50  # Height of the state image
        state_img = np.zeros((img_height, len(states_np)))
        
        # Map states to consecutive indices for better visualization
        state_map = {state: i for i, state in enumerate(unique_states)}
        for t in range(len(states_np)):
            state_img[:, t] = state_map[states_np[t]]
            
        # Display the state image
        img = ax_seq.imshow(state_img, aspect='auto', interpolation='nearest',
                           vmin=-0.5, vmax=len(unique_states)-0.5, cmap=cmap)
        ax_seq.set_yticks([])  # Hide y-axis ticks
        ax_seq.set_title('State Sequence')
        
        # Add colorbar for states
        cbar = plt.colorbar(img, ax=ax_seq, orientation='horizontal', pad=0.2,
                           ticks=range(len(unique_states)))
        cbar.set_label('State')
        cbar.set_ticklabels(unique_states)
        
        # 3. State patterns: one panel per feature
        for f in range(min(n_features, 3)):  # Limit to 3 features to avoid crowding
            ax = ax_patterns[f]
            
            # Plot patterns for each state for this feature
            for i, state in enumerate(unique_states):
                # Get data for this state
                mask = states_np == state
                if np.sum(mask) == 0:
                    continue
                    
                state_data = data_np[mask, f]
                
                # Calculate statistics
                mean_val = np.mean(state_data)
                std_val = np.std(state_data)
                
                # Plot as error bar
                ax.errorbar([i], [mean_val], yerr=[std_val], fmt='o', 
                           color=cmap(i), capsize=5, label=f'State {state}')
                
                # Add a small horizontal jitter for individual points
                jitter = np.random.normal(0, 0.1, size=len(state_data))
                ax.scatter(i + jitter, state_data, color=cmap(i), alpha=0.3, s=20)
                
            ax.set_title(f'Feature {f+1} by State')
            ax.set_xlabel('State')
            ax.set_ylabel('Value')
            ax.set_xticks(range(len(unique_states)))
            ax.set_xticklabels(unique_states)
            ax.grid(True, alpha=0.3)
        
        # 4. Transition probability heatmap
        # Calculate transition matrix
        state_indices = [np.where(unique_states == s)[0][0] for s in states_np]
        trans_matrix = np.zeros((n_states, n_states))
        for t in range(len(state_indices)-1):
            trans_matrix[state_indices[t], state_indices[t+1]] += 1
            
        # Normalize by row sums
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans_matrix = trans_matrix / row_sums
        
        sns.heatmap(trans_matrix, annot=True, cmap='Blues', ax=ax_trans,
                   xticklabels=unique_states, yticklabels=unique_states)
        ax_trans.set_title('State Transition Probabilities')
        ax_trans.set_xlabel('To State')
        ax_trans.set_ylabel('From State')
        
        # 5. State duration histogram
        # Calculate state durations
        durations = []
        state_labels = []
        current_state = states_np[0]
        current_run = 1
        
        for t in range(1, len(states_np)):
            if states_np[t] == current_state:
                current_run += 1
            else:
                durations.append(current_run)
                state_labels.append(current_state)
                current_state = states_np[t]
                current_run = 1
                
        # Add the last run
        if current_run > 0:
            durations.append(current_run)
            state_labels.append(current_state)
            
        # Convert to numpy arrays
        durations = np.array(durations)
        state_labels = np.array(state_labels)
        
        # Plot histogram by state
        for i, state in enumerate(unique_states):
            state_durations = durations[state_labels == state]
            if len(state_durations) > 0:
                ax_dur.hist(state_durations, alpha=0.7, color=cmap(i), 
                           label=f'State {state}', bins=range(1, max(durations)+2))
                
        ax_dur.set_title('State Duration Histogram')
        ax_dur.set_xlabel('Duration (time steps)')
        ax_dur.set_ylabel('Frequency')
        ax_dur.legend()
        ax_dur.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.95)
        
        # Generate save path if not provided
        if save_path is None:
            # Make sure plots directory exists
            os.makedirs('plots', exist_ok=True)
            save_path = f'plots/composite_viz_window_{self.window_count}.png'
            
        try:
            # Save with bbox_inches to ensure all content is captured
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig('plots/latest_composite_viz.png', dpi=300, bbox_inches='tight')
            print(f"Composite visualization saved to {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save composite visualization: {e}")
        finally:
            plt.close(fig)
    
    def visualize_state_time_series(self, data=None, states=None, save_path=None):
        """
        Create visualizations showing time series data for each state separately.
        
        This method:
        1. Groups data points by state
        2. For each state, shows the time series of all features
        3. Highlights where each state appears in the full sequence
        
        Args:
            data: Optional tensor of shape (window_size, n_features). Uses current_data if None.
            states: Optional tensor of state assignments. Uses most recent states if None.
            save_path: Path to save the visualization. If None, generates a default path.
        """
        # Use current data if none provided
        if data is None and self.current_data is not None:
            data = self.current_data
        
        if states is None and self.state_history and len(self.state_history) > 0:
            states = self.state_history[-1]
            
        if data is None or states is None:
            print("No data or states available for state time series visualization")
            return
            
        # Convert to numpy if needed
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
        
        # Identify unique states
        unique_states = np.unique(states_np)
        n_states = len(unique_states)
        n_features = data_np.shape[1]
        
        # Create a separate figure for each state
        for state in unique_states:
            # Create figure with subplots
            fig, axes = plt.subplots(n_features + 1, 1, figsize=(12, 3 * (n_features + 1)))
            
            # First subplot: State mask (where this state occurs in the sequence)
            state_mask = states_np == state
            axes[0].plot(np.arange(len(states_np)), state_mask.astype(int), 'r-', linewidth=2)
            axes[0].fill_between(np.arange(len(states_np)), 0, state_mask.astype(int), color='r', alpha=0.3)
            axes[0].set_ylim(-0.1, 1.1)
            axes[0].set_title(f'State {state} Occurrences')
            axes[0].set_ylabel('Active')
            axes[0].set_yticks([0, 1])
            axes[0].set_yticklabels(['No', 'Yes'])
            
            # Calculate statistics for this state
            mask_indices = np.where(state_mask)[0]
            state_data = data_np[state_mask]
            
            # Skip if no data for this state
            if len(state_data) == 0:
                axes[0].text(0.5, 0.5, "No data for this state", 
                            transform=axes[0].transAxes, ha='center')
                continue
                
            # Calculate basic statistics
            state_mean = np.mean(state_data, axis=0)
            state_std = np.std(state_data, axis=0)
            
            # For each feature, plot time series and statistics
            for f in range(n_features):
                ax = axes[f + 1]
                
                # Plot full time series in light gray
                ax.plot(np.arange(len(data_np)), data_np[:, f], color='gray', alpha=0.3, label='Full Series')
                
                # Highlight the segments belonging to this state
                for start_idx in mask_indices:
                    ax.axvspan(start_idx - 0.5, start_idx + 0.5, color='red', alpha=0.2)
                    
                # Plot points belonging to this state
                ax.scatter(mask_indices, data_np[state_mask, f], color='red', label=f'State {state}', zorder=10)
                
                # Connect consecutive points
                if len(mask_indices) > 1:
                    for i in range(len(mask_indices) - 1):
                        if mask_indices[i+1] - mask_indices[i] == 1:  # Only connect consecutive points
                            ax.plot(mask_indices[i:i+2], data_np[mask_indices[i:i+2], f], 'r-', linewidth=1.5)
                
                # Add mean and standard deviation lines
                ax.axhline(state_mean[f], color='darkred', linestyle='--', 
                          label=f'Mean: {state_mean[f]:.2f}')
                ax.axhline(state_mean[f] + state_std[f], color='darkred', linestyle=':', alpha=0.7,
                          label=f'±1 Std: {state_std[f]:.2f}')
                ax.axhline(state_mean[f] - state_std[f], color='darkred', linestyle=':', alpha=0.7)
                
                ax.set_title(f'Feature {f+1} for State {state}')
                ax.set_ylabel('Value')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add overall title
            plt.suptitle(f'State {state} Time Series Analysis (Found in {np.sum(state_mask)} positions)', 
                        fontsize=16, y=0.98)
            
            # Adjust layout
            plt.subplots_adjust(hspace=0.4, top=0.92)
            
            # Generate save path if not provided
            if save_path is None:
                os.makedirs('plots/state_time_series', exist_ok=True)
                state_save_path = f'plots/state_time_series/state_{state}_window_{self.window_count}.png'
            else:
                # Modify save_path to include state number
                dir_name = os.path.dirname(save_path)
                file_name = os.path.basename(save_path)
                os.makedirs(dir_name, exist_ok=True)
                name_parts = os.path.splitext(file_name)
                state_save_path = os.path.join(dir_name, f"{name_parts[0]}_state_{state}{name_parts[1]}")
            
            # Save the figure
            try:
                plt.savefig(state_save_path, dpi=300, bbox_inches='tight')
                print(f"State {state} time series saved to {state_save_path}")
            except Exception as e:
                print(f"Warning: Failed to save state time series for state {state}: {e}")
            finally:
                plt.close(fig)
        
        # Create a summary figure with all states
        try:
            # Create a figure showing where each state appears across the time series
            plt.figure(figsize=(12, 8))
            
            # Main subplot with data
            plt.subplot(211)
            for f in range(n_features):
                plt.plot(data_np[:, f], label=f'Feature {f+1}', alpha=0.8)
            
            plt.title('Time Series with State Assignments')
            plt.ylabel('Value')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            # State assignment subplot
            plt.subplot(212)
            
            # Create a colormap for states
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab10', max(10, n_states))
            
            # For each time point, add a colored segment for its state
            for t in range(len(states_np)):
                state_idx = np.where(unique_states == states_np[t])[0][0]
                plt.axvspan(t-0.5, t+0.5, color=cmap(state_idx), alpha=0.7)
            
            # Add legend for states
            for i, state in enumerate(unique_states):
                state_mask = states_np == state
                count = np.sum(state_mask)
                plt.plot([], [], color=cmap(i), label=f'State {state} ({count} points)')
            
            plt.yticks([])
            plt.xlabel('Time')
            plt.title('State Assignments')
            plt.legend(loc='upper right')
            
            # Save the summary figure
            os.makedirs('plots/state_time_series', exist_ok=True)
            summary_path = f'plots/state_time_series/all_states_summary_window_{self.window_count}.png'
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            print(f"State time series summary saved to {summary_path}")
        except Exception as e:
            print(f"Warning: Failed to save state time series summary: {e}")
        finally:
            plt.close()
