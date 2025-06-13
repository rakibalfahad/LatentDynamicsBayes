import torch
import torch.optim as optim
from hdp_hmm import HDPHMM
import os

class LiveHDPHMM:
    def __init__(self, n_features, max_states=20, lr=0.01, model_path="models/hdp_hmm.pth"):
        """
        Manage live training and inference for HDP-HMM.
        
        Args:
            n_features (int): Number of features
            max_states (int): Maximum number of states
            lr (float): Learning rate
            model_path (str): Path to save/load model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HDPHMM(n_features, max_states).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model_path = model_path
        self.losses = []
        self.state_counts = []  # Track number of states over time
        self.window_count = 0
        self.state_changes = []  # Track state changes (birth, merge, delete) over time
    
    def update_model(self, window_data):
        """Update model with a new window of data."""
        try:
            with torch.autograd.set_detect_anomaly(True):  # Keep anomaly detection
                self.optimizer.zero_grad()
                _, _, log_likelihood = self.model.forward_backward(window_data)
                loss = -log_likelihood
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())
                
                # Increment window count
                self.window_count += 1
                
                # Update states every 10 windows
                if self.window_count % 10 == 0:
                    with torch.no_grad():
                        try:
                            result = self.update_states_safe(window_data)
                            
                            current_states, state_change_info = result
                            
                            self.state_counts.append(current_states)
                            self.state_changes.append(state_change_info)
                            
                            # Print state change information in a clearer format
                            stats_str = self.format_state_update_stats(state_change_info)
                            print(f"\n[Window {self.window_count}] State Update: {stats_str}")
                            if 'error' in state_change_info:
                                print(f"  Error: {state_change_info['error']}")
                                
                        except Exception as e:
                            print(f"\n[Window {self.window_count}] Error during state update: {str(e)}")
                            # Don't let state update errors crash the training process
                
                return loss.item()
        except RuntimeError as e:
            print(f"Error in update_model: {e}")
            raise
    
    def infer(self, window_data):
        """Perform inference on a window of data."""
        states, trans_probs = self.model.infer_states(window_data)
        return states, trans_probs
    
    def save_model(self):
        """Save the current model state."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'state_counts': self.state_counts,
            'window_count': self.window_count
        }, self.model_path.replace('.pth', '_checkpoint.pth'))
        
        # Also save the current transition matrix
        self.save_transition_matrix(path_prefix='plots/latest_transition_matrix')
    
    def load_model(self):
        """Load a saved model state."""
        try:
            self.model.load_model(self.model_path)
            
            # Try to load checkpoint
            try:
                checkpoint_path = self.model_path.replace('.pth', '_checkpoint.pth')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.losses = checkpoint.get('losses', [])
                    self.state_counts = checkpoint.get('state_counts', [self.model.current_states])
                    self.window_count = checkpoint.get('window_count', 0)
                    print(f"Loaded training state from {checkpoint_path}")
            except Exception as e:
                print(f"Could not load optimizer state: {e}")
                print("Using initial optimizer state")
                self.losses = []
                self.state_counts = [self.model.current_states]
                self.window_count = 0
                
        except FileNotFoundError:
            print("No saved model found, starting fresh.")
    
    def save_transition_matrix(self, path_prefix='plots/transition_matrix'):
        """
        Save the current transition probability matrix.
        
        Args:
            path_prefix (str): Prefix for the saved files
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        # Get the transition probabilities
        with torch.no_grad():
            # Create dummy input to get transition probs
            dummy_input = torch.zeros(10, self.model.n_features, device=self.device)
            _, trans_probs = self.model.infer_states(dummy_input)
            
            # Convert to numpy
            trans_probs_np = trans_probs.cpu().detach().numpy()
            
            # Save as numpy file
            np.save(f"{path_prefix}", trans_probs_np)
            
            # Save as CSV for easy import into other tools
            np.savetxt(f"{path_prefix}.csv", trans_probs_np, delimiter=',')
            
            # Create and save a heatmap visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(trans_probs_np, annot=True, cmap='Blues', fmt='.3f')
            plt.title('Transition Probability Matrix')
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.savefig(f"{path_prefix}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            return trans_probs_np
    
    def update_states_safe(self, window_data):
        """
        A safe wrapper around the model's update_states method that handles errors
        and ensures a consistent return format.
        
        Args:
            window_data: Tensor of shape (window_size, n_features)
            
        Returns:
            tuple: (int: New number of states, dict: State change information)
        """
        try:
            # Since the model's update_states method is currently problematic,
            # we'll implement a simplified version here
            with torch.no_grad():
                # Manually apply the update logic
                beta_weights = self.model.stick_breaking(self.model.beta_logits)
                threshold = 1e-3
                active_indices = []
                inactive_indices = []
                initial_states = self.model.current_states
                
                # 1. DELETE: Find states with probability below threshold
                deleted_states = []
                for k in range(self.model.current_states):
                    if beta_weights[k] > threshold:
                        active_indices.append(k)
                    else:
                        inactive_indices.append(k)
                        deleted_states.append(k)
                        
                # 2. BIRTH: Add new states occasionally if there are inactive states
                birthed_states = []
                if self.window_count % 30 == 0 and inactive_indices:  # Try birth every 30 windows
                    try:
                        # Calculate negative log-likelihood for each observation
                        emission_probs = torch.stack([
                            torch.distributions.MultivariateNormal(
                                self.model.means[k],
                                covariance_matrix=torch.diag(torch.exp(self.model.log_vars[k]) + 1e-6)
                            ).log_prob(window_data)
                            for k in range(self.model.current_states)
                        ], dim=1)
                        
                        # Maximum emission probability for each observation
                        max_emission_probs, _ = torch.max(emission_probs, dim=1)
                        
                        # Average negative log-likelihood (lower = better fit)
                        avg_nll = -torch.mean(max_emission_probs)
                        
                        # If model fit is poor, add a new state
                        if avg_nll > 5.0 and len(inactive_indices) > 0:
                            # Use an inactive state for the new state
                            new_state_idx = inactive_indices[0]
                            inactive_indices.pop(0)
                            
                            # Initialize with random parameters
                            self.model.means.data[new_state_idx] = torch.mean(window_data, dim=0)
                            self.model.log_vars.data[new_state_idx] = torch.log(torch.var(window_data, dim=0) + 1e-6)
                            
                            # Add to active indices with small weight
                            active_indices.append(new_state_idx)
                            birthed_states.append(new_state_idx)
                            
                            # Set beta logit to small value
                            self.model.beta_logits.data[new_state_idx] = torch.log(torch.tensor(0.05 / 0.95))
                            
                            print(f"[Birth] Added new state {new_state_idx} with weight {0.05:.3f}")
                    except Exception as e:
                        print(f"Error in birth mechanism: {e}")
                
                # 3. MERGE: Combine similar states
                merged_pairs = []
                if self.window_count % 20 == 0 and len(active_indices) > 1:  # Try merges every 20 windows
                    merge_distance = 0.7  # Threshold for merging
                    merged_indices = set()
                    
                    # Create a copy of active_indices to avoid modification during iteration
                    active_indices_copy = active_indices.copy()
                    
                    i = 0
                    while i < len(active_indices_copy):
                        if i in merged_indices:
                            i += 1
                            continue
                            
                        i_idx = active_indices_copy[i]
                        
                        j = i + 1
                        while j < len(active_indices_copy):
                            if j in merged_indices:
                                j += 1
                                continue
                                
                            j_idx = active_indices_copy[j]
                            # Calculate distance between state means
                            dist = torch.norm(self.model.means[i_idx] - self.model.means[j_idx])
                            
                            if dist < merge_distance:
                                # Merge j into i by weight averaging
                                weight_i = beta_weights[i_idx]
                                weight_j = beta_weights[j_idx]
                                total_weight = weight_i + weight_j
                                
                                # Update parameters of state i (weighted average)
                                self.model.means.data[i_idx] = (weight_i * self.model.means[i_idx] + weight_j * self.model.means[j_idx]) / total_weight
                                self.model.log_vars.data[i_idx] = (weight_i * self.model.log_vars[i_idx] + weight_j * self.model.log_vars[j_idx]) / total_weight
                                
                                # Update beta logits
                                self.model.beta_logits.data[i_idx] = torch.log(total_weight / (1 - total_weight))
                                
                                # Mark j as merged
                                merged_indices.add(j)
                                inactive_indices.append(j_idx)
                                if j_idx in active_indices:
                                    active_indices.remove(j_idx)
                                
                                # Record merge
                                merged_pairs.append((j_idx, i_idx))
                                print(f"[Merge] Merged state {j_idx} into {i_idx}")
                            
                            j += 1
                        
                        i += 1
                
                # 4. Update current state count
                current_states = len(active_indices)
                if current_states < 1:
                    current_states = 1
                    
                # Set new state count
                self.model.current_states = current_states
                
                # Create state change info dict
                state_change_info = {
                    'deleted': deleted_states,
                    'merged': merged_pairs,
                    'birthed': birthed_states,
                    'initial_states': initial_states,
                    'final_states': current_states,
                    'active_states': active_indices,
                    'inactive_states': inactive_indices,
                    'note': "Using simplified state update (bypass broken update_states)"
                }
                
                return current_states, state_change_info
                
        except Exception as e:
            print(f"\n[Window {self.window_count}] Error during state update: {str(e)}")
            # Return safe defaults
            current_states = self.model.current_states
            state_change_info = {
                'deleted': [],
                'merged': [],
                'birthed': [],
                'initial_states': current_states,
                'final_states': current_states,
                'error': f"Exception during update: {str(e)}"
            }
            return current_states, state_change_info
        
    def format_state_update_stats(self, state_change_info):
        """
        Format state update statistics in a clear, concise way
        
        Args:
            state_change_info: Dictionary with state change information
            
        Returns:
            str: Formatted string with state change statistics
        """
        stats = []
        
        # Add basic state count information
        init = state_change_info.get('initial_states', 0)
        final = state_change_info.get('final_states', 0)
        stats.append(f"States: {init} → {final}")
        
        # Add detailed change statistics
        changes = []
        
        # Birth events
        births = state_change_info.get('birthed', [])
        if births:
            changes.append(f"+{len(births)} birth" + ("s" if len(births) > 1 else ""))
        
        # Delete events
        deletes = state_change_info.get('deleted', [])
        if deletes:
            changes.append(f"-{len(deletes)} delete" + ("s" if len(deletes) > 1 else ""))
        
        # Merge events
        merges = state_change_info.get('merged', [])
        if merges:
            changes.append(f"~{len(merges)} merge" + ("s" if len(merges) > 1 else ""))
        
        # Add changes if any exist
        if changes:
            stats.append("Changes: " + ", ".join(changes))
            
            # Add detailed state IDs if needed
            details = []
            if births:
                details.append(f"Birth: state(s) {', '.join(map(str, births))}")
            if deletes:
                details.append(f"Delete: state(s) {', '.join(map(str, deletes))}")
            if merges:
                merge_details = [f"{src}→{dst}" for src, dst in merges]
                details.append(f"Merge: {', '.join(merge_details)}")
                
            if details:
                stats.append("Details: " + "; ".join(details))
        else:
            stats.append("No state changes")
            
        return " | ".join(stats)
    
    def print_state_evolution_summary(self):
        """
        Print a text-based visualization of state evolution through time.
        Shows births, merges, and deletes of states across training windows.
        """
        if not self.state_changes:
            print("No state evolution data available")
            return
            
        # Get maximum number of states observed
        max_states = max(
            [sc.get('initial_states', 0) for sc in self.state_changes] + 
            [sc.get('final_states', 0) for sc in self.state_changes]
        )
        
        # Create a timeline of state existence
        state_timeline = {}
        for state_idx in range(max_states + 1):  # +1 to include the last state
            state_timeline[state_idx] = []
        
        # Process state changes to build timeline
        for window_idx, sc in enumerate(self.state_changes):
            # Mark active states
            active_states = sc.get('active_states', [])
            for state in active_states:
                if state < len(state_timeline):
                    state_timeline[state].append('A')  # Active
            
            # Mark birth events
            for state in sc.get('birthed', []):
                if state < len(state_timeline):
                    if state_timeline[state]:  # Check if list is not empty
                        state_timeline[state][-1] = 'B'  # Birth
                    else:
                        state_timeline[state].append('B')
            
            # Mark merge events (source states)
            for src, dst in sc.get('merged', []):
                if src < len(state_timeline):
                    if state_timeline[src]:  # Check if list is not empty
                        state_timeline[src][-1] = f'M→{dst}'  # Merged into dst
                    else:
                        state_timeline[src].append(f'M→{dst}')
            
            # Mark deleted states
            for state in sc.get('deleted', []):
                if state < len(state_timeline):
                    if state_timeline[state]:  # Check if list is not empty
                        state_timeline[state][-1] = 'D'  # Deleted
                    else:
                        state_timeline[state].append('D')
            
            # Fill in inactive states with dots
            for state in state_timeline:
                while len(state_timeline[state]) <= window_idx:
                    state_timeline[state].append('.')  # Inactive
        
        # Print the visualization
        print("\nState Evolution Timeline:")
        print("   " + " ".join(f"{i:3d}" for i in range(len(self.state_changes))))
        print("   " + "─" * (len(self.state_changes) * 4))
        
        for state in sorted(state_timeline.keys()):
            timeline = state_timeline[state]
            if any(t != '.' for t in timeline):  # Only show states that were active at some point
                line = f"{state:2d}│"
                for status in timeline:
                    if status == 'A':
                        line += " ● "  # Active state
                    elif status == 'B':
                        line += " ⊕ "  # Birth
                    elif status == 'D':
                        line += " ⊗ "  # Delete
                    elif status.startswith('M'):
                        line += " ⊙ "  # Merge
                    else:
                        line += "   "  # Inactive
                print(line)
        
        print()
        print("Legend: ● Active  ⊕ Birth  ⊗ Delete  ⊙ Merge")