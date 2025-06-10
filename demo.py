"""
Quick Demo Script for HDP-HMM Live Streaming

This script provides a simple demonstration of the HDP-HMM model
on simulated data with minimal configuration.
"""
import torch
import time
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.collector import LiveDataCollector
from src.data.processor import TimeSeriesProcessor
from src.model.trainer import LiveTrainer
from src.model.hdp_hmm import HDPHMM
from src.visualization.visualizer import LiveVisualizer
from src.utils.utils import set_seed

def run_demo(iterations=100, window_size=50, interactive=True):
    """Run a quick demo of the HDP-HMM model."""
    print("Starting HDP-HMM demo...")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    n_features = 3
    max_states = 10
    
    # Initialize components
    collector = LiveDataCollector(
        n_features=n_features,
        window_size=window_size,
        sample_interval=0.1,
        use_real_metrics=False,
        device=device
    )
    
    processor = TimeSeriesProcessor(
        n_features=n_features,
        window_size=window_size,
        normalize=True,
        device=device
    )
    
    trainer = LiveTrainer(
        n_features=n_features,
        max_states=max_states,
        lr=0.01,
        device=device,
        model_dir="models",
        model_name="demo_hdp_hmm"
    )
    
    visualizer = LiveVisualizer(
        n_features=n_features,
        window_size=window_size,
        feature_names=["Feature 1", "Feature 2", "Feature 3"],
        max_states=max_states,
        output_dir="plots",
        interactive=interactive
    )
    
    print("Collecting data and training model...")
    
    # Main loop
    for i in range(iterations):
        # Collect new window
        window_data = collector.collect_window()
        
        if window_data is not None:
            # Process data
            processed_data = processor.process_window(window_data)
            
            # Train and infer
            loss = trainer.update_model(processed_data)
            states, trans_probs = trainer.infer(processed_data)
            
            # Visualize
            visualizer.update_plot(
                processed_data, states, trans_probs, loss, 
                active_states=trainer.get_active_states_count(),
                save=(i == iterations - 1)  # Save only the final plot
            )
            
            # Print progress
            if i % 10 == 0:
                print(f"Iteration {i}/{iterations}, Loss: {loss:.4f}")
            
            # Add a small delay for visualization
            if interactive:
                time.sleep(0.1)
    
    print("Demo complete!")
    
    # Save final model
    trainer.save_model()
    print("Model saved to models/demo_hdp_hmm.pth")
    
    # Close visualizer
    visualizer.close()
    
    return trainer, visualizer

if __name__ == "__main__":
    # Run the demo
    trainer, visualizer = run_demo(iterations=150)
    
    # Keep plots open until user closes them
    if plt.get_fignums():
        print("Close the plot windows to exit...")
        plt.show(block=True)
