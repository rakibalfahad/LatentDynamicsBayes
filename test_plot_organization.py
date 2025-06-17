#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify plot organization in the HDP-HMM visualization system.
This script runs a few windows of the model and checks if the plots are organized correctly.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import LiveDataCollector
from src.model.trainer import LiveHDPHMM
from live_visualize import LiveVisualizer

def test_plot_organization(n_windows=15):
    """
    Run a few windows of the model and verify plot organization.
    
    Args:
        n_windows (int): Number of windows to process
    """
    print("Testing plot organization with", n_windows, "windows")
    
    # Clean existing plots directory
    plots_dir = Path("plots")
    if plots_dir.exists():
        print(f"Cleaning existing plots directory: {plots_dir}")
        shutil.rmtree(plots_dir)
    
    # Create fresh plots directory
    plots_dir.mkdir(exist_ok=True)
    
    # Parameters
    n_features = 3
    window_size = 100
    max_states = 10
    
    # Initialize components
    data_collector = LiveDataCollector(window_size=window_size, n_features=n_features)
    model = LiveHDPHMM(n_features=n_features, max_states=max_states)
    visualizer = LiveVisualizer(n_features=n_features, window_size=window_size)
    
    # Pre-fill buffer with some random data
    # This ensures we have enough data for the first window
    for _ in range(window_size):
        data_collector.collect_window()  # This will add a sample to the buffer
    
    print("Running", n_windows, "windows...")
    # Process windows
    losses = []
    state_counts = []
    state_changes = []
    
    for i in range(n_windows):
        print(f"Processing window {i+1}/{n_windows}")
        
        # Get window of data
        data = data_collector.get_window()
        if data is None:
            # Generate some random data
            data = torch.randn(window_size, n_features)
        
        # Update model
        results = model.update(data)
        loss, states, trans_probs = results["loss"], results["states"], results["trans_probs"]
        losses.append(loss)
        state_counts.append(len(model.get_active_states()))
        
        # Track state changes if available
        if "state_changes" in results:
            state_changes.append(results["state_changes"])
        else:
            state_changes.append({"initial_states": state_counts[-1], "final_states": state_counts[-1]})
        
        # Visualize
        visualizer.update_plot(
            data, states, trans_probs, loss, losses, state_counts, state_changes
        )
        
        # Add more random data
        for _ in range(window_size // 10):  # Add 10% new data points
            data_collector.collect_window()
    
    visualizer.close()
    
    # Check directory structure
    print("\nChecking plot organization:")
    
    # List all subdirectories in plots/
    plot_dirs = [d for d in plots_dir.iterdir() if d.is_dir()]
    print(f"Found {len(plot_dirs)} plot subdirectories:")
    for d in plot_dirs:
        n_files = len(list(d.glob("*")))
        print(f"  - {d.name}: {n_files} files")
    
    # Check for latest plots in main directory
    main_dir_plots = [f for f in plots_dir.glob("*.png") if "latest" in f.name]
    print(f"\nFound {len(main_dir_plots)} 'latest' plots in main directory:")
    for f in main_dir_plots:
        print(f"  - {f.name}")
    
    print("\nPlot organization test completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test plot organization in HDP-HMM visualization")
    parser.add_argument("--windows", type=int, default=15, help="Number of windows to process")
    args = parser.parse_args()
    
    test_plot_organization(args.windows)
