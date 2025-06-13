import torch
import time
import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.processor import LiveDataCollector
from src.model.trainer import LiveHDPHMM
from live_visualize import LiveVisualizer
from csv_processor import CSVDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hdp_hmm")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run HDP-HMM on live or offline data')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI visualization')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--max-windows', type=int, default=1000, help='Maximum number of windows to process')
    parser.add_argument('--data-dir', type=str, help='Directory containing CSV files for offline processing')
    parser.add_argument('--window-size', type=int, default=100, help='Size of sliding window')
    parser.add_argument('--stride', type=int, help='Stride for sliding window in offline mode (default: window_size)')
    parser.add_argument('--n-features', type=int, default=3, help='Number of features in the data')
    args = parser.parse_args()
    # Parameters
    n_features = args.n_features
    window_size = args.window_size
    max_states = 20
    sample_interval = 1.0
    max_iterations = args.max_windows  # Use the max_windows argument
    model_path = "models/hdp_hmm.pth"
    plots_path = "plots"
    
    # Create directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info("Initializing components...")
    logger.info(f"Training on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Determine if we're in offline or live mode
    offline_mode = args.data_dir is not None
    
    # Initialize data source
    if offline_mode:
        logger.info(f"Running in offline mode with data from {args.data_dir}")
        data_source = CSVDataProcessor(args.data_dir, window_size, args.stride)
        if not data_source.load_csv_files():
            logger.error("Failed to load CSV files. Exiting.")
            return
        # Update n_features from the loaded data
        n_features = data_source.n_features
        logger.info(f"Using {n_features} features from CSV files")
        # Update max_iterations if needed
        total_windows = data_source.get_total_windows()
        if total_windows < max_iterations:
            logger.info(f"Adjusting max iterations from {max_iterations} to {total_windows} based on available data")
            max_iterations = total_windows
    else:
        logger.info("Running in live mode with synthetic data")
        data_source = LiveDataCollector(n_features, window_size, sample_interval)
        # Pre-fill the collector with data so we get a window immediately
        for _ in range(window_size):
            data_source.collect_window()
    
    # Initialize trainer
    trainer = LiveHDPHMM(n_features, max_states, lr=0.01, model_path=model_path)
    
    # Initialize visualizer if GUI is enabled
    visualizer = None
    if not args.no_gui:
        visualizer = LiveVisualizer(n_features, window_size)
    
    # Try loading a pre-trained model
    logger.info("Checking for pre-trained model...")
    trainer.load_model()
    
    logger.info("Starting processing...")
    start_time = time.time()
    window_count = 0
    try:
        for i in range(max_iterations):
            # Get next window of data
            if offline_mode:
                window_data = data_source.get_next_window()
                if window_data is None:
                    logger.info("Reached the end of all CSV files")
                    break
            else:
                window_data = data_source.collect_window()
                
            if window_data is not None:
                window_count += 1
                # Train and infer
                loss = trainer.update_model(window_data)
                states, trans_probs = trainer.infer(window_data)
                
                # Visualize if GUI is enabled
                if visualizer:
                    visualizer.update_plot(
                        window_data, 
                        states, 
                        trans_probs, 
                        loss, 
                        trainer.losses, 
                        trainer.state_counts,
                        trainer.state_changes
                    )
                
                # Save model periodically
                if window_count % 10 == 0:
                    trainer.save_model()
                    logger.info(f"Window {window_count}, Loss: {loss:.4f}, States: {trainer.model.current_states}")
                    
                    # Save tile visualization even in no-gui mode
                    if not visualizer:
                        # Create a temporary visualizer just for saving plots
                        temp_visualizer = LiveVisualizer(n_features, window_size)
                        temp_visualizer.window_count = window_count
                        temp_visualizer.state_history = [states.cpu()]  # Add current states
                        temp_visualizer.create_tile_visualization()
                        
                        # Also create state evolution plot
                        if trainer.state_changes:
                            temp_visualizer.create_state_evolution_plot(
                                trainer.state_changes,
                                f'plots/state_evolution_window_{window_count}.png'
                            )
                        
                        # Create learning curve plot
                        if trainer.losses:
                            # Filter out None values from state_counts
                            valid_state_counts = [s for s in trainer.state_counts if s is not None] if trainer.state_counts else []
                            temp_visualizer.create_learning_curve(
                                trainer.losses,
                                valid_state_counts,
                                f'plots/learning_curve_window_{window_count}.png'
                            )
                        
                        # Create state-specific time series visualizations every 10 windows
                        if window_count % 10 == 0:
                            temp_visualizer.current_data = window_data.cpu()  # Store current data
                            temp_visualizer.visualize_state_time_series(window_data, states)
                        
                        del temp_visualizer
                
            # In live mode, wait for the next sample; in offline mode, continue immediately
            if not offline_mode:
                time.sleep(sample_interval)
    
    except KeyboardInterrupt:
        logger.info("Stopping processing...")
        trainer.save_model()
        if visualizer:
            visualizer.close()
    
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        trainer.save_model()
        raise
    
    finally:
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Saving final model...")
        trainer.save_model()
        
        # Save final transition matrix
        logger.info("Saving final transition matrix...")
        trainer.save_transition_matrix(path_prefix='plots/final_transition_matrix')
        logger.info(f"Final transition matrix saved to plots/final_transition_matrix.png and .csv")
        
        # Print state evolution summary
        if trainer.state_changes:
            births = sum(len(sc.get('birthed', [])) for sc in trainer.state_changes)
            merges = sum(len(sc.get('merged', [])) for sc in trainer.state_changes)
            deletes = sum(len(sc.get('deleted', [])) for sc in trainer.state_changes)
            
            # Get detailed count of each type of event
            birth_counts = {}
            merge_counts = {}
            delete_counts = {}
            
            for sc in trainer.state_changes:
                # Count births
                for state in sc.get('birthed', []):
                    birth_counts[state] = birth_counts.get(state, 0) + 1
                    
                # Count merges by destination state
                for src, dst in sc.get('merged', []):
                    merge_counts[dst] = merge_counts.get(dst, 0) + 1
                    
                # Count deletions
                for state in sc.get('deleted', []):
                    delete_counts[state] = delete_counts.get(state, 0) + 1
            
            logger.info("\n========== State Evolution Summary ==========")
            logger.info(f"Initial states: {trainer.state_changes[0]['initial_states'] if trainer.state_changes else 'N/A'}")
            logger.info(f"Final states: {trainer.model.current_states}")
            logger.info(f"Total events: {births} births, {merges} merges, {deletes} deletes")
            
            # Print detailed statistics if there were any changes
            if births > 0:
                logger.info(f"Birth events by state: {dict(sorted(birth_counts.items()))}")
            if merges > 0:
                logger.info(f"Merge destination counts: {dict(sorted(merge_counts.items()))}")
            if deletes > 0:
                logger.info(f"Delete events by state: {dict(sorted(delete_counts.items()))}")
            
            logger.info("============================================\n")
            
            # Print text-based visualization of state evolution
            trainer.print_state_evolution_summary()
            
            # Save final state evolution plot
            if visualizer:
                visualizer.create_state_evolution_plot(
                    trainer.state_changes,
                    'plots/final_state_evolution.png'
                )
                
                # We need to get the final data window and states to create visualizations
                # Create a temporary visualizer with the last processed data
                if hasattr(trainer, 'latest_data') and hasattr(trainer, 'latest_states'):
                    temp_visualizer = LiveVisualizer(trainer.model.n_features, trainer.latest_data.shape[0])
                    temp_visualizer.window_count = window_count
                    temp_visualizer.current_data = trainer.latest_data.cpu()
                    
                    # Create and save state patterns visualization
                    temp_visualizer.visualize_state_patterns(
                        trainer.latest_data, 
                        trainer.latest_states, 
                        'plots/final_state_patterns.png'
                    )
                    
                    # Create and save composite visualization
                    temp_visualizer.create_composite_state_visualization(
                        trainer.latest_data,
                        trainer.latest_states,
                        'plots/final_composite_viz.png'
                    )
                    
                    # Create and save state-specific time series
                    temp_visualizer.visualize_state_time_series(
                        trainer.latest_data,
                        trainer.latest_states,
                        'plots/final_state_time_series.png'
                    )
            else:
                # Create a temporary visualizer
                temp_visualizer = LiveVisualizer(n_features, window_size)
                temp_visualizer.window_count = window_count
                
                # Make sure we also save the learning curve
                if trainer.losses:
                    valid_state_counts = [s for s in trainer.state_counts if s is not None] if trainer.state_counts else []
                    temp_visualizer.create_learning_curve(
                        trainer.losses,
                        valid_state_counts,
                        'plots/final_learning_curve.png'
                    )
                
                # Create final state evolution plot
                if trainer.state_changes:
                    temp_visualizer.create_state_evolution_plot(
                        trainer.state_changes,
                        'plots/final_state_evolution.png'
                    )
                
                # Create final visualizations if we have latest data
                if hasattr(trainer, 'latest_data') and hasattr(trainer, 'latest_states'):
                    temp_visualizer.current_data = trainer.latest_data.cpu()
                    
                    # Create and save state patterns visualization
                    temp_visualizer.visualize_state_patterns(
                        trainer.latest_data, 
                        trainer.latest_states, 
                        'plots/final_state_patterns.png'
                    )
                    
                    # Create and save composite visualization
                    temp_visualizer.create_composite_state_visualization(
                        trainer.latest_data,
                        trainer.latest_states,
                        'plots/final_composite_viz.png'
                    )
                    
                    # Create and save state-specific time series
                    temp_visualizer.visualize_state_time_series(
                        trainer.latest_data,
                        trainer.latest_states,
                        'plots/final_state_time_series.png'
                    )
                
                del temp_visualizer
        
        # Print processing summary
        mode_str = "offline" if offline_mode else "live"
        logger.info(f"Processed {window_count} windows in {processing_time:.1f} seconds ({mode_str} mode)")
        logger.info(f"{mode_str.capitalize()} processing complete")
        if visualizer:
            visualizer.close()

if __name__ == "__main__":
    main()