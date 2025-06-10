"""
HDP-HMM Live Streaming Data Analysis

This script provides real-time clustering of multidimensional time series data
using a Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) with 
stick-breaking construction. It handles incremental training, visualization,
and model persistence for continuous data streams.

Example:
    $ python main.py --config config.json
"""
import torch
import argparse
import time
import signal
import sys
import os
import logging
from datetime import datetime

# Import components
from src.data.collector import LiveDataCollector
from src.data.processor import TimeSeriesProcessor
from src.model.trainer import LiveTrainer
from src.visualization.visualizer import LiveVisualizer
from src.utils.utils import logger, ConfigManager, set_seed, setup_device, PerformanceMonitor

# Global flag for clean shutdown
running = True

def signal_handler(sig, frame):
    """Handle interrupt signals for clean shutdown."""
    global running
    logger.info("Interrupt received, shutting down...")
    running = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='HDP-HMM Live Streaming Analysis')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without GUI visualization')
    parser.add_argument('--use-real', action='store_true',
                        help='Use real system metrics instead of simulated data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Size of sliding window')
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up device (GPU/CPU)
    device = setup_device()
    
    # Get configuration values with command line overrides
    n_features = config_manager.get('model.n_features', 3)
    max_states = config_manager.get('model.max_states', 20)
    window_size = args.window_size or config_manager.get('data.window_size', 100)
    sample_interval = config_manager.get('data.sample_interval', 1.0)
    max_iterations = config_manager.get('training.max_iterations', 1000)
    use_real_metrics = args.use_real or config_manager.get('data.use_real_metrics', False)
    interactive = not args.no_gui and config_manager.get('visualization.interactive', True)
    
    # Create directories
    model_dir = config_manager.get('paths.model_dir', 'models')
    plot_dir = config_manager.get('paths.plot_dir', 'plots')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Data collector
    collector = LiveDataCollector(
        n_features=n_features,
        window_size=window_size,
        sample_interval=sample_interval,
        use_real_metrics=use_real_metrics,
        device=device
    )
    
    # Data processor
    processor = TimeSeriesProcessor(
        n_features=n_features,
        window_size=window_size,
        normalize=True,
        device=device
    )
    
    # Model trainer
    trainer = LiveTrainer(
        n_features=n_features,
        max_states=max_states,
        lr=config_manager.get('model.learning_rate', 0.01),
        device=device,
        model_dir=model_dir,
        model_name="hdp_hmm"
    )
    
    # Visualizer
    feature_names = config_manager.get('visualization.feature_names', None)
    visualizer = LiveVisualizer(
        n_features=n_features,
        window_size=window_size,
        feature_names=feature_names,
        max_states=max_states,
        output_dir=plot_dir,
        interactive=interactive
    )
    
    # Performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Try loading a pre-trained model
    logger.info("Checking for pre-trained model...")
    trainer.load_model()
    
    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Main processing loop
    logger.info("Starting live processing...")
    start_time = time.time()
    iter_count = 0
    
    try:
        while running and iter_count < max_iterations:
            iter_count += 1
            
            # 1. Collect new data window
            window_data = collector.collect_window()
            
            if window_data is not None:
                # 2. Process the data
                processed_data = processor.process_window(window_data)
                
                # 3. Train the model
                perf_monitor.start_timer("train")
                loss = trainer.update_model(processed_data)
                train_time = perf_monitor.stop_timer("train", category="training")
                
                # 4. Perform inference
                perf_monitor.start_timer("infer")
                states, trans_probs = trainer.infer(processed_data)
                infer_time = perf_monitor.stop_timer("infer", category="inference")
                
                # 5. Visualize results
                active_states = trainer.get_active_states_count()
                save_plots = iter_count % config_manager.get('visualization.save_interval', 20) == 0
                visualizer.update_plot(
                    processed_data, states, trans_probs, loss, 
                    active_states=active_states,
                    save=save_plots
                )
                
                # 6. Save model periodically
                save_interval = config_manager.get('training.save_interval', 10)
                if iter_count % save_interval == 0:
                    trainer.save_model()
                    logger.info(f"Iteration {iter_count}/{max_iterations}, "
                               f"Loss: {loss:.4f}, Active States: {active_states}, "
                               f"Train: {train_time:.3f}s, Infer: {infer_time:.3f}s")
                
                # 7. Save checkpoint periodically
                checkpoint_interval = config_manager.get('training.checkpoint_interval', 100)
                if iter_count % checkpoint_interval == 0:
                    checkpoint_path = trainer.save_checkpoint()
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Sleep to maintain desired sample rate
            time.sleep(sample_interval)
    
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
    
    finally:
        # Clean shutdown
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"Processed {iter_count} iterations in {elapsed_time:.1f} seconds")
        
        if perf_monitor.training_times:
            avg_train_time = sum(perf_monitor.training_times) / len(perf_monitor.training_times)
            logger.info(f"Average training time: {avg_train_time:.3f} seconds")
        
        if perf_monitor.inference_times:
            avg_infer_time = sum(perf_monitor.inference_times) / len(perf_monitor.inference_times)
            logger.info(f"Average inference time: {avg_infer_time:.3f} seconds")
        
        visualizer.close()
        logger.info("Live processing complete")

if __name__ == "__main__":
    main()