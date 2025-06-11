import torch
import time
import logging
import os
from src.data.processor import LiveDataCollector
from src.model.trainer import LiveHDPHMM
from live_visualize import LiveVisualizer

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
    # Parameters
    n_features = 3
    window_size = 100
    max_states = 20
    sample_interval = 1.0
    max_iterations = 1000
    model_path = "models/hdp_hmm.pth"
    
    # Create models directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info("Initializing components...")
    logger.info(f"Training on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Initialize components
    collector = LiveDataCollector(n_features, window_size, sample_interval)
    trainer = LiveHDPHMM(n_features, max_states, lr=0.01, model_path=model_path)
    visualizer = LiveVisualizer(n_features, window_size)
    
    # Try loading a pre-trained model
    logger.info("Checking for pre-trained model...")
    trainer.load_model()
    
    logger.info("Starting live processing...")
    start_time = time.time()
    window_count = 0
    try:
        for i in range(max_iterations):
            # Collect new window
            window_data = collector.collect_window()
            if window_data is not None:
                window_count += 1
                # Train and infer
                loss = trainer.update_model(window_data)
                states, trans_probs = trainer.infer(window_data)
                
                # Visualize
                visualizer.update_plot(window_data, states, trans_probs, loss, trainer.losses)
                
                # Save model periodically
                if window_count % 10 == 0:
                    trainer.save_model()
                    logger.info(f"Window {window_count}, Loss: {loss:.4f}")
                
            time.sleep(sample_interval)
    
    except KeyboardInterrupt:
        logger.info("Stopping live processing...")
        trainer.save_model()
        visualizer.close()
    
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        trainer.save_model()
        raise
    
    finally:
        logger.info(f"Saving final model...")
        trainer.save_model()
        logger.info(f"Processed {window_count} windows in {time.time() - start_time:.1f} seconds")
        logger.info("Live processing complete")

if __name__ == "__main__":
    main()