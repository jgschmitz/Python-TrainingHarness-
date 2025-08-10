"""
CIFAR-10 Training Pipeline with Callbacks

This module implements a comprehensive training pipeline for CIFAR-10 classification
with various callbacks for optimal training performance and monitoring.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import numpy as np
import os
import datetime
from typing import Dict, List, Optional, Tuple
import json
import matplotlib.pyplot as plt


class CIFAR10Trainer:
    """
    Training pipeline for CIFAR-10 classification with advanced callbacks and monitoring.
    """
    
    def __init__(self, model: keras.Model, model_name: str = "cifar10_model"):
        """
        Initialize the trainer.
        
        Args:
            model: Compiled Keras model
            model_name: Name for saving models and logs
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        self.best_model_path = None
        
        # Create directories for saving
        self.checkpoint_dir = f"checkpoints/{model_name}"
        self.logs_dir = f"logs/{model_name}"
        self.plots_dir = f"plots/{model_name}"
        
        for directory in [self.checkpoint_dir, self.logs_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def create_callbacks(self, 
                        patience: int = 10,
                        min_lr: float = 1e-7,
                        factor: float = 0.5,
                        monitor: str = 'val_accuracy',
                        save_best_only: bool = True) -> List[callbacks.Callback]:
        """
        Create comprehensive callbacks for training.
        
        Args:
            patience: Patience for early stopping and LR reduction
            min_lr: Minimum learning rate
            factor: Factor for learning rate reduction
            monitor: Metric to monitor
            save_best_only: Whether to save only the best model
            
        Returns:
            List of Keras callbacks
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callback_list = [
            # Early Stopping
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='max' if 'accuracy' in monitor else 'min'
            ),
            
            # Model Checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, f'best_model_{timestamp}.h5'),
                monitor=monitor,
                save_best_only=save_best_only,
                save_weights_only=False,
                verbose=1,
                mode='max' if 'accuracy' in monitor else 'min'
            ),
            
            # Learning Rate Reduction
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=factor,
                patience=patience//2,
                min_lr=min_lr,
                verbose=1,
                mode='max' if 'accuracy' in monitor else 'min'
            ),
            
            # TensorBoard Logging
            callbacks.TensorBoard(
                log_dir=os.path.join(self.logs_dir, timestamp),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            ),
            
            # CSV Logger
            callbacks.CSVLogger(
                filename=os.path.join(self.logs_dir, f'training_log_{timestamp}.csv'),
                separator=',',
                append=False
            ),
            
            # Learning Rate Scheduler (Cosine Annealing)
            callbacks.LearningRateScheduler(
                schedule=self._cosine_annealing_schedule,
                verbose=1
            ),
            
            # Custom Progress Callback
            ProgressCallback()
        ]
        
        return callback_list
    
    def _cosine_annealing_schedule(self, epoch: int, lr: float) -> float:
        """
        Cosine annealing learning rate schedule.
        
        Args:
            epoch: Current epoch
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        if epoch < 10:
            return lr
        
        # Cosine annealing after warmup
        max_lr = 0.001
        min_lr = 1e-6
        cycle_length = 50
        
        cycle_progress = (epoch - 10) % cycle_length / cycle_length
        new_lr = min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * cycle_progress)) / 2
        
        return new_lr
    
    def train(self,
              train_data: tf.data.Dataset,
              validation_data: tf.data.Dataset,
              epochs: int = 100,
              callbacks_config: Optional[Dict] = None) -> keras.callbacks.History:
        """
        Train the model with comprehensive monitoring.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of epochs to train
            callbacks_config: Configuration for callbacks
            
        Returns:
            Training history
        """
        if callbacks_config is None:
            callbacks_config = {}
        
        # Create callbacks
        training_callbacks = self.create_callbacks(**callbacks_config)
        
        print(f"Starting training for {self.model_name}")
        print(f"Training for {epochs} epochs")
        print(f"Model parameters: {self.model.count_params():,}")
        
        # Train the model
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=training_callbacks,
            verbose=1
        )
        
        # Save training history
        self._save_training_history()
        
        # Plot training results
        self._plot_training_history()
        
        return self.history
    
    def _save_training_history(self) -> None:
        """Save training history to JSON file."""
        if self.history is None:
            return
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(self.logs_dir, f'history_{timestamp}.json')
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
    
    def _plot_training_history(self) -> None:
        """Plot and save training history."""
        if self.history is None:
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation loss
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation accuracy
        ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate
        if 'lr' in history:
            ax3.plot(epochs, history['lr'], 'g-', label='Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True)
        
        # Plot top-3 accuracy if available
        if 'top_3_accuracy' in history:
            ax4.plot(epochs, history['top_3_accuracy'], 'b-', label='Training Top-3 Accuracy')
            ax4.plot(epochs, history['val_top_3_accuracy'], 'r-', label='Validation Top-3 Accuracy')
            ax4.set_title('Top-3 Accuracy')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('Top-3 Accuracy')
            ax4.legend()
            ax4.grid(True)
        else:
            # If no top-3 accuracy, show loss again with different scale
            ax4.plot(epochs, history['loss'], 'b-', label='Training Loss')
            ax4.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax4.set_title('Loss (Alternative View)')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('Loss')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.plots_dir, f'training_history_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {plot_path}")
    
    def get_best_metrics(self) -> Dict:
        """
        Get the best metrics from training history.
        
        Returns:
            Dictionary with best metrics
        """
        if self.history is None:
            return {}
        
        history = self.history.history
        
        best_metrics = {
            'best_train_accuracy': max(history['accuracy']),
            'best_val_accuracy': max(history['val_accuracy']),
            'best_train_loss': min(history['loss']),
            'best_val_loss': min(history['val_loss']),
            'total_epochs': len(history['loss'])
        }
        
        if 'top_3_accuracy' in history:
            best_metrics['best_train_top3_accuracy'] = max(history['top_3_accuracy'])
            best_metrics['best_val_top3_accuracy'] = max(history['val_top_3_accuracy'])
        
        return best_metrics


class ProgressCallback(callbacks.Callback):
    """Custom callback for detailed progress reporting."""
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n--- Epoch {epoch + 1} ---")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        print(f"Epoch {epoch + 1} Results:")
        for metric, value in logs.items():
            print(f"  {metric}: {value:.4f}")
        
        if 'lr' in logs:
            print(f"  Learning Rate: {logs['lr']:.2e}")


def create_data_augmentation() -> keras.Sequential:
    """
    Create data augmentation pipeline for CIFAR-10.
    
    Returns:
        Data augmentation model
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomBrightness(0.1)
    ])


class MixupCallback(callbacks.Callback):
    """Mixup data augmentation callback."""

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def on_batch_begin(self, batch, logs=None):
        # Mixup would be implemented here for advanced training
        pass


class WarmupCallback(callbacks.Callback):
    """Learning rate warmup callback."""

    def __init__(self, warmup_epochs=5, base_lr=0.001):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            print(f"Warmup LR: {lr:.6f}")


def setup_mixed_precision():
    """Setup mixed precision training for faster training on modern GPUs."""
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision training enabled")


def get_training_strategy():
    """Get distributed training strategy if multiple GPUs available."""
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
        return strategy
    else:
        print("Using default strategy (single GPU/CPU)")
        return tf.distribute.get_strategy()


if __name__ == "__main__":
    # Example usage
    print("CIFAR-10 Training Pipeline Example")

    # This would typically be used with actual data and model
    print("To use this trainer:")
    print("1. Load your CIFAR-10 data")
    print("2. Create your model using cifar10_model.py")
    print("3. Initialize CIFAR10Trainer with your model")
    print("4. Call trainer.train() with your data")

    # Example configuration
    example_config = {
        'patience': 15,
        'min_lr': 1e-7,
        'factor': 0.5,
        'monitor': 'val_accuracy'
    }

    print(f"\nExample callback configuration: {example_config}")

    # Show available training enhancements
    print("\nAvailable training enhancements:")
    print("- Mixed precision training")
    print("- Multi-GPU distributed training")
    print("- Learning rate warmup")
    print("- Mixup data augmentation")
    print("- Comprehensive callbacks and monitoring")
