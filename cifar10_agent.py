#!/usr/bin/env python3
"""
CIFAR-10 Classification Agent

A comprehensive command-line interface for training, evaluating, and using
CIFAR-10 image classification models.

Usage:
    python cifar10_agent.py train --architecture basic --epochs 50
    python cifar10_agent.py evaluate --model-path checkpoints/best_model.h5
    python cifar10_agent.py predict --image-path test_image.jpg --model-path checkpoints/best_model.h5
    python cifar10_agent.py info --show-classes
"""

import argparse
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import json
from datetime import datetime

# Import our custom modules
from cifar10_model import CIFAR10Model, create_cifar10_model
from cifar10_training import CIFAR10Trainer, setup_mixed_precision, get_training_strategy


class CIFAR10Agent:
    """
    Main CIFAR-10 Classification Agent with CLI interface.
    """
    
    def __init__(self):
        """Initialize the CIFAR-10 agent."""
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.model = None
        self.trainer = None
    
    def load_cifar10_data(self, validation_split: float = 0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and preprocess CIFAR-10 data.
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        print("Loading CIFAR-10 dataset...")
        
        # Load CIFAR-10 data
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # Split training data into train and validation
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]
        
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Test samples: {len(x_test)}")
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        
        # Configure datasets for performance
        BATCH_SIZE = 32
        AUTOTUNE = tf.data.AUTOTUNE
        
        train_dataset = (train_dataset
                        .shuffle(1000)
                        .batch(BATCH_SIZE)
                        .prefetch(AUTOTUNE))
        
        val_dataset = (val_dataset
                      .batch(BATCH_SIZE)
                      .prefetch(AUTOTUNE))
        
        test_dataset = (test_dataset
                       .batch(BATCH_SIZE)
                       .prefetch(AUTOTUNE))
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, args):
        """Train a CIFAR-10 model."""
        print(f"🚀 Starting CIFAR-10 model training...")
        print(f"Architecture: {args.architecture}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning Rate: {args.learning_rate}")
        
        # Setup training enhancements
        if args.mixed_precision:
            setup_mixed_precision()
        
        strategy = get_training_strategy()
        
        # Load data
        train_data, val_data, test_data = self.load_cifar10_data(args.validation_split)
        
        # Create model within strategy scope
        with strategy.scope():
            cifar_model = create_cifar10_model(
                architecture=args.architecture,
                learning_rate=args.learning_rate
            )
            self.model = cifar_model.model
        
        # Initialize trainer
        model_name = f"cifar10_{args.architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trainer = CIFAR10Trainer(self.model, model_name)
        
        # Configure callbacks
        callbacks_config = {
            'patience': args.patience,
            'min_lr': args.min_lr,
            'factor': args.lr_factor,
            'monitor': 'val_accuracy'
        }
        
        # Train the model
        history = self.trainer.train(
            train_data=train_data,
            validation_data=val_data,
            epochs=args.epochs,
            callbacks_config=callbacks_config
        )
        
        # Show final results
        best_metrics = self.trainer.get_best_metrics()
        print("\n🎯 Training Complete!")
        print("Best Metrics:")
        for metric, value in best_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_results = self.model.evaluate(test_data, verbose=0)

        # Handle different numbers of metrics
        if isinstance(test_results, list):
            test_loss = test_results[0]
            test_accuracy = test_results[1]
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # Print additional metrics if available
            if len(test_results) > 2:
                print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
        else:
            print(f"Test Loss: {test_results:.4f}")
        
        return history
    
    def evaluate_model(self, args):
        """Evaluate a trained model."""
        print(f"📊 Evaluating model: {args.model_path}")
        
        # Load model
        self.model = keras.models.load_model(args.model_path)
        
        # Load test data
        _, _, test_data = self.load_cifar10_data()
        
        # Evaluate
        results = self.model.evaluate(test_data, verbose=1)

        print(f"\nEvaluation Results:")
        if isinstance(results, list):
            print(f"Test Loss: {results[0]:.4f}")
            print(f"Test Accuracy: {results[1]:.4f}")

            if len(results) > 2:
                print(f"Test Top-3 Accuracy: {results[2]:.4f}")
        else:
            print(f"Test Loss: {results:.4f}")
        
        return results
    
    def predict_image(self, args):
        """Predict class for a single image."""
        print(f"🔍 Predicting image: {args.image_path}")
        
        # Load model
        self.model = keras.models.load_model(args.model_path)
        
        # Load and preprocess image
        image = keras.preprocessing.image.load_img(
            args.image_path, target_size=(32, 32)
        )
        image_array = keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print(f"\nPrediction Results:")
        print(f"Predicted Class: {self.class_names[predicted_class]}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        print(f"\nTop 3 Predictions:")
        for i, idx in enumerate(top_3_indices, 1):
            print(f"  {i}. {self.class_names[idx]}: {predictions[0][idx]:.4f}")
        
        return predicted_class, confidence
    
    def show_info(self, args):
        """Show information about CIFAR-10 dataset and classes."""
        print("📋 CIFAR-10 Dataset Information")
        print("=" * 50)
        
        if args.show_classes:
            print("Classes:")
            for i, class_name in enumerate(self.class_names):
                print(f"  {i}: {class_name}")
        
        if args.show_stats:
            print("\nDataset Statistics:")
            print("  Training samples: 50,000")
            print("  Test samples: 10,000")
            print("  Image size: 32x32x3 (RGB)")
            print("  Number of classes: 10")
            print("  Samples per class: 5,000 (training), 1,000 (test)")
        
        if args.show_sample:
            self._show_sample_images()
    
    def _show_sample_images(self):
        """Display sample images from each class."""
        print("\n🖼️  Loading sample images...")
        
        # Load a small sample of data
        (x_train, y_train), _ = keras.datasets.cifar10.load_data()
        
        # Create a plot with one image per class
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for class_idx in range(10):
            # Find first image of this class
            sample_idx = np.where(y_train == class_idx)[0][0]
            image = x_train[sample_idx]
            
            axes[class_idx].imshow(image)
            axes[class_idx].set_title(f"{class_idx}: {self.class_names[class_idx]}")
            axes[class_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Sample images saved as 'cifar10_samples.png'")


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Classification Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a basic model
  python cifar10_agent.py train --architecture basic --epochs 50
  
  # Train an advanced model with mixed precision
  python cifar10_agent.py train --architecture advanced --epochs 100 --mixed-precision
  
  # Evaluate a trained model
  python cifar10_agent.py evaluate --model-path checkpoints/best_model.h5
  
  # Predict a single image
  python cifar10_agent.py predict --image-path test.jpg --model-path checkpoints/best_model.h5
  
  # Show dataset information
  python cifar10_agent.py info --show-classes --show-stats --show-sample
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a CIFAR-10 model')
    train_parser.add_argument('--architecture', choices=['basic', 'advanced'], 
                             default='basic', help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--validation-split', type=float, default=0.1, 
                             help='Validation split ratio')
    train_parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    train_parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate')
    train_parser.add_argument('--lr-factor', type=float, default=0.5, help='LR reduction factor')
    train_parser.add_argument('--mixed-precision', action='store_true', 
                             help='Enable mixed precision training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', required=True, help='Path to trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict image class')
    predict_parser.add_argument('--image-path', required=True, help='Path to image file')
    predict_parser.add_argument('--model-path', required=True, help='Path to trained model')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset information')
    info_parser.add_argument('--show-classes', action='store_true', help='Show class names')
    info_parser.add_argument('--show-stats', action='store_true', help='Show dataset statistics')
    info_parser.add_argument('--show-sample', action='store_true', help='Show sample images')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize agent
    agent = CIFAR10Agent()
    
    try:
        if args.command == 'train':
            agent.train_model(args)
        elif args.command == 'evaluate':
            agent.evaluate_model(args)
        elif args.command == 'predict':
            agent.predict_image(args)
        elif args.command == 'info':
            agent.show_info(args)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
