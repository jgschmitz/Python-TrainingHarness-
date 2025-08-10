CIFAR-10 CNN Model Architecture

This module contains the CNN model architecture optimized for CIFAR-10 image classification.
CIFAR-10 consists of 32x32 RGB images across 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional
import numpy as np


class CIFAR10Model:
    """
    CNN Model class for CIFAR-10 image classification.
    
    This model is designed specifically for CIFAR-10's characteristics:
    - Input: 32x32x3 RGB images
    - Output: 10 classes
    - Architecture: Multiple Conv2D blocks with BatchNorm, Dropout, and residual connections
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3), num_classes: int = 10):
        """
        Initialize the CIFAR-10 model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (10 for CIFAR-10)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def build_basic_cnn(self) -> keras.Model:
        """
        Build a basic CNN model for CIFAR-10.

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model
    
    def build_advanced_cnn(self) -> keras.Model:
        """
        Build an advanced CNN model with residual connections for CIFAR-10.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(32, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # First residual block
        x = self._residual_block(x, 32, stride=1)
        x = self._residual_block(x, 32, stride=1)
        
        # Second residual block with downsampling
        x = self._residual_block(x, 64, stride=2)
        x = self._residual_block(x, 64, stride=1)
        
        # Third residual block with downsampling
        x = self._residual_block(x, 128, stride=2)
        x = self._residual_block(x, 128, stride=1)
        
        # Fourth residual block with downsampling
        x = self._residual_block(x, 256, stride=2)
        x = self._residual_block(x, 256, stride=1)
        
        # Global average pooling and output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _residual_block(self, x, filters: int, stride: int = 1):
        """
        Create a residual block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            stride: Stride for convolution
            
        Returns:
            Output tensor
        """
        shortcut = x
        
        # First conv layer
        x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv layer
        x = layers.Conv2D(filters, 3, strides=1, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut and apply activation
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)
        
        return x
    
    def build_model(self, architecture: str = 'basic') -> keras.Model:
        """
        Build and compile the model.
        
        Args:
            architecture: Type of architecture ('basic' or 'advanced')
            
        Returns:
            Compiled Keras model
        """
        if architecture == 'basic':
            self.model = self.build_basic_cnn()
        elif architecture == 'advanced':
            self.model = self.build_advanced_cnn()
        else:
            raise ValueError("Architecture must be 'basic' or 'advanced'")
        
        return self.model
    
    def compile_model(self,
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'categorical_crossentropy',
                     metrics: list = None) -> None:
        """
        Compile the model with specified parameters.

        Args:
            optimizer: Optimizer to use
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def get_model_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        print(f"Model Architecture Summary:")
        print(f"Input Shape: {self.input_shape}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Class Names: {', '.join(self.class_names)}")
        print("\n" + "="*50)
        self.model.summary()
    
    def get_model_config(self) -> dict:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model configuration
        """
        if self.model is None:
            return {"error": "Model not built yet"}
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        return {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_layers": len(self.model.layers),
            "class_names": self.class_names
        }


def create_cifar10_model(architecture: str = 'basic', 
                        compile_model: bool = True,
                        learning_rate: float = 0.001) -> CIFAR10Model:
    """
    Factory function to create and optionally compile a CIFAR-10 model.
    
    Args:
        architecture: Model architecture ('basic' or 'advanced')
        compile_model: Whether to compile the model
        learning_rate: Learning rate for optimizer
        
    Returns:
        CIFAR10Model instance
    """
    cifar_model = CIFAR10Model()
    cifar_model.build_model(architecture=architecture)
    
    if compile_model:
        cifar_model.compile_model(learning_rate=learning_rate)
    
    return cifar_model


if __name__ == "__main__":
    # Example usage
    print("Creating CIFAR-10 CNN Models...")
    
    # Create basic model
    print("\n1. Basic CNN Model:")
    basic_model = create_cifar10_model(architecture='basic')
    basic_model.get_model_summary()
    
    print(f"\nBasic Model Config: {basic_model.get_model_config()}")
    
    # Create advanced model
    print("\n2. Advanced CNN Model with Residual Connections:")
    advanced_model = create_cifar10_model(architecture='advanced')
    advanced_model.get_model_summary()
    
    print(f"\nAdvanced Model Config: {advanced_model.get_model_config()}")
