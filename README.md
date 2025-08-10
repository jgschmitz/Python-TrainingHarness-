# 🖼️ CIFAR-10 Classification Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive TensorFlow/Keras-based agent for CIFAR-10 image classification with advanced training features and CLI interface.

## ✨ Features

- 🏗️ **Multiple CNN Architectures**: Basic and Advanced models with residual connections
- 🚀 **Advanced Training Pipeline**: Mixed precision, multi-GPU support, comprehensive callbacks
- 📊 **Complete Monitoring**: TensorBoard integration, automatic plotting, CSV logging
- 🎯 **Easy CLI Interface**: Train, evaluate, and predict with simple commands
- 🧪 **Built-in Testing**: Comprehensive test suite to verify everything works
- 📚 **Extensive Documentation**: Complete usage guide and examples

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python cifar10_agent.py info --show-classes
```

### Basic Usage

#### 1. Train a Model
```bash
# Train basic CNN model
python cifar10_agent.py train --architecture basic --epochs 50

# Train advanced model with mixed precision
python cifar10_agent.py train --architecture advanced --epochs 100 --mixed-precision
```

#### 2. Evaluate a Model
```bash
python cifar10_agent.py evaluate --model-path checkpoints/best_model.h5
```

#### 3. Predict Single Image
```bash
python cifar10_agent.py predict --image-path test_image.jpg --model-path checkpoints/best_model.h5
```

#### 4. Dataset Information
```bash
python cifar10_agent.py info --show-classes --show-stats --show-sample
```

## 📊 Key Differences from MNIST

| Aspect | MNIST | CIFAR-10 |
|--------|-------|----------|
| **Image Size** | 28×28×1 (grayscale) | 32×32×3 (RGB color) |
| **Classes** | 10 digits (0-9) | 10 objects (airplane, car, etc.) |
| **Complexity** | Simple handwritten digits | Complex real-world objects |
| **Training Time** | Minutes | Hours |
| **Model Architecture** | Simple CNN | Deeper CNN with residual connections |
| **Data Augmentation** | Minimal | Extensive (rotation, flip, zoom, etc.) |
| **Accuracy Target** | >99% achievable | ~95% is excellent |

## 🏗️ Architecture Options

### Basic CNN
- 3 Convolutional blocks
- BatchNormalization + Dropout
- ~500K parameters
- Good for quick experiments

### Advanced CNN
- Residual connections
- Global Average Pooling
- ~1M+ parameters
- Better accuracy, longer training

## 🎯 CIFAR-10 Classes

0. **airplane** ✈️
1. **automobile** 🚗
2. **bird** 🐦
3. **cat** 🐱
4. **deer** 🦌
5. **dog** 🐕
6. **frog** 🐸
7. **horse** 🐎
8. **ship** 🚢
9. **truck** 🚛

## 🔧 Advanced Features

### Training Enhancements
- **Mixed Precision**: Faster training on modern GPUs
- **Multi-GPU Support**: Distributed training
- **Learning Rate Scheduling**: Cosine annealing + warmup
- **Data Augmentation**: Comprehensive image transformations
- **Early Stopping**: Prevent overfitting

### Monitoring & Logging
- **TensorBoard**: Real-time training visualization
- **Automatic Plotting**: Training history graphs
- **Model Checkpointing**: Save best models automatically
- **CSV Logging**: Export training metrics

### Callbacks
- Early stopping with patience
- Learning rate reduction on plateau
- Model checkpointing
- Custom progress reporting
- TensorBoard logging

## 📈 Expected Performance

### Basic Model
- **Training Time**: ~30-60 minutes (GPU)
- **Test Accuracy**: 75-85%
- **Parameters**: ~500K

### Advanced Model
- **Training Time**: 1-3 hours (GPU)
- **Test Accuracy**: 85-95%
- **Parameters**: ~1M+

## 🛠️ Command Reference

### Training Options
```bash
python cifar10_agent.py train [OPTIONS]

Options:
  --architecture [basic|advanced]  Model architecture (default: basic)
  --epochs INTEGER                 Number of epochs (default: 50)
  --learning-rate FLOAT           Learning rate (default: 0.001)
  --validation-split FLOAT        Validation split (default: 0.1)
  --patience INTEGER              Early stopping patience (default: 10)
  --mixed-precision               Enable mixed precision training
```

### Evaluation Options
```bash
python cifar10_agent.py evaluate [OPTIONS]

Options:
  --model-path TEXT  Path to trained model [required]
```

### Prediction Options
```bash
python cifar10_agent.py predict [OPTIONS]

Options:
  --image-path TEXT  Path to image file [required]
  --model-path TEXT  Path to trained model [required]
```

### Info Options
```bash
python cifar10_agent.py info [OPTIONS]

Options:
  --show-classes    Show class names
  --show-stats      Show dataset statistics
  --show-sample     Show sample images
```

## 📁 File Structure

```
cifar10_agent/
├── cifar10_agent.py          # Main CLI interface
├── cifar10_model.py          # Model architectures
├── cifar10_training.py       # Training pipeline
├── setup.py                  # Setup script
├── test_cifar10_agent.py     # Test suite
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── checkpoints/             # Saved models
├── logs/                    # Training logs
└── plots/                   # Training visualizations
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Reduce batch size in training code
   - Enable mixed precision: `--mixed-precision`

2. **Slow Training**
   - Ensure GPU is available: `nvidia-smi`
   - Use mixed precision training
   - Consider reducing model size

3. **Low Accuracy**
   - Train for more epochs
   - Use advanced architecture
   - Check data preprocessing

### Performance Tips

- **Use GPU**: Significantly faster than CPU
- **Mixed Precision**: 1.5-2x speedup on modern GPUs
- **Batch Size**: Larger batches (32-128) often work better
- **Learning Rate**: Start with 0.001, adjust based on results

## 🎓 Usage Examples

### Complete Training Workflow
```bash
# 1. Check dataset info
python cifar10_agent.py info --show-classes --show-stats

# 2. Train model
python cifar10_agent.py train --architecture advanced --epochs 100 --mixed-precision

# 3. Evaluate model
python cifar10_agent.py evaluate --model-path checkpoints/best_model_*.h5

# 4. Test prediction
python cifar10_agent.py predict --image-path sample.jpg --model-path checkpoints/best_model_*.h5
```

### Quick Test Run
```bash
# Fast training for testing
python cifar10_agent.py train --architecture basic --epochs 5
```

## 📚 Next Steps

1. **Experiment with hyperparameters**
2. **Try different architectures**
3. **Implement custom data augmentation**
4. **Add ensemble methods**
5. **Deploy model for inference**

---

**Happy Training! 🎉**
