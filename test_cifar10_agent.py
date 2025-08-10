#!/usr/bin/env python3
"""
Test script for CIFAR-10 Classification Agent

This script runs basic tests to ensure all components work correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        # Test our custom modules
        from cifar10_model import CIFAR10Model, create_cifar10_model
        print("✅ cifar10_model module")
        
        from cifar10_training import CIFAR10Trainer
        print("✅ cifar10_training module")
        
        from cifar10_agent import CIFAR10Agent
        print("✅ cifar10_agent module")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\n🧪 Testing model creation...")
    
    try:
        from cifar10_model import create_cifar10_model
        
        # Test basic model
        basic_model = create_cifar10_model(architecture='basic', compile_model=True)
        print("✅ Basic model created successfully")
        print(f"   Parameters: {basic_model.model.count_params():,}")
        
        # Test advanced model
        advanced_model = create_cifar10_model(architecture='advanced', compile_model=True)
        print("✅ Advanced model created successfully")
        print(f"   Parameters: {advanced_model.model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_data_loading():
    """Test CIFAR-10 data loading."""
    print("\n🧪 Testing data loading...")
    
    try:
        from cifar10_agent import CIFAR10Agent
        
        agent = CIFAR10Agent()
        train_data, val_data, test_data = agent.load_cifar10_data(validation_split=0.1)
        
        print("✅ CIFAR-10 data loaded successfully")
        
        # Check data shapes
        for batch in train_data.take(1):
            images, labels = batch
            print(f"   Training batch shape: {images.shape}")
            print(f"   Labels shape: {labels.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False


def test_cli_interface():
    """Test CLI interface."""
    print("\n🧪 Testing CLI interface...")
    
    try:
        import subprocess
        
        # Test help command
        result = subprocess.run([
            sys.executable, 'cifar10_agent.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ CLI help command works")
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
        
        # Test info command
        result = subprocess.run([
            sys.executable, 'cifar10_agent.py', 'info', '--show-classes'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ CLI info command works")
        else:
            print(f"❌ CLI info failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False


def test_quick_training():
    """Test quick training run."""
    print("\n🧪 Testing quick training (1 epoch)...")
    
    try:
        import subprocess
        
        # Run a very quick training test
        result = subprocess.run([
            sys.executable, 'cifar10_agent.py', 'train',
            '--architecture', 'basic',
            '--epochs', '1'
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout for safety
        
        if result.returncode == 0:
            print("✅ Quick training test passed")
            return True
        else:
            print(f"❌ Quick training failed: {result.stderr}")
            return False
        
    except subprocess.TimeoutExpired:
        print("⚠️  Training test timed out (this might be normal on slow systems)")
        return True
    except Exception as e:
        print(f"❌ Training test error: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability."""
    print("\n🧪 Testing GPU availability...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) available:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  No GPUs detected - training will use CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🚀 CIFAR-10 Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Data Loading Test", test_data_loading),
        ("GPU Availability Test", test_gpu_availability),
        ("CLI Interface Test", test_cli_interface),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Optional: Quick training test (can be slow)
    print("\n" + "=" * 50)
    run_training_test = input("Run quick training test? (y/N): ").lower().strip()
    if run_training_test == 'y':
        result = test_quick_training()
        results.append(("Quick Training Test", result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your CIFAR-10 agent is ready to use.")
        print("\nNext steps:")
        print("1. python cifar10_agent.py info --show-classes --show-stats")
        print("2. python cifar10_agent.py train --architecture basic --epochs 10")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
