#!/usr/bin/env python3
"""
CIFAR-10 Agent Setup Script

This script helps set up the CIFAR-10 classification agent environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def install_requirements():
    """Install required packages."""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing required packages"
    )

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ['checkpoints', 'logs', 'plots', 'data']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directories created successfully")
    return True

def run_tests():
    """Run the test suite."""
    return run_command(
        f"{sys.executable} test_cifar10_agent.py",
        "Running test suite"
    )

def main():
    """Main setup function."""
    print("🚀 CIFAR-10 Agent Setup")
    print("=" * 50)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Create Directories", create_directories),
        ("Install Requirements", install_requirements),
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        if step_func():
            success_count += 1
        else:
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please fix the error and run setup again.")
            return False
        print()
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run tests: python test_cifar10_agent.py")
    print("2. View dataset info: python cifar10_agent.py info --show-classes")
    print("3. Start training: python cifar10_agent.py train --architecture basic --epochs 10")
    print("\nFor help: python cifar10_agent.py --help")
    
    # Ask if user wants to run tests
    run_test = input("\nRun test suite now? (y/N): ").lower().strip()
    if run_test == 'y':
        print("\n" + "=" * 50)
        if run_tests():
            print("🎉 All tests passed! Your CIFAR-10 agent is ready to use.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
