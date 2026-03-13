#!/usr/bin/env python3
"""Simple script to run the quantization-aware training demo."""

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'streamlit', 'matplotlib', 
        'plotly', 'numpy', 'pandas', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def run_demo():
    """Run the Streamlit demo."""
    if not check_requirements():
        return False
    
    print("Starting Quantization-Aware Training Demo...")
    print("The demo will open in your web browser.")
    print("Press Ctrl+C to stop the demo.")
    
    try:
        # Run Streamlit demo
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "demo.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
        return False
    
    return True


def run_training():
    """Run the training script."""
    if not check_requirements():
        return False
    
    print("Starting Quantization-Aware Training...")
    
    try:
        subprocess.run([sys.executable, "train.py"])
    except Exception as e:
        print(f"Error running training: {e}")
        return False
    
    return True


def main():
    """Main function."""
    print("Quantization-Aware Training")
    print("=" * 40)
    print("1. Run Interactive Demo")
    print("2. Run Training")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == "1":
            run_demo()
            break
        elif choice == "2":
            run_training()
            break
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
