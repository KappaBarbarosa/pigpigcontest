#!/usr/bin/env python3
"""
Quick start script for TAICA CVPDL 2025 Homework 1
This script provides easy commands to get started with the project.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False
    return True


def main():
    """Main function to run quick start commands."""
    print("🚀 TAICA CVPDL 2025 Homework 1 - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('train') or not os.path.exists('test'):
        print("❌ Error: Please run this script from the project root directory")
        print("Make sure 'train' and 'test' directories exist")
        return
    
    # Check if uv is available
    try:
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
        print("✅ uv is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv not found. Please install uv first:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        return
    
    # Commands to run
    commands = [
        ("uv run python example.py", "Running example script to test the environment"),
        ("uv run python main.py --mode inference --model_type custom --confidence_threshold 0.3", "Running inference with custom model"),
    ]
    
    print(f"\n📋 Running {len(commands)} commands...")
    
    success_count = 0
    for cmd, description in commands:
        if run_command(cmd, description):
            success_count += 1
    
    print(f"\n🎉 Quick start completed!")
    print(f"✅ {success_count}/{len(commands)} commands succeeded")
    
    if success_count == len(commands):
        print("\n🎯 Next steps:")
        print("1. Try training a model:")
        print("   uv run python main.py --mode train --model_type custom --epochs 5")
        print("\n2. Run inference on test images:")
        print("   uv run python main.py --mode inference --model_type custom")
        print("\n3. Explore the code in the 'src/' directory")
        print("\n4. Check the README.md for more detailed instructions")
    else:
        print("\n⚠️  Some commands failed. Please check the errors above.")


if __name__ == "__main__":
    main()
