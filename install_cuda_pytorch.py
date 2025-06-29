#!/usr/bin/env python3
"""
CUDA and PyTorch GPU Installation Helper Script

This script helps install CUDA Toolkit and PyTorch with GPU support for RTX 2060.
"""

import subprocess
import sys
import os
import webbrowser
from pathlib import Path

def check_nvidia_gpu():
    """Check if NVIDIA GPU is detected."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA GPU not detected or nvidia-smi not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found. CUDA may not be installed.")
        return False

def check_cuda_installation():
    """Check if CUDA is installed."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA Toolkit detected!")
            print(result.stdout)
            return True
        else:
            print("❌ CUDA Toolkit not detected")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ CUDA Toolkit not installed")
        return False

def check_pytorch_cuda():
    """Check if PyTorch has CUDA support."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("❌ PyTorch CUDA support not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_cuda_instructions():
    """Provide instructions for installing CUDA."""
    print("\n" + "="*60)
    print("📋 CUDA INSTALLATION INSTRUCTIONS")
    print("="*60)
    print("1. Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
    print("2. Choose your system configuration:")
    print("   - Operating System: Windows")
    print("   - Architecture: x86_64")
    print("   - Version: 11 or 12 (recommended)")
    print("   - Installer Type: exe (local)")
    print("3. Download and run the installer")
    print("4. Follow the installation wizard")
    print("5. Restart your computer after installation")
    print("6. Run this script again to verify installation")
    print("="*60)
    
    # Open the CUDA download page
    response = input("\nWould you like to open the CUDA download page? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")

def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    print("\n" + "="*60)
    print("🔧 INSTALLING PYTORCH WITH CUDA SUPPORT")
    print("="*60)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment. Consider creating one first.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Uninstall current PyTorch
    print("Uninstalling current PyTorch...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'])
    
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA support...")
    
    # Try CUDA 12.1 first (latest stable)
    try:
        cmd = [
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio', 
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ]
        result = subprocess.run(cmd, check=True)
        print("✅ PyTorch with CUDA 12.1 installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  CUDA 12.1 installation failed, trying CUDA 11.8...")
        try:
            cmd = [
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio', 
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ]
            result = subprocess.run(cmd, check=True)
            print("✅ PyTorch with CUDA 11.8 installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyTorch with CUDA support")
            return False

def main():
    """Main installation process."""
    print("🚀 CUDA and PyTorch GPU Installation Helper")
    print("="*60)
    
    # Check current status
    print("\n📊 CURRENT STATUS:")
    gpu_detected = check_nvidia_gpu()
    cuda_installed = check_cuda_installation()
    pytorch_cuda = check_pytorch_cuda()
    
    print("\n" + "="*60)
    print("📋 RECOMMENDED ACTIONS:")
    print("="*60)
    
    if not gpu_detected:
        print("❌ NVIDIA GPU not detected. Please check your hardware.")
        return
    
    if not cuda_installed:
        print("1️⃣  Install CUDA Toolkit")
        install_cuda_instructions()
        return
    
    if not pytorch_cuda:
        print("2️⃣  Install PyTorch with CUDA support")
        response = input("Would you like to install PyTorch with CUDA support now? (y/n): ")
        if response.lower() == 'y':
            if install_pytorch_cuda():
                print("\n✅ Installation complete! Please restart your Python environment.")
                print("Run this script again to verify the installation.")
            else:
                print("\n❌ Installation failed. Please check the error messages above.")
        return
    
    if gpu_detected and cuda_installed and pytorch_cuda:
        print("🎉 Everything is set up correctly!")
        print("Your RTX 2060 is ready for GPU-accelerated training!")
        
        # Test GPU computation
        print("\n🧪 Testing GPU computation...")
        try:
            import torch
            device = torch.device('cuda')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("✅ GPU computation test successful!")
            print(f"Matrix multiplication result shape: {z.shape}")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")

if __name__ == "__main__":
    main() 