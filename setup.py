#!/usr/bin/env python3
"""
Setup script for Supply Chain Intelligence Demo
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed")

def main():
    print("🚚 Supply Chain Intelligence Demo Setup")
    print("=" * 50)
    
    if install_requirements():
        check_cuda()
        print("\n🎉 Setup complete!")
        print("\nTo run the application:")
        print("  streamlit run app.py")
    else:
        print("\n❌ Setup failed!")

if __name__ == "__main__":
    main()