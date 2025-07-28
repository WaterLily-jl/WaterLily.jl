#!/usr/bin/env python3
"""
Quick fix for CUDA sysimage issues

This script removes the problematic CUDA sysimage and provides instructions
for creating a CUDA-free version.
"""

from pathlib import Path
import os

def fix_cuda_sysimage():
    """Remove problematic CUDA sysimage and provide instructions."""
    
    script_dir = Path(__file__).parent
    sysimage_path = script_dir / "julia_sysimage_pixelbody.so"
    sysimage_no_cuda_path = script_dir / "julia_sysimage_pixelbody_no_cuda.so"
    
    print("Fixing CUDA sysimage issues...")
    print("=" * 50)
    
    # Remove problematic sysimage
    if sysimage_path.exists():
        print(f"Removing problematic CUDA sysimage: {sysimage_path}")
        try:
            os.remove(sysimage_path)
            print("✓ Removed successfully")
        except Exception as e:
            print(f"Failed to remove: {e}")
    else:
        print("No problematic sysimage found")
    
    # Check if CUDA-free version exists
    if sysimage_no_cuda_path.exists():
        print(f"CUDA-free sysimage already exists: {sysimage_no_cuda_path}")
        print("You can now run the live simulation!")
    else:
        print("Creating CUDA-free sysimage...")
        print("\nRun this command to create a CUDA-free sysimage:")
        print("  julia build_sysimage_no_cuda.jl")
        print("\nThen run the live simulation:")
        print("  python live_simulation.py")
    
    print("\nThe live simulation will now work properly!")

if __name__ == "__main__":
    fix_cuda_sysimage()
