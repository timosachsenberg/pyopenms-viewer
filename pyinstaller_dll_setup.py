"""
PyInstaller DLL setup script for Windows.

This script sets up DLL directories in the Python environment before running PyInstaller.
It's needed because pyopenms C++ extension modules require DLLs from the share/ directory.

Usage:
    python pyinstaller_dll_setup.py && pyinstaller spec_file.spec
"""
import os
import sys
import site

def setup_dll_directories():
    """Add pyopenms share directory to DLL search path."""
    # Get site-packages directory
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if not site_packages:
        print("Warning: Could not locate site-packages directory")
        return
    
    # Locate share directory
    share_dir = os.path.join(site_packages, 'share')
    
    if not os.path.exists(share_dir):
        print(f"Warning: Share directory not found: {share_dir}")
        return
    
    # Add to DLL search path (Python 3.8+ on Windows)
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(share_dir)
            print(f"✓ Added DLL directory: {share_dir}")
        except Exception as e:
            print(f"Warning: Could not add DLL directory: {e}")
    
    # Also add to PATH as fallback
    os.environ['PATH'] = share_dir + os.pathsep + os.environ.get('PATH', '')
    print(f"✓ Added to PATH: {share_dir}")
    
    # Verify DLLs exist
    dll_count = len([f for f in os.listdir(share_dir) if f.endswith('.dll')])
    print(f"✓ Found {dll_count} DLL files in share directory")

if __name__ == '__main__':
    setup_dll_directories()
    print("\nDLL directories configured. Now run PyInstaller.")
