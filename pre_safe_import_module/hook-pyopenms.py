# Pre-safe-import-module hook for pyopenms
# This runs in the PARENT process BEFORE PyInstaller attempts to import pyopenms.
# Any changes to os.environ ARE inherited by subprocesses.
# Critical for Windows where OpenMS.dll must be in PATH before importing.

from PyInstaller.utils.hooks import get_package_paths
import os
import sys

def pre_safe_import_module(api):
    """
    Pre-safe-import hook called by PyInstaller BEFORE importing pyopenms.
    
    This runs in the parent process. Environment variable changes (os.environ)
    are inherited by any subprocesses spawned after this point.
    """
    print("[HOOK PRE-SAFE-IMPORT] hook-pyopenms.py executing!", flush=True)
    
    if sys.platform == 'win32':
        try:
            pkg_base, pkg_dir = get_package_paths('pyopenms')
            pyopenms_dir = pkg_dir  # site-packages/pyopenms
            share_dir = os.path.join(pkg_base, 'share')
            
            print(f"[HOOK PRE-SAFE-IMPORT] Found pyopenms at: {pyopenms_dir}", flush=True)
            
            # Add directories to PATH environment variable
            # This WILL be inherited by subprocesses that PyInstaller spawns
            paths_to_add = []
            if os.path.exists(pyopenms_dir):
                paths_to_add.append(pyopenms_dir)
            if os.path.exists(share_dir):
                paths_to_add.append(share_dir)
            
            if paths_to_add:
                # Prepend to PATH so DLLs are found first
                current_path = os.environ.get('PATH', '')
                new_path = os.pathsep.join(paths_to_add)
                if current_path:
                    new_path += os.pathsep + current_path
                os.environ['PATH'] = new_path
                print(f"[HOOK PRE-SAFE-IMPORT] Modified PATH, added: {paths_to_add}", flush=True)
            
            # Also try add_dll_directory for the parent process
            # (won't affect subprocess, but good for consistency)
            if hasattr(os, 'add_dll_directory'):
                for path in paths_to_add:
                    try:
                        os.add_dll_directory(path)
                        print(f"[HOOK PRE-SAFE-IMPORT] Added DLL directory: {path}", flush=True)
                    except (OSError, FileNotFoundError) as e:
                        print(f"[HOOK PRE-SAFE-IMPORT] Failed to add DLL directory: {e}", flush=True)
                        
        except Exception as e:
            # Silently fail - don't break PyInstaller if this doesn't work
            print(f"[HOOK PRE-SAFE-IMPORT] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()


