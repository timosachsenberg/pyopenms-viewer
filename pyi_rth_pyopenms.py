# PyInstaller runtime hook for pyopenms.
#
# This hook runs when the frozen application starts, BEFORE any user code.
# It sets up the Windows DLL search path so pyopenms can find its dependency DLLs.
#
# CRITICAL PROBLEM: PyQt6 and pyopenms both bundle Qt6 DLLs. If both are in PATH,
# Windows may load mismatched Qt6 versions causing symbol resolution failures.
#
# SOLUTION: Ensure pyopenms's DLLs (especially Qt6) are loaded FIRST by modifying
# PATH before any imports happen.
import os
import sys

def debug_print(msg):
    """Print with explicit flush and ASCII-only characters for Windows console."""
    try:
        print(msg, flush=True)
        sys.stdout.flush()
    except Exception:
        # If even this fails, silently continue
        pass

if getattr(sys, 'frozen', False):
    debug_print("[pyi_rth_pyopenms] Runtime hook starting...")
    
    # Wait for PyInstaller to set up the extraction environment
    import time
    time.sleep(0.5)  # Give PyInstaller time to create the temp directory
    
    # sys._MEIPASS is PyInstaller's temporary extraction directory
    try:
        exe_dir = sys._MEIPASS
        debug_print(f"[pyi_rth_pyopenms] Extraction directory: {exe_dir}")
        
        # Verify the extraction directory exists and is accessible
        if not os.path.exists(exe_dir):
            debug_print(f"[pyi_rth_pyopenms] ERROR: Extraction directory does not exist: {exe_dir}")
            time.sleep(1.0)  # Wait longer
            if not os.path.exists(exe_dir):
                debug_print(f"[pyi_rth_pyopenms] ERROR: Extraction directory still does not exist after waiting")
                exe_dir = None
    except AttributeError:
        debug_print("[pyi_rth_pyopenms] ERROR: sys._MEIPASS not available")
        exe_dir = None
    except Exception as e:
        debug_print(f"[pyi_rth_pyopenms] ERROR: Failed to access extraction directory: {e}")
        exe_dir = None
    
    if exe_dir is None:
        debug_print("[pyi_rth_pyopenms] Runtime hook aborting - no extraction directory available")
        # Don't exit - let the app try to run anyway
        pass  # Continue with minimal setup
    
    # CRITICAL: Set up PATH to ensure pyopenms DLLs are found FIRST
    # This prevents Qt6 version conflicts between pyopenms and PyQt6
    current_path = os.environ.get('PATH', '')
    
    # Add pyopenms_dlls subdirectory to PATH (contains pyopenms DLLs)
    pyopenms_dlls_dir = os.path.join(exe_dir, 'pyopenms_dlls')
    if os.path.exists(pyopenms_dlls_dir):
        # Remove existing pyopenms_dlls from PATH to avoid duplicates
        path_parts = current_path.split(os.pathsep)
        filtered_parts = [p for p in path_parts if not p.endswith('pyopenms_dlls')]
        cleaned_path = os.pathsep.join(filtered_parts)
        
        # PREPEND pyopenms_dlls directory to ensure our DLLs are found first
        os.environ['PATH'] = pyopenms_dlls_dir + os.pathsep + exe_dir + os.pathsep + cleaned_path
        debug_print(f"[pyi_rth_pyopenms] PATH updated (pyopenms_dlls and exe_dir prepended)")
        
        # Use Windows DLL search path API
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(pyopenms_dlls_dir)
                os.add_dll_directory(exe_dir)
                debug_print(f"[pyi_rth_pyopenms] Added DLL directories: pyopenms_dlls and exe_dir")
            except Exception as e:
                debug_print(f"[pyi_rth_pyopenms] WARNING: add_dll_directory() failed: {e}")
    else:
        # Fallback: just prepend exe_dir
        path_parts = current_path.split(os.pathsep)
        filtered_parts = [p for p in path_parts if p != exe_dir]
        cleaned_path = os.pathsep.join(filtered_parts)
        os.environ['PATH'] = exe_dir + os.pathsep + cleaned_path
        debug_print(f"[pyi_rth_pyopenms] PATH updated (exe_dir prepended, pyopenms_dlls not found)")
        
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(exe_dir)
                debug_print(f"[pyi_rth_pyopenms] Added DLL directory: {exe_dir}")
            except Exception as e:
                debug_print(f"[pyi_rth_pyopenms] WARNING: add_dll_directory() failed: {e}")
    
    # Set Qt plugin path to our directory (if it exists)
    qt_plugins_dir = os.path.join(exe_dir, 'Qt6', 'plugins')
    if os.path.exists(qt_plugins_dir):
        os.environ['QT_PLUGIN_PATH'] = qt_plugins_dir
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_dir
        debug_print(f"[pyi_rth_pyopenms] QT_PLUGIN_PATH set to: {qt_plugins_dir}")
    
    debug_print("[pyi_rth_pyopenms] Runtime hook completed - PATH setup only")
