# PyInstaller runtime hook for pyopenms DLL loading
import os
import sys

def _add_dll_dir():
    # Find the directory containing pyopenms DLLs
    try:
        import pyopenms
        dll_dir = os.path.dirname(pyopenms.__file__)
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)
        else:
            os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
    except Exception as e:
        print(f"[pyi_rth_pyopenms] Could not set DLL directory: {e}", file=sys.stderr)

_add_dll_dir()
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
    
    # sys._MEIPASS is PyInstaller's temporary extraction directory
    try:
        exe_dir = sys._MEIPASS
        debug_print(f"[pyi_rth_pyopenms] Extraction directory: {exe_dir}")
        
        # Verify the extraction directory exists and is accessible
        if not os.path.exists(exe_dir):
            debug_print(f"[pyi_rth_pyopenms] ERROR: Extraction directory does not exist: {exe_dir}")
            # Try to wait a moment for extraction to complete
            import time
            time.sleep(0.5)
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
        # Exit early if we can't access the extraction directory
        sys.exit(1)
    
    # STEP 1: PREPEND exe_dir to PATH FIRST (before any file operations)
    # This ensures pyopenms DLLs are found before PyQt6's
    current_path = os.environ.get('PATH', '')
    
    # Remove any existing exe_dir from PATH to avoid duplicates
    path_parts = current_path.split(os.pathsep)
    filtered_parts = [p for p in path_parts if p != exe_dir]
    cleaned_path = os.pathsep.join(filtered_parts)
    
    # Prepend exe_dir to ensure our DLLs are found first
    os.environ['PATH'] = exe_dir + os.pathsep + cleaned_path
    debug_print(f"[pyi_rth_pyopenms] PATH updated (exe_dir prepended)")
    
    # STEP 2: Use Windows DLL search path API (Python 3.8+)
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(exe_dir)
            debug_print(f"[pyi_rth_pyopenms] Added DLL directory: {exe_dir}")
        except Exception as e:
            debug_print(f"[pyi_rth_pyopenms] WARNING: add_dll_directory() failed: {e}")
    
    # STEP 3: Handle PyQt6 Qt6 conflicts (more carefully)
    # Instead of deleting immediately, try to hide PyQt6 paths from PATH
    pyqt6_qt_dir = os.path.join(exe_dir, 'PyQt6', 'Qt6', 'bin')
    if os.path.exists(pyqt6_qt_dir):
        debug_print(f"[pyi_rth_pyopenms] Found PyQt6 Qt6 directory, attempting to mitigate conflicts")
        
        # First, try to remove PyQt6 paths from PATH
        pyqt6_parent = os.path.join(exe_dir, 'PyQt6')
        if pyqt6_parent in os.environ.get('PATH', ''):
            path_parts = os.environ['PATH'].split(os.pathsep)
            filtered_parts = [p for p in path_parts if pyqt6_parent not in p]
            os.environ['PATH'] = os.pathsep.join(filtered_parts)
            debug_print(f"[pyi_rth_pyopenms] Removed PyQt6 from PATH")
        
        # Only try to delete if we're confident it's safe (wait longer for extraction to complete)
        import time
        time.sleep(1.0)  # Give extraction more time to complete
        
        import shutil
        try:
            # Check if directory is still accessible
            if os.path.exists(pyqt6_qt_dir):
                shutil.rmtree(pyqt6_qt_dir)
                debug_print(f"[pyi_rth_pyopenms] Successfully removed PyQt6 Qt6/bin directory")
            else:
                debug_print(f"[pyi_rth_pyopenms] PyQt6 Qt6/bin directory no longer exists")
        except Exception as e:
            debug_print(f"[pyi_rth_pyopenms] WARNING: Could not remove PyQt6 Qt6/bin: {e}")
            # If deletion fails, at least we've removed it from PATH
    
    # STEP 4: Verify critical DLLs are present
    critical_dlls = ['OpenMS.dll', 'Qt6Core.dll', 'Qt6Network.dll']
    missing_dlls = []
    found_dlls = []
    
    for dll in critical_dlls:
        dll_path = os.path.join(exe_dir, dll)
        if os.path.exists(dll_path):
            found_dlls.append(dll)
        else:
            missing_dlls.append(dll)
    
    debug_print(f"[pyi_rth_pyopenms] Found {len(found_dlls)}/{len(critical_dlls)} critical DLLs")
    if missing_dlls:
        debug_print(f"[pyi_rth_pyopenms] WARNING: Missing: {', '.join(missing_dlls)}")
    
    # STEP 5: Check for Qt6 plugins directory
    qt_plugins_dir = os.path.join(exe_dir, 'Qt6', 'plugins')
    if os.path.exists(qt_plugins_dir):
        # Set Qt plugin path environment variable
        os.environ['QT_PLUGIN_PATH'] = qt_plugins_dir
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_dir
        debug_print(f"[pyi_rth_pyopenms] QT_PLUGIN_PATH set to: {qt_plugins_dir}")
    
    debug_print("[pyi_rth_pyopenms] Runtime hook completed successfully")
