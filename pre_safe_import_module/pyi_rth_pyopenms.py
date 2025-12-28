"""
PyInstaller runtime hook for pyopenms.

This hook runs when the frozen application starts, BEFORE any user code.
It sets up the Windows DLL search path so pyopenms can find its dependency DLLs.

CRITICAL PROBLEM: PyQt6 and pyopenms both bundle Qt6 DLLs. If both are in PATH,
Windows may load mismatched Qt6 versions causing symbol resolution failures.

SOLUTION: Ensure pyopenms's DLLs (especially Qt6) are loaded FIRST by modifying
PATH before any imports happen.
"""
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
    exe_dir = sys._MEIPASS
    debug_print(f"[pyi_rth_pyopenms] Extraction directory: {exe_dir}")
    
    # CRITICAL: Delete PyQt6's Qt6 DLLs to force use of pyopenms's Qt6
    # This is the ONLY reliable way to prevent DLL conflicts on Windows
    pyqt6_qt_dir = os.path.join(exe_dir, 'PyQt6', 'Qt6', 'bin')
    if os.path.exists(pyqt6_qt_dir):
        import shutil
        try:
            shutil.rmtree(pyqt6_qt_dir)
            debug_print(f"[pyi_rth_pyopenms] Removed PyQt6 Qt6/bin directory")
        except Exception as e:
            debug_print(f"[pyi_rth_pyopenms] WARNING: Could not remove PyQt6 Qt6/bin: {e}")
    
    # STEP 1: Verify critical DLLs are present (all should be in root now)
    critical_dlls = ['OpenMS.dll', 'Qt6Core.dll', 'Qt6Network.dll', 'msvcp140.dll', 'vcomp140.dll']
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
    
    # STEP 2: PREPEND exe_dir to PATH (ensures pyopenms DLLs are found FIRST)
    current_path = os.environ.get('PATH', '')
    
    # Remove any remaining PyQt6 paths from PATH (though Qt6/bin should be deleted)
    path_parts = current_path.split(os.pathsep)
    cleaned_parts = [p for p in path_parts if 'PyQt6' not in p]
    cleaned_path = os.pathsep.join(cleaned_parts)
    
    # Prepend exe_dir
    os.environ['PATH'] = exe_dir + os.pathsep + cleaned_path
    debug_print(f"[pyi_rth_pyopenms] PATH updated (exe_dir prepended, PyQt6 removed)")
    
    # STEP 3: Use Windows DLL search path API (Python 3.8+)
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(exe_dir)
            debug_print(f"[pyi_rth_pyopenms] Added DLL directory: {exe_dir}")
        except Exception as e:
            debug_print(f"[pyi_rth_pyopenms] WARNING: add_dll_directory() failed: {e}")
    
    # STEP 4: Check for Qt6 plugins directory (if collected by hook)
    qt_plugins_dir = os.path.join(exe_dir, 'Qt6', 'plugins')
    if os.path.exists(qt_plugins_dir):
        # Set Qt plugin path environment variable
        os.environ['QT_PLUGIN_PATH'] = qt_plugins_dir
        debug_print(f"[pyi_rth_pyopenms] QT_PLUGIN_PATH set to: {qt_plugins_dir}")
    
    # STEP 5: Force Qt to use our plugins (not PyQt6's)
    # This prevents Qt from loading mismatched plugins from PyQt6
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_dir if os.path.exists(qt_plugins_dir) else exe_dir
    
    debug_print("[pyi_rth_pyopenms] Runtime hook completed successfully")
