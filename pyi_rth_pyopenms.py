# PyInstaller runtime hook for pyopenms.
#
# This hook runs when the frozen application starts, BEFORE any user code.
# It sets up the Windows DLL search path so pyopenms can find its dependency DLLs.
#
# CRITICAL PROBLEM: PyQt6 and pyopenms both bundle Qt6 DLLs. If both are in PATH,
# Windows may load mismatched Qt6 versions causing symbol resolution failures.
#
# SOLUTION: Pre-load pyopenms DLLs BEFORE any other imports to ensure correct versions.
#
# TROUBLESHOOTING: If app fails on second run:
# 1. Check Task Manager for running pyopenms-viewer.exe processes
# 2. Delete %TEMP%\_MEI* directories
# 3. Temporarily disable antivirus and try again
import ctypes
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


if getattr(sys, "frozen", False) and sys.platform == "win32":
    debug_print("[pyi_rth_pyopenms] Runtime hook starting...")

    # sys._MEIPASS is PyInstaller's temporary extraction directory
    exe_dir = None
    try:
        exe_dir = sys._MEIPASS
        debug_print(f"[pyi_rth_pyopenms] Extraction directory: {exe_dir}")

        # Verify the extraction directory exists and is accessible
        if not os.path.exists(exe_dir):
            debug_print(f"[pyi_rth_pyopenms] ERROR: Extraction directory does not exist: {exe_dir}")
            exe_dir = None
        elif not os.access(exe_dir, os.R_OK):
            debug_print(f"[pyi_rth_pyopenms] ERROR: Extraction directory not readable: {exe_dir}")
            exe_dir = None
    except AttributeError:
        debug_print("[pyi_rth_pyopenms] ERROR: sys._MEIPASS not available")
        exe_dir = None
    except Exception as e:
        debug_print(f"[pyi_rth_pyopenms] ERROR: Failed to access extraction directory: {e}")
        exe_dir = None

    if exe_dir is None:
        debug_print("[pyi_rth_pyopenms] Runtime hook aborting - no extraction directory available")
    else:
        # CRITICAL: Set up PATH to ensure pyopenms DLLs are found FIRST
        # This prevents Qt6 version conflicts between pyopenms and PyQt6
        current_path = os.environ.get("PATH", "")

        # Add pyopenms_dlls subdirectory to PATH (contains pyopenms DLLs)
        pyopenms_dlls_dir = os.path.join(exe_dir, "pyopenms_dlls")
        pyopenms_pkg_dir = os.path.join(exe_dir, "pyopenms")

        # Collect all DLL directories we need
        dll_dirs = []

        if os.path.exists(pyopenms_dlls_dir):
            dll_dirs.append(pyopenms_dlls_dir)
            # List contents for debugging
            try:
                dlls = os.listdir(pyopenms_dlls_dir)
                debug_print(f"[pyi_rth_pyopenms] Found {len(dlls)} files in pyopenms_dlls/")
                for dll in dlls:
                    debug_print(f"[pyi_rth_pyopenms]   - {dll}")
            except Exception as e:
                debug_print(f"[pyi_rth_pyopenms] WARNING: Could not list pyopenms_dlls: {e}")
        else:
            debug_print("[pyi_rth_pyopenms] WARNING: pyopenms_dlls directory not found!")

        if os.path.exists(pyopenms_pkg_dir):
            dll_dirs.append(pyopenms_pkg_dir)
            debug_print("[pyi_rth_pyopenms] Found pyopenms package directory")

        dll_dirs.append(exe_dir)

        # Remove existing references from PATH to avoid duplicates
        path_parts = current_path.split(os.pathsep)
        filtered_parts = [p for p in path_parts if p and p not in dll_dirs]

        # PREPEND our directories to PATH
        new_path = os.pathsep.join(dll_dirs + filtered_parts)
        os.environ["PATH"] = new_path
        debug_print(f"[pyi_rth_pyopenms] PATH updated with {len(dll_dirs)} directories prepended")

        # Use Windows DLL search path API (Windows 10+)
        # CRITICAL: Call add_dll_directory for EACH directory
        if hasattr(os, "add_dll_directory"):
            for dll_dir in dll_dirs:
                try:
                    os.add_dll_directory(dll_dir)
                    debug_print(f"[pyi_rth_pyopenms] add_dll_directory: {dll_dir}")
                except Exception as e:
                    debug_print(f"[pyi_rth_pyopenms] WARNING: add_dll_directory failed for {dll_dir}: {e}")

        # CRITICAL: Pre-load the OpenMS DLLs in the correct order BEFORE Python tries to import
        # This ensures the correct versions are loaded from our directory
        if os.path.exists(pyopenms_dlls_dir):
            # Load order matters! Dependencies must be loaded before dependents
            dll_load_order = [
                # MSVC runtime (required by everything)
                "vcruntime140.dll",
                "vcruntime140_1.dll",
                "msvcp140.dll",
                "msvcp140_1.dll",
                "msvcp140_2.dll",
                "msvcp140_atomic_wait.dll",
                "msvcp140_codecvt_ids.dll",
                "concrt140.dll",
                "vcomp140.dll",
                # Other dependencies
                "zlib.dll",
                # Qt6 (pyopenms's version, not PyQt6's)
                "Qt6Core.dll",
                "Qt6Network.dll",
                # OpenMS core libraries
                "OpenSwathAlgo.dll",
                "OpenMS.dll",
            ]

            debug_print(f"[pyi_rth_pyopenms] Pre-loading DLLs from {pyopenms_dlls_dir}")
            loaded_count = 0
            for dll_name in dll_load_order:
                dll_path = os.path.join(pyopenms_dlls_dir, dll_name)
                if os.path.exists(dll_path):
                    try:
                        # Use LOAD_WITH_ALTERED_SEARCH_PATH to load from the DLL's directory
                        ctypes.WinDLL(dll_path)
                        loaded_count += 1
                        debug_print(f"[pyi_rth_pyopenms]   Loaded: {dll_name}")
                    except Exception as e:
                        debug_print(f"[pyi_rth_pyopenms]   FAILED to load {dll_name}: {e}")
                else:
                    debug_print(f"[pyi_rth_pyopenms]   Not found: {dll_name}")

            debug_print(f"[pyi_rth_pyopenms] Pre-loaded {loaded_count} DLLs")

        # Set Qt plugin path to our directory (if it exists)
        qt_plugins_dir = os.path.join(exe_dir, "Qt6", "plugins")
        if os.path.exists(qt_plugins_dir):
            os.environ["QT_PLUGIN_PATH"] = qt_plugins_dir
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugins_dir
            debug_print(f"[pyi_rth_pyopenms] QT_PLUGIN_PATH set to: {qt_plugins_dir}")

        # CRITICAL: Set OPENMS_DATA_PATH so pyopenms can find its CV/data files
        # Without this, MzMLFile().load() may fail CV validation or error on
        # the first mzML load because it cannot locate psi-ms.obo and similar files.
        openms_share_dir = os.path.join(exe_dir, "share", "OpenMS")
        if os.path.exists(openms_share_dir):
            os.environ["OPENMS_DATA_PATH"] = openms_share_dir
            debug_print(f"[pyi_rth_pyopenms] OPENMS_DATA_PATH set to: {openms_share_dir}")
        else:
            # Fallback: point at the share/ root and let OpenMS search sub-dirs
            share_dir = os.path.join(exe_dir, "share")
            if os.path.exists(share_dir):
                os.environ["OPENMS_DATA_PATH"] = share_dir
                debug_print(f"[pyi_rth_pyopenms] OPENMS_DATA_PATH set to share/ root: {share_dir}")
            else:
                debug_print("[pyi_rth_pyopenms] WARNING: share/OpenMS not found, OPENMS_DATA_PATH not set")

        debug_print("[pyi_rth_pyopenms] Runtime hook completed successfully")

if getattr(sys, "frozen", False) and sys.platform == "darwin":
    exe_dir = getattr(sys, "_MEIPASS", None)
    if exe_dir:
        # Set OPENMS_DATA_PATH so MzMLFile().load() can find CV/data files (psi-ms.obo etc.)
        openms_share_dir = os.path.join(exe_dir, "share", "OpenMS")
        if os.path.exists(openms_share_dir):
            os.environ["OPENMS_DATA_PATH"] = openms_share_dir
        else:
            share_dir = os.path.join(exe_dir, "share")
            if os.path.exists(share_dir):
                os.environ["OPENMS_DATA_PATH"] = share_dir

        # Help the dynamic linker find bundled dylibs from pyopenms_dlls/
        pyopenms_dlls_dir = os.path.join(exe_dir, "pyopenms_dlls")
        if os.path.exists(pyopenms_dlls_dir):
            existing = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = pyopenms_dlls_dir + (":" + existing if existing else "")
