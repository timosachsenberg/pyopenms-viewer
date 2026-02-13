# PyInstaller hook for pyopenms
# Collect the entire package (dylibs, data files, hidden imports) WITHOUT importing it.
# This avoids build-time DLL loading failures on Windows.

import os

from PyInstaller.utils.hooks import get_package_paths

# CRITICAL: DO NOT use collect_all() as it may try to import pyopenms
# which fails on Windows due to DLL dependencies during build.
# Instead, manually collect files without importing.

datas = []
binaries = []
hiddenimports = []

# Find pyopenms installation directory without importing it
try:
    pkg_base, pkg_dir = get_package_paths("pyopenms")

    # Collect ALL files from pyopenms directory WITHOUT importing
    if os.path.exists(pkg_dir):
        print(f"hook-pyopenms: Collecting from {pkg_dir} (no import)")

        # Collect all Python files as datas to preserve package structure
        for root, _dirs, files in os.walk(pkg_dir):
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(root, pkg_dir)
                dest_dir = os.path.join("pyopenms", rel_path) if rel_path != "." else "pyopenms"

                if file.endswith(".py"):
                    # Python source files go to pyopenms directory
                    datas.append((src, dest_dir))
                elif file.endswith((".pyd", ".so")):
                    # Python extension modules (.pyd on Windows, .so on Linux)
                    # These MUST be in the pyopenms directory for Python to import them
                    binaries.append((src, dest_dir))
                    print(f"hook-pyopenms: Found extension module: {file} -> {dest_dir}/")
                elif file.endswith((".dll", ".dylib")):
                    # DLLs go to a subdirectory to avoid conflicts with PyQt6's Qt6 DLLs
                    # The runtime hook will add this directory to PATH
                    binaries.append((src, "pyopenms_dlls"))
                    print(f"hook-pyopenms: Found DLL: {file} -> pyopenms_dlls/")
                elif file.endswith((".pyi", ".json", ".xml", ".txt", ".dat")):
                    # Data files preserve structure
                    datas.append((src, dest_dir))

        # CRITICAL: Manually collect Qt6 plugins that pyopenms needs
        # These are in pyopenms's Qt6 installation, not PyQt6's
        qt_plugins_dir = os.path.join(pkg_dir, "Qt6", "plugins")
        if os.path.exists(qt_plugins_dir):
            print(f"hook-pyopenms: Collecting Qt6 plugins from {qt_plugins_dir}")
            for root, _dirs, files in os.walk(qt_plugins_dir):
                for file in files:
                    if file.endswith(".dll"):
                        src = os.path.join(root, file)
                        # Preserve plugins directory structure: Qt6/plugins/platforms/qwindows.dll
                        rel_path = os.path.relpath(root, pkg_dir)
                        binaries.append((src, rel_path))
                        print(f"hook-pyopenms: Found Qt6 plugin: {file}")

    # Also collect share/ directory if it exists (OpenMS THIRDPARTY libs)
    share_dir = os.path.join(pkg_base, "share")
    if os.path.exists(share_dir):
        print(f"hook-pyopenms: Collecting from share directory: {share_dir}")
        for root, _dirs, files in os.walk(share_dir):
            for file in files:
                src = os.path.join(root, file)
                if file.endswith((".dll", ".so", ".dylib")):
                    # Share DLLs go to pyopenms_dlls subdirectory to avoid conflicts
                    binaries.append((src, "pyopenms_dlls"))
                    print(f"hook-pyopenms: Found share DLL: {file} -> pyopenms_dlls/")
                else:
                    # Preserve directory structure for data files
                    rel_path = os.path.relpath(root, pkg_base)
                    datas.append((src, rel_path))

    # Discover hidden imports by scanning __init__.py without importing
    # This avoids the import-time DLL failure
    init_file = os.path.join(pkg_dir, "__init__.py")
    if os.path.exists(init_file):
        with open(init_file) as f:
            content = f.read()
            # Look for common import patterns
            if "_pyopenms" in content:
                hiddenimports.append("pyopenms._pyopenms_1")
            if "version" in content:
                hiddenimports.append("pyopenms.version")
            if "Constants" in content:
                hiddenimports.append("pyopenms.Constants")

    dll_count = len([b for b in binaries if b[0].endswith(".dll")])
    pyd_count = len([b for b in binaries if b[0].endswith(".pyd")])
    print(f"hook-pyopenms: Collected {dll_count} DLLs and {pyd_count} extension modules")
    print(f"hook-pyopenms: Total binaries: {len(binaries)}, datas: {len(datas)}")


except Exception as e:
    # If we can't locate pyopenms, log it but don't fail
    print(f"hook-pyopenms ERROR: Could not collect pyopenms files: {e}")
    import traceback

    traceback.print_exc()

# Ensure all hidden imports for pyopenms submodules
# These are critical for runtime but we collect them WITHOUT importing
hiddenimports += [
    "pyopenms",
    "pyopenms._pyopenms_1",  # Main extension module
    "pyopenms._pyopenms_2",  # Additional extension modules
    "pyopenms._pyopenms_3",
    "pyopenms._pyopenms_4",
    "pyopenms._pyopenms_5",
    "pyopenms._pyopenms_6",
    "pyopenms._pyopenms_7",
    "pyopenms._pyopenms_8",
    "pyopenms.version",
    "pyopenms.Constants",
    "pyopenms.plotting",
]

# CRITICAL: Exclude imports that would trigger pyopenms to load during build
# We only want to collect files, not actually import the module
excludedimports = []
