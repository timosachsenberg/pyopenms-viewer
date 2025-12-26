# -*- mode: python ; coding: utf-8 -*-
# Windows-specific spec file for single-file executable
# CRITICAL: This spec avoids importing pyopenms during build to prevent DLL failures
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs
import sys
import os
import site

# EARLY SETUP: Configure PATH for any subprocess spawned by PyInstaller
# This helps (but doesn't fully solve) the pyopenms import issue during build
if sys.platform == 'win32':
    from PyInstaller.utils.hooks import get_package_paths
    try:
        pkg_base, pkg_dir = get_package_paths('pyopenms')
        pyopenms_dir = pkg_dir
        
        print(f"[SPEC] pyopenms directory: {pyopenms_dir}", flush=True)
        
        # Add to PATH (inherited by subprocesses)
        if os.path.exists(pyopenms_dir):
            os.environ['PATH'] = pyopenms_dir + os.pathsep + os.environ.get('PATH', '')
            print(f"[SPEC] Added pyopenms directory to PATH", flush=True)
            
            # Verify OpenMS.dll exists
            openms_dll = os.path.join(pyopenms_dir, 'OpenMS.dll')
            if os.path.exists(openms_dll):
                print(f"[SPEC] Verified OpenMS.dll exists: {openms_dll}", flush=True)
            else:
                print(f"[SPEC] WARNING: OpenMS.dll not found at {openms_dll}", flush=True)
        
        # Add DLL directory for main process
        if hasattr(os, 'add_dll_directory') and os.path.exists(pyopenms_dir):
            try:
                os.add_dll_directory(pyopenms_dir)
                print(f"[SPEC] Added DLL search directory", flush=True)
            except Exception as e:
                print(f"[SPEC] WARNING: add_dll_directory failed: {e}", flush=True)
                
    except Exception as e:
        print(f"[SPEC] ERROR: Failed to configure pyopenms paths: {e}", flush=True)

datas = []
binaries = []
hiddenimports = []

# DO NOT use collect_all('pyopenms') here - it will try to import pyopenms
# which fails on Windows due to DLL issues during build.
# Instead, let hook-pyopenms.py do all the collection WITHOUT importing.

# Collect plotly resources (safe to import)
tmp_ret = collect_all('plotly')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect nicegui resources (safe to import)
tmp_ret = collect_all('nicegui')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Add explicit hidden imports for pyopenms
# These tell PyInstaller what to look for, but don't actually import
hiddenimports += [
    'pyopenms',
    'pyopenms._pyopenms_1',
    'pyopenms._pyopenms_2',
    'pyopenms._pyopenms_3',
    'pyopenms._pyopenms_4',
    'pyopenms._pyopenms_5',
    'pyopenms.version',
    'pyopenms.Constants',
    'pyopenms.plotting',
]

print(f"[SPEC] Starting Analysis with {len(binaries)} binaries, {len(datas)} datas", flush=True)


a = Analysis(
    ['pyopenms_viewer/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.', 'pre_safe_import_module'],  # Include both standard hooks and pre-safe-import hooks
    hooksconfig={},
    runtime_hooks=['pyi_rth_pyopenms.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Single-file executable for Windows
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pyopenms-viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
