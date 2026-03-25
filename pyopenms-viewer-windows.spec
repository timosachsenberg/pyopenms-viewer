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

# Get the directory containing this spec file (project root)
spec_dir = os.path.dirname(os.path.abspath(SPEC))

datas = []
binaries = []
hiddenimports = []

# DO NOT use collect_all('pyopenms') here - it will try to import pyopenms
# which fails on Windows due to DLL issues during build.
# Instead, manually collect pyopenms files here AND in hook-pyopenms.py.

# CRITICAL: Manually collect pyopenms binaries WITHOUT importing the module
# This ensures DLLs are collected even if PyInstaller's binary analysis fails
if sys.platform == 'win32':
    try:
        from PyInstaller.utils.hooks import get_package_paths
        pkg_base, pkg_dir = get_package_paths('pyopenms')
        
        print(f"[SPEC] Manually collecting pyopenms binaries from {pkg_dir}", flush=True)
        
        for root, dirs, files in os.walk(pkg_dir):
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(root, pkg_dir)
                dest_dir = os.path.join('pyopenms', rel_path) if rel_path != '.' else 'pyopenms'
                
                if file.endswith('.pyd'):
                    # Python extension modules go to pyopenms/
                    binaries.append((src, dest_dir))
                    print(f"[SPEC] Collected .pyd: {file} -> {dest_dir}/", flush=True)
                elif file.endswith('.dll'):
                    # DLLs go to pyopenms_dlls/ to avoid Qt6 conflicts
                    binaries.append((src, 'pyopenms_dlls'))
                    print(f"[SPEC] Collected .dll: {file} -> pyopenms_dlls/", flush=True)
                elif file.endswith('.py'):
                    # Python source files
                    datas.append((src, dest_dir))

        # Also collect pyopenms.libs/ — on some pyopenms builds (e.g. 3.6 dev)
        # native DLLs live in a sibling directory rather than inside the package.
        pyopenms_libs_dir = os.path.join(pkg_base, 'pyopenms.libs')
        if os.path.exists(pyopenms_libs_dir):
            print(f"[SPEC] Collecting pyopenms.libs DLLs from {pyopenms_libs_dir}", flush=True)
            for root, dirs, files in os.walk(pyopenms_libs_dir):
                for file in files:
                    if file.endswith('.dll'):
                        src = os.path.join(root, file)
                        binaries.append((src, 'pyopenms_dlls'))
                        print(f"[SPEC] Collected pyopenms.libs DLL: {file} -> pyopenms_dlls/", flush=True)
            
        print(f"[SPEC] Total binaries collected: {len(binaries)}", flush=True)
        
    except Exception as e:
        print(f"[SPEC] WARNING: Could not collect pyopenms: {e}", flush=True)

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

# Collect pywebview (native window mode)
tmp_ret = collect_all('webview')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

hiddenimports += [
    'webview',
    'webview.platforms.cocoa',
    'webview.platforms.gtk',
    'webview.platforms.winforms',
    'pywebview',
]

# Add explicit hidden imports for pyopenms
# These tell PyInstaller what to look for, but don't actually import
hiddenimports += [
    'pyopenms',
    'pyopenms._pyopenms_1',
    'pyopenms._pyopenms_2',
    'pyopenms._pyopenms_3',
    'pyopenms._pyopenms_4',
    'pyopenms._pyopenms_5',
    'pyopenms._pyopenms_6',
    'pyopenms._pyopenms_7',
    'pyopenms._pyopenms_8',
    'pyopenms.version',
    'pyopenms.Constants',
    'pyopenms.plotting',
]

# Add explicit hidden imports for pyopenms_viewer package
# This ensures all submodules are bundled even if not directly imported at top level
hiddenimports += [
    'pyopenms_viewer',
    'pyopenms_viewer.app',
    'pyopenms_viewer.cli',
    # Core modules
    'pyopenms_viewer.core',
    'pyopenms_viewer.core.state',
    'pyopenms_viewer.core.events',
    'pyopenms_viewer.core.config',
    'pyopenms_viewer.core.data_manager',
    # Loaders - ALL submodules
    'pyopenms_viewer.loaders',
    'pyopenms_viewer.loaders.mzml_loader',
    'pyopenms_viewer.loaders.feature_loader',
    'pyopenms_viewer.loaders.id_loader',
    'pyopenms_viewer.loaders.chromatogram_loader',
    'pyopenms_viewer.loaders.ion_mobility_loader',
    'pyopenms_viewer.loaders.spectrum_extractor',
    # Panels - ALL submodules
    'pyopenms_viewer.panels',
    'pyopenms_viewer.panels.base_panel',
    'pyopenms_viewer.panels.spectrum_panel',
    'pyopenms_viewer.panels.chromatogram_panel',
    'pyopenms_viewer.panels.peak_map_panel',
    'pyopenms_viewer.panels.tic_panel',
    'pyopenms_viewer.panels.spectra_table_panel',
    'pyopenms_viewer.panels.features_table_panel',
    'pyopenms_viewer.panels.im_peak_map_panel',
    'pyopenms_viewer.panels.faims_panel',
    # Annotation - ALL submodules
    'pyopenms_viewer.annotation',
    'pyopenms_viewer.annotation.spectrum_annotator',
    'pyopenms_viewer.annotation.theoretical_spectrum',
    'pyopenms_viewer.annotation.tick_formatter',
    # Rendering - ALL submodules
    'pyopenms_viewer.rendering',
    'pyopenms_viewer.rendering.peak_map_renderer',
    'pyopenms_viewer.rendering.axis_renderer',
    'pyopenms_viewer.rendering.minimap_renderer',
    'pyopenms_viewer.rendering.overlay_renderer',
    # Components - ALL submodules
    'pyopenms_viewer.components',
    'pyopenms_viewer.components.local_file_picker',
    # Utils - ALL submodules
    'pyopenms_viewer.utils',
    'pyopenms_viewer.utils.coordinate_transform',
]

print(f"[SPEC] Starting Analysis with {len(binaries)} binaries, {len(datas)} datas", flush=True)


a = Analysis(
    ['pyopenms_viewer/__main__.py'],
    pathex=[spec_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.', 'pre_safe_import_module'],  # Include both standard hooks and pre-safe-import hooks
    hooksconfig={},
    runtime_hooks=['pyi_rth_pyopenms.py'],  # Re-enable runtime hook with updated logic
    excludes=[
        # Exclude Qt6 WebEngine components - they are huge (~100MB+) and not needed for this app
        # This also prevents extraction failures with Qt6WebEngineCore.dll
        'PyQt6.QtWebEngine',
        'PyQt6.QtWebEngineCore',
        'PyQt6.QtWebEngineWidgets',
        'PyQt6.QtWebChannel',
        'PyQt6.QtPositioning',
    ],
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
    upx=False,  # Disable UPX compression to avoid DLL issues
    upx_exclude=[],
    runtime_tmpdir=None,  # Let PyInstaller use default temp directory
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
