# -*- mode: python ; coding: utf-8 -*-
# macOS-specific spec file for app bundle
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas = []
binaries = []
hiddenimports = []

# Collect all pyopenms resources
tmp_ret = collect_all('pyopenms')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Explicitly collect dynamic libraries from pyopenms
binaries += collect_dynamic_libs('pyopenms')

# Collect plotly resources
tmp_ret = collect_all('plotly')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect nicegui resources
tmp_ret = collect_all('nicegui')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect pywebview (native window mode)
tmp_ret = collect_all('webview')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Strip any PyQt6 binaries/datas that collect_all may have pulled in transitively.
# pyopenms ships its own Qt dylibs; loading both causes a duplicate-QtCore SIGSEGV.
binaries = [(dest, src, typ) for (dest, src, typ) in binaries if 'PyQt6' not in dest and 'PyQt6' not in src]
datas    = [(src, dest) for (src, dest) in datas    if 'PyQt6' not in src and 'PyQt6' not in dest]

hiddenimports += [
    'webview',
    'webview.platforms.cocoa',
    'webview.platforms.gtk',
    'webview.platforms.winforms',
    'pywebview',
]

# Add explicit hidden imports for pyopenms extension modules
hiddenimports += [
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

a = Analysis(
    ['pyopenms_viewer/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.', 'pre_safe_import_module'],
    hooksconfig={},
    runtime_hooks=['pyi_rth_pyopenms.py'],
    # Exclude PyQt6 entirely on macOS: pyopenms bundles its own Qt dylibs in
    # __dot__dylibs/, and the app uses NiceGUI (web) + pywebview (WebKit) which
    # do not need PyQt6. Bundling both causes a duplicate QtCore crash:
    # PyQt6/QtCore.abi3.so and pyopenms/__dot__dylibs/QtCore are both loaded,
    # their global constructors collide, and CFBundleCopyBundleURL segfaults.
    excludes=['PyQt6', 'PyQt5', 'PySide2', 'PySide6'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# macOS app bundle
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pyopenms-viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='pyopenms-viewer',
)

app = BUNDLE(
    coll,
    name='pyopenms-viewer.app',
    icon=None,
    bundle_identifier=None,
)