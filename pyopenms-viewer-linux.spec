# -*- mode: python ; coding: utf-8 -*-
# Linux-specific spec file for single-file executable
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

# Add explicit hidden imports for pyopenms extension modules
hiddenimports += [
    'pyopenms._pyopenms_1',
    'pyopenms.version',
]

a = Analysis(
    ['pyopenms_viewer/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Single-file executable for Linux
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
