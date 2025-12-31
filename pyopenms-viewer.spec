# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs
import sys

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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pyopenms-viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pyopenms-viewer',
)
app = BUNDLE(
    coll,
    name='pyopenms-viewer.app',
    icon=None,
    bundle_identifier=None,
)
