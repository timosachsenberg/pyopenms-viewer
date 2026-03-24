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
    excludes=[],
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