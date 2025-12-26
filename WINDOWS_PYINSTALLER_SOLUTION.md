# Windows PyInstaller Solution - Technical Analysis

## Problem Analysis

### Root Cause
The Windows build fails with `0xc0000139` (STATUS_ENTRYPOINT_NOT_FOUND) during PyInstaller's analysis phase when trying to import `pyopenms`. This error indicates a DLL symbol resolution failure.

**Why it fails:**
1. **PyInstaller's isolated subprocess imports**: PyInstaller runs isolated Python subprocesses during analysis to discover dependencies by importing modules
2. **DLL loading conflicts**: Both PyQt6 (our UI framework) and pyopenms bundle Qt6 DLLs. When both are accessible, Windows may load mismatched versions
3. **PATH inheritance limitations**: While `os.environ['PATH']` changes ARE inherited by subprocesses, `os.add_dll_directory()` is NOT (it's process-local API)
4. **Missing DLL search paths**: PyInstaller's isolated subprocess doesn't see `.pth` files or site.py modifications

### Key Evidence from Workflow #37

```
Windows fatal exception: code 0xc0000139
File "C:\...\pyopenms\_all_modules.py", line 1 in <module>
```

The error occurs when `_pyopenms_1.pyd` (Python extension) tries to load `OpenMS.dll`, which in turn needs specific Qt6 DLL symbols. If PyQt6's Qt6 DLLs are loaded first, the symbols don't match.

**Dependency chain:**
```
_pyopenms_1.pyd → OpenMS.dll → Qt6Core.dll (pyopenms version)
                                    ↑
                                    Conflict with PyQt6's Qt6Core.dll
```

## Solution Architecture

### 1. **Avoid Importing pyopenms During Build** (hook-pyopenms.py)

**Change:** Remove `collect_all('pyopenms')` which triggers import.

**Implementation:**
- Use `get_package_paths('pyopenms')` to find package location WITHOUT importing
- Manually walk directory tree to collect all files
- Explicitly gather:
  - All `.dll` files → place in root directory (`.`)
  - All `.pyd` extension modules → place in root directory (`.`)
  - All `.py` source files → preserve directory structure (`pyopenms/`)
  - Qt6 plugins if present → preserve structure (`Qt6/plugins/`)

**Benefits:**
- Eliminates build-time DLL loading failures
- PyInstaller analysis completes successfully
- All files still collected comprehensively

### 2. **Runtime DLL Isolation** (pyi_rth_pyopenms.py)

**Change:** Aggressively control DLL search order at runtime.

**Implementation:**
```python
# Remove PyQt6 Qt6 paths from PATH
path_parts = current_path.split(os.pathsep)
cleaned_parts = [p for p in path_parts if 'PyQt6' not in p and 'Qt6\\bin' not in p]

# Prepend pyopenms DLL directory
os.environ['PATH'] = exe_dir + os.pathsep + cleaned_path

# Set Qt plugin paths to use pyopenms's Qt6, not PyQt6's
os.environ['QT_PLUGIN_PATH'] = qt_plugins_dir
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_dir
```

**Benefits:**
- Ensures pyopenms's Qt6 DLLs are loaded first
- Prevents PyQt6 Qt6 DLL conflicts
- Uses ASCII-only output (no Unicode encoding errors)

### 3. **Simplified Spec File** (pyopenms-viewer-windows.spec)

**Change:** Remove `collect_all('pyopenms')` from spec file.

**Implementation:**
- Only collect safe-to-import packages (plotly, nicegui)
- Declare pyopenms hidden imports WITHOUT importing
- Let hook-pyopenms.py handle all pyopenms collection

**Benefits:**
- Spec file doesn't trigger pyopenms import
- Cleaner separation of concerns
- Faster analysis phase

### 4. **Hidden Imports Without Import** (hook-pyopenms.py)

**Change:** Scan `__init__.py` content to discover imports instead of actually importing.

**Implementation:**
```python
with open(init_file, 'r') as f:
    content = f.read()
    if '_pyopenms' in content:
        hiddenimports.append('pyopenms._pyopenms_1')
```

**Benefits:**
- PyInstaller knows what modules to include
- No actual import during build
- Avoids DLL loading

## Expected Behavior After Fix

### Build Phase
1. Spec file adds pyopenms directory to PATH
2. hook-pyopenms.py collects all DLLs/extensions WITHOUT importing
3. Analysis completes without `0xc0000139` error
4. ~28 DLLs and 5 extension modules (.pyd) collected
5. Executable builds successfully

### Runtime Phase
1. Runtime hook removes PyQt6 from PATH
2. Runtime hook prepends extraction directory to PATH
3. Qt environment variables set to use pyopenms's Qt6
4. `import pyopenms` succeeds with correct DLLs loaded
5. Application starts without symbol resolution errors

## Testing the Fix

### Commit and Push
```bash
git add hook-pyopenms.py pyi_rth_pyopenms.py pyopenms-viewer-windows.spec
git commit -m "Fix Windows PyInstaller build by avoiding pyopenms import during analysis

- Refactor hook-pyopenms.py to collect files WITHOUT importing module
- Add DLL isolation in runtime hook to prevent Qt6 conflicts with PyQt6
- Remove collect_all('pyopenms') from spec file
- Use ASCII-only output in runtime hook for Windows console
- Scan __init__.py content to discover hidden imports

This solves the 0xc0000139 (STATUS_ENTRYPOINT_NOT_FOUND) error during
PyInstaller's analysis phase by avoiding build-time DLL loading."

git push origin fix/windows-pyinstaller-dll
```

### Expected Workflow #38 Output
```
[SPEC] pyopenms directory: C:\...\pyopenms
[SPEC] Verified OpenMS.dll exists: C:\...\pyopenms\OpenMS.dll
hook-pyopenms: Collecting from C:\...\pyopenms (no import)
hook-pyopenms: Found DLL: OpenMS.dll
hook-pyopenms: Found DLL: Qt6Core.dll
... [28 total DLLs]
hook-pyopenms: Collected 28 DLLs and 5 extension modules
INFO: Looking for dynamic libraries
INFO: Building EXE completed successfully
```

Runtime test:
```
[pyi_rth_pyopenms] Runtime hook starting...
[pyi_rth_pyopenms] Found 5/5 critical DLLs
[pyi_rth_pyopenms] PATH updated (PyQt6 Qt6 paths removed, exe_dir prepended)
[pyi_rth_pyopenms] QT_PLUGIN_PATH set to: ...
[pyi_rth_pyopenms] Runtime hook completed successfully

pyOpenMS version: 3.5.0  ← SUCCESS!
```

## Fallback Plan

If this solution still fails with symbol resolution errors, next steps:

1. **Exclude PyQt6 Qt6 binaries entirely** - Force pyopenms to provide all Qt6
2. **Use Nuitka instead of PyInstaller** - Better DLL dependency handling
3. **Create conda-based installer** - Ship full conda environment
4. **Docker container** - Most reliable cross-platform distribution

## References

- PyInstaller DLL handling: https://pyinstaller.org/en/stable/runtime-information.html
- Windows error 0xc0000139: https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes
- Qt plugin loading: https://doc.qt.io/qt-6/deployment-plugins.html
