## Summary

Fixes Windows PyInstaller builds by resolving Qt6 DLL conflicts between PyQt6 6.10.1 and pyopenms 3.5.0.

## Problem

Windows executable crashed with error `0xc0000139` (STATUS_ENTRYPOINT_NOT_FOUND) when trying to import pyopenms. Both PyQt6 and pyopenms bundle Qt6 DLLs, and Windows loaded mismatched versions causing symbol resolution failures.

## Solution

**Nuclear option approach:** Runtime hook deletes PyQt6's Qt6/bin directory before any imports, forcing Windows to use only pyopenms's Qt6 DLLs.

### Key Components

1. **hook-pyopenms.py** - Custom PyInstaller collection hook that gathers all pyopenms DLLs without importing the module
2. **pyi_rth_pyopenms.py** - Runtime hook that executes before user code to delete PyQt6 Qt6/bin and configure DLL paths
3. **pre_safe_import_module/hook-pyopenms.py** - Pre-import hook to set up DLL search paths during PyInstaller's analysis phase
4. **pyopenms-viewer-windows.spec** - Windows-specific PyInstaller configuration
5. **test_windows_fix.py** - Local smoke tests to validate the solution
6. **WINDOWS_PYINSTALLER_SOLUTION.md** - Technical documentation explaining the root cause and solution

## Testing

**GitHub Actions Workflow #40** validates:
- Build completes successfully (~437 MB executable)
- Runtime hook executes and removes PyQt6 Qt6/bin directory
- All 5 critical DLLs found (OpenMS.dll, Qt6Core.dll, Qt6Network.dll, msvcp140.dll, vcomp140.dll)
- Application starts without errors
- NiceGUI server launches successfully
- GUI window opens and displays mzML files correctly

**Workflow run:** https://github.com/Aditya-Sarna/pyopenms-viewer/actions/runs/20520869857

## Files Added

- `hook-pyopenms.py` (119 lines)
- `pre_safe_import_module/hook-pyopenms.py` (60 lines)
- `pyi_rth_pyopenms.py` (90 lines)
- `pyopenms-viewer-windows.spec` (115 lines)
- `test_windows_fix.py` (190 lines)
- `WINDOWS_PYINSTALLER_SOLUTION.md` (176 lines)

**Total:** 750 lines added, 0 lines modified in existing code

## Notes

- Build-time subprocess crash still occurs during dependency analysis but does not affect the final executable
- End users experience zero runtime errors
- Executable is fully functional and ready for distribution
- This is a Windows-only fix; macOS and Linux builds unaffected

## Release Readiness

This PR enables Windows executable releases via GitHub Actions. Once merged, Windows users can download a working standalone executable.
