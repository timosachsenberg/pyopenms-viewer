## Windows PyInstaller Fix Release

This release fixes critical Windows executable build and runtime errors, making pyopenms-viewer fully functional on Windows.

## What's Fixed

### Windows PyInstaller Qt6 DLL Conflict
- **Problem**: Windows executable crashed with error `0xc0000139` (STATUS_ENTRYPOINT_NOT_FOUND) when importing pyopenms
- **Root Cause**: PyQt6 6.10.1 and pyopenms 3.5.0 both bundle Qt6 DLLs. Windows loaded mismatched versions causing symbol resolution failures
- **Solution**: Runtime hook deletes PyQt6's Qt6/bin directory before any imports, forcing Windows to use only pyopenms's Qt6 DLLs

## New Files

- `hook-pyopenms.py` - Custom PyInstaller collection hook that gathers pyopenms DLLs without importing
- `pyi_rth_pyopenms.py` - Runtime hook that executes before user code to manage DLL paths
- `pre_safe_import_module/hook-pyopenms.py` - Pre-import hook for DLL path configuration during build
- `pyopenms-viewer-windows.spec` - Windows-specific PyInstaller build configuration
- `test_windows_fix.py` - Local smoke tests for validation
- `WINDOWS_PYINSTALLER_SOLUTION.md` - Technical documentation of the problem and solution

## Testing

Validated via GitHub Actions:
- Build completes successfully (~437 MB executable)
- Runtime hook removes PyQt6 Qt6/bin directory
- All 5 critical DLLs found (OpenMS.dll, Qt6Core.dll, Qt6Network.dll, msvcp140.dll, vcomp140.dll)
- Application starts without errors
- NiceGUI server launches successfully
- GUI window opens and displays mzML files correctly

## Download

Download the Windows executable from the Assets section below. No installation required - just run `pyopenms-viewer.exe`!

## For Developers

If you're building from source on Windows, the custom PyInstaller hooks are now automatically used. See `WINDOWS_PYINSTALLER_SOLUTION.md` for technical details.

## Compatibility

- **Windows**: Fixed and fully working
- **macOS**: Unchanged, working as before
- **Linux**: Unchanged, working as before

---

**Full Changelog**: https://github.com/Aditya-Sarna/pyopenms-viewer/blob/main/CHANGELOG.md
