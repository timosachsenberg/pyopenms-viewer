"""
macOS libomp.dylib symlink fix.

On macOS, pyopenms bundles its own libomp.dylib, but if Homebrew's libomp is also
loaded (e.g., by other packages), it can cause a segfault due to:
  "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized."

This module symlinks pyopenms's libomp.dylib to the Homebrew version to ensure only
one copy is loaded.
"""

import os
import sys
import subprocess
from pathlib import Path


def fix_libomp_symlink():
    """
    Symlink pyopenms's libomp.dylib to Homebrew's libomp to avoid double-loading.

    This prevents the segfault caused by loading two different libomp.dylib instances.
    Only runs on macOS when both pyopenms and Homebrew libomp are present.
    """
    if sys.platform != "darwin":
        return  # Only needed on macOS

    try:
        # Find pyopenms's libomp.dylib
        import site

        site_packages = site.getsitepackages()
        if not site_packages:
            return

        pyopenms_libomp = None
        for sp in site_packages:
            candidate = Path(sp) / "pyopenms" / "libomp.dylib"
            if candidate.exists():
                pyopenms_libomp = candidate
                break

        if not pyopenms_libomp:
            # No pyopenms libomp found, nothing to fix
            return

        # Check if it's already a symlink (already fixed)
        if pyopenms_libomp.is_symlink():
            return

        # Find Homebrew's libomp
        try:
            result = subprocess.run(
                ["brew", "--prefix", "libomp"], capture_output=True, text=True, check=True, timeout=5
            )
            brew_libomp_prefix = result.stdout.strip()
            brew_libomp = Path(brew_libomp_prefix) / "lib" / "libomp.dylib"

            if not brew_libomp.exists():
                # Homebrew libomp not found, nothing to fix
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # brew not installed or libomp not found
            return

        # Backup the original pyopenms libomp
        backup = pyopenms_libomp.with_suffix(".dylib.bak")
        if not backup.exists():
            pyopenms_libomp.rename(backup)
        else:
            # Backup already exists, just remove the current one
            pyopenms_libomp.unlink()

        # Create symlink
        pyopenms_libomp.symlink_to(brew_libomp)

        print(f"✓ Symlinked {pyopenms_libomp} -> {brew_libomp}", file=sys.stderr)

    except Exception as e:
        # Don't fail if the fix doesn't work, just warn
        print(f"Warning: Could not fix libomp symlink: {e}", file=sys.stderr)


if __name__ == "__main__":
    fix_libomp_symlink()
