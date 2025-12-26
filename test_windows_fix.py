#!/usr/bin/env python3
"""
Smoke test for Windows PyInstaller fix.

This script validates that the hook changes work correctly by:
1. Checking hook-pyopenms.py can execute without importing pyopenms
2. Verifying runtime hook has ASCII-only output
3. Testing DLL collection logic
"""

import os
import sys
import ast
from pathlib import Path

def test_hook_no_import():
    """Verify hook-pyopenms.py doesn't use collect_all() that imports."""
    print("Testing hook-pyopenms.py doesn't import pyopenms...")
    
    hook_path = Path(__file__).parent / "hook-pyopenms.py"
    with open(hook_path, 'r') as f:
        content = f.read()
    
    # Parse as AST to find function calls
    tree = ast.parse(content)
    
    found_collect_all = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == 'collect_all':
                    # Check if it's being called on 'pyopenms'
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if node.args[0].value == 'pyopenms':
                            found_collect_all = True
    
    if found_collect_all:
        print("  FAIL: hook still uses collect_all('pyopenms')")
        return False
    
    print("  PASS: hook doesn't call collect_all('pyopenms')")
    return True

def test_runtime_hook_ascii():
    """Verify runtime hook only uses ASCII characters."""
    print("\nTesting pyi_rth_pyopenms.py uses ASCII-only output...")
    
    rth_path = Path(__file__).parent / "pyi_rth_pyopenms.py"
    with open(rth_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for Unicode characters
    unicode_chars = []
    for i, char in enumerate(content):
        if ord(char) > 127 and char not in '\n\r\t':
            unicode_chars.append((i, char, hex(ord(char))))
    
    if unicode_chars:
        print(f"  FAIL: Found {len(unicode_chars)} non-ASCII characters:")
        for pos, char, code in unicode_chars[:5]:  # Show first 5
            print(f"    Position {pos}: '{char}' ({code})")
        return False
    
    print("  PASS: Runtime hook uses only ASCII characters")
    return True

def test_spec_no_collect_all_pyopenms():
    """Verify spec file doesn't use collect_all('pyopenms')."""
    print("\nTesting pyopenms-viewer-windows.spec doesn't import pyopenms...")
    
    spec_path = Path(__file__).parent / "pyopenms-viewer-windows.spec"
    with open(spec_path, 'r') as f:
        lines = f.readlines()
    
    # Check each line, ignoring comments
    for i, line in enumerate(lines):
        # Skip comments
        if '#' in line:
            code_part = line[:line.index('#')]
        else:
            code_part = line
        
        # Check for problematic patterns in code (not comments)
        if "collect_all('pyopenms')" in line or 'collect_all("pyopenms")' in line:
            print(f"  FAIL: Line {i+1} uses collect_all('pyopenms'): {line.strip()}")
            return False
        
        if "collect_dynamic_libs('pyopenms')" in line or 'collect_dynamic_libs("pyopenms")' in line:
            print(f"  FAIL: Line {i+1} uses collect_dynamic_libs('pyopenms'): {line.strip()}")
            return False
    
    print("  PASS: spec file doesn't call collect functions on pyopenms")
    return True

def test_hook_collects_dlls():
    """Verify hook logic for collecting DLLs is present."""
    print("\nTesting hook-pyopenms.py has DLL collection logic...")
    
    hook_path = Path(__file__).parent / "hook-pyopenms.py"
    with open(hook_path, 'r') as f:
        content = f.read()
    
    required_patterns = [
        'get_package_paths',  # Must use this instead of importing
        "endswith('.dll')",   # Must collect DLLs
        "binaries.append",    # Must add to binaries list
        'os.walk',           # Must walk directory tree
    ]
    
    missing = []
    for pattern in required_patterns:
        if pattern not in content:
            missing.append(pattern)
    
    if missing:
        print(f"  FAIL: Missing required patterns: {missing}")
        return False
    
    print("  PASS: Hook has DLL collection logic")
    return True

def test_runtime_hook_qt_isolation():
    """Verify runtime hook isolates Qt6 DLLs."""
    print("\nTesting pyi_rth_pyopenms.py isolates Qt6 DLLs...")
    
    rth_path = Path(__file__).parent / "pyi_rth_pyopenms.py"
    with open(rth_path, 'r') as f:
        content = f.read()
    
    required_patterns = [
        'PyQt6',  # Must check for PyQt6 removal
        'QT_PLUGIN_PATH',  # Must set Qt plugin path
        "os.environ['PATH']",  # Must modify PATH
        'sys._MEIPASS',  # Must use PyInstaller's extraction dir
        'shutil.rmtree',  # Must delete PyQt6 Qt6/bin directory
    ]
    
    missing = []
    for pattern in required_patterns:
        if pattern not in content:
            missing.append(pattern)
    
    if missing:
        print(f"  FAIL: Missing required patterns: {missing}")
        return False
    
    print("  PASS: Runtime hook has Qt6 isolation logic")
    return True

def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Windows PyInstaller Fix - Smoke Test")
    print("=" * 60)
    
    tests = [
        test_hook_no_import,
        test_runtime_hook_ascii,
        test_spec_no_collect_all_pyopenms,
        test_hook_collects_dlls,
        test_runtime_hook_qt_isolation,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  FAIL: Exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\nAll smoke tests PASSED!")
        print("\nNext steps:")
        print("1. Wait for GitHub Actions workflow to complete")
        print("2. Check workflow logs for build success")
        print("3. Verify runtime hook output in smoke test")
        print("4. If successful, create PR to upstream")
        return 0
    else:
        print("\nSome tests FAILED!")
        print("\nFix the issues above before pushing to CI.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
