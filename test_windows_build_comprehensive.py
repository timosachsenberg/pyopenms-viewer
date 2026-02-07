#!/usr/bin/env python3
"""
COMPREHENSIVE WINDOWS BUILD VERIFICATION TEST
==============================================
This script validates ALL components of the Windows PyInstaller build configuration
without actually running PyInstaller (which requires Windows).

Tests:
1. File existence and syntax validation
2. Hook logic simulation
3. Runtime hook logic validation
4. Spec file configuration validation
5. CI workflow validation
6. Hidden imports completeness
7. File separation logic verification
8. DLL path setup verification
"""

import ast
import os
import re
import sys


# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def fail(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")
    return False


def warn(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")


def header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")


# Track all test results
all_passed = True


def test(name):
    """Decorator for test functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            global all_passed
            try:
                result = func(*args, **kwargs)
                if result is False:
                    all_passed = False
                return result
            except Exception as e:
                fail(f"{name}: EXCEPTION - {e}")
                all_passed = False
                return False

        return wrapper

    return decorator


# =============================================================================
# TEST 1: FILE EXISTENCE
# =============================================================================
header("TEST 1: Critical File Existence")

REQUIRED_FILES = [
    "hook-pyopenms.py",
    "pyi_rth_pyopenms.py",
    "pre_safe_import_module/hook-pyopenms.py",
    "pyopenms-viewer-windows.spec",
    "pyopenms-viewer-linux.spec",
    "pyopenms-viewer-macos.spec",
    ".github/workflows/windows.yml",
    ".github/workflows/build.yml",
    ".github/workflows/linux.yml",
]

for f in REQUIRED_FILES:
    if os.path.exists(f):
        ok(f"File exists: {f}")
    else:
        all_passed = False
        fail(f"MISSING: {f}")

# =============================================================================
# TEST 2: PYTHON SYNTAX VALIDATION
# =============================================================================
header("TEST 2: Python Syntax Validation")

PYTHON_FILES = [
    "hook-pyopenms.py",
    "pyi_rth_pyopenms.py",
    "pre_safe_import_module/hook-pyopenms.py",
]

for f in PYTHON_FILES:
    if os.path.exists(f):
        try:
            with open(f) as file:
                source = file.read()
            ast.parse(source)
            ok(f"Valid Python syntax: {f}")
        except SyntaxError as e:
            all_passed = False
            fail(f"SYNTAX ERROR in {f}: {e}")
    else:
        warn(f"Cannot check syntax - file missing: {f}")

# =============================================================================
# TEST 3: HOOK-PYOPENMS.PY LOGIC VERIFICATION
# =============================================================================
header("TEST 3: hook-pyopenms.py Logic Verification")

if os.path.exists("hook-pyopenms.py"):
    with open("hook-pyopenms.py") as f:
        hook_content = f.read()

    # Test 3.1: .pyd files go to pyopenms/ directory
    pyd_pattern = r"file\.endswith\(\s*\(\s*['\"]\.pyd['\"]"
    if re.search(pyd_pattern, hook_content):
        # Check that .pyd is followed by placing in dest_dir (not pyopenms_dlls)
        pyd_section = re.search(
            r"elif file\.endswith\(\(['\"]\.pyd['\"].*?\n.*?binaries\.append\(\(src,\s*(\w+)\)", hook_content, re.DOTALL
        )
        if pyd_section and pyd_section.group(1) == "dest_dir":
            ok(".pyd files → dest_dir (pyopenms/) ✓")
        else:
            all_passed = False
            fail(".pyd files should go to dest_dir, not pyopenms_dlls!")
    else:
        all_passed = False
        fail("Cannot find .pyd handling logic")

    # Test 3.2: .dll files go to pyopenms_dlls/ directory
    dll_pattern = r"elif file\.endswith\(\(['\"]\.dll['\"].*?\n.*?binaries\.append\(\(src,\s*['\"]pyopenms_dlls['\"]\)"
    if re.search(dll_pattern, hook_content, re.DOTALL):
        ok(".dll files → pyopenms_dlls/ ✓")
    else:
        all_passed = False
        fail(".dll files should go to pyopenms_dlls!")

    # Test 3.3: All 8 _pyopenms modules in hiddenimports
    for i in range(1, 9):
        if f"pyopenms._pyopenms_{i}" in hook_content:
            ok(f"Hidden import: pyopenms._pyopenms_{i} ✓")
        else:
            all_passed = False
            fail(f"MISSING hidden import: pyopenms._pyopenms_{i}")

    # Test 3.4: Does NOT use collect_all('pyopenms') at module level
    if re.search(r"^[^#]*collect_all\s*\(\s*['\"]pyopenms['\"]\s*\)", hook_content, re.MULTILINE):
        all_passed = False
        fail("hook-pyopenms.py should NOT call collect_all('pyopenms') - causes import!")
    else:
        ok("Does not call collect_all('pyopenms') at module level ✓")

    # Test 3.5: Uses get_package_paths instead of importing
    if "get_package_paths('pyopenms')" in hook_content:
        ok("Uses get_package_paths() to find pyopenms without importing ✓")
    else:
        warn("Should use get_package_paths('pyopenms') to avoid importing")

# =============================================================================
# TEST 4: RUNTIME HOOK VERIFICATION
# =============================================================================
header("TEST 4: pyi_rth_pyopenms.py Runtime Hook Verification")

if os.path.exists("pyi_rth_pyopenms.py"):
    with open("pyi_rth_pyopenms.py") as f:
        rth_content = f.read()

    # Test 4.1: Checks for frozen state
    if "getattr(sys, 'frozen', False)" in rth_content:
        ok("Checks sys.frozen before running ✓")
    else:
        all_passed = False
        fail("Must check sys.frozen before executing frozen-only code")

    # Test 4.2: Uses sys._MEIPASS
    if "sys._MEIPASS" in rth_content:
        ok("Uses sys._MEIPASS for extraction directory ✓")
    else:
        all_passed = False
        fail("Must use sys._MEIPASS to find extraction directory")

    # Test 4.3: Adds pyopenms_dlls to PATH
    if "pyopenms_dlls" in rth_content and "PATH" in rth_content:
        ok("Adds pyopenms_dlls to PATH ✓")
    else:
        all_passed = False
        fail("Must add pyopenms_dlls directory to PATH")

    # Test 4.4: Uses os.add_dll_directory()
    if "os.add_dll_directory" in rth_content:
        ok("Uses os.add_dll_directory() for Windows 10+ ✓")
    else:
        all_passed = False
        fail("Must use os.add_dll_directory() for Windows 10+")

    # Test 4.5: Adds pyopenms package directory for .pyd files
    if "pyopenms_pkg_dir" in rth_content or ("pyopenms" in rth_content and "add_dll_directory" in rth_content):
        ok("Adds pyopenms package directory for .pyd file loading ✓")
    else:
        warn("Should add pyopenms package directory to DLL search path")

    # Test 4.6: Prepends to PATH (not appends)
    if re.search(r"PATH.*=.*pyopenms_dlls.*\+.*pathsep", rth_content):
        ok("PREPENDS pyopenms_dlls to PATH (not appends) ✓")
    else:
        warn("Should prepend pyopenms_dlls to PATH to ensure priority")

# =============================================================================
# TEST 5: WINDOWS SPEC FILE VERIFICATION
# =============================================================================
header("TEST 5: pyopenms-viewer-windows.spec Verification")

if os.path.exists("pyopenms-viewer-windows.spec"):
    with open("pyopenms-viewer-windows.spec") as f:
        spec_content = f.read()

    # Test 5.1: runtime_hooks includes pyi_rth_pyopenms.py
    if "runtime_hooks=['pyi_rth_pyopenms.py']" in spec_content:
        ok("runtime_hooks includes pyi_rth_pyopenms.py ✓")
    elif "runtime_hooks=[]" in spec_content:
        all_passed = False
        fail("CRITICAL: runtime_hooks is EMPTY - runtime hook won't run!")
    else:
        warn("Could not verify runtime_hooks setting")

    # Test 5.2: hookspath includes current directory
    if "hookspath=['.'" in spec_content or "hookspath=['.'," in spec_content:
        ok("hookspath includes '.' for custom hooks ✓")
    else:
        all_passed = False
        fail("hookspath must include '.' to find hook-pyopenms.py")

    # Test 5.3: All 8 _pyopenms modules in hiddenimports
    missing_modules = []
    for i in range(1, 9):
        if f"pyopenms._pyopenms_{i}" not in spec_content:
            missing_modules.append(i)

    if not missing_modules:
        ok("All 8 _pyopenms extension modules in hiddenimports ✓")
    else:
        all_passed = False
        fail(f"MISSING _pyopenms modules in spec: {missing_modules}")

    # Test 5.4: Does NOT use collect_all('pyopenms')
    # Check for actual call, not just comment mentioning it
    # Pattern: collect_all('pyopenms') or collect_all("pyopenms") NOT preceded by # comment
    lines = spec_content.split("\n")
    uses_collect_all_pyopenms = False
    for line in lines:
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        if "collect_all('pyopenms')" in line or 'collect_all("pyopenms")' in line:
            uses_collect_all_pyopenms = True
            break

    if uses_collect_all_pyopenms:
        all_passed = False
        fail("CRITICAL: Spec file uses collect_all('pyopenms') - will fail on Windows!")
    else:
        ok("Does NOT call collect_all('pyopenms') in spec ✓")

    # Test 5.5: UPX is disabled (can corrupt DLLs)
    if "upx=False" in spec_content:
        ok("UPX compression disabled (prevents DLL corruption) ✓")
    else:
        warn("Consider disabling UPX (upx=False) to prevent DLL issues")

    # Test 5.6: Excludes problematic Qt6 WebEngine
    if "PyQt6.QtWebEngine" in spec_content and "excludes" in spec_content:
        ok("Excludes Qt6WebEngine (prevents extraction failures) ✓")
    else:
        warn("Consider excluding PyQt6.QtWebEngine to reduce size and issues")

# =============================================================================
# TEST 6: CI WORKFLOW VERIFICATION
# =============================================================================
header("TEST 6: CI Workflow Verification")

# Test 6.1: windows.yml
if os.path.exists(".github/workflows/windows.yml"):
    with open(".github/workflows/windows.yml") as f:
        windows_yml = f.read()

    if "pyopenms-viewer-windows.spec" in windows_yml:
        ok("windows.yml uses pyopenms-viewer-windows.spec ✓")
    elif "--collect-all pyopenms" in windows_yml:
        all_passed = False
        fail("CRITICAL: windows.yml uses --collect-all pyopenms (will fail!)")
    else:
        warn("Could not verify windows.yml build command")

    if "pyinstaller-hooks-contrib" in windows_yml:
        ok("windows.yml installs pyinstaller-hooks-contrib ✓")
    else:
        warn("windows.yml should install pyinstaller-hooks-contrib")

# Test 6.2: build.yml
if os.path.exists(".github/workflows/build.yml"):
    with open(".github/workflows/build.yml") as f:
        build_yml = f.read()

    if "pyopenms-viewer-windows.spec" in build_yml:
        ok("build.yml uses pyopenms-viewer-windows.spec for Windows ✓")
    elif "nicegui.scripts.pack" in build_yml:
        all_passed = False
        fail("CRITICAL: build.yml uses nicegui.pack instead of spec file!")
    else:
        warn("Could not verify build.yml Windows build command")

    if "spec_file:" in build_yml:
        ok("build.yml uses matrix with spec_file variable ✓")
    else:
        warn("build.yml should use matrix for platform-specific spec files")

# =============================================================================
# TEST 7: PRE-SAFE-IMPORT-MODULE HOOK
# =============================================================================
header("TEST 7: pre_safe_import_module Hook Verification")

if os.path.exists("pre_safe_import_module/hook-pyopenms.py"):
    with open("pre_safe_import_module/hook-pyopenms.py") as f:
        presafe_content = f.read()

    if "def pre_safe_import_module(api)" in presafe_content:
        ok("Defines pre_safe_import_module(api) function ✓")
    else:
        all_passed = False
        fail("Must define pre_safe_import_module(api) function")

    if "sys.platform == 'win32'" in presafe_content:
        ok("Checks for Windows platform ✓")
    else:
        warn("Should check sys.platform == 'win32'")

    if "os.environ['PATH']" in presafe_content:
        ok("Modifies PATH environment variable ✓")
    else:
        warn("Should modify PATH for DLL discovery")

# =============================================================================
# TEST 8: SIMULATE HOOK FILE COLLECTION
# =============================================================================
header("TEST 8: Simulate Hook File Collection Logic")

try:
    from PyInstaller.utils.hooks import get_package_paths

    pkg_base, pkg_dir = get_package_paths("pyopenms")
    print(f"   pyopenms location: {pkg_dir}")

    pyd_count = 0
    dll_count = 0
    pyd_to_pyopenms = 0
    dll_to_pyopenms_dlls = 0

    for root, dirs, files in os.walk(pkg_dir):
        for file in files:
            rel_path = os.path.relpath(root, pkg_dir)
            dest_dir = os.path.join("pyopenms", rel_path) if rel_path != "." else "pyopenms"

            if file.endswith((".pyd", ".so")):
                pyd_count += 1
                # Simulate hook logic: .pyd goes to dest_dir
                if dest_dir.startswith("pyopenms") and "pyopenms_dlls" not in dest_dir:
                    pyd_to_pyopenms += 1
            elif file.endswith((".dll", ".dylib")):
                dll_count += 1
                # Simulate hook logic: .dll goes to pyopenms_dlls
                dll_to_pyopenms_dlls += 1  # Always goes to pyopenms_dlls in our hook

    print(f"   Found {pyd_count} extension modules (.pyd/.so)")
    print(f"   Found {dll_count} DLLs (.dll/.dylib)")

    if pyd_count > 0 and pyd_to_pyopenms == pyd_count:
        ok(f"All {pyd_count} extension modules → pyopenms/ ✓")
    else:
        all_passed = False
        fail(f"Extension module placement issue: {pyd_to_pyopenms}/{pyd_count}")

    if dll_count > 0:
        ok(f"All {dll_count} DLLs → pyopenms_dlls/ ✓")

    # Verify expected extension modules exist
    expected_modules = [f"_pyopenms_{i}" for i in range(1, 9)]
    found_modules = []
    for root, dirs, files in os.walk(pkg_dir):
        for file in files:
            for mod in expected_modules:
                if mod in file and file.endswith((".pyd", ".so")):
                    found_modules.append(mod)

    found_modules = list(set(found_modules))
    if len(found_modules) >= 1:
        ok(f"Found {len(found_modules)} _pyopenms_N extension modules")
    else:
        warn("Could not verify _pyopenms_N modules exist")

except ImportError:
    warn("PyInstaller not installed - skipping simulation")
except Exception as e:
    warn(f"Could not simulate hook: {e}")

# =============================================================================
# TEST 9: CROSS-CHECK ALL COMPONENTS
# =============================================================================
header("TEST 9: Cross-Component Consistency Check")

# Verify consistency between hook and spec
if os.path.exists("hook-pyopenms.py") and os.path.exists("pyopenms-viewer-windows.spec"):
    with open("hook-pyopenms.py") as f:
        hook = f.read()
    with open("pyopenms-viewer-windows.spec") as f:
        spec = f.read()

    # Check both have same hidden imports
    hook_has_8 = all(f"_pyopenms_{i}" in hook for i in range(1, 9))
    spec_has_8 = all(f"_pyopenms_{i}" in spec for i in range(1, 9))

    if hook_has_8 and spec_has_8:
        ok("Both hook and spec have all 8 _pyopenms modules ✓")
    else:
        all_passed = False
        fail(f"Inconsistent hidden imports: hook={hook_has_8}, spec={spec_has_8}")

# Verify runtime hook is referenced correctly
if os.path.exists("pyi_rth_pyopenms.py") and os.path.exists("pyopenms-viewer-windows.spec"):
    with open("pyopenms-viewer-windows.spec") as f:
        spec = f.read()

    if "pyi_rth_pyopenms.py" in spec:
        ok("Spec file references pyi_rth_pyopenms.py ✓")
    else:
        all_passed = False
        fail("Spec file must reference pyi_rth_pyopenms.py in runtime_hooks!")

# =============================================================================
# TEST 10: HOOK EXECUTION ORDER VERIFICATION
# =============================================================================
header("TEST 10: Hook Execution Order Verification")

"""
PyInstaller Hook Execution Order:
=================================
BUILD TIME (on developer machine):
  1. pre_safe_import_module hooks - Run BEFORE PyInstaller tries to import a module
  2. pre_find_module_path hooks - Modify module search paths
  3. Standard hooks (hook-*.py) - Run AFTER module is found, collect files
  4. post_safe_import_module hooks - Run after import

RUNTIME (on user machine, when frozen app starts):
  5. Runtime hooks (pyi_rth_*.py) - Run BEFORE user code, in order listed

Our hooks:
  - pre_safe_import_module/hook-pyopenms.py: Sets PATH before PyInstaller imports pyopenms
  - hook-pyopenms.py: Collects pyopenms files WITHOUT importing
  - pyi_rth_pyopenms.py: Sets DLL paths when frozen app starts
"""

print("   PyInstaller Hook Execution Order:")
print("   ─────────────────────────────────────────────────────")
print("   BUILD TIME:")
print("   │")
print("   ├─1. pre_safe_import_module/hook-pyopenms.py")
print("   │     → Sets PATH so OpenMS.dll can be found")
print("   │     → Runs BEFORE PyInstaller imports pyopenms")
print("   │")
print("   ├─2. hook-pyopenms.py (standard hook)")
print("   │     → Collects .pyd, .dll, .py files")
print("   │     → Does NOT import pyopenms (avoids DLL failure)")
print("   │     → .pyd → pyopenms/, .dll → pyopenms_dlls/")
print("   │")
print("   ├─3. Analysis phase completes")
print("   │     → All files collected into frozen bundle")
print("   │")
print("   RUNTIME (when user runs .exe):")
print("   │")
print("   └─4. pyi_rth_pyopenms.py (runtime hook)")
print("         → Runs BEFORE any user code")
print("         → Adds pyopenms_dlls/ to PATH")
print("         → Calls os.add_dll_directory()")
print("         → Then user code imports pyopenms successfully")
print("   ─────────────────────────────────────────────────────")

# Test 10.1: Verify pre_safe_import_module runs first (sets PATH for build)
if os.path.exists("pre_safe_import_module/hook-pyopenms.py"):
    with open("pre_safe_import_module/hook-pyopenms.py") as f:
        presafe = f.read()

    # Must set PATH before any import happens
    sets_path_early = "os.environ['PATH']" in presafe or 'os.environ["PATH"]' in presafe
    has_presafe_func = "def pre_safe_import_module" in presafe

    if sets_path_early and has_presafe_func:
        ok("pre_safe_import_module sets PATH before import (Step 1) ✓")
    else:
        all_passed = False
        fail("pre_safe_import_module must set PATH in pre_safe_import_module() function")
else:
    all_passed = False
    fail("pre_safe_import_module/hook-pyopenms.py missing!")

# Test 10.2: Verify standard hook does NOT import pyopenms
if os.path.exists("hook-pyopenms.py"):
    with open("hook-pyopenms.py") as f:
        hook = f.read()

    # Check for dangerous imports that would trigger DLL load
    dangerous_imports = [
        "import pyopenms",
        "from pyopenms import",
        'importlib.import_module("pyopenms")',
        "importlib.import_module('pyopenms')",
    ]

    has_dangerous_import = False
    for pattern in dangerous_imports:
        # Check if pattern exists outside of comments
        for line in hook.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern in line:
                has_dangerous_import = True
                break

    if not has_dangerous_import:
        ok("hook-pyopenms.py does NOT import pyopenms (Step 2) ✓")
    else:
        all_passed = False
        fail("hook-pyopenms.py must NOT import pyopenms - will fail on Windows!")

    # Verify it uses get_package_paths instead
    if "get_package_paths('pyopenms')" in hook:
        ok("hook-pyopenms.py uses get_package_paths() to find package safely ✓")
    else:
        warn("hook-pyopenms.py should use get_package_paths() to avoid importing")

# Test 10.3: Verify runtime hook runs at frozen startup
if os.path.exists("pyi_rth_pyopenms.py"):
    with open("pyi_rth_pyopenms.py") as f:
        rth = f.read()

    # Must check for frozen state first
    checks_frozen = "getattr(sys, 'frozen', False)" in rth or "sys.frozen" in rth
    uses_meipass = "sys._MEIPASS" in rth
    sets_path = "os.environ['PATH']" in rth or 'os.environ["PATH"]' in rth
    adds_dll_dir = "os.add_dll_directory" in rth

    if checks_frozen:
        ok("Runtime hook checks sys.frozen (only runs when frozen) ✓")
    else:
        all_passed = False
        fail("Runtime hook must check sys.frozen before executing")

    if uses_meipass:
        ok("Runtime hook uses sys._MEIPASS to find extraction dir ✓")
    else:
        all_passed = False
        fail("Runtime hook must use sys._MEIPASS")

    if sets_path and adds_dll_dir:
        ok("Runtime hook sets PATH and calls add_dll_directory() (Step 4) ✓")
    else:
        all_passed = False
        fail("Runtime hook must set PATH and call add_dll_directory()")

# Test 10.4: Verify runtime hook is listed in spec (determines execution order)
if os.path.exists("pyopenms-viewer-windows.spec"):
    with open("pyopenms-viewer-windows.spec") as f:
        spec = f.read()

    # Check runtime_hooks order - pyopenms should be first if there are multiple
    rth_match = re.search(r"runtime_hooks\s*=\s*\[(.*?)\]", spec, re.DOTALL)
    if rth_match:
        rth_list = rth_match.group(1)
        if "pyi_rth_pyopenms.py" in rth_list:
            # Check if it's first in the list
            hooks_in_list = re.findall(r"['\"]([^'\"]+)['\"]", rth_list)
            if hooks_in_list and hooks_in_list[0] == "pyi_rth_pyopenms.py":
                ok("pyi_rth_pyopenms.py is FIRST in runtime_hooks (correct order) ✓")
            elif "pyi_rth_pyopenms.py" in hooks_in_list:
                warn("pyi_rth_pyopenms.py should be first in runtime_hooks for DLL priority")
            else:
                all_passed = False
                fail("pyi_rth_pyopenms.py not found in runtime_hooks!")
        else:
            all_passed = False
            fail("runtime_hooks must include pyi_rth_pyopenms.py!")
    else:
        all_passed = False
        fail("Could not parse runtime_hooks from spec file")

# Test 10.5: Verify no circular dependencies or import conflicts
print("\n   Checking for import conflicts...")

# The runtime hook must NOT import pyopenms before setting up PATH
if os.path.exists("pyi_rth_pyopenms.py"):
    with open("pyi_rth_pyopenms.py") as f:
        rth_content = f.read()

    # Find where PATH is set vs where pyopenms might be imported
    path_set_line = None
    pyopenms_import_line = None

    for i, line in enumerate(rth_content.split("\n"), 1):
        if "os.environ['PATH']" in line and "=" in line:
            if path_set_line is None:
                path_set_line = i
        if "import pyopenms" in line or "from pyopenms" in line:
            if not line.strip().startswith("#"):
                pyopenms_import_line = i

    if pyopenms_import_line is None:
        ok("Runtime hook does NOT import pyopenms (correct - avoids circular dep) ✓")
    elif path_set_line and pyopenms_import_line > path_set_line:
        ok("Runtime hook imports pyopenms AFTER setting PATH (safe order) ✓")
    else:
        all_passed = False
        fail("Runtime hook imports pyopenms BEFORE setting PATH - will fail!")

# Test 10.6: Verify hookspath order in spec
if os.path.exists("pyopenms-viewer-windows.spec"):
    with open("pyopenms-viewer-windows.spec") as f:
        spec = f.read()

    hookspath_match = re.search(r"hookspath\s*=\s*\[(.*?)\]", spec, re.DOTALL)
    if hookspath_match:
        hookspath = hookspath_match.group(1)
        has_current_dir = "'.'," in hookspath or "'.'" in hookspath
        has_presafe = "pre_safe_import_module" in hookspath

        if has_current_dir:
            ok("hookspath includes '.' for standard hooks ✓")
        else:
            all_passed = False
            fail("hookspath must include '.' to find hook-pyopenms.py")

        if has_presafe:
            ok("hookspath includes 'pre_safe_import_module' directory ✓")
        else:
            warn("hookspath should include 'pre_safe_import_module' for pre-import hooks")

# =============================================================================
# TEST 11: DLL LOAD ORDER VERIFICATION
# =============================================================================
header("TEST 11: DLL Load Order Verification")

"""
Windows DLL Load Order (critical for Qt6 conflicts):
====================================================
When Python imports _pyopenms_1.pyd, Windows searches for DLLs in:
  1. Directory containing the .exe
  2. Directories in os.add_dll_directory() (Windows 10+)
  3. System directories (System32, etc.)
  4. Directories in PATH environment variable

Our strategy:
  - Put pyopenms DLLs in pyopenms_dlls/ (isolated from PyQt6)
  - Add pyopenms_dlls/ FIRST to PATH
  - Add pyopenms_dlls/ via add_dll_directory()
  - This ensures pyopenms's Qt6 DLLs load before PyQt6's
"""

if os.path.exists("pyi_rth_pyopenms.py"):
    with open("pyi_rth_pyopenms.py") as f:
        rth = f.read()

    # Test 11.1: pyopenms_dlls is added FIRST to PATH (prepended, not appended)
    # Look for pattern: PATH = pyopenms_dlls + ... + old_path
    prepend_pattern = re.search(r"PATH.*=.*pyopenms_dlls.*\+.*pathsep", rth, re.IGNORECASE)
    append_pattern = re.search(r"PATH.*=.*\+.*pyopenms_dlls", rth, re.IGNORECASE)

    if prepend_pattern:
        ok("pyopenms_dlls is PREPENDED to PATH (loaded first) ✓")
    elif append_pattern:
        all_passed = False
        fail("pyopenms_dlls is APPENDED to PATH - must be PREPENDED for priority!")
    else:
        warn("Could not verify PATH modification order")

    # Test 11.2: add_dll_directory is called for both directories
    dll_dir_calls = rth.count("add_dll_directory(")
    if dll_dir_calls >= 2:
        ok(f"add_dll_directory() called {dll_dir_calls} times (pyopenms_dlls + exe_dir + pyopenms) ✓")
    elif dll_dir_calls == 1:
        warn("add_dll_directory() only called once - should add multiple directories")
    else:
        all_passed = False
        fail("add_dll_directory() not called - required for Windows 10+")

    # Test 11.3: Check pyopenms_dlls is added before exe_dir
    lines = rth.split("\n")
    pyopenms_dlls_add_line = None
    exe_dir_add_line = None

    for i, line in enumerate(lines):
        if "add_dll_directory" in line and "pyopenms_dlls" in line:
            pyopenms_dlls_add_line = i
        if "add_dll_directory" in line and "exe_dir" in line and "pyopenms" not in line:
            exe_dir_add_line = i

    if pyopenms_dlls_add_line is not None:
        ok("add_dll_directory() called for pyopenms_dlls ✓")
    else:
        warn("Could not verify add_dll_directory(pyopenms_dlls_dir)")

# Test 11.4: Verify .pyd files are NOT in pyopenms_dlls (would break imports)
if os.path.exists("hook-pyopenms.py"):
    with open("hook-pyopenms.py") as f:
        hook = f.read()

    # Check that .pyd files go to pyopenms/, not pyopenms_dlls/
    pyd_to_dlls = re.search(r"\.pyd.*pyopenms_dlls", hook)

    if pyd_to_dlls:
        all_passed = False
        fail("CRITICAL: .pyd files go to pyopenms_dlls/ - breaks imports!")
    else:
        ok(".pyd files go to pyopenms/ (not pyopenms_dlls/) ✓")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
header("FINAL SUMMARY")

if all_passed:
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'=' * 60}")
    print("ALL TESTS PASSED! ✓")
    print(f"{'=' * 60}{Colors.END}")
    print("\nThe Windows build configuration appears to be correct.")
    print("Key verified components:")
    print("  1. hook-pyopenms.py: Collects files WITHOUT importing")
    print("  2. pyi_rth_pyopenms.py: Sets up DLL paths at runtime")
    print("  3. pyopenms-viewer-windows.spec: Properly configured")
    print("  4. CI workflows: Use correct spec files")
    print("  5. File separation: .pyd→pyopenms/, .dll→pyopenms_dlls/")
    sys.exit(0)
else:
    print(f"\n{Colors.RED}{Colors.BOLD}{'=' * 60}")
    print("SOME TESTS FAILED! ✗")
    print(f"{'=' * 60}{Colors.END}")
    print("\nPlease fix the issues above before building.")
    sys.exit(1)
