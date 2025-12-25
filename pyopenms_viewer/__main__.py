"""Entry point for running pyopenms-viewer as a module.

Usage:
    python -m pyopenms_viewer [options] [files...]
"""

import sys
from multiprocessing import freeze_support

from pyopenms_viewer.cli import (
    _run_embedded_python_snippet,
    _sanitize_pyinstaller_args,
    main,
)


if __name__ == "__main__":
    freeze_support()
    argv = _sanitize_pyinstaller_args(sys.argv)
    if _run_embedded_python_snippet(argv):
        sys.exit(0)
    sys.argv[:] = argv
    main()
