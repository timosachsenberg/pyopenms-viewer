"""Command-line interface for pyopenms-viewer.

This module provides the Click-based CLI for launching the viewer.
"""

import os
import sys
from pathlib import Path

import click

# Fix for PyInstaller windowed mode
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

_PYINSTALLER_ARG_EXACT = {"-B", "-S", "-I"}
_PYINSTALLER_ARG_PREFIXES = ("-psn_", "--multiprocessing-", "tracker_fd=", "pipe_handle=")


def _sanitize_pyinstaller_args(argv: list[str]) -> list[str]:
    """Strip arguments injected by macOS launch services or multiprocessing.

    PyInstaller GUI binaries on macOS receive synthetic flags such as ``-psn_*``
    from LaunchServices. When multiprocessing spins up helper processes it also
    re-executes the entry point with ``--multiprocessing-*`` and ``-B``. These
    confuse Click's argument parser and previously caused the app to abort
    immediately. We filter them out before Click inspects ``sys.argv``.
    """

    if len(argv) <= 1:
        return argv

    sanitized = [argv[0]]
    for arg in argv[1:]:
        if arg in _PYINSTALLER_ARG_EXACT:
            continue
        if arg.startswith(_PYINSTALLER_ARG_PREFIXES):
            continue
        sanitized.append(arg)
    log_path = os.environ.get("PYOPENMS_VIEWER_DEBUG_ARGS")
    if log_path:
        try:
            with Path(log_path).expanduser().open("a", encoding="utf-8") as fh:
                fh.write(f"original={argv!r}\n sanitized={sanitized!r}\n")
        except OSError:
            pass
    return sanitized


def _run_embedded_python_snippet(argv: list[str]) -> bool:
    """Execute ``python -c ...`` style invocations used by multiprocessing.

    When PyInstaller re-executes the frozen app to launch helpers such as the
    multiprocessing resource tracker it passes ``-c" <code>``. Running Click in
    that situation is wrong; instead we execute the snippet just like the
    regular interpreter would and exit immediately.
    """

    if len(argv) >= 3 and argv[1] == "-c":
        code = argv[2]
        exec(compile(code, "<pyinstaller-cmd>", "exec"), {"__name__": "__main__"})
        return True
    return False

# Set OpenMP threads for pyOpenMS
cpu_count = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(cpu_count))

# Global for CLI files and options (loaded after UI starts)
_cli_files = {"mzml": None, "featurexml": None, "idxml": None}
_cli_options = {"out_of_core": False, "cache_dir": None}


def get_cli_files() -> dict:
    """Get CLI files to load on startup."""
    return _cli_files


def get_cli_options() -> dict:
    """Get CLI options for application configuration."""
    return _cli_options


def _check_native_available() -> bool:
    """Check if pywebview is available for native mode."""
    try:
        import webview  # noqa: F401

        return True
    except ImportError:
        return False


@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--port", "-p", default=8080, help="Port to run the server on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to bind to")
@click.option("--open/--no-open", "-o/-n", "open_browser", default=True, help="Open browser automatically")
@click.option("--native/--browser", default=True, help="Run in native window (default) or browser mode")
@click.option("--dark/--light", default=True, help="Use dark mode (default) or light mode")
@click.option(
    "--out-of-core/--in-memory",
    default=False,
    help="Use disk-based caching for large datasets (reduces RAM usage)",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=None,
    help="Directory for cache files (default: temp directory)",
)
def main(files, port, host, open_browser, native, dark, out_of_core, cache_dir):
    """pyopenms-viewer - Fast visualization of mass spectrometry data.

    Load mzML, featureXML, and idXML files for visualization.

    \b
    Examples:
        pyopenms-viewer                              # Start empty
        pyopenms-viewer sample.mzML                  # Load mzML file
        pyopenms-viewer sample.mzML features.featureXML  # Load with features
        pyopenms-viewer sample.mzML ids.idXML        # Load with IDs
    """
    global _cli_files, _cli_options

    # Store CLI options for app initialization
    _cli_options["out_of_core"] = out_of_core
    _cli_options["cache_dir"] = cache_dir

    # Parse file arguments by extension
    for filepath in files:
        path = Path(filepath)
        ext = path.suffix.lower()
        if ext == ".mzml":
            _cli_files["mzml"] = str(path.absolute())
        elif ext == ".featurexml":
            _cli_files["featurexml"] = str(path.absolute())
        elif ext == ".idxml":
            _cli_files["idxml"] = str(path.absolute())
        else:
            click.echo(f"Warning: Unknown file type: {ext}", err=True)

    # Check native mode availability and fallback if needed
    use_native = native
    if native and not _check_native_available():
        click.echo(
            "Warning: Native mode requested but pywebview is not installed. "
            "Falling back to browser mode.\n"
            "Install native dependencies with: uv sync --extra native",
            err=True,
        )
        use_native = False

    # Import here to avoid circular imports and slow startup for --help
    from nicegui import ui
    from nicegui.core import sio

    import pyopenms_viewer.app  # noqa: F401

    # Increase Socket.IO buffer size for large Plotly figure updates
    # Default is 1MB, increase to 10MB for spectra with many peaks
    sio.eio.max_http_buffer_size = 10 * 1024 * 1024

    # Store dark mode preference
    os.environ["PYOPENMS_VIEWER_DARK_MODE"] = "1" if dark else "0"

    # Run the UI
    ui.run(
        title="pyopenms-viewer",
        host=host,
        port=port,
        reload=False,
        show=open_browser and not use_native,
        native=use_native,
        window_size=(1400, 900) if use_native else None,
        dark=dark,
        reconnect_timeout=60.0,
    )


if __name__ == "__main__":
    main()
