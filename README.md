# pyopenms-viewer

Fast mzML peak map viewer using NiceGUI, Datashader, and pyOpenMS.

Designed to handle **50+ million peaks** with smooth zooming and panning using server-side rendering.

## Features

- **Peak Map Visualization** - Datashader-powered rendering for massive datasets
- **FeatureMap Overlay** - Display centroids, bounding boxes, and convex hulls
- **idXML Overlay** - Show peptide identification precursor positions
- **MS2 Spectrum Viewer** - Annotated spectrum viewer for peptide identifications
- **TIC Display** - Total Ion Chromatogram with clickable MS1 spectrum viewer

## Prerequisites

Before installing pyopenms-viewer, ensure you have:

- **Python 3.10 or higher** - Check with `python --version` or `python3 --version`
- **Operating System**: Windows, macOS, or Linux

## Installation

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. If you don't have it installed:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

Once uv is installed, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/timosachsenberg/pyopenms-viewer.git
cd pyopenms-viewer

# Install dependencies
uv sync
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/timosachsenberg/pyopenms-viewer.git
cd pyopenms-viewer

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate

# Install the package
pip install -e .
```

### Optional Extras

```bash
# For native desktop window (instead of browser)
uv sync --extra native
# or with pip
pip install -e ".[native]"

# For development (testing, linting)
uv sync --extra dev
# or with pip
pip install -e ".[dev]"
```

## Quick Start

### Step 1: Start the Viewer

```bash
# Using uv
uv run pyopenms-viewer

# Or if installed with pip (with virtual environment activated)
pyopenms-viewer
```

This opens a web browser at `http://localhost:8080` with an empty viewer.

### Step 2: Load Your Data

You can load files in two ways:

**Via command line:**
```bash
# Load an mzML file
uv run pyopenms-viewer sample.mzML

# Load mzML with feature overlay
uv run pyopenms-viewer sample.mzML features.featureXML

# Load mzML with peptide identifications
uv run pyopenms-viewer sample.mzML ids.idXML

# Load all three file types
uv run pyopenms-viewer sample.mzML features.featureXML ids.idXML
```

**Via the web interface:**
- Drag and drop files onto the upload area
- Supports `.mzML`, `.featureXML`, and `.idXML` files

### Step 3: Navigate the Peak Map

- **Zoom**: Scroll with mouse wheel
- **Pan**: Click and drag
- **Reset View**: Double-click or press `Home`
- **Keyboard Shortcuts**:
  - `+` or `=` : Zoom in
  - `-` : Zoom out
  - `Arrow keys` : Pan

## Command Line Options

```bash
uv run pyopenms-viewer [FILES] [OPTIONS]

Options:
  -p, --port INTEGER     Port to run the server on (default: 8080)
  -H, --host TEXT        Host to bind to (default: 0.0.0.0)
  -o, --open / -n, --no-open
                         Open browser automatically (default: open)
  --native               Run as native desktop app (requires pywebview)
  --help                 Show this message and exit
```

**Examples:**
```bash
# Start on a different port
uv run pyopenms-viewer --port 3000

# Don't open browser automatically
uv run pyopenms-viewer --no-open sample.mzML

# Run as native desktop application
uv run pyopenms-viewer --native sample.mzML
```

## Development

```bash
# Run tests
uv run pytest

# Lint code
uv run ruff check .

# Format code
uv run ruff format .
```

## Troubleshooting

### Common Issues

**"uv: command not found"**
- Restart your terminal after installing uv, or add it to your PATH manually

**"Python 3.10+ required"**
- Install a newer Python version from [python.org](https://www.python.org/downloads/) or use pyenv

**"Port 8080 already in use"**
- Use a different port: `uv run pyopenms-viewer --port 3000`

**"pywebview not found" (when using --native)**
- Install the native extra: `uv sync --extra native`

**Slow loading with large files**
- Large mzML files (>1GB) may take time to load initially
- Progress is shown in the status bar

### Getting Help

- Check existing [GitHub Issues](https://github.com/timosachsenberg/pyopenms-viewer/issues)
- Open a new issue with your error message and Python version

## License

BSD-3-Clause
