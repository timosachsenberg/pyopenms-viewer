"""Configuration constants, colormaps, and default settings."""

import colorcet as cc
import matplotlib

# Ion type colors for spectrum annotation
ION_COLORS = {
    "b": "#1f77b4",  # Blue
    "y": "#d62728",  # Red
    "a": "#2ca02c",  # Green
    "c": "#9467bd",  # Purple
    "x": "#8c564b",  # Brown
    "z": "#e377c2",  # Pink
    "precursor": "#ff7f0e",  # Orange
    "unknown": "#7f7f7f",  # Gray
}

# Neutral gray for muted plot/UI chrome (axes, gridlines, secondary text);
# mirrors DEFAULTS.AXIS_COLOR (136, 136, 136) as a CSS hex.
NEUTRAL_GRAY_HEX = "#888"

# Available colormaps for peak map visualization
COLORMAPS = {
    "jet": matplotlib.colormaps["jet"],
    "hot": matplotlib.colormaps["hot"],
    "fire": cc.fire,
    "viridis": matplotlib.colormaps["viridis"],
    "plasma": matplotlib.colormaps["plasma"],
    "inferno": matplotlib.colormaps["inferno"],
    "magma": matplotlib.colormaps["magma"],
}


def get_colormap_background(colormap_name: str) -> str:
    """Get the lowest color from a colormap as a hex string for background.

    Args:
        colormap_name: Name of the colormap (e.g., "jet", "viridis")

    Returns:
        Hex color string (e.g., "#000080") or "black" if not found
    """
    cmap = COLORMAPS.get(colormap_name)
    if cmap is None:
        return "black"

    # Check if it's a matplotlib colormap (has __call__ method)
    if callable(cmap):
        # Matplotlib colormap - get color at 0
        rgba = cmap(0)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    elif isinstance(cmap, list) and len(cmap) > 0:
        # Colorcet list - first element is the lowest color
        return cmap[0]
    else:
        return "black"


# Default display settings
class DEFAULTS:
    """Default configuration values."""

    # Image dimensions
    PLOT_WIDTH = 1100
    PLOT_HEIGHT = 550

    # Margins
    MARGIN_LEFT = 80
    MARGIN_RIGHT = 20
    MARGIN_TOP = 20
    MARGIN_BOTTOM = 50

    # Minimap dimensions
    MINIMAP_WIDTH = 400
    MINIMAP_HEIGHT = 200

    # Mobilogram
    MOBILOGRAM_WIDTH = 150

    # Display options
    SHOW_CENTROIDS = True
    SHOW_BOUNDING_BOXES = False
    SHOW_CONVEX_HULLS = False
    SHOW_IDS = True
    SHOW_ID_SEQUENCES = False
    SHOW_SPECTRUM_MARKER = True
    SWAP_AXES = True  # m/z on x-axis, RT on y-axis
    COLORMAP = "jet"
    RT_IN_MINUTES = False
    SPECTRUM_INTENSITY_PERCENT = True
    SPECTRUM_AUTO_SCALE = False
    PEAKMAP_DOWNSAMPLING = False
    ANNOTATE_PEAKS = True
    ANNOTATION_TOLERANCE_DA = 0.05
    MIRROR_ANNOTATION_VIEW = False
    SHOW_UNMATCHED_THEORETICAL = True
    SHOW_ALL_HITS = False

    # 3D view settings
    MAX_3D_PEAKS = 5000
    RT_THRESHOLD_3D = 120.0  # seconds
    MZ_THRESHOLD_3D = 50.0

    # Zoom history
    MAX_ZOOM_HISTORY = 10

    # Out-of-core settings
    OUT_OF_CORE = False  # Enable disk-based caching
    CACHE_DIR = None  # Cache directory (None = temp dir)
    CACHE_COMPRESSION = "snappy"  # Compression: snappy, zstd, gzip, none

    # Performance acceleration settings
    USE_GPU_IF_AVAILABLE = True  # Use cuDF for GPU-accelerated rendering if available
    USE_DASK_IF_AVAILABLE = True  # Use Dask for multi-threaded CPU rendering if available
    DASK_N_PARTITIONS = 4  # Number of Dask partitions (typically set to CPU core count)

    # Rasterization settings
    # If 0: always use rasterization. If non-zero: use point rendering when RT < threshold AND mz < threshold
    DEEP_ZOOM_RT_THRESHOLD = 60.0  # RT range threshold for point rendering (seconds, 0 = always raster)
    DEEP_ZOOM_MZ_THRESHOLD = 50.0  # m/z range threshold for point rendering (0 = always raster)

    # Colors (RGBA tuples)
    CENTROID_COLOR = (0, 255, 100, 255)
    HOVER_COLOR = (255, 200, 0, 255)  # Orange/yellow for hover highlight
    BBOX_COLOR = (255, 255, 0, 200)
    HULL_COLOR = (0, 200, 255, 150)
    SELECTED_COLOR = (255, 100, 255, 255)
    ID_COLOR = (255, 150, 50, 255)
    ID_SELECTED_COLOR = (255, 50, 50, 255)

    # Feature selection
    HOVER_SNAP_DISTANCE_PX = 15  # Pixel threshold for snapping to centroids

    # Neutral gray colors for axes (work on light and dark backgrounds)
    AXIS_COLOR = (136, 136, 136, 255)
    TICK_COLOR = (136, 136, 136, 255)
    LABEL_COLOR = (136, 136, 136, 255)
    GRID_COLOR = (60, 60, 60, 255)


# Panel definitions
PANEL_DEFINITIONS = {
    "tic": {"name": "TIC", "icon": "show_chart"},
    "imaging": {"name": "Ion Image", "icon": "image"},
    "chromatograms": {"name": "Chromatograms", "icon": "timeline"},
    "peakmap": {"name": "2D Peak Map", "icon": "grid_on"},
    "im_peakmap": {"name": "Ion Mobility Frame", "icon": "blur_on"},
    "spectrum": {"name": "1D Spectrum", "icon": "ssid_chart"},
    "spectra_table": {"name": "Spectra", "icon": "list"},
    "features_table": {"name": "Features", "icon": "scatter_plot"},
    "custom_range": {"name": "Custom Range", "icon": "tune"},
    "export": {"name": "Data Export", "icon": "file_download"},
    "log": {"name": "Algorithm Log", "icon": "terminal"},
    "scripting": {"name": "Python Scripting", "icon": "code"},
    "legend": {"name": "Help", "icon": "help"},
}

# Default panel order
DEFAULT_PANEL_ORDER = [
    "tic",
    "imaging",
    "chromatograms",
    "peakmap",
    "im_peakmap",
    "spectrum",
    "spectra_table",
    "features_table",
    "custom_range",
    "export",
    "log",
    "scripting",
    "legend",
]

# Default panel visibility
# True = always show, False = always hide, "auto" = show only when data exists
DEFAULT_PANEL_VISIBILITY = {
    "tic": True,
    "imaging": "auto",
    "chromatograms": "auto",
    "peakmap": True,
    "im_peakmap": "auto",
    "spectrum": True,
    "spectra_table": True,
    "features_table": "auto",
    "custom_range": True,
    "export": "auto",
    "log": "auto",
    "scripting": "auto",
    "legend": True,
}
