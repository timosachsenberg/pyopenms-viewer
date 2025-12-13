"""Central state container for pyopenms-viewer.

This module defines ViewerState, the single source of truth for all data
in the application. All components receive a reference to the same ViewerState
instance, ensuring memory efficiency for large datasets (10GB+).

MEMORY SAFETY: Data structures are stored as references, never copied.
Components access data via properties that return references or views (masks).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pandas as pd

from pyopenms_viewer.core.config import (
    DEFAULT_PANEL_ORDER,
    DEFAULT_PANEL_VISIBILITY,
    DEFAULTS,
    PANEL_DEFINITIONS,
)
from pyopenms_viewer.core.events import EventBus

if TYPE_CHECKING:
    from pyopenms_viewer.core.data_manager import DataManager


@dataclass
class ViewBounds:
    """Current view bounds for RT/m/z/IM ranges."""

    rt_min: float = 0.0
    rt_max: float = 1.0
    mz_min: float = 0.0
    mz_max: float = 1.0
    im_min: float = 0.0
    im_max: float = 1.0


@dataclass
class DataBounds:
    """Full data extent (set once after loading, never changes)."""

    rt_min: float = 0.0
    rt_max: float = 1.0
    mz_min: float = 0.0
    mz_max: float = 1.0
    im_min: float = 0.0
    im_max: float = 1.0


class ViewerState:
    """Central state container holding all shared data.

    MEMORY SAFETY: All data structures are stored as references.
    Components access data via properties that return references or views (masks).
    No data is ever copied when passing to panels.

    Attributes are organized into groups:
    - Primary data (large, never copied)
    - Metadata (small, safe to access)
    - View bounds
    - Selection state
    - Display options
    - Event bus

    Example usage:
        state = ViewerState()
        state.on_data_loaded(lambda dt: print(f"Loaded: {dt}"))

        # Load data (sets state.exp, state.df, etc.)
        loader = MzMLLoader(state)
        loader.load_sync("data.mzML")

        # Get peaks in current view (returns view, not copy)
        view_df = state.get_peaks_in_view()
    """

    def __init__(self):
        # ========== DATA MANAGER (unified DuckDB interface) ==========
        self.data_manager: Optional[DataManager] = None
        self.out_of_core: bool = DEFAULTS.OUT_OF_CORE
        self._cache_dir: Optional[Path] = None

        # ========== PRIMARY DATA (NEVER COPIED) ==========
        # These are the large data structures that must be shared
        # In out-of-core mode, df and im_df may be None after registration
        self.exp = None  # MSExperiment - pyOpenMS C++ object (~500MB)
        self.df: Optional[pd.DataFrame] = None  # All peaks: rt, mz, intensity, log_intensity (~2GB)
        self.im_df: Optional[pd.DataFrame] = None  # Ion mobility peaks: mz, im, intensity, log_intensity
        self.chromatogram_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # idx -> (rt, intensity) arrays

        # ========== OVERLAY DATA ==========
        self.feature_map = None  # PyOpenMS FeatureMap object
        self.peptide_ids: list = []  # List of PeptideIdentification objects
        self.protein_ids: list = []  # List of ProteinIdentification objects

        # ========== METADATA (small, safe to access) ==========
        self.spectrum_data: list[dict] = []  # Spectrum metadata for table (~10MB)
        self.feature_data: list[dict] = []  # Feature metadata for table
        self.id_data: list[dict] = []  # ID metadata for table
        self.id_meta_keys: list[str] = []  # Discovered meta value keys
        self.chromatograms: list[dict] = []  # Chromatogram metadata

        # ========== TIC DATA ==========
        self.tic_rt: Optional[np.ndarray] = None
        self.tic_intensity: Optional[np.ndarray] = None
        self.tic_source: str = "MS1 TIC"  # Description (e.g., "MS1 TIC", "MS2 BPC")
        self.faims_tic: dict[float, tuple[np.ndarray, np.ndarray]] = {}  # CV -> (rt, intensity)

        # ========== DATA BOUNDS (full extent) ==========
        self.rt_min: float = 0.0
        self.rt_max: float = 1.0
        self.mz_min: float = 0.0
        self.mz_max: float = 1.0
        self.im_min: float = 0.0
        self.im_max: float = 1.0

        # ========== VIEW BOUNDS (current view) ==========
        self.view_rt_min: Optional[float] = None
        self.view_rt_max: Optional[float] = None
        self.view_mz_min: Optional[float] = None
        self.view_mz_max: Optional[float] = None
        self.view_im_min: Optional[float] = None
        self.view_im_max: Optional[float] = None

        # ========== FLAGS ==========
        self.has_faims: bool = False
        self.has_ion_mobility: bool = False
        self.has_chromatograms: bool = False
        self.faims_cvs: list[float] = []
        self.im_type: Optional[str] = None  # "ion mobility", "inverse reduced ion mobility", etc.
        self.im_unit: str = ""
        self.show_faims_view: bool = False
        self.selected_faims_cv: Optional[float] = None  # Currently selected CV for filtering peak map

        # ========== FILE PATHS ==========
        self.current_file: Optional[str] = None
        self.features_file: Optional[str] = None
        self.id_file: Optional[str] = None

        # ========== SELECTION STATE ==========
        self.selected_spectrum_idx: Optional[int] = None
        self.selected_feature_idx: Optional[int] = None
        self.selected_id_idx: Optional[int] = None
        self.selected_chromatogram_indices: list[int] = []
        self.hover_feature_idx: Optional[int] = None
        self.hover_id_idx: Optional[int] = None

        # ========== SPECTRUM MEASUREMENT STATE ==========
        self.spectrum_measure_mode: bool = False
        self.spectrum_measure_start: Optional[tuple[float, float]] = None
        self.spectrum_measurements: dict[int, list] = {}  # spectrum_idx -> list of measurements
        self.spectrum_hover_peak: Optional[tuple[float, float]] = None
        self.spectrum_selected_measurement_idx: Optional[int] = None
        self.spectrum_dragging: bool = False
        self.spectrum_zoom_range: Optional[tuple[float, float]] = None

        # ========== PEAK ANNOTATION STATE ==========
        self.peak_annotations: dict[int, list[dict]] = {}  # spectrum_idx -> list of annotations
        self.peak_annotation_mode: bool = False
        self.show_mz_labels: bool = False

        # ========== ZOOM HISTORY ==========
        self.zoom_history: list[tuple] = []  # List of (rt_min, rt_max, mz_min, mz_max, label)
        self.max_zoom_history: int = DEFAULTS.MAX_ZOOM_HISTORY

        # ========== DISPLAY OPTIONS ==========
        self.show_centroids: bool = DEFAULTS.SHOW_CENTROIDS
        self.show_bounding_boxes: bool = DEFAULTS.SHOW_BOUNDING_BOXES
        self.show_convex_hulls: bool = DEFAULTS.SHOW_CONVEX_HULLS
        self.show_ids: bool = DEFAULTS.SHOW_IDS
        self.show_id_sequences: bool = DEFAULTS.SHOW_ID_SEQUENCES
        self.show_spectrum_marker: bool = DEFAULTS.SHOW_SPECTRUM_MARKER
        self.swap_axes: bool = DEFAULTS.SWAP_AXES
        self.colormap: str = DEFAULTS.COLORMAP
        self.rt_in_minutes: bool = DEFAULTS.RT_IN_MINUTES
        self.spectrum_intensity_percent: bool = DEFAULTS.SPECTRUM_INTENSITY_PERCENT
        self.spectrum_auto_scale: bool = DEFAULTS.SPECTRUM_AUTO_SCALE
        self.peakmap_downsampling: bool = DEFAULTS.PEAKMAP_DOWNSAMPLING
        self.annotate_peaks: bool = DEFAULTS.ANNOTATE_PEAKS
        self.annotation_tolerance_da: float = DEFAULTS.ANNOTATION_TOLERANCE_DA
        self.mirror_annotation_view: bool = DEFAULTS.MIRROR_ANNOTATION_VIEW
        self.show_unmatched_theoretical: bool = DEFAULTS.SHOW_UNMATCHED_THEORETICAL
        self.show_all_hits: bool = DEFAULTS.SHOW_ALL_HITS
        self.link_spectrum_mz_to_im: bool = False
        self.show_mobilogram: bool = True

        # ========== 3D VIEW OPTIONS ==========
        self.show_3d_view: bool = False
        self.max_3d_peaks: int = DEFAULTS.MAX_3D_PEAKS
        self.rt_threshold_3d: float = DEFAULTS.RT_THRESHOLD_3D
        self.mz_threshold_3d: float = DEFAULTS.MZ_THRESHOLD_3D

        # ========== COLORS ==========
        self.centroid_color: tuple = DEFAULTS.CENTROID_COLOR
        self.hover_color: tuple = DEFAULTS.HOVER_COLOR
        self.bbox_color: tuple = DEFAULTS.BBOX_COLOR
        self.hull_color: tuple = DEFAULTS.HULL_COLOR
        self.selected_color: tuple = DEFAULTS.SELECTED_COLOR
        self.id_color: tuple = DEFAULTS.ID_COLOR
        self.id_selected_color: tuple = DEFAULTS.ID_SELECTED_COLOR
        self.axis_color: tuple = DEFAULTS.AXIS_COLOR
        self.tick_color: tuple = DEFAULTS.TICK_COLOR
        self.label_color: tuple = DEFAULTS.LABEL_COLOR
        self.grid_color: tuple = DEFAULTS.GRID_COLOR

        # ========== FEATURE SELECTION ==========
        self.hover_snap_distance_px: int = DEFAULTS.HOVER_SNAP_DISTANCE_PX

        # ========== IMAGE DIMENSIONS ==========
        self.plot_width: int = DEFAULTS.PLOT_WIDTH
        self.plot_height: int = DEFAULTS.PLOT_HEIGHT
        self.margin_left: int = DEFAULTS.MARGIN_LEFT
        self.margin_right: int = DEFAULTS.MARGIN_RIGHT
        self.margin_top: int = DEFAULTS.MARGIN_TOP
        self.margin_bottom: int = DEFAULTS.MARGIN_BOTTOM
        self.minimap_width: int = DEFAULTS.MINIMAP_WIDTH
        self.minimap_height: int = DEFAULTS.MINIMAP_HEIGHT
        self.mobilogram_plot_width: int = DEFAULTS.MOBILOGRAM_WIDTH

        # ========== PANEL CONFIGURATION ==========
        self.panel_definitions: dict = PANEL_DEFINITIONS.copy()
        self.panel_order: list[str] = DEFAULT_PANEL_ORDER.copy()
        self.panel_visibility: dict = DEFAULT_PANEL_VISIBILITY.copy()

        # ========== EVENT BUS ==========
        self._event_bus = EventBus()

        # ========== UI UPDATE FLAGS ==========
        self._updating_from_tic: bool = False
        self._hover_update_pending: bool = False

    # ========== COMPUTED PROPERTIES ==========

    @property
    def canvas_width(self) -> int:
        """Total canvas width including margins."""
        return self.plot_width + self.margin_left + self.margin_right

    @property
    def canvas_height(self) -> int:
        """Total canvas height including margins."""
        return self.plot_height + self.margin_top + self.margin_bottom

    # ========== DATA MANAGER METHODS ==========

    def init_data_manager(
        self,
        out_of_core: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the data manager for unified DuckDB queries.

        This should be called early in application startup, before loading data.

        Args:
            out_of_core: If True, cache data to Parquet files on disk.
                        If False, register DataFrames directly with DuckDB.
            cache_dir: Directory for cache files. If None, uses a temp directory.
        """
        from pyopenms_viewer.core.data_manager import DataManager

        self.out_of_core = out_of_core
        self._cache_dir = Path(cache_dir) if cache_dir else None

        # Clean up existing data manager if present
        if self.data_manager is not None:
            self.data_manager.cleanup()

        self.data_manager = DataManager(
            out_of_core=out_of_core,
            cache_dir=self._cache_dir,
            compression=DEFAULTS.CACHE_COMPRESSION,
        )

    def get_cache_size_mb(self) -> float:
        """Get current cache size in megabytes.

        Returns:
            Cache size in MB, or 0 if not in out-of-core mode
        """
        if self.data_manager is None:
            return 0.0
        return self.data_manager.get_cache_size_mb()

    # ========== VIEW ACCESSORS (return views, not copies) ==========

    def get_peaks_in_view(self) -> pd.DataFrame:
        """Return peaks in current view bounds via unified DuckDB query.

        This method provides unified access to peak data regardless of storage mode.
        All queries go through DuckDB for consistency and simpler code paths:

        IN-MEMORY MODE: DuckDB queries zero-copy registered DataFrame (~30ms for 5M peaks)
        OUT-OF-CORE MODE: DuckDB queries from Parquet files (~74ms for 5M peaks)

        Returns:
            DataFrame of peaks within current RT/m/z bounds
        """
        rt_min = self.view_rt_min if self.view_rt_min is not None else self.rt_min
        rt_max = self.view_rt_max if self.view_rt_max is not None else self.rt_max
        mz_min = self.view_mz_min if self.view_mz_min is not None else self.mz_min
        mz_max = self.view_mz_max if self.view_mz_max is not None else self.mz_max

        if self.data_manager is None:
            return pd.DataFrame()

        cv = self.selected_faims_cv if self.has_faims else None
        return self.data_manager.query_peaks_in_view(rt_min, rt_max, mz_min, mz_max, cv)

    def get_im_peaks_in_view(self) -> pd.DataFrame:
        """Return ion mobility peaks in current view bounds via unified DuckDB query.

        This method provides unified access to IM peak data regardless of storage mode.
        All queries go through DuckDB for consistency and simpler code paths.

        Returns:
            DataFrame of ion mobility peaks within current m/z/IM bounds
        """
        mz_min = self.view_mz_min if self.view_mz_min is not None else self.mz_min
        mz_max = self.view_mz_max if self.view_mz_max is not None else self.mz_max
        im_min = self.view_im_min if self.view_im_min is not None else self.im_min
        im_max = self.view_im_max if self.view_im_max is not None else self.im_max

        if self.data_manager is None:
            return pd.DataFrame()

        return self.data_manager.query_im_peaks_in_view(mz_min, mz_max, im_min, im_max)

    def get_view_bounds(self) -> ViewBounds:
        """Get current view bounds as a ViewBounds object."""
        return ViewBounds(
            rt_min=self.view_rt_min if self.view_rt_min is not None else self.rt_min,
            rt_max=self.view_rt_max if self.view_rt_max is not None else self.rt_max,
            mz_min=self.view_mz_min if self.view_mz_min is not None else self.mz_min,
            mz_max=self.view_mz_max if self.view_mz_max is not None else self.mz_max,
            im_min=self.view_im_min if self.view_im_min is not None else self.im_min,
            im_max=self.view_im_max if self.view_im_max is not None else self.im_max,
        )

    def get_data_bounds(self) -> DataBounds:
        """Get full data bounds as a DataBounds object."""
        return DataBounds(
            rt_min=self.rt_min,
            rt_max=self.rt_max,
            mz_min=self.mz_min,
            mz_max=self.mz_max,
            im_min=self.im_min,
            im_max=self.im_max,
        )

    # ========== VIEW MANIPULATION ==========

    def set_view(
        self,
        rt_min: Optional[float] = None,
        rt_max: Optional[float] = None,
        mz_min: Optional[float] = None,
        mz_max: Optional[float] = None,
        im_min: Optional[float] = None,
        im_max: Optional[float] = None,
        emit_event: bool = True,
    ) -> None:
        """Set view bounds.

        Args:
            rt_min, rt_max, mz_min, mz_max, im_min, im_max: New bounds (None = keep current)
            emit_event: If True, emit view_changed event
        """
        if rt_min is not None:
            self.view_rt_min = rt_min
        if rt_max is not None:
            self.view_rt_max = rt_max
        if mz_min is not None:
            self.view_mz_min = mz_min
        if mz_max is not None:
            self.view_mz_max = mz_max
        if im_min is not None:
            self.view_im_min = im_min
        if im_max is not None:
            self.view_im_max = im_max

        if emit_event:
            self.emit_view_changed()

    def reset_view(self, emit_event: bool = True) -> None:
        """Reset view to full data extent."""
        self.view_rt_min = self.rt_min
        self.view_rt_max = self.rt_max
        self.view_mz_min = self.mz_min
        self.view_mz_max = self.mz_max
        self.view_im_min = self.im_min
        self.view_im_max = self.im_max

        if emit_event:
            self.emit_view_changed()

    # ========== PANEL VISIBILITY ==========

    def should_panel_be_visible(self, panel_id: str) -> bool:
        """Determine if a panel should be visible based on visibility setting and data.

        Args:
            panel_id: The panel identifier (e.g., "tic", "im_peakmap")

        Returns:
            True if panel should be visible
        """
        visibility = self.panel_visibility.get(panel_id, True)

        if visibility is True:
            return True
        elif visibility is False:
            return False
        elif visibility == "auto":
            # Auto-visibility based on data availability
            if panel_id == "im_peakmap":
                return self.has_ion_mobility
            elif panel_id == "chromatograms":
                return self.has_chromatograms
            elif panel_id == "features_table":
                return len(self.feature_data) > 0
            else:
                return True
        return True

    # ========== EVENT BUS DELEGATION ==========

    def on_data_loaded(self, callback: Callable) -> Callable:
        """Register a callback for when data is loaded (mzML, features, or IDs).

        Callback signature: callback(data_type: str)
        """
        return self._event_bus.subscribe("data_loaded", callback)

    def on_view_changed(self, callback: Callable) -> Callable:
        """Register a callback for when the view (zoom/pan) changes.

        Callback signature: callback()
        """
        return self._event_bus.subscribe("view_changed", callback)

    def on_selection_changed(self, callback: Callable) -> Callable:
        """Register a callback for when selection (spectrum, feature, ID) changes.

        Callback signature: callback(selection_type: str, index: int | None)
        """
        return self._event_bus.subscribe("selection_changed", callback)

    def on_display_options_changed(self, callback: Callable) -> Callable:
        """Register a callback for when display options change.

        Callback signature: callback(option_name: str, value: Any)
        """
        return self._event_bus.subscribe("display_options_changed", callback)

    def emit_data_loaded(self, data_type: str) -> None:
        """Emit data loaded event.

        Args:
            data_type: "mzml", "features", or "ids"
        """
        self._event_bus.emit("data_loaded", data_type=data_type)

    def emit_view_changed(self) -> None:
        """Emit view changed event."""
        self._event_bus.emit("view_changed")

    def emit_selection_changed(self, selection_type: str, index: Optional[int]) -> None:
        """Emit selection changed event.

        Args:
            selection_type: "spectrum", "feature", or "id"
            index: Index of selected item, or None if deselected
        """
        self._event_bus.emit("selection_changed", selection_type=selection_type, index=index)

    def emit_display_options_changed(self, option_name: str, value: Any) -> None:
        """Emit display options changed event.

        Args:
            option_name: Name of the option that changed
            value: New value
        """
        self._event_bus.emit("display_options_changed", option_name=option_name, value=value)

    # ========== SELECTION HELPERS ==========

    def select_spectrum(self, index: Optional[int], emit_event: bool = True) -> None:
        """Select a spectrum by index.

        Args:
            index: Spectrum index, or None to deselect
            emit_event: If True, emit selection_changed event
        """
        self.selected_spectrum_idx = index
        if emit_event:
            self.emit_selection_changed("spectrum", index)

    def select_feature(self, index: Optional[int], emit_event: bool = True) -> None:
        """Select a feature by index.

        Args:
            index: Feature index, or None to deselect
            emit_event: If True, emit selection_changed event
        """
        self.selected_feature_idx = index
        if emit_event:
            self.emit_selection_changed("feature", index)

    def select_id(self, index: Optional[int], emit_event: bool = True) -> None:
        """Select an identification by index.

        Args:
            index: ID index, or None to deselect
            emit_event: If True, emit selection_changed event
        """
        self.selected_id_idx = index
        if emit_event:
            self.emit_selection_changed("id", index)

    # ========== DATA CLEARING ==========

    def clear_mzml_data(self) -> None:
        """Clear all mzML-related data."""
        # Clear data manager cache
        if self.data_manager is not None:
            self.data_manager.clear()

        self.exp = None
        self.df = None
        self.current_file = None
        self.tic_rt = None
        self.tic_intensity = None
        self.tic_source = "MS1 TIC"
        self.spectrum_data = []
        self.chromatograms = []
        self.chromatogram_data = {}
        self.selected_chromatogram_indices = []
        self.has_chromatograms = False
        self.im_df = None
        self.has_ion_mobility = False
        self.im_type = None
        self.im_unit = ""
        self.faims_cvs = []
        self.faims_tic = {}
        self.has_faims = False
        self.show_faims_view = False
        self.selected_faims_cv = None
        self.selected_spectrum_idx = None
        self.zoom_history = []
        self.spectrum_measurements = {}
        self.peak_annotations = {}

    def clear_feature_data(self) -> None:
        """Clear feature-related data."""
        self.feature_map = None
        self.features_file = None
        self.feature_data = []
        self.selected_feature_idx = None
        self.hover_feature_idx = None

    def clear_id_data(self) -> None:
        """Clear identification-related data."""
        # Store meta keys before clearing (for clearing from spectrum_data)
        old_meta_keys = self.id_meta_keys.copy()

        self.peptide_ids = []
        self.protein_ids = []
        self.id_file = None
        self.id_data = []
        self.id_meta_keys = []
        self.selected_id_idx = None
        self.hover_id_idx = None

        # Clear ID linkage from spectrum data
        for spec_row in self.spectrum_data:
            spec_row["sequence"] = "-"
            spec_row["full_sequence"] = ""
            spec_row["score"] = "-"
            spec_row["id_idx"] = None
            spec_row["hit_rank"] = "-"
            spec_row["all_hits"] = []
            # Clear meta value fields
            for meta_key in old_meta_keys:
                if meta_key in spec_row:
                    spec_row[meta_key] = "-"

    def clear_all(self) -> None:
        """Clear all data."""
        self.clear_mzml_data()
        self.clear_feature_data()
        self.clear_id_data()

    # ========== UI ELEMENT REFERENCES ==========
    # These are set by panels when they build their UI
    # Panels can use these to coordinate updates

    @property
    def panel_elements(self) -> dict:
        """Dictionary of panel expansion elements."""
        if not hasattr(self, "_panel_elements"):
            self._panel_elements = {}
        return self._panel_elements

    @panel_elements.setter
    def panel_elements(self, value: dict) -> None:
        self._panel_elements = value

    # These attributes are set by panels
    image_element: Any = None  # Main peak map image
    minimap_image: Any = None  # Minimap image
    coord_label: Any = None  # Coordinate display label
    breadcrumb_label: Any = None  # Breadcrumb navigation label
    scene_3d_container: Any = None  # 3D view container
    plot_3d: Any = None  # 3D plot element
    view_3d_status: Any = None  # 3D view status label
    dark: Any = None  # Dark mode toggle

    # ========== ZOOM HISTORY METHODS ==========

    def push_zoom_history(self) -> None:
        """Save current view state to zoom history."""
        if self.view_rt_min is None:
            return

        # Create label for this view state
        rt_range = self.view_rt_max - self.view_rt_min
        mz_range = self.view_mz_max - self.view_mz_min
        full_rt = self.rt_max - self.rt_min
        full_mz = self.mz_max - self.mz_min

        # Check if this is approximately full view
        if rt_range >= full_rt * 0.95 and mz_range >= full_mz * 0.95:
            label = "Full"
        else:
            label = f"RT {self.view_rt_min:.0f}-{self.view_rt_max:.0f}"

        state = (self.view_rt_min, self.view_rt_max, self.view_mz_min, self.view_mz_max, label)

        # Don't add if same as last entry
        if self.zoom_history and self.zoom_history[-1][:4] == state[:4]:
            return

        self.zoom_history.append(state)

        # Limit history size
        if len(self.zoom_history) > self.max_zoom_history:
            self.zoom_history = self.zoom_history[-self.max_zoom_history :]

    def go_to_zoom_history(self, index: int, emit_event: bool = False) -> None:
        """Jump to a specific point in zoom history.

        Args:
            index: History index to jump to
            emit_event: If True, emit view_changed event
        """
        if index < 0 or index >= len(self.zoom_history):
            return

        state = self.zoom_history[index]
        self.view_rt_min, self.view_rt_max, self.view_mz_min, self.view_mz_max, _ = state

        # Truncate history to this point (forward history is lost)
        self.zoom_history = self.zoom_history[: index + 1]

        if emit_event:
            self.emit_view_changed()

    def zoom_at_point(self, x_frac: float, y_frac: float, zoom_in: bool = True, emit_event: bool = False) -> None:
        """Zoom centered on a specific point (given as fraction of plot area).

        Args:
            x_frac: Horizontal position (0=left, 1=right) in plot area
            y_frac: Vertical position (0=top, 1=bottom) in plot area
            zoom_in: True to zoom in, False to zoom out
            emit_event: If True, emit view_changed event
        """
        # Check if we have data
        if self.data_manager is None or self.data_manager.get_peak_count() == 0:
            return

        # Save current state to zoom history
        self.push_zoom_history()

        # Current ranges
        rt_range = self.view_rt_max - self.view_rt_min
        mz_range = self.view_mz_max - self.view_mz_min

        # Zoom factor
        factor = 0.7 if zoom_in else 1.4

        # New ranges
        new_rt_range = rt_range * factor
        new_mz_range = mz_range * factor

        if self.swap_axes:
            # swap_axes=True: m/z on x-axis, RT on y-axis
            mz_point = self.view_mz_min + x_frac * mz_range
            rt_point = self.view_rt_max - y_frac * rt_range  # Y is inverted

            # Keep the point under cursor at same position
            new_mz_min = mz_point - x_frac * new_mz_range
            new_mz_max = mz_point + (1 - x_frac) * new_mz_range
            new_rt_min = rt_point - (1 - y_frac) * new_rt_range
            new_rt_max = rt_point + y_frac * new_rt_range
        else:
            # swap_axes=False: RT on x-axis, m/z on y-axis
            rt_point = self.view_rt_min + x_frac * rt_range
            mz_point = self.view_mz_max - y_frac * mz_range  # Y is inverted

            # Keep the point under cursor at same position
            new_rt_min = rt_point - x_frac * new_rt_range
            new_rt_max = rt_point + (1 - x_frac) * new_rt_range
            new_mz_min = mz_point - (1 - y_frac) * new_mz_range
            new_mz_max = mz_point + y_frac * new_mz_range

        # Clamp to data bounds
        self.view_rt_min = max(self.rt_min, new_rt_min)
        self.view_rt_max = min(self.rt_max, new_rt_max)
        self.view_mz_min = max(self.mz_min, new_mz_min)
        self.view_mz_max = min(self.mz_max, new_mz_max)

        if emit_event:
            self.emit_view_changed()

    def minimap_click_to_view(self, x_frac: float, y_frac: float, emit_event: bool = False) -> None:
        """Center the main view on the clicked position in minimap.

        Args:
            x_frac: Horizontal fraction (0-1) of minimap click
            y_frac: Vertical fraction (0-1) of minimap click
            emit_event: If True, emit view_changed event
        """
        # Check if we have data (in-memory or out-of-core mode)
        if self.data_manager is None or self.data_manager.get_peak_count() == 0:
            return

        # Convert minimap fractions to data coordinates (depends on axis orientation)
        if self.swap_axes:
            # m/z on x-axis, RT on y-axis
            mz_click = self.mz_min + x_frac * (self.mz_max - self.mz_min)
            rt_click = self.rt_max - y_frac * (self.rt_max - self.rt_min)
        else:
            # RT on x-axis, m/z on y-axis (traditional)
            rt_click = self.rt_min + x_frac * (self.rt_max - self.rt_min)
            mz_click = self.mz_max - y_frac * (self.mz_max - self.mz_min)

        # Center current view on this point
        rt_half_range = (self.view_rt_max - self.view_rt_min) / 2
        mz_half_range = (self.view_mz_max - self.view_mz_min) / 2

        new_rt_min = rt_click - rt_half_range
        new_rt_max = rt_click + rt_half_range
        new_mz_min = mz_click - mz_half_range
        new_mz_max = mz_click + mz_half_range

        # Clamp to data bounds
        if new_rt_min < self.rt_min:
            new_rt_max += self.rt_min - new_rt_min
            new_rt_min = self.rt_min
        if new_rt_max > self.rt_max:
            new_rt_min -= new_rt_max - self.rt_max
            new_rt_max = self.rt_max

        if new_mz_min < self.mz_min:
            new_mz_max += self.mz_min - new_mz_min
            new_mz_min = self.mz_min
        if new_mz_max > self.mz_max:
            new_mz_min -= new_mz_max - self.mz_max
            new_mz_max = self.mz_max

        # Final clamp
        self.view_rt_min = max(self.rt_min, new_rt_min)
        self.view_rt_max = min(self.rt_max, new_rt_max)
        self.view_mz_min = max(self.mz_min, new_mz_min)
        self.view_mz_max = min(self.mz_max, new_mz_max)

        if emit_event:
            self.emit_view_changed()

    def pixel_to_data_coords(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        """Convert pixel coordinates to RT/m/z data coordinates.

        Args:
            pixel_x: Pixel x coordinate
            pixel_y: Pixel y coordinate

        Returns:
            Tuple of (rt, mz) data coordinates
        """
        # Account for margins
        plot_x = pixel_x - self.margin_left
        plot_y = pixel_y - self.margin_top

        # Clamp to plot area
        plot_x = max(0, min(self.plot_width, plot_x))
        plot_y = max(0, min(self.plot_height, plot_y))

        # Convert to data coordinates
        rt = self.view_rt_min + (plot_x / self.plot_width) * (self.view_rt_max - self.view_rt_min)
        mz = self.view_mz_max - (plot_y / self.plot_height) * (self.view_mz_max - self.view_mz_min)

        return rt, mz

    # ========== ZOOM/PAN HELPERS ==========

    def zoom_in(self, emit_event: bool = True) -> None:
        """Zoom in by 10% on all axes."""
        # Check if we have data (in-memory or out-of-core mode)
        if self.data_manager is None or self.data_manager.get_peak_count() == 0:
            return

        rt_range = (self.view_rt_max - self.view_rt_min) * 0.1
        mz_range = (self.view_mz_max - self.view_mz_min) * 0.1

        self.view_rt_min += rt_range
        self.view_rt_max -= rt_range
        self.view_mz_min += mz_range
        self.view_mz_max -= mz_range

        if emit_event:
            self.emit_view_changed()

    def zoom_out(self, emit_event: bool = True) -> None:
        """Zoom out by 10% on all axes."""
        # Check if we have data (in-memory or out-of-core mode)
        if self.data_manager is None or self.data_manager.get_peak_count() == 0:
            return

        rt_range = (self.view_rt_max - self.view_rt_min) * 0.1
        mz_range = (self.view_mz_max - self.view_mz_min) * 0.1

        self.view_rt_min = max(self.rt_min, self.view_rt_min - rt_range)
        self.view_rt_max = min(self.rt_max, self.view_rt_max + rt_range)
        self.view_mz_min = max(self.mz_min, self.view_mz_min - mz_range)
        self.view_mz_max = min(self.mz_max, self.view_mz_max + mz_range)

        if emit_event:
            self.emit_view_changed()

    def pan(self, rt_frac: float = 0.0, mz_frac: float = 0.0, emit_event: bool = True) -> None:
        """Pan the view by a fraction of the current range.

        Args:
            rt_frac: Fraction of RT range to pan (positive = right)
            mz_frac: Fraction of m/z range to pan (positive = up)
            emit_event: If True, emit view_changed event
        """
        # Check if we have data (in-memory or out-of-core mode)
        if self.data_manager is None or self.data_manager.get_peak_count() == 0:
            return

        rt_range = self.view_rt_max - self.view_rt_min
        mz_range = self.view_mz_max - self.view_mz_min

        rt_shift = rt_range * rt_frac
        mz_shift = mz_range * mz_frac

        new_rt_min = self.view_rt_min + rt_shift
        new_rt_max = self.view_rt_max + rt_shift
        new_mz_min = self.view_mz_min + mz_shift
        new_mz_max = self.view_mz_max + mz_shift

        # Clamp to data bounds
        if new_rt_min < self.rt_min:
            new_rt_max += self.rt_min - new_rt_min
            new_rt_min = self.rt_min
        if new_rt_max > self.rt_max:
            new_rt_min -= new_rt_max - self.rt_max
            new_rt_max = self.rt_max

        if new_mz_min < self.mz_min:
            new_mz_max += self.mz_min - new_mz_min
            new_mz_min = self.mz_min
        if new_mz_max > self.mz_max:
            new_mz_min -= new_mz_max - self.mz_max
            new_mz_max = self.mz_max

        self.view_rt_min = max(self.rt_min, new_rt_min)
        self.view_rt_max = min(self.rt_max, new_rt_max)
        self.view_mz_min = max(self.mz_min, new_mz_min)
        self.view_mz_max = min(self.mz_max, new_mz_max)

        if emit_event:
            self.emit_view_changed()

    # ========== CONVENIENCE CLEAR METHODS ==========

    def clear_features(self) -> None:
        """Clear features (alias for clear_feature_data)."""
        self.clear_feature_data()

    def clear_ids(self) -> None:
        """Clear IDs (alias for clear_id_data)."""
        self.clear_id_data()

    # ========== PANEL VISIBILITY UPDATE ==========

    def update_panel_visibility(self) -> None:
        """Update visibility of all panels based on current settings and data."""
        for panel_id, element in self.panel_elements.items():
            if panel_id == "legend":
                continue  # Help panel is always visible
            visible = self.should_panel_be_visible(panel_id)
            if hasattr(element, "set_visibility"):
                element.set_visibility(visible)

    # ========== FAIMS ==========

    def update_faims_plots(self) -> None:
        """Update FAIMS plots. Placeholder for FAIMS panel implementation."""
        pass

    # ========== MEASUREMENT MANAGEMENT ==========

    def delete_selected_measurement(self) -> None:
        """Delete the currently selected measurement in spectrum panel."""
        if self.selected_spectrum_idx is None or self.spectrum_selected_measurement_idx is None:
            return

        measurements = self.spectrum_measurements.get(self.selected_spectrum_idx, [])
        if 0 <= self.spectrum_selected_measurement_idx < len(measurements):
            measurements.pop(self.spectrum_selected_measurement_idx)
            self.spectrum_measurements[self.selected_spectrum_idx] = measurements
            self.spectrum_selected_measurement_idx = None
            self.emit_view_changed()

    # ========== PEPTIDE ID MATCHING ==========

    def find_matching_id_for_spectrum(
        self,
        spectrum_idx: int,
        rt_tolerance: float = 5.0,
        mz_tolerance: float = 0.5,
    ) -> Optional[int]:
        """Find peptide ID matching the given spectrum.

        Matches based on RT and precursor m/z within tolerance.

        Args:
            spectrum_idx: Spectrum index to find ID for
            rt_tolerance: RT tolerance in seconds
            mz_tolerance: m/z tolerance in Da

        Returns:
            ID index if found, None otherwise
        """
        if self.exp is None or not self.peptide_ids:
            return None

        if spectrum_idx < 0 or spectrum_idx >= len(self.exp):
            return None

        spec = self.exp[spectrum_idx]

        # Only MS2 spectra can have peptide IDs
        if spec.getMSLevel() != 2:
            return None

        spec_rt = spec.getRT()
        precursors = spec.getPrecursors()
        if not precursors:
            return None

        spec_prec_mz = precursors[0].getMZ()

        # Find best matching ID
        best_id_idx = None
        best_rt_diff = float("inf")

        for i, pep_id in enumerate(self.peptide_ids):
            id_rt = pep_id.getRT()
            id_mz = pep_id.getMZ()

            if abs(id_rt - spec_rt) <= rt_tolerance and abs(id_mz - spec_prec_mz) <= mz_tolerance:
                rt_diff = abs(id_rt - spec_rt)
                if rt_diff < best_rt_diff:
                    best_rt_diff = rt_diff
                    best_id_idx = i

        return best_id_idx

    def find_spectrum_for_id(
        self,
        id_idx: int,
        rt_tolerance: float = 5.0,
        mz_tolerance: float = 0.5,
    ) -> Optional[int]:
        """Find spectrum index matching the given peptide ID.

        Args:
            id_idx: Peptide ID index
            rt_tolerance: RT tolerance in seconds
            mz_tolerance: m/z tolerance in Da

        Returns:
            Spectrum index if found, None otherwise
        """
        if self.exp is None or not self.peptide_ids:
            return None

        if id_idx < 0 or id_idx >= len(self.peptide_ids):
            return None

        pep_id = self.peptide_ids[id_idx]
        id_rt = pep_id.getRT()
        id_mz = pep_id.getMZ()

        best_spec_idx = None
        best_rt_diff = float("inf")

        for i in range(len(self.exp)):
            spec = self.exp[i]
            if spec.getMSLevel() != 2:
                continue

            spec_rt = spec.getRT()
            if abs(spec_rt - id_rt) > rt_tolerance:
                continue

            precursors = spec.getPrecursors()
            if precursors:
                prec_mz = precursors[0].getMZ()
                if abs(prec_mz - id_mz) <= mz_tolerance:
                    rt_diff = abs(spec_rt - id_rt)
                    if rt_diff < best_rt_diff:
                        best_rt_diff = rt_diff
                        best_spec_idx = i

        return best_spec_idx
