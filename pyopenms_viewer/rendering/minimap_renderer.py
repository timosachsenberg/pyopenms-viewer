"""Minimap renderer for overview navigation.

Renders a small overview of the full data extent with a rectangle
showing the current view bounds. Supports cached rasterization for efficient
re-shading with different colormaps.
"""

import base64
import io
from typing import Optional

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import xarray as xr
from PIL import ImageDraw

from pyopenms_viewer.core.config import COLORMAPS, get_colormap_background
from pyopenms_viewer.utils.gpu import to_accelerated_dataframe


class MinimapRenderer:
    """Renders a minimap showing full data extent with current view rectangle.

    The minimap provides an overview of the entire dataset with a rectangle
    overlay showing the current zoom region. Supports click-to-center navigation.
    """

    def __init__(self, width: int = 400, height: int = 200):
        """Initialize minimap renderer.

        Args:
            width: Minimap width in pixels
            height: Minimap height in pixels
        """
        self.width = width
        self.height = height

    def render(self, state) -> Optional[str]:
        """Render the minimap showing full data extent with view rectangle overlay.

        Uses cached rasterization when available for efficient re-shading with different
        colormaps. Falls back to datashader point rendering if rasterization is not
        available.

        Args:
            state: ViewerState with data and view bounds

        Returns:
            Base64-encoded PNG string, or None if no data
        """
        # Try rasterization-based approach first (cached)
        if state.exp is not None and hasattr(state.exp, "rasterizeRTMZ"):
            return self._render_with_rasterization(state)

        # Fallback to datashader point rendering
        return self._render_with_datashader(state)

    def _render_with_rasterization(self, state) -> Optional[str]:
        """Render minimap using cached rasterization from pyOpenMS.

        When a FAIMS CV is selected, uses the per-CV experiment and its cached raster
        instead of the global experiment/cache.

        Args:
            state: ViewerState with data and view bounds

        Returns:
            Base64-encoded PNG string, or None if no data
        """
        # Select experiment and cache based on FAIMS CV selection
        if state.has_faims and state.selected_faims_cv is not None:
            cv = state.selected_faims_cv
            raster_exp = state.faims_experiments.get(cv)
            if raster_exp is None:
                return self._render_with_datashader(state)
            cached_raster = state.faims_minimap_rasters.get(cv)
        else:
            raster_exp = state.exp
            cached_raster = state.cached_minimap_raster

        # Check cache - if empty, rasterize
        if cached_raster is None:
            # Rasterize at full data extent
            # Request appropriate bin counts based on swap_axes
            if state.swap_axes:
                # Swapped: m/z on X (width), RT on Y (height)
                # Request (self.width mz bins, self.height RT bins) = (400, 200)
                output = np.zeros((self.width, self.height), dtype=np.float32)
            else:
                # Default: RT on X (width), m/z on Y (height)
                # Request (self.height mz bins, self.width RT bins) = (200, 400)
                output = np.zeros((self.height, self.width), dtype=np.float32)

            try:
                raster_exp.rasterizeRTMZ(
                    output,
                    state.rt_min,
                    state.rt_max,
                    state.mz_min,
                    state.mz_max,
                    ms_level=1,
                    aggregation="sum",
                )
                # Transpose if swapped to get standard (height, width) shape
                if state.swap_axes:
                    output = output.T
                cached_raster = output
            except Exception:
                # If rasterization fails, fall back to datashader
                return self._render_with_datashader(state)

            # Store in appropriate cache
            if state.has_faims and state.selected_faims_cv is not None:
                state.faims_minimap_rasters[state.selected_faims_cv] = cached_raster
            else:
                state.cached_minimap_raster = cached_raster

        # Create xarray DataArray from cached raster
        # Use actual raster shape for coordinates (per-CV rasters may differ in size)
        n_rows, n_cols = cached_raster.shape

        if state.swap_axes:
            # Swapped view: rows=RT, cols=m/z
            xr_data = xr.DataArray(
                cached_raster,
                coords={
                    "rt": np.linspace(state.rt_min, state.rt_max, n_rows),
                    "mz": np.linspace(state.mz_min, state.mz_max, n_cols),
                },
                dims=["rt", "mz"],
            )
        else:
            # Default view: rows=m/z, cols=RT
            xr_data = xr.DataArray(
                cached_raster,
                coords={
                    "mz": np.linspace(state.mz_min, state.mz_max, n_rows),
                    "rt": np.linspace(state.rt_min, state.rt_max, n_cols),
                },
                dims=["mz", "rt"],
            )

        # Apply colormap with histogram equalization for better contrast
        img = tf.shade(xr_data, cmap=COLORMAPS[state.colormap], how="eq_hist")
        img = tf.dynspread(img, threshold=0.5, max_px=4)
        img = tf.set_background(img, get_colormap_background(state.colormap))

        # Convert to PIL
        plot_img = img.to_pil()

        # Draw overlays (view rectangle and spectrum marker)
        self._draw_view_rectangle(plot_img, state)
        self._draw_spectrum_marker(plot_img, state)

        # Convert to base64
        buffer = io.BytesIO()
        plot_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _render_with_datashader(self, state) -> Optional[str]:
        """Render minimap using datashader point rendering (fallback).

        Args:
            state: ViewerState with data and view bounds

        Returns:
            Base64-encoded PNG string, or None if no data
        """
        # Get data for minimap
        # If downsampling is enabled, use data_manager's downsampled query
        # Otherwise, get full data for accurate representation
        if state.data_manager is not None:
            if state.peakmap_downsampling:
                minimap_df = state.data_manager.query_peaks_for_minimap()
            else:
                # No downsampling - get all peaks
                minimap_df = state.data_manager.query_all_peaks()
        elif state.df is not None:
            minimap_df = state.df
        elif state.exp is not None and hasattr(state.exp, "get2DPeakDataLong"):
            # Phase 1 rasterization path: extract data on-demand when df is None
            try:
                rt_array, mz_array, intensity_array = state.exp.get2DPeakDataLong(
                    state.rt_min, state.rt_max, state.mz_min, state.mz_max, ms_level=1
                )
                minimap_df = pd.DataFrame(
                    {
                        "rt": rt_array,
                        "mz": mz_array,
                        "intensity": intensity_array,
                        "log_intensity": np.log1p(intensity_array),
                    }
                )
            except Exception:
                return None
        else:
            minimap_df = None

        if minimap_df is None or len(minimap_df) == 0:
            return None

        # Convert to accelerated DataFrame for faster rendering
        minimap_df = to_accelerated_dataframe(minimap_df)

        # Create minimap canvas - swap axes to match main view
        if state.swap_axes:
            # m/z on x-axis, RT on y-axis
            cvs = ds.Canvas(
                plot_width=self.width,
                plot_height=self.height,
                x_range=(state.mz_min, state.mz_max),
                y_range=(state.rt_min, state.rt_max),
            )
            agg = cvs.points(minimap_df, "mz", "rt", agg=ds.max("log_intensity"))
        else:
            # RT on x-axis, m/z on y-axis (traditional)
            cvs = ds.Canvas(
                plot_width=self.width,
                plot_height=self.height,
                x_range=(state.rt_min, state.rt_max),
                y_range=(state.mz_min, state.mz_max),
            )
            agg = cvs.points(minimap_df, "rt", "mz", agg=ds.max("log_intensity"))

        # Apply color map with histogram equalization for better contrast
        img = tf.shade(agg, cmap=COLORMAPS[state.colormap], how="eq_hist")
        img = tf.dynspread(img, threshold=0.5, max_px=4)
        img = tf.set_background(img, get_colormap_background(state.colormap))

        # Convert to PIL
        plot_img = img.to_pil()

        # Draw view rectangle
        self._draw_view_rectangle(plot_img, state)

        # Draw spectrum marker
        self._draw_spectrum_marker(plot_img, state)

        # Convert to base64
        buffer = io.BytesIO()
        plot_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _draw_view_rectangle(self, plot_img, state):
        """Draw the current view rectangle on the minimap.

        Args:
            plot_img: PIL Image to draw on
            state: ViewerState with view bounds
        """
        if (
            state.view_rt_min is None
            or state.view_rt_max is None
            or state.view_mz_min is None
            or state.view_mz_max is None
        ):
            return

        draw = ImageDraw.Draw(plot_img)

        # Convert data coords to pixel coords
        rt_range = state.rt_max - state.rt_min
        mz_range = state.mz_max - state.mz_min

        if rt_range <= 0 or mz_range <= 0:
            return

        if state.swap_axes:
            # m/z on x-axis, RT on y-axis
            x1 = int((state.view_mz_min - state.mz_min) / mz_range * self.width)
            x2 = int((state.view_mz_max - state.mz_min) / mz_range * self.width)
            y1 = int((state.rt_max - state.view_rt_max) / rt_range * self.height)
            y2 = int((state.rt_max - state.view_rt_min) / rt_range * self.height)
        else:
            # RT on x-axis, m/z on y-axis (traditional)
            x1 = int((state.view_rt_min - state.rt_min) / rt_range * self.width)
            x2 = int((state.view_rt_max - state.rt_min) / rt_range * self.width)
            y1 = int((state.mz_max - state.view_mz_max) / mz_range * self.height)
            y2 = int((state.mz_max - state.view_mz_min) / mz_range * self.height)

        # Clamp to minimap bounds and ensure x1 <= x2, y1 <= y2
        x1, x2 = max(0, x1), min(self.width - 1, x2)
        y1, y2 = max(0, y1), min(self.height - 1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Draw two concentric rectangles with complementary colors for visibility
        # Outer rectangle (blue)
        draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(0, 100, 255, 255), width=2)
        # Inner rectangle (yellow) - only draw if there's enough space
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=(255, 255, 0, 255), width=2)

    def _draw_spectrum_marker(self, plot_img, state):
        """Draw a marker at the selected spectrum RT position.

        Args:
            plot_img: PIL Image to draw on
            state: ViewerState with selection and experiment data
        """
        if state.selected_spectrum_idx is None or state.exp is None:
            return

        spec = state.exp[state.selected_spectrum_idx]
        rt = spec.getRT()
        ms_level = spec.getMSLevel()

        rt_range = state.rt_max - state.rt_min
        if rt_range <= 0:
            return

        draw = ImageDraw.Draw(plot_img)

        # Use different colors for MS1 vs MS2
        if ms_level == 1:
            color = (0, 255, 0, 255)  # Green for MS1
        else:
            color = (255, 0, 255, 255)  # Magenta for MS2

        if state.swap_axes:
            # RT is on y-axis - draw horizontal lines
            y = int((state.rt_max - rt) / rt_range * self.height)
            y = max(0, min(self.height - 1, y))
            draw.line([(0, y - 2), (self.width, y - 2)], fill=(0, 0, 0, 200), width=1)
            draw.line([(0, y - 1), (self.width, y - 1)], fill=color, width=1)
            draw.line([(0, y + 1), (self.width, y + 1)], fill=color, width=1)
            draw.line([(0, y + 2), (self.width, y + 2)], fill=(0, 0, 0, 200), width=1)
        else:
            # RT is on x-axis - draw vertical lines
            x = int((rt - state.rt_min) / rt_range * self.width)
            x = max(0, min(self.width - 1, x))
            draw.line([(x - 2, 0), (x - 2, self.height)], fill=(0, 0, 0, 200), width=1)
            draw.line([(x - 1, 0), (x - 1, self.height)], fill=color, width=1)
            draw.line([(x + 1, 0), (x + 1, self.height)], fill=color, width=1)
            draw.line([(x + 2, 0), (x + 2, self.height)], fill=(0, 0, 0, 200), width=1)

    def render_for_cv(self, state, cv: float, width: int = None, height: int = None) -> Optional[str]:
        """Render a minimap for a specific FAIMS CV value.

        Tries rasterization from per-CV MSExperiment first (with caching),
        falls back to DataFrame-based datashader rendering.

        Args:
            state: ViewerState with FAIMS data
            cv: The compensation voltage value to render
            width: Optional custom width (uses self.width if not specified)
            height: Optional custom height (uses half of self.height if not specified)

        Returns:
            Base64-encoded PNG string, or None if no data for this CV
        """
        if not state.has_faims:
            return None

        render_width = width or self.width
        render_height = height or max(40, self.height // 2)

        # Try rasterization from per-CV experiment first (with caching)
        cv_exp = state.faims_experiments.get(cv)
        if cv_exp is not None and hasattr(cv_exp, "rasterizeRTMZ"):
            result = self._render_cv_with_rasterization(state, cv, cv_exp, render_width, render_height)
            if result is not None:
                return result

        # Fallback: DataFrame-based rendering
        if state.data_manager is not None:
            cv_df = state.data_manager.query_peaks_for_cv(cv, downsample=state.peakmap_downsampling)
        else:
            # Check if cv_df is in faims_data cache
            if cv in state.faims_data:
                cv_df = state.faims_data[cv]
            else:
                # Extract on-demand if faims_data is empty (Phase 1 rasterization path)
                cv_df = state.get_faims_peaks_for_cv(cv, state.rt_min, state.rt_max, state.mz_min, state.mz_max)

        if cv_df is None or len(cv_df) == 0:
            return None

        # Convert to accelerated DataFrame if available
        cv_df = to_accelerated_dataframe(cv_df)

        # Create minimap canvas - swap axes to match main view
        if state.swap_axes:
            # m/z on x-axis, RT on y-axis
            cvs = ds.Canvas(
                plot_width=render_width,
                plot_height=render_height,
                x_range=(state.mz_min, state.mz_max),
                y_range=(state.rt_min, state.rt_max),
            )
            agg = cvs.points(cv_df, "mz", "rt", agg=ds.max("log_intensity"))
        else:
            # RT on x-axis, m/z on y-axis (traditional)
            cvs = ds.Canvas(
                plot_width=render_width,
                plot_height=render_height,
                x_range=(state.rt_min, state.rt_max),
                y_range=(state.mz_min, state.mz_max),
            )
            agg = cvs.points(cv_df, "rt", "mz", agg=ds.max("log_intensity"))

        # Apply color map with histogram equalization for better contrast
        img = tf.shade(agg, cmap=COLORMAPS[state.colormap], how="eq_hist")
        img = tf.dynspread(img, threshold=0.5, max_px=4)
        img = tf.set_background(img, get_colormap_background(state.colormap))

        # Convert to PIL
        plot_img = img.to_pil()

        # Convert to base64
        buffer = io.BytesIO()
        plot_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _render_cv_with_rasterization(
        self, state, cv: float, cv_exp, render_width: int, render_height: int
    ) -> Optional[str]:
        """Render per-CV minimap using rasterization with caching.

        Args:
            state: ViewerState
            cv: Compensation voltage value
            cv_exp: Per-CV MSExperiment object
            render_width: Render width in pixels
            render_height: Render height in pixels

        Returns:
            Base64-encoded PNG string, or None if rasterization fails
        """
        cached_raster = state.faims_minimap_rasters.get(cv)

        if cached_raster is None:
            if state.swap_axes:
                output = np.zeros((render_width, render_height), dtype=np.float32)
            else:
                output = np.zeros((render_height, render_width), dtype=np.float32)

            try:
                cv_exp.rasterizeRTMZ(
                    output,
                    state.rt_min,
                    state.rt_max,
                    state.mz_min,
                    state.mz_max,
                    ms_level=1,
                    aggregation="sum",
                )
                if state.swap_axes:
                    output = output.T
                cached_raster = output
                state.faims_minimap_rasters[cv] = cached_raster
            except Exception:
                return None

        if cached_raster.max() == 0.0:
            return None

        if state.swap_axes:
            xr_data = xr.DataArray(
                cached_raster,
                coords={
                    "rt": np.linspace(state.rt_min, state.rt_max, render_height),
                    "mz": np.linspace(state.mz_min, state.mz_max, render_width),
                },
                dims=["rt", "mz"],
            )
        else:
            xr_data = xr.DataArray(
                cached_raster,
                coords={
                    "mz": np.linspace(state.mz_min, state.mz_max, render_height),
                    "rt": np.linspace(state.rt_min, state.rt_max, render_width),
                },
                dims=["mz", "rt"],
            )

        img = tf.shade(xr_data, cmap=COLORMAPS[state.colormap], how="eq_hist")
        img = tf.dynspread(img, threshold=0.5, max_px=4)
        img = tf.set_background(img, get_colormap_background(state.colormap))

        plot_img = img.to_pil()

        buffer = io.BytesIO()
        plot_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
