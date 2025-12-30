"""Peak map rendering using Datashader.

This module provides the main renderer for 2D peak maps using Datashader
for server-side aggregation of millions of peaks.
"""

import base64
import io

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pyopenms_viewer.annotation.tick_formatter import calculate_nice_ticks, format_tick_label
from pyopenms_viewer.core.config import COLORMAPS, get_colormap_background
from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.rendering.overlay_renderer import OverlayRenderer


def _get_font(size: int = 11):
    """Get a font for rendering, with fallbacks."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


class PeakMapRenderer:
    """Renders 2D peak maps using Datashader.

    This is a stateless renderer that takes ViewerState and produces images.
    All data access is via state references (no copies).

    Example:
        renderer = PeakMapRenderer()
        base64_png = renderer.render(state, fast=False)
    """

    def __init__(
        self,
        plot_width: int = 1100,
        plot_height: int = 550,
        margin_left: int = 80,
        margin_right: int = 20,
        margin_top: int = 20,
        margin_bottom: int = 50,
    ):
        """Initialize renderer with dimensions.

        Args:
            plot_width: Width of the plot area (excluding margins)
            plot_height: Height of the plot area (excluding margins)
            margin_left: Left margin for axis labels
            margin_right: Right margin
            margin_top: Top margin
            margin_bottom: Bottom margin for axis labels
        """
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.canvas_width = plot_width + margin_left + margin_right
        self.canvas_height = plot_height + margin_top + margin_bottom

        # Overlay renderer for features, IDs, markers
        self.overlay_renderer = OverlayRenderer(
            plot_width=plot_width,
            plot_height=plot_height,
            margin_left=0,  # Overlay renders on plot area directly
            margin_top=0,
        )

    def render(
        self,
        state: ViewerState,
        fast: bool = False,
        draw_overlays: bool = True,
        draw_axes: bool = True,
    ) -> str:
        """Render the peak map to a base64-encoded PNG string.

        Args:
            state: ViewerState containing all data and view bounds
            fast: If True, render at reduced resolution for panning
            draw_overlays: If True, draw features/IDs/markers (skipped in fast mode)
            draw_axes: If True, draw axis labels (skipped in fast mode)

        Returns:
            Base64-encoded PNG string, or empty string if no data
        """
        # Get view bounds
        view_rt_min = state.view_rt_min if state.view_rt_min is not None else state.rt_min
        view_rt_max = state.view_rt_max if state.view_rt_max is not None else state.rt_max
        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max

        # Get peaks in view - two paths for performance:
        #
        # IN-MEMORY MODE (state.df is not None):
        #   Use direct pandas boolean masking for best performance (~22ms for 5M peaks).
        #   This is the fast path for datasets that fit in RAM.
        #
        # OUT-OF-CORE MODE (state.df is None):
        #   Use state.get_peaks_in_view() which queries via DuckDB from Parquet files.
        #   Slower (~74ms) but enables viewing datasets larger than available RAM.
        #   The DataManager handles the DuckDB queries transparently.
        #
        if state.df is not None:
            # Fast path: direct pandas filtering (in-memory mode)
            # Determine which DataFrame to use (CV-filtered or full)
            if state.has_faims and state.selected_faims_cv is not None and state.selected_faims_cv in state.faims_data:
                source_df = state.faims_data[state.selected_faims_cv]
            else:
                source_df = state.df

            mask = (
                (source_df["rt"] >= view_rt_min)
                & (source_df["rt"] <= view_rt_max)
                & (source_df["mz"] >= view_mz_min)
                & (source_df["mz"] <= view_mz_max)
            )
            view_df = source_df[mask]
        else:
            # Out-of-core path: query via DuckDB (handles FAIMS CV filtering internally)
            view_df = state.get_peaks_in_view()

        if view_df is None or len(view_df) == 0:
            return ""

        # Render with Datashader
        resolution_factor = 4 if fast else 1
        render_width = self.plot_width // resolution_factor
        render_height = self.plot_height // resolution_factor

        # Adaptive downsampling for very large views (>2M peaks).
        # Uses stride sampling which is fast and preserves RT distribution since
        # MS data is sorted by RT. Target ~3 points per pixel for good visual quality.
        if state.peakmap_downsampling:
            target_points = render_width * render_height * 3
            if len(view_df) > target_points:
                sample_rate = len(view_df) // target_points
                view_df = view_df.iloc[::sample_rate]

        if state.swap_axes:
            # m/z on x-axis, RT on y-axis
            ds_canvas = ds.Canvas(
                plot_width=render_width,
                plot_height=render_height,
                x_range=(view_mz_min, view_mz_max),
                y_range=(view_rt_min, view_rt_max),
            )
            agg = ds_canvas.points(view_df, "mz", "rt", ds.max("log_intensity"))
        else:
            # RT on x-axis, m/z on y-axis
            ds_canvas = ds.Canvas(
                plot_width=render_width,
                plot_height=render_height,
                x_range=(view_rt_min, view_rt_max),
                y_range=(view_mz_min, view_mz_max),
            )
            agg = ds_canvas.points(view_df, "rt", "mz", ds.max("log_intensity"))

        # Apply colormap
        img = tf.shade(agg, cmap=COLORMAPS[state.colormap], how="linear")
        if not fast:
            img = tf.dynspread(img, threshold=0.5, max_px=3)
        img = tf.set_background(img, get_colormap_background(state.colormap))

        plot_img = img.to_pil()

        # Upscale if fast mode
        if fast and resolution_factor > 1:
            plot_img = plot_img.resize((self.plot_width, self.plot_height), Image.Resampling.NEAREST)

        # Draw overlays on plot image (features, IDs, spectrum markers)
        plot_img_rgba = plot_img.convert("RGBA")
        if draw_overlays and not fast:
            plot_img_rgba = self.overlay_renderer.draw_all(plot_img_rgba, state)

        # Compose final canvas
        canvas = Image.new("RGBA", (self.canvas_width, self.canvas_height), (0, 0, 0, 0))
        canvas.paste(plot_img_rgba, (self.margin_left, self.margin_top))

        # Draw axes unless in fast mode
        if draw_axes and not fast:
            canvas = self._draw_axes(canvas, state, view_rt_min, view_rt_max, view_mz_min, view_mz_max)

        # Encode to base64
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _draw_axes(
        self,
        canvas: Image.Image,
        state: ViewerState,
        view_rt_min: float,
        view_rt_max: float,
        view_mz_min: float,
        view_mz_max: float,
    ) -> Image.Image:
        """Draw axes, tick marks, and labels on the canvas.

        Args:
            canvas: PIL Image to draw on
            state: ViewerState for settings
            view_rt_min/max: RT range
            view_mz_min/max: m/z range

        Returns:
            Modified canvas with axes drawn
        """
        draw = ImageDraw.Draw(canvas)
        font = _get_font(11)
        title_font = _get_font(12)

        axis_color = (136, 136, 136, 255)
        tick_color = (136, 136, 136, 255)
        label_color = (136, 136, 136, 255)

        plot_left = self.margin_left
        plot_right = self.margin_left + self.plot_width
        plot_top = self.margin_top
        plot_bottom = self.margin_top + self.plot_height

        # Draw border rectangle
        draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=axis_color, width=1)

        if state.swap_axes:
            # X-axis: m/z, Y-axis: RT
            x_ticks = calculate_nice_ticks(view_mz_min, view_mz_max, num_ticks=8)
            x_range = view_mz_max - view_mz_min
            x_min, x_max = view_mz_min, view_mz_max

            for tick_val in x_ticks:
                if x_min <= tick_val <= x_max:
                    x_frac = (tick_val - x_min) / x_range
                    x = plot_left + int(x_frac * self.plot_width)
                    draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=tick_color, width=1)
                    label = format_tick_label(tick_val, x_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    draw.text((x - label_width // 2, plot_bottom + 8), label, fill=label_color, font=font)

            x_title = "m/z"
            bbox = draw.textbbox((0, 0), x_title, font=title_font)
            title_width = bbox[2] - bbox[0]
            draw.text(
                (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
                x_title,
                fill=label_color,
                font=title_font,
            )

            # Y-axis: RT
            if state.rt_in_minutes:
                display_rt_min = view_rt_min / 60.0
                display_rt_max = view_rt_max / 60.0
                y_ticks_display = calculate_nice_ticks(display_rt_min, display_rt_max, num_ticks=8)
                y_ticks = [t * 60.0 for t in y_ticks_display]
            else:
                y_ticks = calculate_nice_ticks(view_rt_min, view_rt_max, num_ticks=8)
            y_range = view_rt_max - view_rt_min
            y_min, y_max = view_rt_min, view_rt_max

            for tick_val in y_ticks:
                if y_min <= tick_val <= y_max:
                    y_frac = 1 - (tick_val - y_min) / y_range
                    y = plot_top + int(y_frac * self.plot_height)
                    draw.line([(plot_left - 5, y), (plot_left, y)], fill=tick_color, width=1)
                    display_val = tick_val / 60.0 if state.rt_in_minutes else tick_val
                    display_range = y_range / 60.0 if state.rt_in_minutes else y_range
                    label = format_tick_label(display_val, display_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    label_height = bbox[3] - bbox[1]
                    draw.text((plot_left - label_width - 10, y - label_height // 2), label, fill=label_color, font=font)

            y_title = "RT (min)" if state.rt_in_minutes else "RT (s)"
        else:
            # X-axis: RT, Y-axis: m/z
            if state.rt_in_minutes:
                display_rt_min = view_rt_min / 60.0
                display_rt_max = view_rt_max / 60.0
                rt_ticks_display = calculate_nice_ticks(display_rt_min, display_rt_max, num_ticks=8)
                rt_ticks = [t * 60.0 for t in rt_ticks_display]
            else:
                rt_ticks = calculate_nice_ticks(view_rt_min, view_rt_max, num_ticks=8)
            rt_range = view_rt_max - view_rt_min

            for tick_val in rt_ticks:
                if view_rt_min <= tick_val <= view_rt_max:
                    x_frac = (tick_val - view_rt_min) / rt_range
                    x = plot_left + int(x_frac * self.plot_width)
                    draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=tick_color, width=1)
                    display_val = tick_val / 60.0 if state.rt_in_minutes else tick_val
                    display_range = rt_range / 60.0 if state.rt_in_minutes else rt_range
                    label = format_tick_label(display_val, display_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    draw.text((x - label_width // 2, plot_bottom + 8), label, fill=label_color, font=font)

            x_title = "RT (min)" if state.rt_in_minutes else "RT (s)"
            bbox = draw.textbbox((0, 0), x_title, font=title_font)
            title_width = bbox[2] - bbox[0]
            draw.text(
                (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
                x_title,
                fill=label_color,
                font=title_font,
            )

            # Y-axis: m/z
            mz_ticks = calculate_nice_ticks(view_mz_min, view_mz_max, num_ticks=8)
            mz_range = view_mz_max - view_mz_min

            for tick_val in mz_ticks:
                if view_mz_min <= tick_val <= view_mz_max:
                    y_frac = 1 - (tick_val - view_mz_min) / mz_range
                    y = plot_top + int(y_frac * self.plot_height)
                    draw.line([(plot_left - 5, y), (plot_left, y)], fill=tick_color, width=1)
                    label = format_tick_label(tick_val, mz_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    label_height = bbox[3] - bbox[1]
                    draw.text((plot_left - label_width - 10, y - label_height // 2), label, fill=label_color, font=font)

            y_title = "m/z"

        # Draw rotated Y-axis title
        txt_img = Image.new("RGBA", (100, 30), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), y_title, fill=label_color, font=title_font)
        txt_img = txt_img.rotate(90, expand=True)

        y_title_x = 5
        y_title_y = plot_top + self.plot_height // 2 - txt_img.height // 2
        canvas.paste(txt_img, (y_title_x, y_title_y), txt_img)

        return canvas

    def render_faims(self, state: ViewerState, cv: float) -> str:
        """Render a single FAIMS CV peak map.

        Args:
            state: ViewerState with FAIMS data
            cv: Compensation voltage value

        Returns:
            Base64-encoded PNG string
        """
        if cv not in state.faims_data or len(state.faims_data[cv]) == 0:
            return ""

        cv_df = state.faims_data[cv]

        view_rt_min = state.view_rt_min if state.view_rt_min is not None else state.rt_min
        view_rt_max = state.view_rt_max if state.view_rt_max is not None else state.rt_max
        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max

        mask = (
            (cv_df["rt"] >= view_rt_min)
            & (cv_df["rt"] <= view_rt_max)
            & (cv_df["mz"] >= view_mz_min)
            & (cv_df["mz"] <= view_mz_max)
        )
        view_df = cv_df[mask]

        if len(view_df) == 0:
            return ""

        # Smaller dimensions for FAIMS panels
        faims_width = self.plot_width // 2
        faims_height = self.plot_height // 2

        if state.swap_axes:
            ds_canvas = ds.Canvas(
                plot_width=faims_width,
                plot_height=faims_height,
                x_range=(view_mz_min, view_mz_max),
                y_range=(view_rt_min, view_rt_max),
            )
            agg = ds_canvas.points(view_df, "mz", "rt", ds.max("log_intensity"))
        else:
            ds_canvas = ds.Canvas(
                plot_width=faims_width,
                plot_height=faims_height,
                x_range=(view_rt_min, view_rt_max),
                y_range=(view_mz_min, view_mz_max),
            )
            agg = ds_canvas.points(view_df, "rt", "mz", ds.max("log_intensity"))

        img = tf.shade(agg, cmap=COLORMAPS[state.colormap], how="linear")
        img = tf.dynspread(img, threshold=0.5, max_px=2)
        img = tf.set_background(img, get_colormap_background(state.colormap))

        plot_img = img.to_pil()

        buffer = io.BytesIO()
        plot_img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class IMPeakMapRenderer:
    """Renders ion mobility peak maps using Datashader."""

    def __init__(
        self,
        plot_width: int = 1100,
        plot_height: int = 550,
        margin_left: int = 80,
        margin_right: int = 20,
        margin_top: int = 20,
        margin_bottom: int = 50,
    ):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.canvas_width = plot_width + margin_left + margin_right
        self.canvas_height = plot_height + margin_top + margin_bottom

    def render(self, state: ViewerState) -> str:
        """Render ion mobility peak map.

        Args:
            state: ViewerState with IM data

        Returns:
            Base64-encoded PNG string
        """
        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max
        view_im_min = state.view_im_min if state.view_im_min is not None else state.im_min
        view_im_max = state.view_im_max if state.view_im_max is not None else state.im_max

        # Get IM peaks in view - two paths for performance:
        #
        # IN-MEMORY MODE (state.im_df is not None):
        #   Use direct pandas boolean masking for best performance.
        #
        # OUT-OF-CORE MODE (state.im_df is None):
        #   Use state.get_im_peaks_in_view() which queries via DuckDB from Parquet.
        #   The DataManager handles the DuckDB queries transparently.
        #
        if state.im_df is not None:
            # Fast path: direct pandas filtering (in-memory mode)
            mask = (
                (state.im_df["mz"] >= view_mz_min)
                & (state.im_df["mz"] <= view_mz_max)
                & (state.im_df["im"] >= view_im_min)
                & (state.im_df["im"] <= view_im_max)
            )
            view_df = state.im_df[mask]
        else:
            # Out-of-core path: query via DuckDB
            view_df = state.get_im_peaks_in_view()

        if view_df is None or len(view_df) == 0:
            return ""

        # m/z on x-axis, IM on y-axis
        ds_canvas = ds.Canvas(
            plot_width=self.plot_width,
            plot_height=self.plot_height,
            x_range=(view_mz_min, view_mz_max),
            y_range=(view_im_min, view_im_max),
        )
        agg = ds_canvas.points(view_df, "mz", "im", ds.max("log_intensity"))

        img = tf.shade(agg, cmap=COLORMAPS[state.colormap], how="linear")
        img = tf.dynspread(img, threshold=0.5, max_px=3)
        img = tf.set_background(img, get_colormap_background(state.colormap))

        plot_img = img.to_pil()

        # Calculate canvas width - add space for mobilogram if enabled
        mobilogram_space = state.mobilogram_plot_width + 20 if state.show_mobilogram else 0
        total_canvas_width = self.canvas_width + mobilogram_space

        # Compose final canvas
        canvas = Image.new("RGBA", (total_canvas_width, self.canvas_height), (0, 0, 0, 0))
        plot_img_rgba = plot_img.convert("RGBA")
        canvas.paste(plot_img_rgba, (self.margin_left, self.margin_top))

        # Draw axes
        canvas = self._draw_axes(canvas, state, view_mz_min, view_mz_max, view_im_min, view_im_max)

        # Draw mobilogram on the right side if enabled
        if state.show_mobilogram:
            canvas = self._draw_mobilogram(canvas, state, view_mz_min, view_mz_max, view_im_min, view_im_max)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _draw_axes(
        self,
        canvas: Image.Image,
        state: ViewerState,
        view_mz_min: float,
        view_mz_max: float,
        view_im_min: float,
        view_im_max: float,
    ) -> Image.Image:
        """Draw axes, tick marks, and labels on the IM canvas.

        Args:
            canvas: PIL Image to draw on
            state: ViewerState for settings
            view_mz_min/max: m/z range
            view_im_min/max: IM range

        Returns:
            Modified canvas with axes drawn
        """
        draw = ImageDraw.Draw(canvas)
        font = _get_font(11)
        title_font = _get_font(12)

        axis_color = (136, 136, 136, 255)
        tick_color = (136, 136, 136, 255)
        label_color = (136, 136, 136, 255)

        plot_left = self.margin_left
        plot_right = self.margin_left + self.plot_width
        plot_top = self.margin_top
        plot_bottom = self.margin_top + self.plot_height

        # Draw border rectangle
        draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=axis_color, width=1)

        # X-axis: m/z
        x_ticks = calculate_nice_ticks(view_mz_min, view_mz_max, num_ticks=8)
        x_range = view_mz_max - view_mz_min

        for tick_val in x_ticks:
            if view_mz_min <= tick_val <= view_mz_max:
                x_frac = (tick_val - view_mz_min) / x_range
                x = plot_left + int(x_frac * self.plot_width)
                draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=tick_color, width=1)
                label = format_tick_label(tick_val, x_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                draw.text((x - label_width // 2, plot_bottom + 8), label, fill=label_color, font=font)

        x_title = "m/z"
        bbox = draw.textbbox((0, 0), x_title, font=title_font)
        title_width = bbox[2] - bbox[0]
        draw.text(
            (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
            x_title,
            fill=label_color,
            font=title_font,
        )

        # Y-axis: Ion mobility
        y_ticks = calculate_nice_ticks(view_im_min, view_im_max, num_ticks=8)
        y_range = view_im_max - view_im_min

        for tick_val in y_ticks:
            if view_im_min <= tick_val <= view_im_max:
                y_frac = 1 - (tick_val - view_im_min) / y_range
                y = plot_top + int(y_frac * self.plot_height)
                draw.line([(plot_left - 5, y), (plot_left, y)], fill=tick_color, width=1)
                label = format_tick_label(tick_val, y_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                draw.text((plot_left - label_width - 10, y - label_height // 2), label, fill=label_color, font=font)

        # Y-axis title with unit
        y_title = f"IM ({state.im_unit})" if state.im_unit else "Ion Mobility"

        # Draw rotated Y-axis title
        txt_img = Image.new("RGBA", (120, 30), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), y_title, fill=label_color, font=title_font)
        txt_img = txt_img.rotate(90, expand=True)

        y_title_x = 5
        y_title_y = plot_top + self.plot_height // 2 - txt_img.height // 2
        canvas.paste(txt_img, (y_title_x, y_title_y), txt_img)

        return canvas

    def _draw_mobilogram(
        self,
        canvas: Image.Image,
        state: ViewerState,
        view_mz_min: float,
        view_mz_max: float,
        view_im_min: float,
        view_im_max: float,
    ) -> Image.Image:
        """Draw mobilogram (summed intensity vs IM) on the right side of the IM peakmap."""
        draw = ImageDraw.Draw(canvas)

        # Get mobilogram data
        im_values, intensities = self._extract_mobilogram(state, view_mz_min, view_mz_max, view_im_min, view_im_max)
        if len(im_values) == 0 or len(intensities) == 0:
            return canvas

        # Sort by IM values for proper line drawing (prevents jumps)
        sort_idx = np.argsort(im_values)
        im_values = im_values[sort_idx]
        intensities = intensities[sort_idx]

        # Mobilogram plot area (to the right of the main plot)
        mob_left = self.margin_left + self.plot_width + 10
        mob_right = mob_left + state.mobilogram_plot_width
        mob_top = self.margin_top
        mob_bottom = self.margin_top + self.plot_height

        # Draw border for mobilogram area
        axis_color = (136, 136, 136, 255)
        label_color = (136, 136, 136, 255)
        draw.rectangle([mob_left, mob_top, mob_right, mob_bottom], outline=axis_color, width=1)

        # Normalize intensities to plot width
        max_intensity = np.max(intensities) if len(intensities) > 0 else 1.0
        if max_intensity == 0:
            max_intensity = 1.0

        # Draw filled mobilogram as horizontal bars
        im_range = view_im_max - view_im_min
        if im_range == 0:
            im_range = 1.0

        # Draw as a filled area plot
        points = []
        for im_val, intensity in zip(im_values, intensities):
            # Y position (IM axis, inverted - low IM at bottom)
            y_frac = 1.0 - (im_val - view_im_min) / im_range
            y = mob_top + int(y_frac * self.plot_height)

            # X position (intensity, starting from left edge)
            x_frac = intensity / max_intensity
            x = mob_left + int(x_frac * state.mobilogram_plot_width)

            points.append((x, y))

        # Draw line connecting all points
        if len(points) > 1:
            # Build polygon path: bottom-left -> data points (bottom to top) -> top-left -> close
            fill_points = [(mob_left, mob_bottom)]  # Start at bottom-left
            fill_points.extend(points)  # Go through data points (bottom to top)
            fill_points.append((mob_left, mob_top))  # End at top-left

            # Fill with semi-transparent cyan
            draw.polygon(fill_points, fill=(0, 200, 255, 80))

            # Draw the line on top
            draw.line(points, fill=(0, 200, 255, 255), width=2)

        # Draw axis label at top
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 10)
            except OSError:
                font = ImageFont.load_default()

        label = "Mobilogram"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        draw.text(
            (mob_left + (state.mobilogram_plot_width - label_width) // 2, mob_top - 15),
            label,
            fill=label_color,
            font=font,
        )

        return canvas

    def _extract_mobilogram(
        self,
        state: ViewerState,
        mz_min: float,
        mz_max: float,
        im_min: float,
        im_max: float,
    ) -> tuple:
        """Extract mobilogram (summed intensity vs ion mobility) from IM data.

        Returns:
            Tuple of (im_values, intensities) arrays for plotting
        """
        # Get IM peaks - use same dual-path approach as render():
        # IN-MEMORY: direct pandas filtering, OUT-OF-CORE: DuckDB query
        if state.im_df is not None:
            # Fast path: direct pandas filtering (in-memory mode)
            mask = (
                (state.im_df["mz"] >= mz_min)
                & (state.im_df["mz"] <= mz_max)
                & (state.im_df["im"] >= im_min)
                & (state.im_df["im"] <= im_max)
            )
            filtered_df = state.im_df[mask]
        else:
            # Out-of-core path: query via DuckDB
            filtered_df = state.get_im_peaks_in_view()

        if filtered_df is None or len(filtered_df) == 0:
            return np.array([]), np.array([])

        # Bin IM values and sum intensities
        n_bins = min(200, max(50, int(len(filtered_df) / 100)))  # Adaptive bin count

        bin_edges = np.linspace(im_min, im_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Digitize IM values into bins
        bin_indices = np.digitize(filtered_df["im"].values, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Sum intensities per bin
        intensities = np.zeros(n_bins, dtype=np.float64)
        np.add.at(intensities, bin_indices, filtered_df["intensity"].values)

        return bin_centers, intensities
