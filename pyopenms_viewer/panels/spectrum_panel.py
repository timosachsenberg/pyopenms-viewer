"""Spectrum (1D) panel component.

This panel displays individual MS spectra with interactive features
including peak measurement, annotation, and navigation.
"""

from typing import Callable, Optional

import numpy as np
import plotly.graph_objects as go
from nicegui import ui

from pyopenms_viewer.annotation.spectrum_annotator import (
    annotate_spectrum_with_id,
    compute_spectrum_annotation,
    create_annotated_spectrum_plot,
    get_external_peak_annotations_from_hit,
)
from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders.mzml_loader import get_cv_from_spectrum
from pyopenms_viewer.panels.base_panel import BasePanel


class SpectrumPanel(BasePanel):
    """1D Spectrum viewer panel.

    Features:
    - Navigation between spectra (prev/next, by MS level)
    - Intensity display modes (% or absolute)
    - Auto Y-axis scaling
    - Peak measurement mode (click two peaks to measure Δm/z)
    - Peak annotation/labeling
    - Precursor display for MS2+
    - Integration with ID data for annotated views
    """

    # Maximum peaks to display before downsampling kicks in
    MAX_DISPLAY_PEAKS = 5000

    def __init__(self, state: ViewerState):
        """Initialize spectrum panel.

        Args:
            state: ViewerState instance (shared reference)
        """
        super().__init__(state, "spectrum", "1D Spectrum", "show_chart")

        # UI elements
        self.spectrum_plot: Optional[ui.plotly] = None
        self.nav_label: Optional[ui.label] = None
        self.info_label: Optional[ui.label] = None
        self.measure_btn = None
        self.annotation_btn = None

        # References to external update callbacks
        self._on_spectrum_changed_callback: Optional[Callable] = None

        # Plotly config (must be included in figure dict)
        self._plotly_config = {
            "modeBarButtonsToRemove": ["autoScale2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "spectrum",
                "width": 1200,
                "height": 600,
                "scale": 1,
            },
        }

    def _figure_with_config(self, fig: go.Figure) -> dict:
        """Convert go.Figure to dict and add config for modebar customization."""
        fig_dict = fig.to_plotly_json()
        fig_dict["config"] = self._plotly_config
        return fig_dict

    def _downsample_spectrum(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        max_peaks: int = MAX_DISPLAY_PEAKS,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Downsample spectrum for display while preserving important peaks.

        Uses a hybrid approach:
        1. Keep uniformly spaced peaks for m/z coverage (70% of budget)
        2. Keep top N peaks by intensity (30% of budget)

        The uniform sampling is weighted higher to prevent large gaps in the display.

        Args:
            mz_array: Full m/z array
            int_array: Full intensity array
            max_peaks: Maximum number of peaks to keep

        Returns:
            Tuple of (downsampled_mz, downsampled_int, indices) where indices
            maps back to the original arrays for snap-to-peak functionality
        """
        n_peaks = len(mz_array)

        if n_peaks <= max_peaks:
            return mz_array, int_array, np.arange(n_peaks)

        # Split budget: 70% for uniform coverage, 30% for top intensity
        n_uniform = int(max_peaks * 0.7)
        n_top = max_peaks - n_uniform

        # Get uniformly spaced indices for m/z coverage (higher priority)
        uniform_indices = np.linspace(0, n_peaks - 1, n_uniform, dtype=int)

        # Get indices of top N peaks by intensity
        top_indices = np.argsort(int_array)[-n_top:]

        # Combine and deduplicate
        all_indices = np.unique(np.concatenate([uniform_indices, top_indices]))

        # Sort by m/z for proper display
        all_indices = all_indices[np.argsort(mz_array[all_indices])]

        return mz_array[all_indices], int_array[all_indices], all_indices

    def build(self, container: ui.element) -> ui.expansion:
        """Build the spectrum panel UI.

        Args:
            container: Parent element to build panel in

        Returns:
            The expansion element created
        """
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                with ui.column().classes("w-full items-center"):
                    self._build_navigation_row()
                    self._build_spectrum_plot()

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)
        self.state.on_selection_changed(self._on_selection_changed)

        self._is_built = True
        return self.expansion

    def _build_navigation_row(self):
        """Build the navigation and controls row."""
        with ui.row().classes("w-full items-center gap-2 mb-1").style(f"max-width: {self.state.canvas_width}px;"):
            # Navigation buttons
            ui.button("|<", on_click=lambda: self._navigate_to(0)).props("dense size=sm").tooltip("First")

            ui.button("< MS1", on_click=lambda: self._navigate_by_ms_level(-1, 1)).props(
                "dense size=sm color=cyan"
            ).tooltip("Prev MS1")

            ui.button("<", on_click=lambda: self._navigate(-1)).props("dense size=sm").tooltip("Prev")

            self.nav_label = ui.label("No spectrum").classes("mx-2 text-gray-400 text-sm")

            ui.button(">", on_click=lambda: self._navigate(1)).props("dense size=sm").tooltip("Next")

            ui.button("MS1 >", on_click=lambda: self._navigate_by_ms_level(1, 1)).props(
                "dense size=sm color=cyan"
            ).tooltip("Next MS1")

            ui.button(
                ">|",
                on_click=lambda: self._navigate_to(len(self.state.exp) - 1 if self.state.exp else 0),
            ).props("dense size=sm").tooltip("Last")

            ui.label("|").classes("mx-1 text-gray-600")

            ui.button("< MS2", on_click=lambda: self._navigate_by_ms_level(-1, 2)).props(
                "dense size=sm color=orange"
            ).tooltip("Prev MS2")

            ui.button("MS2 >", on_click=lambda: self._navigate_by_ms_level(1, 2)).props(
                "dense size=sm color=orange"
            ).tooltip("Next MS2")

            ui.element("div").classes("flex-grow")  # Spacer

            # Intensity display toggle
            ui.label("Intensity:").classes("text-xs text-gray-400")
            ui.toggle(
                ["%", "abs"],
                value="%" if self.state.spectrum_intensity_percent else "abs",
                on_change=self._toggle_intensity_mode,
            ).props("dense size=sm color=grey").tooltip("Toggle between relative (%) and absolute intensity")

            # Auto-scale checkbox
            ui.checkbox(
                "Auto Y",
                value=self.state.spectrum_auto_scale,
                on_change=self._toggle_auto_scale,
            ).props("dense size=sm color=grey").classes("text-xs").tooltip(
                "Auto-scale Y-axis to fit visible peaks (highest peak at 95%)"
            )

            # Measurement mode toggle
            self.measure_btn = (
                ui.button("📏 Measure", on_click=self._toggle_measure_mode)
                .props("dense size=sm color=grey")
                .tooltip("Toggle measurement mode - click two peaks to measure Δm/z")
            )

            ui.button(
                "Clear Δ",
                on_click=self._clear_measurements,
            ).props("dense size=sm color=grey").tooltip("Clear measurements for this spectrum")

            ui.label("|").classes("mx-1 text-gray-600")

            # Annotation mode toggle
            self.annotation_btn = (
                ui.button("🏷️ Label", on_click=self._toggle_annotation_mode)
                .props("dense size=sm color=grey")
                .tooltip("Toggle label mode - click peaks to add/edit custom labels")
            )

            # Show m/z labels toggle
            ui.checkbox(
                "m/z",
                value=self.state.show_mz_labels,
                on_change=self._toggle_mz_labels,
            ).props("dense size=sm color=grey").classes("text-xs").tooltip("Show m/z values on top peaks")

            ui.button("Clear 🏷️", on_click=self._clear_annotations).props("dense size=sm color=grey").tooltip(
                "Clear all labels for this spectrum"
            )

            # Show unmatched theoretical peaks toggle (for mirror mode)
            ui.checkbox(
                "Unmatched",
                value=self.state.show_unmatched_theoretical,
                on_change=self._toggle_show_unmatched,
            ).props("dense size=sm color=grey").classes("text-xs").tooltip(
                "Show unmatched theoretical ions as dashed lines in mirror mode"
            )

            ui.label("|").classes("mx-1 text-gray-600")

            self.info_label = ui.label("Click TIC to select spectrum").classes("text-xs text-gray-500")

    def _build_spectrum_plot(self):
        """Build the spectrum plot area."""
        self.spectrum_plot = ui.plotly(self._figure_with_config(go.Figure())).classes("w-full")

        # Event handlers
        self.spectrum_plot.on("plotly_click", self._on_plot_click)
        self.spectrum_plot.on("plotly_relayout", self._on_plot_relayout)
        # Hover handlers for snap-to-peak highlighting
        self.spectrum_plot.on("plotly_hover", self._on_plot_hover)
        self.spectrum_plot.on("plotly_unhover", self._on_plot_unhover)

        # Store reference in state for cross-panel access
        self.state.spectrum_browser_plot = self.spectrum_plot

    def update(self) -> None:
        """Update the spectrum display for current selection."""
        if self.state.selected_spectrum_idx is not None:
            self.show_spectrum(self.state.selected_spectrum_idx)

    def _has_data(self) -> bool:
        """Check if panel has data to display."""
        return self.state.exp is not None and len(self.state.exp) > 0

    def show_spectrum(self, spectrum_idx: int) -> None:
        """Display a spectrum in the panel.

        Shows annotated spectrum if a matching peptide ID exists and annotation is enabled.

        Args:
            spectrum_idx: Index of spectrum to display
        """
        if self.state.exp is None or spectrum_idx < 0 or spectrum_idx >= len(self.state.exp):
            return

        self.state.selected_spectrum_idx = spectrum_idx
        spec = self.state.exp[spectrum_idx]

        mz_array, int_array = spec.get_peaks()
        rt = spec.getRT()
        ms_level = spec.getMSLevel()

        if len(mz_array) == 0:
            ui.notify("Spectrum is empty", type="warning")
            return

        # Check if there's a matching peptide ID for annotation
        matching_id_idx = self.state.find_matching_id_for_spectrum(spectrum_idx)
        print(
            f"DEBUG show_spectrum: spectrum_idx={spectrum_idx}, ms_level={ms_level}, matching_id_idx={matching_id_idx}"
        )

        if matching_id_idx is not None:
            # Use annotated spectrum display (shows sequence in title even if peak coloring is off)
            pep_id = self.state.peptide_ids[matching_id_idx]
            hits = pep_id.getHits()
            if hits:
                fig = self._create_annotated_spectrum_figure(spec, mz_array, int_array, spectrum_idx, matching_id_idx)
                # Update info label with ID info
                if self.info_label is not None:
                    best_hit = hits[0]
                    sequence_str = best_hit.getSequence().toString()
                    charge = best_hit.getCharge()
                    precursors = spec.getPrecursors()
                    prec_mz = precursors[0].getMZ() if precursors else pep_id.getMZ()
                    self.info_label.set_text(
                        f"RT: {rt:.2f}s | ID: {sequence_str} | Charge: {charge}+ | Precursor: {prec_mz:.4f}"
                    )
            else:
                # No hits, fall back to regular display
                matching_id_idx = None

        if matching_id_idx is None:
            # Regular spectrum display (no annotation)
            fig = self._create_spectrum_figure(spec, mz_array, int_array, spectrum_idx)
            # Update info label
            if self.info_label is not None:
                tic = float(np.sum(int_array))
                mz_range = f"{mz_array.min():.2f} - {mz_array.max():.2f}" if len(mz_array) > 0 else "-"
                self.info_label.set_text(
                    f"RT: {rt:.2f}s | MS Level: {ms_level} | Peaks: {len(mz_array):,} | TIC: {tic:.2e} | m/z: {mz_range}"
                )

        # Update plot
        if self.spectrum_plot is not None:
            self.spectrum_plot.update_figure(self._figure_with_config(fig))

        # Update navigation label
        if self.nav_label is not None:
            self.nav_label.set_text(f"Spectrum {spectrum_idx + 1} of {len(self.state.exp)}")

    def _create_spectrum_figure(
        self, spec, mz_array: np.ndarray, int_array: np.ndarray, spectrum_idx: int
    ) -> go.Figure:
        """Create a Plotly figure for the spectrum.

        Args:
            spec: pyOpenMS spectrum object
            mz_array: Array of m/z values
            int_array: Array of intensity values
            spectrum_idx: Spectrum index for title

        Returns:
            Plotly Figure object
        """
        rt = spec.getRT()
        ms_level = spec.getMSLevel()
        cv = get_cv_from_spectrum(spec)
        total_peaks = len(mz_array)
        max_int = float(int_array.max()) if total_peaks > 0 else 1.0

        # If zoomed, filter to visible range first, then downsample
        if self.state.spectrum_zoom_range is not None:
            xmin, xmax = self.state.spectrum_zoom_range
            visible_mask = (mz_array >= xmin) & (mz_array <= xmax)
            mz_visible = mz_array[visible_mask]
            int_visible = int_array[visible_mask]
            visible_peaks = len(mz_visible)
        else:
            mz_visible, int_visible = mz_array, int_array
            visible_peaks = total_peaks

        # Downsample for display if too many peaks (and downsampling is enabled)
        if self.state.peakmap_downsampling:
            mz_display, int_display_raw, _ = self._downsample_spectrum(mz_visible, int_visible)
            is_downsampled = len(mz_display) < visible_peaks
        else:
            mz_display, int_display_raw = mz_visible, int_visible
            is_downsampled = False

        # Choose intensity values based on display mode
        if self.state.spectrum_intensity_percent:
            int_display = (int_display_raw / max_int) * 100
            y_title = "Relative Intensity (%)"
            hover_fmt = "m/z: %{x:.4f}<br>Intensity: %{y:.1f}%<extra></extra>"
            y_range = [0, 105]
        else:
            int_display = int_display_raw
            y_title = "Intensity"
            hover_fmt = "m/z: %{x:.4f}<br>Intensity: %{y:.2e}<extra></extra>"
            y_range = [0, max_int * 1.05]

        # Create figure
        fig = go.Figure()

        # Color based on theme
        is_dark = self.state.dark.value if self.state.dark else True
        color = "#00d4ff" if is_dark else "#000000"

        # Add spectrum as vertical lines (stem plot)
        x_stems = []
        y_stems = []
        for mz, intensity in zip(mz_display, int_display):
            x_stems.extend([mz, mz, None])
            y_stems.extend([0, intensity, None])

        fig.add_trace(
            go.Scatter(
                x=x_stems, y=y_stems, mode="lines", line={"color": color, "width": 1}, hoverinfo="skip", name="spectrum"
            )
        )

        # Add invisible hover points at peak tops (opacity 0 hides markers but keeps hover)
        fig.add_trace(
            go.Scatter(
                x=mz_display,
                y=int_display,
                mode="markers",
                marker={"color": color, "size": 8, "opacity": 0},
                hovertemplate=hover_fmt,
            )
        )

        # Title with spectrum info (show downsampling indicator)
        if self.state.spectrum_zoom_range is not None and visible_peaks < total_peaks:
            # Zoomed view
            if is_downsampled:
                title = f"Spectrum #{spectrum_idx} | MS{ms_level} | RT={rt:.2f}s | {visible_peaks:,}/{total_peaks:,} peaks ({len(mz_display):,} shown)"
            else:
                title = (
                    f"Spectrum #{spectrum_idx} | MS{ms_level} | RT={rt:.2f}s | {visible_peaks:,}/{total_peaks:,} peaks"
                )
        elif is_downsampled:
            title = f"Spectrum #{spectrum_idx} | MS{ms_level} | RT={rt:.2f}s | {total_peaks:,} peaks ({len(mz_display):,} shown)"
        else:
            title = f"Spectrum #{spectrum_idx} | MS{ms_level} | RT={rt:.2f}s | {total_peaks:,} peaks"

        # Add precursor line for MS2+
        if ms_level > 1:
            precursors = spec.getPrecursors()
            if precursors:
                prec_mz = precursors[0].getMZ()
                fig.add_vline(
                    x=prec_mz, line_dash="dash", line_color="orange", annotation_text=f"Precursor ({prec_mz:.2f})"
                )
                title += f" | Precursor: {prec_mz:.4f}"

        # Add CV if available
        if cv is not None:
            title += f" | CV={cv:.1f}V"

        # Layout
        fig.update_layout(
            title={"text": title, "font": {"size": 14, "color": "#888"}},
            xaxis_title="m/z",
            yaxis_title=y_title,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin={"l": 60, "r": 20, "t": 50, "b": 50},
            showlegend=False,
            modebar={"remove": ["lasso2d", "select2d"]},
            font={"color": "#888"},
            uirevision="spectrum_stable",
        )

        # Apply saved zoom range if available
        if self.state.spectrum_zoom_range is not None:
            fig.update_xaxes(
                range=list(self.state.spectrum_zoom_range),
                showgrid=False,
                linecolor="#888",
                tickcolor="#888",
                ticks="outside",
                ticklen=5,
            )
            # Auto-scale y-axis to visible peaks if enabled (use full array for accuracy)
            if self.state.spectrum_auto_scale:
                xmin, xmax = self.state.spectrum_zoom_range
                visible_mask = (mz_array >= xmin) & (mz_array <= xmax)
                if np.any(visible_mask):
                    # Use original int_array for accurate max calculation
                    if self.state.spectrum_intensity_percent:
                        visible_max = float(int_array[visible_mask].max() / max_int) * 100
                    else:
                        visible_max = float(int_array[visible_mask].max())
                    y_range = [0, visible_max / 0.95]
        else:
            fig.update_xaxes(showgrid=False, linecolor="#888", tickcolor="#888", ticks="outside", ticklen=5)

        fig.update_yaxes(
            range=y_range,
            showgrid=False,
            fixedrange=True,
            linecolor="#888",
            tickcolor="#888",
            ticks="outside",
            ticklen=5,
        )

        # Add measurements, annotations, m/z labels, and hover highlights
        self._add_measurements_to_figure(fig, spectrum_idx, mz_array, int_array)
        self._add_annotations_to_figure(fig, spectrum_idx, mz_array, int_array)
        self._add_mz_labels_to_figure(fig, spectrum_idx, mz_array, int_array, max_int)
        self._add_highlight_to_figure(fig, mz_array, int_array)

        return fig

    def _create_annotated_spectrum_figure(
        self,
        spec,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        spectrum_idx: int,
        id_idx: int,
    ) -> go.Figure:
        """Create an annotated Plotly figure for MS2 spectrum with peptide ID.

        Args:
            spec: pyOpenMS spectrum object
            mz_array: Array of m/z values
            int_array: Array of intensity values
            spectrum_idx: Spectrum index for title
            id_idx: Index of matching peptide ID

        Returns:
            Plotly Figure object with ion annotations
        """
        rt = spec.getRT()
        cv = get_cv_from_spectrum(spec)
        pep_id = self.state.peptide_ids[id_idx]
        hits = pep_id.getHits()

        if not hits:
            # No hits, fall back to regular spectrum
            return self._create_spectrum_figure(spec, mz_array, int_array, spectrum_idx)

        best_hit = hits[0]
        sequence_str = best_hit.getSequence().toString()
        charge = best_hit.getCharge()
        precursors = spec.getPrecursors()
        prec_mz = precursors[0].getMZ() if precursors else pep_id.getMZ()

        # Compute annotation data
        annotation_data = None
        if self.state.annotate_peaks:
            # First check for external peak annotations (from specialized tools like OpenNuXL)
            external_annotations = get_external_peak_annotations_from_hit(
                best_hit, mz_array, tolerance_da=self.state.annotation_tolerance_da
            )

            if not external_annotations:
                # Check for annotations from SpectrumAnnotator
                external_annotations = annotate_spectrum_with_id(
                    spec, best_hit, tolerance_da=self.state.annotation_tolerance_da
                )

            # Compute annotation data
            annotation_data = compute_spectrum_annotation(
                exp_mz=mz_array,
                exp_int=int_array,
                sequence_str=sequence_str,
                charge=charge,
                precursor_mz=prec_mz,
                tolerance_da=self.state.annotation_tolerance_da,
                external_annotations=external_annotations if external_annotations else None,
            )

        # Create annotated spectrum plot
        fig = create_annotated_spectrum_plot(
            mz_array,
            int_array,
            sequence_str,
            charge,
            prec_mz,
            annotate=self.state.annotate_peaks,
            mirror_mode=self.state.mirror_annotation_view,
            show_unmatched=self.state.show_unmatched_theoretical,
            annotation_data=annotation_data,
        )

        # Update title to include spectrum index and CV if available
        title = f"Spectrum #{spectrum_idx} | {sequence_str} (z={charge}+) | RT={rt:.2f}s"
        if cv is not None:
            title += f" | CV={cv:.1f}V"
        fig.update_layout(
            title={"text": title, "font": {"size": 14}},
            height=350,
            uirevision="spectrum_stable",  # Stable key to preserve zoom/pan state
        )

        # Apply saved zoom range if available
        if self.state.spectrum_zoom_range is not None:
            fig.update_xaxes(range=list(self.state.spectrum_zoom_range))

        # Add measurements, custom annotations, and hover highlights
        # (same as regular spectrum - these overlay on top of ion annotations)
        self._add_measurements_to_figure(fig, spectrum_idx, mz_array, int_array)
        self._add_annotations_to_figure(fig, spectrum_idx, mz_array, int_array)
        self._add_highlight_to_figure(fig, mz_array, int_array)

        return fig

    def _add_measurements_to_figure(
        self, fig: go.Figure, spectrum_idx: int, mz_array: np.ndarray, int_array: np.ndarray
    ):
        """Add stored measurements to the figure.

        Uses shapes instead of traces to match old implementation and avoid
        adding extra data to the figure.

        Args:
            fig: Plotly figure to modify
            spectrum_idx: Current spectrum index
            mz_array: Array of m/z values
            int_array: Array of intensity values
        """
        if spectrum_idx not in self.state.spectrum_measurements:
            return

        max_int = float(int_array.max()) if len(int_array) > 0 else 1.0

        for mz1, int1, mz2, int2 in self.state.spectrum_measurements[spectrum_idx]:
            # Convert to display intensities (percentage if enabled)
            if self.state.spectrum_intensity_percent:
                y1 = (int1 / max_int) * 100
                y2 = (int2 / max_int) * 100
                max_bracket = 90  # Cap at 90% to stay in visible area
            else:
                y1, y2 = int1, int2
                max_bracket = max_int * 0.9

            # Draw horizontal bracket at height slightly above the higher peak, capped at 90%
            bracket_y = min(max(y1, y2) * 1.1, max_bracket)

            # Horizontal line between the two m/z values (orange works in light/dark)
            fig.add_shape(
                type="line",
                x0=mz1,
                y0=bracket_y,
                x1=mz2,
                y1=bracket_y,
                line={"color": "#ff8800", "width": 2},
            )

            # Vertical lines down to each peak
            fig.add_shape(
                type="line",
                x0=mz1,
                y0=y1,
                x1=mz1,
                y1=bracket_y,
                line={"color": "#ff8800", "width": 1, "dash": "dot"},
            )
            fig.add_shape(
                type="line",
                x0=mz2,
                y0=y2,
                x1=mz2,
                y1=bracket_y,
                line={"color": "#ff8800", "width": 1, "dash": "dot"},
            )

            # Calculate delta m/z
            delta_mz = abs(mz2 - mz1)
            mid_mz = (mz1 + mz2) / 2

            # Add annotation with delta m/z
            fig.add_annotation(
                x=mid_mz,
                y=bracket_y,
                text=f"Δ{delta_mz:.4f}",
                showarrow=False,
                yshift=12,
                font={"color": "#ff8800", "size": 11},
                borderpad=2,
            )

    def _add_annotations_to_figure(
        self, fig: go.Figure, spectrum_idx: int, mz_array: np.ndarray, int_array: np.ndarray
    ):
        """Add peak annotations to the figure.

        Args:
            fig: Plotly figure to modify
            spectrum_idx: Current spectrum index
            mz_array: Array of m/z values
            int_array: Array of intensity values
        """
        if spectrum_idx not in self.state.peak_annotations:
            return

        max_int = float(int_array.max()) if len(int_array) > 0 else 1.0
        if self.state.spectrum_intensity_percent:
            scale = 100 / max_int
        else:
            scale = 1.0

        for ann in self.state.peak_annotations[spectrum_idx]:
            mz = ann["mz"]
            intensity = ann["intensity"]
            label = ann.get("label", f"{mz:.4f}")

            fig.add_annotation(
                x=mz,
                y=intensity * scale,
                text=label,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="green",
                font={"size": 9, "color": "green"},
                yshift=10,
            )

    def _add_mz_labels_to_figure(
        self,
        fig: go.Figure,
        spectrum_idx: int,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        max_int: float,
    ):
        """Add m/z labels to interesting peaks if enabled.

        Uses scipy.signal.find_peaks to find prominent peaks and labels them
        with their m/z values. Only considers peaks in the visible range.

        Args:
            fig: Plotly figure to modify
            spectrum_idx: Current spectrum index
            mz_array: Array of m/z values
            int_array: Array of intensity values
            max_int: Maximum intensity for scaling (from full spectrum)
        """
        if not self.state.show_mz_labels or len(mz_array) == 0:
            return

        from scipy.signal import find_peaks

        # Filter to visible range if zoomed
        if self.state.spectrum_zoom_range is not None:
            xmin, xmax = self.state.spectrum_zoom_range
            visible_mask = (mz_array >= xmin) & (mz_array <= xmax)
            mz_visible = mz_array[visible_mask]
            int_visible = int_array[visible_mask]
        else:
            mz_visible = mz_array
            int_visible = int_array

        if len(mz_visible) == 0:
            return

        # Use visible max for prominence calculation (find interesting peaks in current view)
        visible_max = float(int_visible.max()) if len(int_visible) > 0 else max_int

        # Get m/z values that already have custom annotations
        custom_annotations = self.state.peak_annotations.get(spectrum_idx, [])
        annotated_mz = {ann["mz"] for ann in custom_annotations}

        # Use scipy.signal.find_peaks to find interesting peaks in visible range
        min_prominence = visible_max * 0.05  # 5% of visible max intensity
        peak_indices, properties = find_peaks(
            int_visible,
            prominence=min_prominence,
            distance=max(1, len(mz_visible) // 50),  # At least 2% spacing
        )

        # If find_peaks returns too many or too few, fall back to top N by prominence
        if len(peak_indices) > 15:
            # Sort by prominence and keep top 15
            prominences = properties.get("prominences", int_visible[peak_indices])
            sorted_idx = np.argsort(prominences)[-15:]
            peak_indices = peak_indices[sorted_idx]
        elif len(peak_indices) == 0:
            # Fallback: just use top 10 by intensity in visible range
            peak_indices = np.argsort(int_visible)[-10:]

        # Calculate y values based on display mode
        if self.state.spectrum_intensity_percent:
            scale = 100 / max_int
        else:
            scale = 1.0

        for idx in peak_indices:
            mz = float(mz_visible[idx])
            # Skip if this peak already has a custom annotation
            if any(abs(mz - ann_mz) < 0.01 for ann_mz in annotated_mz):
                continue

            intensity = float(int_visible[idx])
            y_val = intensity * scale

            fig.add_annotation(
                x=mz,
                y=y_val,
                text=f"{mz:.2f}",
                showarrow=False,
                yshift=10,
                font={"color": "#888", "size": 9},
            )

    def _add_highlight_to_figure(self, fig: go.Figure, mz_array: np.ndarray, int_array: np.ndarray):
        """Add hover highlight marker and measurement preview to figure.

        Matches old implementation behavior for snap-to-peak highlighting.

        Args:
            fig: Plotly figure to modify
            mz_array: Array of m/z values
            int_array: Array of intensity values
        """
        if len(mz_array) == 0:
            return

        max_int = float(int_array.max()) if len(int_array) > 0 else 1.0

        # Add marker for locked-in start point (distinct from hover)
        if self.state.spectrum_measure_start is not None and self.state.spectrum_measure_mode:
            start_mz, start_int = self.state.spectrum_measure_start
            if self.state.spectrum_intensity_percent:
                start_y = (start_int / max_int) * 100
            else:
                start_y = start_int

            # Orange circle marker for the locked-in start point
            fig.add_trace(
                go.Scatter(
                    x=[start_mz],
                    y=[start_y],
                    mode="markers",
                    marker={"color": "#ff8800", "size": 10, "symbol": "circle", "line": {"width": 1, "color": "#333"}},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Add hover highlight marker to show nearest (snap) peak
        if self.state.spectrum_hover_peak is not None:
            hover_mz, hover_int = self.state.spectrum_hover_peak
            if self.state.spectrum_intensity_percent:
                hover_y = (hover_int / max_int) * 100
            else:
                hover_y = hover_int

            # Highlighted ring around the hovered peak
            highlight_color = "#ff8800" if self.state.spectrum_measure_mode else "#0077cc"
            fig.add_trace(
                go.Scatter(
                    x=[hover_mz],
                    y=[hover_y],
                    mode="markers",
                    marker={"color": highlight_color, "size": 12, "symbol": "circle-open", "line": {"width": 2}},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Add preview line from start point to hover point
        if self.state.spectrum_measure_start is not None and self.state.spectrum_hover_peak is not None:
            start_mz, start_int = self.state.spectrum_measure_start
            hover_mz, hover_int = self.state.spectrum_hover_peak

            if self.state.spectrum_intensity_percent:
                y1 = (start_int / max_int) * 100
                y2 = (hover_int / max_int) * 100
                max_bracket = 90
            else:
                y1, y2 = start_int, hover_int
                max_bracket = max_int * 0.9

            bracket_y = min(max(y1, y2) * 1.1, max_bracket)

            # Preview horizontal line (dashed, semi-transparent orange)
            fig.add_shape(
                type="line",
                x0=start_mz,
                y0=bracket_y,
                x1=hover_mz,
                y1=bracket_y,
                line={"color": "rgba(255, 136, 0, 0.6)", "width": 2, "dash": "dash"},
            )

            # Preview vertical connectors
            fig.add_shape(
                type="line",
                x0=start_mz,
                y0=y1,
                x1=start_mz,
                y1=bracket_y,
                line={"color": "rgba(255, 136, 0, 0.6)", "width": 1, "dash": "dot"},
            )
            fig.add_shape(
                type="line",
                x0=hover_mz,
                y0=y2,
                x1=hover_mz,
                y1=bracket_y,
                line={"color": "rgba(255, 136, 0, 0.6)", "width": 1, "dash": "dot"},
            )

            # Preview delta text
            delta_mz = abs(hover_mz - start_mz)
            fig.add_annotation(
                x=(start_mz + hover_mz) / 2,
                y=bracket_y,
                text=f"Δ{delta_mz:.4f}",
                showarrow=False,
                yshift=12,
                font={"color": "rgba(255, 136, 0, 0.8)", "size": 10},
            )

    # === Event handlers ===

    def _on_data_loaded(self, data_type: str):
        """Handle data loaded event."""
        if data_type == "mzml":
            if self._has_data():
                # Show first spectrum
                self.show_spectrum(0)
                # Auto-expand panel
                if self.expansion:
                    self.expansion.value = True
            else:
                # Clear display when data is removed
                self._clear_display()
                if self.expansion:
                    self.expansion.value = False
        elif data_type == "ids":
            # Refresh current spectrum to update annotations
            if self.state.selected_spectrum_idx is not None:
                self.show_spectrum(self.state.selected_spectrum_idx)

    def _clear_display(self):
        """Clear the spectrum display."""
        if self.spectrum_plot is not None:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis_title="m/z",
                yaxis_title="Intensity",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            self.spectrum_plot.update_figure(self._figure_with_config(empty_fig))
        if self.info_label is not None:
            self.info_label.set_text("No spectrum loaded")

    def _on_selection_changed(self, selection_type: str, index: Optional[int]):
        """Handle selection changed event."""
        if selection_type == "spectrum" and index is not None:
            self.show_spectrum(index)

    def _on_plot_click(self, e):
        """Handle click on spectrum plot for measurement and annotation."""
        try:
            if not e.args:
                return

            # Get clicked point
            points = e.args.get("points", [])
            if not points:
                return

            clicked_mz = points[0].get("x")
            clicked_y = points[0].get("y")
            if clicked_mz is None:
                return

            # Get current spectrum data
            if self.state.selected_spectrum_idx is None or self.state.exp is None:
                return

            spec = self.state.exp[self.state.selected_spectrum_idx]
            mz_array, int_array = spec.get_peaks()

            if len(mz_array) == 0:
                return

            # Check if clicking on an existing measurement line (for selection)
            if clicked_y is not None:
                measurement_idx = self._find_measurement_at_position(clicked_mz, clicked_y, mz_array, int_array)
                if measurement_idx is not None:
                    # Toggle selection
                    if self.state.spectrum_selected_measurement_idx == measurement_idx:
                        self.state.spectrum_selected_measurement_idx = None
                        ui.notify("Measurement deselected", type="info")
                    else:
                        self.state.spectrum_selected_measurement_idx = measurement_idx
                        ui.notify("Measurement selected - press Delete to remove", type="info")
                    self.show_spectrum(self.state.selected_spectrum_idx)
                    return

            # Clear selection if clicking elsewhere (not on a measurement)
            if self.state.spectrum_selected_measurement_idx is not None:
                self.state.spectrum_selected_measurement_idx = None
                self.show_spectrum(self.state.selected_spectrum_idx)

            # Handle annotation mode - click to add/edit peak labels
            if self.state.peak_annotation_mode:
                snapped = self._snap_to_peak(clicked_mz, mz_array, int_array, clicked_y)
                if snapped is None:
                    ui.notify("No peak found near click position", type="warning")
                    return

                snapped_mz, snapped_int = snapped
                self._show_annotation_dialog(snapped_mz, snapped_int)
                return

            # Only handle peak measurement when measurement mode is active
            if not self.state.spectrum_measure_mode:
                return

            # Snap to nearest peak
            snapped = self._snap_to_peak(clicked_mz, mz_array, int_array)
            if snapped is None:
                ui.notify("No peak found near click position", type="warning")
                return

            snapped_mz, snapped_int = snapped

            if self.state.spectrum_measure_start is None:
                # First click - set start point
                self.state.spectrum_measure_start = (snapped_mz, snapped_int)
                ui.notify(f"Start: m/z {snapped_mz:.4f} - click second peak", type="info")
                # Re-render to show start marker (since hover is disabled)
                self.show_spectrum(self.state.selected_spectrum_idx)
            else:
                # Second click - complete measurement
                start_mz, start_int = self.state.spectrum_measure_start
                self.state.spectrum_measure_start = None
                self.state.spectrum_hover_peak = None

                # Store the measurement
                spectrum_idx = self.state.selected_spectrum_idx
                if spectrum_idx not in self.state.spectrum_measurements:
                    self.state.spectrum_measurements[spectrum_idx] = []
                self.state.spectrum_measurements[spectrum_idx].append((start_mz, start_int, snapped_mz, snapped_int))

                delta_mz = abs(snapped_mz - start_mz)
                ui.notify(f"Δm/z = {delta_mz:.4f}", type="positive")

                # Refresh display to show the measurement
                self.show_spectrum(spectrum_idx)

        except Exception:
            pass

    def _snap_to_peak(
        self, target_mz: float, mz_array: np.ndarray, int_array: np.ndarray, target_int: Optional[float] = None
    ) -> Optional[tuple[float, float]]:
        """Snap to the nearest peak using 2D distance (m/z and intensity).

        Args:
            target_mz: Target m/z value from click
            mz_array: Array of m/z values in spectrum
            int_array: Array of intensity values in spectrum
            target_int: Optional target intensity for 2D snapping

        Returns:
            Tuple of (mz, intensity) for snapped peak, or None if no peak nearby
        """
        if len(mz_array) == 0:
            return None

        # Normalize both dimensions to [0, 1] for fair distance comparison
        mz_min, mz_max = float(mz_array.min()), float(mz_array.max())
        int_min, int_max = float(int_array.min()), float(int_array.max())

        mz_range = mz_max - mz_min if mz_max > mz_min else 1.0
        int_range = int_max - int_min if int_max > int_min else 1.0

        # Normalize arrays
        mz_norm = (mz_array - mz_min) / mz_range
        target_mz_norm = (target_mz - mz_min) / mz_range

        if target_int is not None:
            # Use 2D Euclidean distance (m/z and intensity)
            int_norm = (int_array - int_min) / int_range
            target_int_norm = (target_int - int_min) / int_range
            distances = np.sqrt((mz_norm - target_mz_norm) ** 2 + (int_norm - target_int_norm) ** 2)
        else:
            # Fall back to m/z-only distance
            distances = np.abs(mz_norm - target_mz_norm)

        idx = distances.argmin()
        snapped_mz = float(mz_array[idx])
        snapped_int = float(int_array[idx])

        # Only snap if within a reasonable tolerance (1% of m/z range or 1 Da, whichever is larger)
        tolerance = max(1.0, mz_range * 0.01)

        if abs(snapped_mz - target_mz) > tolerance:
            return None

        return (snapped_mz, snapped_int)

    def _find_measurement_at_position(
        self, mz: float, y: float, mz_array: np.ndarray, int_array: np.ndarray
    ) -> Optional[int]:
        """Find if a click position is near an existing measurement line.

        Args:
            mz: Clicked m/z position
            y: Clicked y position (intensity, possibly in %)
            mz_array: Array of m/z values in spectrum
            int_array: Array of intensity values in spectrum

        Returns:
            Measurement index if found, None otherwise
        """
        if self.state.selected_spectrum_idx is None:
            return None
        if self.state.selected_spectrum_idx not in self.state.spectrum_measurements:
            return None

        if len(mz_array) == 0:
            return None

        max_int = float(int_array.max())
        mz_range = float(mz_array.max() - mz_array.min())
        mz_tolerance = mz_range * 0.02  # 2% of m/z range

        measurements = self.state.spectrum_measurements[self.state.selected_spectrum_idx]
        for i, (mz1, int1, mz2, int2) in enumerate(measurements):
            # Convert to display intensities
            if self.state.spectrum_intensity_percent:
                y1 = (int1 / max_int) * 100
                y2 = (int2 / max_int) * 100
            else:
                y1, y2 = int1, int2

            # Calculate bracket y position (same logic as in drawing)
            bracket_y = max(y1, y2) * 1.1

            # Check if click is near the horizontal line
            if min(mz1, mz2) - mz_tolerance <= mz <= max(mz1, mz2) + mz_tolerance:
                # Check if y is near the bracket line (within 10% of bracket height)
                y_tolerance = bracket_y * 0.1
                if abs(y - bracket_y) < y_tolerance:
                    return i

        return None

    def _show_annotation_dialog(self, mz: float, intensity: float):
        """Show dialog to add/edit a peak annotation.

        Args:
            mz: m/z value of the peak
            intensity: Intensity value of the peak
        """
        spectrum_idx = self.state.selected_spectrum_idx

        # Check if annotation already exists at this m/z
        existing_label = ""
        if spectrum_idx in self.state.peak_annotations:
            for ann in self.state.peak_annotations[spectrum_idx]:
                if abs(ann["mz"] - mz) < 0.01:
                    existing_label = ann.get("label", "")
                    break

        # Create dialog
        with ui.dialog() as dialog, ui.card().classes("min-w-[300px]"):
            ui.label("Peak Annotation").classes("text-lg font-bold")
            ui.label(f"m/z: {mz:.4f}").classes("text-sm text-gray-400")

            label_input = ui.input(
                "Label",
                value=existing_label,
                placeholder=f"{mz:.4f}",
            ).classes("w-full")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):

                def delete_annotation():
                    self._remove_peak_annotation(spectrum_idx, mz)
                    dialog.close()
                    self.show_spectrum(spectrum_idx)
                    ui.notify("Annotation removed", type="info")

                def save_annotation():
                    label = label_input.value.strip() if label_input.value else None
                    self._add_or_edit_peak_annotation(spectrum_idx, mz, intensity, label)
                    dialog.close()
                    self.show_spectrum(spectrum_idx)
                    ui.notify("Annotation saved", type="positive")

                if existing_label:
                    ui.button("Delete", on_click=delete_annotation, color="red").props("flat")

                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save", on_click=save_annotation, color="primary")

        dialog.open()

    def _add_or_edit_peak_annotation(self, spectrum_idx: int, mz: float, intensity: float, label: Optional[str] = None):
        """Add or edit a peak annotation.

        Args:
            spectrum_idx: Spectrum index
            mz: m/z value of the peak
            intensity: Intensity value of the peak
            label: Label text (defaults to m/z value)
        """
        if spectrum_idx not in self.state.peak_annotations:
            self.state.peak_annotations[spectrum_idx] = []

        annotations = self.state.peak_annotations[spectrum_idx]

        # Check if annotation already exists at this m/z (within tolerance)
        for ann in annotations:
            if abs(ann["mz"] - mz) < 0.01:
                # Update existing annotation
                ann["label"] = label if label is not None else f"{mz:.4f}"
                return

        # Add new annotation
        annotations.append({"mz": mz, "intensity": intensity, "label": label if label is not None else f"{mz:.4f}"})

    def _remove_peak_annotation(self, spectrum_idx: int, mz: float):
        """Remove a peak annotation at the given m/z.

        Args:
            spectrum_idx: Spectrum index
            mz: m/z value of the annotation to remove
        """
        if spectrum_idx not in self.state.peak_annotations:
            return

        annotations = self.state.peak_annotations[spectrum_idx]
        self.state.peak_annotations[spectrum_idx] = [ann for ann in annotations if abs(ann["mz"] - mz) >= 0.01]

    def _on_plot_relayout(self, e):
        """Handle zoom/pan changes on spectrum plot."""
        try:
            if not e.args:
                return
            xmin = e.args.get("xaxis.range[0]")
            xmax = e.args.get("xaxis.range[1]")
            if xmin is not None and xmax is not None:
                self.state.spectrum_zoom_range = (xmin, xmax)
                # Re-render to apply auto-scale, re-downsample, or update m/z labels for visible range
                if (
                    self.state.spectrum_auto_scale or self.state.peakmap_downsampling or self.state.show_mz_labels
                ) and self.state.selected_spectrum_idx is not None:
                    self.show_spectrum(self.state.selected_spectrum_idx)
                # Sync m/z range to IM peakmap if linking is enabled
                if self.state.link_spectrum_mz_to_im and self.state.has_ion_mobility:
                    self.state.view_mz_min = max(self.state.mz_min, xmin)
                    self.state.view_mz_max = min(self.state.mz_max, xmax)
                    self.state.emit_view_changed()
            elif e.args.get("xaxis.autorange"):
                self.state.spectrum_zoom_range = None
                # Re-render to reset y-axis, re-downsample full spectrum, or update m/z labels
                if (
                    self.state.spectrum_auto_scale or self.state.peakmap_downsampling or self.state.show_mz_labels
                ) and self.state.selected_spectrum_idx is not None:
                    self.show_spectrum(self.state.selected_spectrum_idx)
                # Reset IM m/z range if linking is enabled
                if self.state.link_spectrum_mz_to_im and self.state.has_ion_mobility:
                    self.state.view_mz_min = self.state.mz_min
                    self.state.view_mz_max = self.state.mz_max
                    self.state.emit_view_changed()
        except Exception:
            pass

    def _on_plot_hover(self, e):
        """Handle hover on spectrum plot to highlight nearest peak.

        Shows preview line in measure mode. Matches old implementation behavior.
        """
        try:
            if not e.args:
                return

            points = e.args.get("points", [])
            if not points:
                return

            hovered_mz = points[0].get("x")
            hovered_int = points[0].get("y")
            if hovered_mz is None:
                return

            if self.state.selected_spectrum_idx is None or self.state.exp is None:
                return

            spec = self.state.exp[self.state.selected_spectrum_idx]
            mz_array, int_array = spec.get_peaks()

            if len(mz_array) == 0:
                return

            # Convert hovered_int from display units to raw units if in percentage mode
            if hovered_int is not None and self.state.spectrum_intensity_percent:
                max_int = float(int_array.max()) if len(int_array) > 0 else 1.0
                hovered_int_raw = (hovered_int / 100.0) * max_int
            else:
                hovered_int_raw = hovered_int

            # Snap to nearest peak using 2D distance (m/z and intensity)
            snapped = self._snap_to_peak(hovered_mz, mz_array, int_array, hovered_int_raw)
            if snapped is None:
                return

            # Only refresh if the hovered peak actually changed (optimization)
            if self.state.spectrum_hover_peak == snapped:
                return

            self.state.spectrum_hover_peak = snapped
            # Refresh to show preview - zoom will be preserved by uirevision
            self.show_spectrum(self.state.selected_spectrum_idx)

        except Exception:
            pass

    def _on_plot_unhover(self, e):
        """Handle unhover - clear hover highlight."""
        if self.state.spectrum_hover_peak is not None:
            self.state.spectrum_hover_peak = None
            if self.state.selected_spectrum_idx is not None:
                self.show_spectrum(self.state.selected_spectrum_idx)

    # === Navigation methods ===

    def _navigate(self, direction: int):
        """Navigate to prev/next spectrum."""
        if self.state.exp is None or len(self.state.exp) == 0:
            return

        if self.state.selected_spectrum_idx is None:
            new_idx = 0
        else:
            new_idx = self.state.selected_spectrum_idx + direction

        new_idx = max(0, min(len(self.state.exp) - 1, new_idx))
        self.show_spectrum(new_idx)

    def _navigate_to(self, index: int):
        """Navigate to a specific spectrum index."""
        if self.state.exp is None or len(self.state.exp) == 0:
            return
        index = max(0, min(len(self.state.exp) - 1, index))
        self.show_spectrum(index)

    def _navigate_by_ms_level(self, direction: int, ms_level: int):
        """Navigate to prev/next spectrum of specific MS level."""
        if self.state.exp is None or len(self.state.exp) == 0:
            return

        start_idx = self.state.selected_spectrum_idx if self.state.selected_spectrum_idx is not None else -1

        if direction > 0:
            for i in range(start_idx + 1, len(self.state.exp)):
                if self.state.exp[i].getMSLevel() == ms_level:
                    self.show_spectrum(i)
                    return
        else:
            for i in range(start_idx - 1, -1, -1):
                if self.state.exp[i].getMSLevel() == ms_level:
                    self.show_spectrum(i)
                    return

    # === Toggle methods ===

    def _toggle_intensity_mode(self, e):
        """Toggle between % and absolute intensity."""
        self.state.spectrum_intensity_percent = e.value == "%"
        if self.state.selected_spectrum_idx is not None:
            self.show_spectrum(self.state.selected_spectrum_idx)

    def _toggle_auto_scale(self, e):
        """Toggle auto Y-axis scaling."""
        self.state.spectrum_auto_scale = e.value
        if self.state.selected_spectrum_idx is not None:
            self.show_spectrum(self.state.selected_spectrum_idx)

    def _toggle_measure_mode(self):
        """Toggle measurement mode."""
        self.state.spectrum_measure_mode = not self.state.spectrum_measure_mode
        self.state.spectrum_measure_start = None
        self.state.spectrum_hover_peak = None

        # Disable annotation mode when measure mode is active
        if self.state.spectrum_measure_mode and self.state.peak_annotation_mode:
            self.state.peak_annotation_mode = False
            if self.annotation_btn:
                self.annotation_btn.props("color=grey")

        if self.measure_btn:
            color = "yellow" if self.state.spectrum_measure_mode else "grey"
            self.measure_btn.props(f"color={color}")

        if self.state.spectrum_measure_mode:
            ui.notify("Measure mode ON - click two peaks to measure Δm/z", type="info")
        else:
            ui.notify("Measure mode OFF", type="info")

        if self.state.selected_spectrum_idx is not None:
            self.show_spectrum(self.state.selected_spectrum_idx)

    def _toggle_annotation_mode(self):
        """Toggle annotation mode."""
        self.state.peak_annotation_mode = not self.state.peak_annotation_mode

        # Disable measure mode when annotation mode is active
        if self.state.peak_annotation_mode and self.state.spectrum_measure_mode:
            self.state.spectrum_measure_mode = False
            self.state.spectrum_measure_start = None
            if self.measure_btn:
                self.measure_btn.props("color=grey")

        if self.annotation_btn:
            color = "green" if self.state.peak_annotation_mode else "grey"
            self.annotation_btn.props(f"color={color}")

        if self.state.peak_annotation_mode:
            ui.notify("Label mode ON - click peaks to add labels", type="info")
        else:
            ui.notify("Label mode OFF", type="info")

    def _toggle_mz_labels(self, e):
        """Toggle m/z label display."""
        self.state.show_mz_labels = e.value
        if self.state.selected_spectrum_idx is not None:
            self.show_spectrum(self.state.selected_spectrum_idx)

    def _toggle_show_unmatched(self, e):
        """Toggle unmatched theoretical peaks display in mirror mode."""
        self.state.show_unmatched_theoretical = e.value
        if self.state.selected_spectrum_idx is not None:
            self.show_spectrum(self.state.selected_spectrum_idx)

    def _clear_measurements(self):
        """Clear measurements for current spectrum."""
        if self.state.selected_spectrum_idx is not None:
            if self.state.selected_spectrum_idx in self.state.spectrum_measurements:
                del self.state.spectrum_measurements[self.state.selected_spectrum_idx]
            self.state.spectrum_selected_measurement_idx = None
            self.show_spectrum(self.state.selected_spectrum_idx)
            ui.notify("Measurements cleared", type="info")

    def _clear_annotations(self):
        """Clear annotations for current spectrum."""
        if self.state.selected_spectrum_idx is not None:
            if self.state.selected_spectrum_idx in self.state.peak_annotations:
                del self.state.peak_annotations[self.state.selected_spectrum_idx]
            self.show_spectrum(self.state.selected_spectrum_idx)
            ui.notify("Annotations cleared", type="info")
