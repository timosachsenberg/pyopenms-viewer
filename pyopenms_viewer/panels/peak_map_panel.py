"""Peak map (2D RT vs m/z) panel component.

This panel displays the main 2D peak map visualization using datashader,
with interactive mouse controls for zoom, pan, and measurement.
"""

import time
from typing import Callable, Optional

from nicegui import ui
from nicegui.events import MouseEventArguments

from pyopenms_viewer.core.config import COLORMAPS
from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel
from pyopenms_viewer.rendering import MinimapRenderer, PeakMapRenderer


class PeakMapPanel(BasePanel):
    """2D Peak Map visualization panel.

    Features:
    - Datashader-rendered peak map (RT vs m/z)
    - Feature/ID overlay options
    - Mouse drag-to-zoom, shift+drag to measure, ctrl+drag to pan
    - Mouse wheel zoom
    - Minimap navigation
    - 3D view toggle
    """

    def __init__(self, state: ViewerState):
        """Initialize peak map panel.

        Args:
            state: ViewerState instance (shared reference)
        """
        super().__init__(state, "peakmap", "2D Peak Map", "grid_on")

        # UI elements
        self.image_element: Optional[ui.interactive_image] = None
        self.minimap_image: Optional[ui.image] = None
        self.coord_label: Optional[ui.label] = None
        self.breadcrumb_label: Optional[ui.label] = None
        self.scene_3d_container: Optional[ui.column] = None
        self.plot_3d = None
        self.view_3d_status: Optional[ui.label] = None
        self.view_3d_btn = None

        # Checkboxes for overlay options
        self.centroid_cb = None
        self.bbox_cb = None
        self.hull_cb = None
        self.ids_cb = None
        self.id_seq_cb = None
        self.swap_axes_cb = None
        self.spectrum_marker_cb = None

        # FAIMS UI elements
        self.faims_checkbox: Optional[ui.checkbox] = None
        self.faims_container: Optional[ui.column] = None
        self.faims_cv_minimaps: dict[float, ui.image] = {}
        self.faims_cv_labels: dict[float, ui.label] = {}

        # Renderers
        self.peak_map_renderer = PeakMapRenderer(
            plot_width=state.plot_width,
            plot_height=state.plot_height,
            margin_left=state.margin_left,
            margin_right=state.margin_right,
            margin_top=state.margin_top,
            margin_bottom=state.margin_bottom,
        )
        self.minimap_renderer = MinimapRenderer(
            width=state.minimap_width,
            height=state.minimap_height,
        )

        # Drag state for mouse interactions
        self._drag_state = {
            "dragging": False,
            "measuring": False,
            "panning": False,
            "start_x": 0,
            "start_y": 0,
            "pan_rt_min": 0,
            "pan_rt_max": 0,
            "pan_mz_min": 0,
            "pan_mz_max": 0,
            "last_pan_render": 0.0,
        }

        # Callback for external update triggers
        self._on_update_callback: Optional[Callable] = None

        # Last hover update time for debouncing
        self._last_hover_update: float = 0.0
        self._hover_debounce_ms: float = 50.0  # Minimum ms between hover updates

        # Plotly config for 3D view (must be included in figure dict)
        self._plotly_config = {
            "modeBarButtonsToRemove": ["autoScale2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "peak_map_3d",
                "width": 1200,
                "height": 800,
                "scale": 1,
            },
        }

    def _figure_with_config(self, fig) -> dict:
        """Convert go.Figure to dict and add config for modebar customization."""
        fig_dict = fig.to_plotly_json()
        fig_dict["config"] = self._plotly_config
        return fig_dict

    def build(self, container: ui.element) -> ui.expansion:
        """Build the peak map panel UI.

        Args:
            container: Parent element to build panel in

        Returns:
            The expansion element created
        """
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                self._build_options_row()
                self._build_navigation_row()
                self._build_help_text()
                self._build_peak_map_area()

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)
        self.state.on_view_changed(self._on_view_changed)
        self.state.on_selection_changed(self._on_selection_changed)

        self._is_built = True
        return self.expansion

    def _build_options_row(self):
        """Build the display options row with checkboxes and controls."""
        with ui.row().classes("w-full items-center gap-4 mb-2 flex-wrap"):
            ui.label("Overlay:").classes("text-xs text-gray-400")

            # Centroids checkbox
            self.centroid_cb = (
                ui.checkbox("Centroids", value=True, on_change=self._toggle_centroids)
                .props("dense")
                .classes("text-green-400")
            )

            # Bounding boxes checkbox
            self.bbox_cb = (
                ui.checkbox("Bounding Boxes", value=False, on_change=self._toggle_bboxes)
                .props("dense")
                .classes("text-yellow-400")
            )

            # Convex hulls checkbox
            self.hull_cb = (
                ui.checkbox("Convex Hulls", value=False, on_change=self._toggle_hulls)
                .props("dense")
                .classes("text-cyan-400")
            )

            # IDs checkbox
            self.ids_cb = (
                ui.checkbox("Identifications", value=True, on_change=self._toggle_ids)
                .props("dense")
                .classes("text-orange-400")
            )

            # Sequences checkbox
            self.id_seq_cb = (
                ui.checkbox("Sequences", value=False, on_change=self._toggle_id_sequences)
                .props("dense")
                .classes("text-orange-300")
            )
            ui.tooltip("Show peptide sequences on 2D peakmap")

            ui.label("|").classes("text-gray-600 mx-2")
            ui.label("Colormap:").classes("text-xs text-gray-400")

            # Colormap selector
            colormap_options = list(COLORMAPS.keys())
            ui.select(colormap_options, value=self.state.colormap, on_change=self._change_colormap).props(
                "dense outlined"
            ).classes("w-28")

            ui.label("|").classes("text-gray-600 mx-2")
            ui.label("RT:").classes("text-xs text-gray-400")

            # RT unit toggle
            ui.toggle(
                ["sec", "min"], value="min" if self.state.rt_in_minutes else "sec", on_change=self._toggle_rt_unit
            ).props("dense")

            ui.label("|").classes("text-gray-600 mx-2")

            # Swap axes checkbox
            self.swap_axes_cb = (
                ui.checkbox("Swap Axes", value=self.state.swap_axes, on_change=self._toggle_swap_axes)
                .props("dense")
                .classes("text-purple-400")
            )
            ui.tooltip(
                "When checked: m/z on x-axis, RT on y-axis (default). When unchecked: RT on x-axis, m/z on y-axis."
            )

            # Spectrum marker checkbox
            self.spectrum_marker_cb = (
                ui.checkbox("Marker", value=self.state.show_spectrum_marker, on_change=self._toggle_spectrum_marker)
                .props("dense")
                .classes("text-cyan-400")
            )
            ui.tooltip("Show/hide the spectrum position marker (crosshair) on the 2D peakmap.")

            ui.element("div").classes("flex-grow")

            # Save PNG button
            ui.button(icon="download", on_click=self._save_png).props("dense flat size=sm").tooltip(
                "Save peak map as PNG"
            )

    def _build_navigation_row(self):
        """Build the breadcrumb trail and coordinate display row."""
        with ui.row().classes("w-full items-center justify-between mb-1"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("navigation", size="xs").classes("text-gray-400")
                self.breadcrumb_label = ui.label("Full view").classes(
                    "text-xs text-gray-400 cursor-pointer hover:text-cyan-400"
                )
                self.breadcrumb_label.on("click", self._open_range_popover)
                ui.tooltip("Click to set exact range (or press G)")

                # Range popover (hidden by default)
                self._build_range_popover()

            self.coord_label = ui.label("RT: --  m/z: --").classes("text-xs text-cyan-400 font-mono")

        # Store reference in state for other components
        self.state.coord_label = self.coord_label
        self.state.breadcrumb_label = self.breadcrumb_label

    def _build_range_popover(self):
        """Build the compact range input popover."""
        self.range_dialog = ui.dialog().props("persistent")
        with self.range_dialog:
            with ui.card().classes("p-4"):
                ui.label("Go to Range").classes("text-sm font-bold mb-2")
                with ui.row().classes("gap-2 items-end"):
                    ui.label("RT:").classes("text-xs text-gray-400")
                    self.range_rt_min = (
                        ui.number(format="%.1f").props("dense outlined").classes("w-24").style("font-size: 12px;")
                    )
                    ui.label("–").classes("text-gray-400")
                    self.range_rt_max = (
                        ui.number(format="%.1f").props("dense outlined").classes("w-24").style("font-size: 12px;")
                    )
                    self.range_rt_unit_label = ui.label("s").classes("text-xs text-gray-400")

                with ui.row().classes("gap-2 items-end mt-2"):
                    ui.label("m/z:").classes("text-xs text-gray-400")
                    self.range_mz_min = (
                        ui.number(format="%.2f").props("dense outlined").classes("w-24").style("font-size: 12px;")
                    )
                    ui.label("–").classes("text-gray-400")
                    self.range_mz_max = (
                        ui.number(format="%.2f").props("dense outlined").classes("w-24").style("font-size: 12px;")
                    )

                with ui.row().classes("gap-2 mt-3 justify-end"):
                    ui.button("Reset", on_click=self._reset_range_from_dialog).props("flat dense color=grey")
                    ui.button("Cancel", on_click=self.range_dialog.close).props("flat dense")
                    ui.button("Apply", on_click=self._apply_range_from_dialog).props("dense color=primary")

    def _open_range_popover(self):
        """Open the range input popover and populate with current values."""
        if not self._has_data():
            ui.notify("No data loaded", type="warning")
            return

        # Populate with current view values
        if self.state.rt_in_minutes:
            self.range_rt_min.value = self.state.view_rt_min / 60
            self.range_rt_max.value = self.state.view_rt_max / 60
            self.range_rt_unit_label.set_text("min")
        else:
            self.range_rt_min.value = self.state.view_rt_min
            self.range_rt_max.value = self.state.view_rt_max
            self.range_rt_unit_label.set_text("s")

        self.range_mz_min.value = self.state.view_mz_min
        self.range_mz_max.value = self.state.view_mz_max

        self.range_dialog.open()

    def _apply_range_from_dialog(self):
        """Apply the range from the dialog inputs."""
        rt_min = self.range_rt_min.value
        rt_max = self.range_rt_max.value
        mz_min = self.range_mz_min.value
        mz_max = self.range_mz_max.value

        # Convert from minutes if needed
        if self.state.rt_in_minutes:
            rt_min = rt_min * 60
            rt_max = rt_max * 60

        # Validate ranges
        if rt_min >= rt_max:
            ui.notify("RT Min must be less than RT Max", type="warning")
            return
        if mz_min >= mz_max:
            ui.notify("m/z Min must be less than m/z Max", type="warning")
            return

        # Clamp to data bounds
        self.state.view_rt_min = max(self.state.rt_min, rt_min)
        self.state.view_rt_max = min(self.state.rt_max, rt_max)
        self.state.view_mz_min = max(self.state.mz_min, mz_min)
        self.state.view_mz_max = min(self.state.mz_max, mz_max)

        self.state.emit_view_changed()
        self.range_dialog.close()
        ui.notify("Range applied", type="positive")

    def _reset_range_from_dialog(self):
        """Reset to full data range from dialog."""
        self.state.reset_view()
        self.range_dialog.close()
        ui.notify("Reset to full range", type="info")

    def _build_help_text(self):
        """Build the help text."""
        ui.label(
            "Scroll to zoom, drag to select region, Shift+drag to measure, double-click to reset, "
            "click centroid to select feature"
        ).classes("text-xs text-gray-500 mb-1")

    def _build_peak_map_area(self):
        """Build the peak map image and minimap."""
        with ui.row().classes("w-full items-start gap-2"):
            # Peak map image with mouse handlers
            with ui.column().classes("flex-none"):
                self.image_element = (
                    ui.interactive_image(
                        on_mouse=self._on_peakmap_mouse,
                        events=["mousedown", "mousemove", "mouseup"],
                        cross=False,
                    )
                    .classes("w-full")
                    .style(
                        f"width: {self.state.canvas_width}px; height: {self.state.canvas_height}px; "
                        f"background: transparent; cursor: crosshair;"
                    )
                )

                # Store reference in state
                self.state.image_element = self.image_element

                # Mouse wheel zoom handler
                self.image_element.on("wheel.prevent", self._on_wheel)

                # Double-click to reset
                self.image_element.on("dblclick", lambda e: self._reset_view())

                # Mouse leave handler
                self.image_element.on("mouseleave", self._on_mouseleave)

                # Ctrl key release during panning
                self.image_element.on("keyup", self._on_keyup)

                # Note: Removed document-level mouseup handler as it triggers on ALL mouseups
                # and can cause excessive websocket traffic. Panning cleanup is handled
                # by mouseleave and keyup events instead.

                # 3D View Container (hidden by default)
                self._build_3d_container()

            # Minimap panel
            self._build_minimap()

    def _build_3d_container(self):
        """Build the 3D view container (hidden by default)."""
        import plotly.graph_objects as go

        self.scene_3d_container = ui.column().classes("w-full mt-1")
        self.scene_3d_container.set_visibility(False)

        with self.scene_3d_container:
            self.view_3d_status = ui.label("").classes("text-xs text-yellow-400")

            # Create empty plotly figure for 3D view
            empty_fig = go.Figure()
            empty_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#888"},
                width=self.state.canvas_width,
                height=500,
                autosize=False,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            )

            with ui.element("div").style(f"width: {self.state.canvas_width}px; height: 500px;"):
                self.plot_3d = ui.plotly(self._figure_with_config(empty_fig)).classes("w-full h-full")

        # Store reference in state
        self.state.scene_3d_container = self.scene_3d_container
        self.state.plot_3d = self.plot_3d
        self.state.view_3d_status = self.view_3d_status

    def _build_minimap(self):
        """Build the minimap navigation panel."""
        with ui.column().classes("flex-none"):
            ui.label("Overview").classes("text-xs text-gray-400 mb-1")

            self.minimap_image = ui.image().style(
                f"width: {self.state.minimap_width}px; height: {self.state.minimap_height}px; "
                f"background: transparent; cursor: pointer; border: 1px solid #888;"
            )
            self.minimap_image.on("click", self._on_minimap_click)

            # Store reference in state
            self.state.minimap_image = self.minimap_image

            # Back and 3D View buttons
            with ui.row().classes("mt-1 gap-1"):
                ui.button("← Back", on_click=self._go_back).props("dense size=sm color=grey").tooltip(
                    "Go to previous view"
                )

                self.view_3d_btn = (
                    ui.button("3D", on_click=self._toggle_3d_view)
                    .props("dense size=sm color=grey")
                    .tooltip("Toggle 3D peak view")
                )

            # FAIMS checkbox (hidden until FAIMS data is detected)
            self.faims_checkbox = (
                ui.checkbox("FAIMS CV Filter", value=False, on_change=self._toggle_faims_filter)
                .props("dense")
                .classes("text-purple-400 mt-2")
            )
            self.faims_checkbox.set_visibility(False)

            # Container for per-CV minimaps (hidden until FAIMS filter is enabled)
            self.faims_container = ui.column().classes("mt-1 gap-1")
            self.faims_container.set_visibility(False)

    def update(self) -> None:
        """Update the peak map display."""
        if not self._has_data() or self.image_element is None:
            return

        # Render peak map
        base64_img = self.peak_map_renderer.render(self.state, fast=False)
        if base64_img:
            self.image_element.set_source(f"data:image/png;base64,{base64_img}")

        # Update minimap
        self.update_minimap()

        # Update breadcrumb
        self._update_breadcrumb()

    def update_minimap(self) -> None:
        """Update the minimap display."""
        if not self._has_data() or self.minimap_image is None:
            return

        base64_img = self.minimap_renderer.render(self.state)
        if base64_img:
            self.minimap_image.set_source(f"data:image/png;base64,{base64_img}")

        # Also update FAIMS CV minimaps if present
        if self.state.has_faims:
            self._update_faims_cv_minimaps()

    def update_lightweight(self) -> None:
        """Update with faster rendering (for panning)."""
        if not self._has_data() or self.image_element is None:
            return

        base64_img = self.peak_map_renderer.render(self.state, fast=True)
        if base64_img:
            self.image_element.set_source(f"data:image/png;base64,{base64_img}")

    def _has_data(self) -> bool:
        """Check if panel has data to display.

        Works for both in-memory mode (state.df is not None) and
        out-of-core mode (state.df is None but data_manager has data).
        """
        if self.state.df is not None:
            return True
        # Out-of-core mode: check data_manager
        if self.state.data_manager is not None:
            return self.state.data_manager.get_peak_count() > 0
        return False

    # === Event handlers ===

    def _on_data_loaded(self, data_type: str):
        """Handle data loaded event."""
        if data_type == "mzml":
            if self._has_data():
                self.update()
                # Auto-expand panel when data loaded
                if self.expansion:
                    self.expansion.value = True
            else:
                # Clear display when data is removed
                self._clear_display()
                if self.expansion:
                    self.expansion.value = False
            # Show/hide FAIMS checkbox based on data
            if self.faims_checkbox:
                self.faims_checkbox.set_visibility(self.state.has_faims)
                if self.state.has_faims:
                    self._create_faims_cv_minimaps()

    def _clear_display(self):
        """Clear the peak map display."""
        if self.image_element is not None:
            self.image_element.set_source("")
        if self.minimap_image is not None:
            self.minimap_image.set_source("")
        if self.breadcrumb_label is not None:
            self.breadcrumb_label.set_text("")

    def _on_view_changed(self):
        """Handle view changed event."""
        self.update()

    def _on_selection_changed(self, selection_type: str, index):
        """Handle selection changed event - redraw spectrum marker."""
        if selection_type == "spectrum" and self.state.show_spectrum_marker:
            # Redraw to update the spectrum marker position
            self.update()

    # === Toggle handlers ===

    def _toggle_centroids(self):
        """Toggle centroid overlay."""
        self.state.show_centroids = self.centroid_cb.value
        if self._has_data():
            self.update()

    def _toggle_bboxes(self):
        """Toggle bounding box overlay."""
        self.state.show_bounding_boxes = self.bbox_cb.value
        if self._has_data():
            self.update()

    def _toggle_hulls(self):
        """Toggle convex hull overlay."""
        self.state.show_convex_hulls = self.hull_cb.value
        if self._has_data():
            self.update()

    def _toggle_ids(self):
        """Toggle ID overlay."""
        self.state.show_ids = self.ids_cb.value
        if self._has_data():
            self.update()

    def _toggle_id_sequences(self):
        """Toggle ID sequence labels."""
        self.state.show_id_sequences = self.id_seq_cb.value
        if self._has_data():
            self.update()

    def _change_colormap(self, e):
        """Change colormap."""
        self.state.colormap = e.value
        if self._has_data():
            self.update()
            self.update_minimap()

    def _toggle_rt_unit(self, e):
        """Toggle RT unit between seconds and minutes."""
        self.state.rt_in_minutes = e.value == "min"
        if self._has_data():
            self.update()
            self.update_minimap()
            # Notify other panels
            self.state.emit_display_options_changed("rt_in_minutes", self.state.rt_in_minutes)

    def _toggle_swap_axes(self):
        """Toggle axis swap."""
        self.state.swap_axes = self.swap_axes_cb.value
        if self._has_data():
            self.update()

    def _toggle_spectrum_marker(self):
        """Toggle spectrum position marker."""
        self.state.show_spectrum_marker = self.spectrum_marker_cb.value
        if self._has_data():
            self.update()

    def _save_png(self):
        """Save peak map as PNG file."""
        if self.image_element is None:
            ui.notify("No image to save", type="warning")
            return

        # Get current image source (base64 data URL)
        src = self.image_element._props.get("src", "")
        if not src or not src.startswith("data:image/png;base64,"):
            ui.notify("No image data available", type="warning")
            return

        # Trigger download via JavaScript
        ui.run_javascript(f'''
            const link = document.createElement("a");
            link.href = "{src}";
            link.download = "peak_map.png";
            link.click();
        ''')
        ui.notify("Downloading peak_map.png", type="positive")

    # === FAIMS handlers ===

    def _toggle_faims_filter(self):
        """Toggle FAIMS CV filter mode."""
        if self.faims_checkbox is None or self.faims_container is None:
            return

        enabled = self.faims_checkbox.value
        self.faims_container.set_visibility(enabled)

        if not enabled:
            # Clear CV filter and show all data
            self.state.selected_faims_cv = None
            # Reset all label highlights
            for _cv, label in self.faims_cv_labels.items():
                label.classes(remove="bg-purple-800", add="")
            self.update()
        else:
            # Update CV minimaps
            self._update_faims_cv_minimaps()

    def _create_faims_cv_minimaps(self):
        """Create minimap images for each FAIMS CV value."""
        if self.faims_container is None or not self.state.has_faims:
            return

        self.faims_container.clear()
        self.faims_cv_minimaps = {}
        self.faims_cv_labels = {}

        # Calculate minimap size - smaller than main minimap
        mini_width = self.state.minimap_width
        mini_height = max(40, self.state.minimap_height // 2)

        with self.faims_container:
            ui.label("Click a CV to filter:").classes("text-xs text-purple-400")
            for cv in self.state.faims_cvs:
                with ui.column().classes("gap-0"):
                    # CV label (clickable)
                    label = (
                        ui.label(f"CV: {cv:.1f}V")
                        .classes("text-xs text-purple-300 cursor-pointer hover:text-purple-100")
                        .style("padding: 2px 4px;")
                    )
                    label.on("click", lambda e, c=cv: self._select_faims_cv(c))
                    self.faims_cv_labels[cv] = label

                    # Minimap image (also clickable)
                    img = ui.image().style(
                        f"width: {mini_width}px; height: {mini_height}px; "
                        f"background: rgba(30,30,30,0.8); cursor: pointer; "
                        f"border: 1px solid #666;"
                    )
                    img.on("click", lambda e, c=cv: self._select_faims_cv(c))
                    self.faims_cv_minimaps[cv] = img

    def _update_faims_cv_minimaps(self):
        """Update the FAIMS CV minimap images."""
        if not self.state.has_faims or not self.faims_cv_minimaps:
            return

        for cv in self.state.faims_cvs:
            if cv in self.faims_cv_minimaps and self.faims_cv_minimaps[cv] is not None:
                # Render minimap for this CV using the per-CV data
                img_data = self.minimap_renderer.render_for_cv(self.state, cv)
                if img_data:
                    self.faims_cv_minimaps[cv].set_source(f"data:image/png;base64,{img_data}")

    def _select_faims_cv(self, cv: float):
        """Select a FAIMS CV to filter the peak map."""
        # Toggle selection - if already selected, deselect
        if self.state.selected_faims_cv == cv:
            self.state.selected_faims_cv = None
        else:
            self.state.selected_faims_cv = cv

        # Update label highlights
        for c, label in self.faims_cv_labels.items():
            if c == self.state.selected_faims_cv:
                label.classes(remove="", add="bg-purple-800 rounded")
            else:
                label.classes(remove="bg-purple-800 rounded", add="")

        # Update the peak map
        self.update()

    # === Mouse handlers ===

    def _pixel_to_data(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coordinates to (rt, mz) respecting swap_axes."""
        plot_x = px - self.state.margin_left
        plot_y = py - self.state.margin_top
        plot_x = max(0, min(self.state.plot_width, plot_x))
        plot_y = max(0, min(self.state.plot_height, plot_y))

        x_frac = plot_x / self.state.plot_width
        y_frac = plot_y / self.state.plot_height
        rt_range = self.state.view_rt_max - self.state.view_rt_min
        mz_range = self.state.view_mz_max - self.state.view_mz_min

        if self.state.swap_axes:
            # swap_axes=True: m/z on x-axis, RT on y-axis (inverted)
            mz = self.state.view_mz_min + x_frac * mz_range
            rt = self.state.view_rt_max - y_frac * rt_range
        else:
            # swap_axes=False: RT on x-axis, m/z on y-axis (inverted)
            rt = self.state.view_rt_min + x_frac * rt_range
            mz = self.state.view_mz_max - y_frac * mz_range
        return rt, mz

    def _data_to_pixel(self, rt: float, mz: float) -> tuple[int, int]:
        """Convert RT/m/z data coordinates to pixel coordinates."""
        rt_range = self.state.view_rt_max - self.state.view_rt_min
        mz_range = self.state.view_mz_max - self.state.view_mz_min

        if rt_range == 0 or mz_range == 0:
            return (0, 0)

        if self.state.swap_axes:
            # m/z on x-axis, RT on y-axis (inverted)
            x = int((mz - self.state.view_mz_min) / mz_range * self.state.plot_width)
            y = int((1 - (rt - self.state.view_rt_min) / rt_range) * self.state.plot_height)
        else:
            # RT on x-axis, m/z on y-axis (inverted)
            x = int((rt - self.state.view_rt_min) / rt_range * self.state.plot_width)
            y = int((1 - (mz - self.state.view_mz_min) / mz_range) * self.state.plot_height)

        return (x + self.state.margin_left, y + self.state.margin_top)

    def _find_nearest_feature(self, pixel_x: int, pixel_y: int) -> Optional[int]:
        """Find the nearest feature centroid to the given pixel position.

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate

        Returns:
            Feature index if within snap distance, None otherwise
        """
        if self.state.feature_map is None or self.state.feature_map.size() == 0:
            return None

        if not self.state.show_centroids:
            return None

        min_dist_sq = self.state.hover_snap_distance_px**2
        nearest_idx = None

        # Get current view bounds
        view_rt_min = self.state.view_rt_min if self.state.view_rt_min is not None else self.state.rt_min
        view_rt_max = self.state.view_rt_max if self.state.view_rt_max is not None else self.state.rt_max
        view_mz_min = self.state.view_mz_min if self.state.view_mz_min is not None else self.state.mz_min
        view_mz_max = self.state.view_mz_max if self.state.view_mz_max is not None else self.state.mz_max

        # Limit search to reasonable number of features
        max_features_to_check = 10000

        for idx, feature in enumerate(self.state.feature_map):
            if idx >= max_features_to_check:
                break

            rt = feature.getRT()
            mz = feature.getMZ()

            # Skip if outside current view
            if not (view_rt_min <= rt <= view_rt_max and view_mz_min <= mz <= view_mz_max):
                continue

            # Convert feature position to pixel coordinates
            fx, fy = self._data_to_pixel(rt, mz)

            # Calculate squared distance
            dist_sq = (pixel_x - fx) ** 2 + (pixel_y - fy) ** 2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_idx = idx

        return nearest_idx

    def _on_peakmap_mouse(self, e: MouseEventArguments):
        """Handle mouse events on the peakmap."""
        if e.type == "mousedown":
            self._handle_mousedown(e)
        elif e.type == "mousemove":
            self._handle_mousemove(e)
        elif e.type == "mouseup":
            self._handle_mouseup(e)

    def _handle_mousedown(self, e: MouseEventArguments):
        """Handle mouse down event."""
        plot_x = e.image_x - self.state.margin_left
        plot_y = e.image_y - self.state.margin_top

        if 0 <= plot_x <= self.state.plot_width and 0 <= plot_y <= self.state.plot_height:
            self._drag_state["dragging"] = True
            self._drag_state["measuring"] = e.shift

            # Check if zoomed in for panning
            is_zoomed_in = (
                self.state.view_rt_min > self.state.rt_min + 0.01
                or self.state.view_rt_max < self.state.rt_max - 0.01
                or self.state.view_mz_min > self.state.mz_min + 0.01
                or self.state.view_mz_max < self.state.mz_max - 0.01
            )
            self._drag_state["panning"] = e.ctrl and is_zoomed_in
            self._drag_state["start_x"] = e.image_x
            self._drag_state["start_y"] = e.image_y

            # Store initial view bounds for panning
            if e.ctrl and is_zoomed_in:
                self._drag_state["pan_rt_min"] = self.state.view_rt_min
                self._drag_state["pan_rt_max"] = self.state.view_rt_max
                self._drag_state["pan_mz_min"] = self.state.view_mz_min
                self._drag_state["pan_mz_max"] = self.state.view_mz_max

    def _handle_mousemove(self, e: MouseEventArguments):
        """Handle mouse move event."""
        # Update coordinate display
        self._update_coord_display(e.image_x, e.image_y)

        # Draw overlay if dragging
        if self._drag_state["dragging"]:
            if self._drag_state["measuring"]:
                self._draw_measurement_overlay(e)
            elif self._drag_state["panning"]:
                self._handle_panning(e)
            else:
                self._draw_selection_overlay(e)
        else:
            # Not dragging - check for feature hover
            self._handle_feature_hover(e)

    def _handle_feature_hover(self, e: MouseEventArguments):
        """Handle feature hover detection and highlighting."""
        # Debounce hover updates
        current_time = time.time() * 1000  # Convert to ms
        if current_time - self._last_hover_update < self._hover_debounce_ms:
            return

        # Find nearest feature
        nearest_idx = self._find_nearest_feature(int(e.image_x), int(e.image_y))

        # Only update if hover changed
        if nearest_idx != self.state.hover_feature_idx:
            self.state.hover_feature_idx = nearest_idx
            self._last_hover_update = current_time

            # Update cursor style using element props
            if self.image_element:
                if nearest_idx is not None:
                    self.image_element.style("cursor: pointer")
                else:
                    self.image_element.style("cursor: crosshair")

            # Re-render to show hover highlight
            self.update()

    def _handle_mouseup(self, e: MouseEventArguments):
        """Handle mouse up event."""
        # Clear overlay
        if self.image_element:
            self.image_element.content = ""

        if self._drag_state["dragging"]:
            was_measuring = self._drag_state["measuring"]
            was_panning = self._drag_state["panning"]
            start_x = self._drag_state["start_x"]
            start_y = self._drag_state["start_y"]
            self._drag_state["dragging"] = False
            self._drag_state["measuring"] = False
            self._drag_state["panning"] = False

            # Skip zoom if we were measuring or panning
            if was_measuring or was_panning:
                if was_panning:
                    self.update()  # Full resolution render
                return

            # Check if this was a click (minimal drag distance) vs a drag-to-zoom
            dx = abs(e.image_x - start_x)
            dy = abs(e.image_y - start_y)

            # If it's a small movement, treat as click for feature selection
            if dx < 5 and dy < 5:
                self._handle_feature_click(e)
                return

            # Handle zoom selection
            self._handle_zoom_selection(e)

    def _handle_feature_click(self, e: MouseEventArguments):
        """Handle click to select a feature."""
        # If we have a hovered feature, select it
        if self.state.hover_feature_idx is not None:
            feature_idx = self.state.hover_feature_idx
            self.state.select_feature(feature_idx)

            # Show feature info in notification
            if self.state.feature_map is not None and feature_idx < self.state.feature_map.size():
                feature = self.state.feature_map[feature_idx]
                rt = feature.getRT()
                mz = feature.getMZ()
                intensity = feature.getIntensity()
                charge = feature.getCharge()

                if self.state.rt_in_minutes:
                    rt_str = f"{rt / 60:.2f} min"
                else:
                    rt_str = f"{rt:.1f} s"

                charge_str = f"+{charge}" if charge > 0 else ""
                ui.notify(
                    f"Selected feature #{feature_idx}: RT={rt_str}, m/z={mz:.4f}{charge_str}, I={intensity:.2e}",
                    type="info",
                    position="bottom-right",
                    timeout=3000,
                )

            self.update()

    def _update_coord_display(self, image_x: float, image_y: float):
        """Update the coordinate display label."""
        if self.coord_label is None:
            return

        try:
            rt, mz = self._pixel_to_data(image_x, image_y)
            if self.state.rt_in_minutes:
                rt_text = f"RT: {rt / 60.0:.3f} min"
            else:
                rt_text = f"RT: {rt:.2f} s"
            self.coord_label.set_text(f"{rt_text}  m/z: {mz:.4f}")
        except Exception:
            pass

    def _draw_measurement_overlay(self, e: MouseEventArguments):
        """Draw measurement line overlay."""
        if self.image_element is None:
            return

        x1, y1 = self._drag_state["start_x"], self._drag_state["start_y"]
        x2, y2 = e.image_x, e.image_y

        rt1, mz1 = self._pixel_to_data(x1, y1)
        rt2, mz2 = self._pixel_to_data(x2, y2)
        delta_rt = abs(rt2 - rt1)
        delta_mz = abs(mz2 - mz1)

        if self.state.rt_in_minutes:
            rt_text = f"ΔRT: {delta_rt / 60.0:.3f} min"
        else:
            rt_text = f"ΔRT: {delta_rt:.2f} s"
        mz_text = f"Δm/z: {delta_mz:.4f}"

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        label_offset = 15

        self.image_element.content = f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                  stroke="yellow" stroke-width="2"/>
            <circle cx="{x1}" cy="{y1}" r="4" fill="yellow"/>
            <circle cx="{x2}" cy="{y2}" r="4" fill="yellow"/>
            <rect x="{mid_x - 5}" y="{mid_y + label_offset - 2}"
                  width="130" height="36" rx="3"
                  fill="rgba(0, 0, 0, 0.8)" stroke="yellow" stroke-width="1"/>
            <text x="{mid_x}" y="{mid_y + label_offset + 12}"
                  fill="yellow" font-size="12" font-family="monospace">{rt_text}</text>
            <text x="{mid_x}" y="{mid_y + label_offset + 28}"
                  fill="yellow" font-size="12" font-family="monospace">{mz_text}</text>
        """

    def _handle_panning(self, e: MouseEventArguments):
        """Handle panning during drag."""
        delta_px = e.image_x - self._drag_state["start_x"]
        delta_py = e.image_y - self._drag_state["start_y"]

        rt_range = self._drag_state["pan_rt_max"] - self._drag_state["pan_rt_min"]
        mz_range = self._drag_state["pan_mz_max"] - self._drag_state["pan_mz_min"]

        if self.state.swap_axes:
            delta_mz = -(delta_px / self.state.plot_width) * mz_range
            delta_rt = (delta_py / self.state.plot_height) * rt_range
        else:
            delta_rt = -(delta_px / self.state.plot_width) * rt_range
            delta_mz = (delta_py / self.state.plot_height) * mz_range

        # Calculate new bounds with clamping
        new_rt_min = self._drag_state["pan_rt_min"] + delta_rt
        new_rt_max = self._drag_state["pan_rt_max"] + delta_rt
        new_mz_min = self._drag_state["pan_mz_min"] + delta_mz
        new_mz_max = self._drag_state["pan_mz_max"] + delta_mz

        # Clamp to data limits
        if new_rt_min < self.state.rt_min:
            shift = self.state.rt_min - new_rt_min
            new_rt_min += shift
            new_rt_max += shift
        if new_rt_max > self.state.rt_max:
            shift = new_rt_max - self.state.rt_max
            new_rt_min -= shift
            new_rt_max -= shift
        if new_mz_min < self.state.mz_min:
            shift = self.state.mz_min - new_mz_min
            new_mz_min += shift
            new_mz_max += shift
        if new_mz_max > self.state.mz_max:
            shift = new_mz_max - self.state.mz_max
            new_mz_min -= shift
            new_mz_max -= shift

        # Update view bounds
        self.state.view_rt_min = new_rt_min
        self.state.view_rt_max = new_rt_max
        self.state.view_mz_min = new_mz_min
        self.state.view_mz_max = new_mz_max

        # Throttle rendering
        current_time = time.time()
        if current_time - self._drag_state["last_pan_render"] >= 0.05:
            self._drag_state["last_pan_render"] = current_time
            self.update_lightweight()

        # Show panning cursor indicator
        if self.image_element:
            cx, cy = e.image_x, e.image_y
            self.image_element.content = f"""
                <circle cx="{cx}" cy="{cy}" r="8" fill="none"
                        stroke="orange" stroke-width="2"/>
                <line x1="{cx - 12}" y1="{cy}" x2="{cx + 12}" y2="{cy}"
                      stroke="orange" stroke-width="2"/>
                <line x1="{cx}" y1="{cy - 12}" x2="{cx}" y2="{cy + 12}"
                      stroke="orange" stroke-width="2"/>
            """

    def _draw_selection_overlay(self, e: MouseEventArguments):
        """Draw zoom selection rectangle overlay."""
        if self.image_element is None:
            return

        x = min(self._drag_state["start_x"], e.image_x)
        y = min(self._drag_state["start_y"], e.image_y)
        w = abs(e.image_x - self._drag_state["start_x"])
        h = abs(e.image_y - self._drag_state["start_y"])

        self.image_element.content = f"""
            <rect x="{x}" y="{y}" width="{w}" height="{h}"
                  fill="rgba(0, 200, 255, 0.15)"
                  stroke="cyan" stroke-width="2" stroke-dasharray="5,5"/>
        """

    def _handle_zoom_selection(self, e: MouseEventArguments):
        """Handle zoom selection on mouse up."""
        end_x = e.image_x
        end_y = e.image_y

        # Calculate selection in plot coordinates
        start_plot_x = self._drag_state["start_x"] - self.state.margin_left
        start_plot_y = self._drag_state["start_y"] - self.state.margin_top
        end_plot_x = end_x - self.state.margin_left
        end_plot_y = end_y - self.state.margin_top

        # Ensure within bounds
        start_plot_x = max(0, min(self.state.plot_width, start_plot_x))
        start_plot_y = max(0, min(self.state.plot_height, start_plot_y))
        end_plot_x = max(0, min(self.state.plot_width, end_plot_x))
        end_plot_y = max(0, min(self.state.plot_height, end_plot_y))

        # Only zoom if dragged a meaningful distance
        dx = abs(end_plot_x - start_plot_x)
        dy = abs(end_plot_y - start_plot_y)

        if dx > 10 and dy > 10:
            # Save current state to zoom history
            self.state.push_zoom_history()

            # Convert to data coordinates
            rt_range = self.state.view_rt_max - self.state.view_rt_min
            mz_range = self.state.view_mz_max - self.state.view_mz_min

            x1_frac = min(start_plot_x, end_plot_x) / self.state.plot_width
            x2_frac = max(start_plot_x, end_plot_x) / self.state.plot_width
            y1_frac = min(start_plot_y, end_plot_y) / self.state.plot_height
            y2_frac = max(start_plot_y, end_plot_y) / self.state.plot_height

            if self.state.swap_axes:
                new_mz_min = self.state.view_mz_min + x1_frac * mz_range
                new_mz_max = self.state.view_mz_min + x2_frac * mz_range
                new_rt_max = self.state.view_rt_max - y1_frac * rt_range
                new_rt_min = self.state.view_rt_max - y2_frac * rt_range
            else:
                new_rt_min = self.state.view_rt_min + x1_frac * rt_range
                new_rt_max = self.state.view_rt_min + x2_frac * rt_range
                new_mz_max = self.state.view_mz_max - y1_frac * mz_range
                new_mz_min = self.state.view_mz_max - y2_frac * mz_range

            self.state.view_rt_min = new_rt_min
            self.state.view_rt_max = new_rt_max
            self.state.view_mz_min = new_mz_min
            self.state.view_mz_max = new_mz_max

            # Save new state to zoom history
            self.state.push_zoom_history()
            self.update()

    def _on_wheel(self, e):
        """Handle mouse wheel zoom."""
        try:
            offset_x = e.args.get("offsetX", 0)
            offset_y = e.args.get("offsetY", 0)
            delta_y = e.args.get("deltaY", 0)

            plot_x = offset_x - self.state.margin_left
            plot_y = offset_y - self.state.margin_top

            if 0 <= plot_x <= self.state.plot_width and 0 <= plot_y <= self.state.plot_height:
                x_frac = plot_x / self.state.plot_width
                y_frac = plot_y / self.state.plot_height
                zoom_in = delta_y < 0
                self.state.zoom_at_point(x_frac, y_frac, zoom_in)
                self.update()
        except Exception:
            pass

    def _on_mouseleave(self, e):
        """Handle mouse leave event."""
        was_panning = self._drag_state["panning"]
        had_hover = self.state.hover_feature_idx is not None

        self._drag_state["dragging"] = False
        self._drag_state["measuring"] = False
        self._drag_state["panning"] = False

        # Clear hover state
        self.state.hover_feature_idx = None

        if self.image_element:
            self.image_element.content = ""
            # Reset cursor
            self.image_element.style("cursor: crosshair")

        if self.coord_label:
            self.coord_label.set_text("RT: --  m/z: --")

        if was_panning or had_hover:
            self.update()

    def _on_keyup(self, e):
        """Handle key release during panning."""
        key = e.args.get("key", "")
        if key == "Control" and self._drag_state["panning"]:
            self._drag_state["dragging"] = False
            self._drag_state["measuring"] = False
            self._drag_state["panning"] = False
            if self.image_element:
                self.image_element.content = ""
            self.update()

    def _on_minimap_click(self, e):
        """Handle minimap click to center view."""
        try:
            offset_x = e.args.get("offsetX", 0)
            offset_y = e.args.get("offsetY", 0)
            x_frac = offset_x / self.state.minimap_width
            y_frac = offset_y / self.state.minimap_height
            self.state.minimap_click_to_view(x_frac, y_frac)
            self.update()
        except Exception:
            pass

    def _go_back(self):
        """Go back in zoom history."""
        if len(self.state.zoom_history) > 1:
            self.state.go_to_zoom_history(len(self.state.zoom_history) - 2)
            self.update()

    def _toggle_3d_view(self):
        """Toggle 3D view visibility."""
        self.state.show_3d_view = not self.state.show_3d_view

        if self.scene_3d_container:
            self.scene_3d_container.set_visibility(self.state.show_3d_view)

        if self.state.show_3d_view and self._has_data():
            self._update_3d_view()

        # Update button appearance
        if self.view_3d_btn:
            if self.state.show_3d_view:
                self.view_3d_btn.props("color=purple")
            else:
                self.view_3d_btn.props("color=grey")

    def _update_3d_view(self):
        """Update the 3D visualization with current view data using pyopenms-viz."""
        if not self.state.show_3d_view or self.plot_3d is None or not self._has_data():
            return

        # Check if region is small enough
        if not self._is_small_region():
            # Show message that region is too large
            if self.view_3d_status:
                rt_range = self.state.view_rt_max - self.state.view_rt_min
                mz_range = self.state.view_mz_max - self.state.view_mz_min
                self.view_3d_status.set_text(
                    f"Zoom in more for 3D (current: RT={rt_range:.0f}s, m/z={mz_range:.0f} | "
                    f"need: RT≤{self.state.rt_threshold_3d:.0f}s, m/z≤{self.state.mz_threshold_3d:.0f})"
                )
            return

        # Get peaks in current view
        df = self.state.df
        mask = (
            (df["rt"] >= self.state.view_rt_min)
            & (df["rt"] <= self.state.view_rt_max)
            & (df["mz"] >= self.state.view_mz_min)
            & (df["mz"] <= self.state.view_mz_max)
        )
        view_df = df[mask].copy()

        if len(view_df) == 0:
            if self.view_3d_status:
                self.view_3d_status.set_text("No peaks in view")
            return

        # Subsample if too many peaks
        num_peaks_total = len(view_df)
        if len(view_df) > self.state.max_3d_peaks:
            view_df = view_df.nlargest(self.state.max_3d_peaks, "intensity")
        num_peaks_shown = len(view_df)

        try:
            # Use pyopenms-viz for 3D plotting
            from pyopenms_viz._plotly.core import PLOTLYPeakMapPlot

            # Rename columns to match pyopenms-viz expectations
            plot_df = view_df.rename(columns={"rt": "RT", "mz": "mz", "intensity": "int"})

            # Create 3D plot (no title - header shows info)
            plot = PLOTLYPeakMapPlot(plot_df, x="RT", y="mz", z="int", plot_3d=True, title="")
            plot.plot()

            # Get the plotly figure
            fig = plot.fig

            # Update layout for light/dark mode compatibility
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#888"},
                scene={
                    "xaxis": {"title": "RT (s)", "backgroundcolor": "rgba(128,128,128,0.1)", "gridcolor": "#888"},
                    "yaxis": {"title": "m/z", "backgroundcolor": "rgba(128,128,128,0.1)", "gridcolor": "#888"},
                    "zaxis": {"title": "Intensity", "backgroundcolor": "rgba(128,128,128,0.1)", "gridcolor": "#888"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "aspectmode": "manual",
                    "aspectratio": {"x": 1.5, "y": 1, "z": 0.8},
                },
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                width=self.state.canvas_width,
                height=500,
                autosize=False,
                showlegend=True,
                legend={"x": 0, "y": 1, "bgcolor": "rgba(128,128,128,0.3)"},
                modebar={"orientation": "v", "bgcolor": "rgba(0,0,0,0)"},
            )

            # Add feature markers if available
            if self.state.feature_map is not None and self.state.show_centroids:
                self._add_features_to_3d_plot(fig)

            # Update the plotly element
            self.plot_3d.update_figure(self._figure_with_config(fig))

            # Update status
            if self.view_3d_status:
                if num_peaks_shown < num_peaks_total:
                    self.view_3d_status.set_text(
                        f"Showing {num_peaks_shown:,} of {num_peaks_total:,} peaks (top intensity)"
                    )
                else:
                    self.view_3d_status.set_text(f"Showing {num_peaks_shown:,} peaks")

        except Exception as e:
            if self.view_3d_status:
                self.view_3d_status.set_text(f"3D plot error: {str(e)[:50]}")

    def _is_small_region(self) -> bool:
        """Check if the current view is small enough for 3D rendering."""
        rt_range = self.state.view_rt_max - self.state.view_rt_min
        mz_range = self.state.view_mz_max - self.state.view_mz_min
        return rt_range <= self.state.rt_threshold_3d and mz_range <= self.state.mz_threshold_3d

    def _add_features_to_3d_plot(self, fig):
        """Add feature bounding boxes to the 3D plotly figure."""
        import plotly.graph_objects as go

        if self.state.feature_map is None:
            return

        # Collect all box edges for a single trace (more efficient)
        box_x = []
        box_y = []
        box_z = []

        for feature in self.state.feature_map:
            rt = feature.getRT()
            mz = feature.getMZ()

            # Check if feature is in current view
            if not (
                self.state.view_rt_min <= rt <= self.state.view_rt_max
                and self.state.view_mz_min <= mz <= self.state.view_mz_max
            ):
                continue

            # Get RT and m/z bounds from convex hull
            hulls = feature.getConvexHulls()
            if hulls and len(hulls) > 0:
                hull_points = hulls[0].getHullPoints()
                if len(hull_points) > 0:
                    rt_vals = [p[0] for p in hull_points]
                    mz_vals = [p[1] for p in hull_points]
                    rt_min, rt_max = min(rt_vals), max(rt_vals)
                    mz_min, mz_max = min(mz_vals), max(mz_vals)
                else:
                    rt_min, rt_max = rt - 5, rt + 5
                    mz_min, mz_max = mz - 0.5, mz + 0.5
            else:
                rt_min, rt_max = rt - 5, rt + 5
                mz_min, mz_max = mz - 0.5, mz + 0.5

            # Draw box edges on the baseline (z=0)
            z_base = 0
            # Edge 1: rt_min to rt_max at mz_min
            box_x.extend([rt_min, rt_max, None])
            box_y.extend([mz_min, mz_min, None])
            box_z.extend([z_base, z_base, None])
            # Edge 2: rt_max at mz_min to mz_max
            box_x.extend([rt_max, rt_max, None])
            box_y.extend([mz_min, mz_max, None])
            box_z.extend([z_base, z_base, None])
            # Edge 3: rt_max to rt_min at mz_max
            box_x.extend([rt_max, rt_min, None])
            box_y.extend([mz_max, mz_max, None])
            box_z.extend([z_base, z_base, None])
            # Edge 4: rt_min at mz_max to mz_min
            box_x.extend([rt_min, rt_min, None])
            box_y.extend([mz_max, mz_min, None])
            box_z.extend([z_base, z_base, None])

        if box_x:
            # Add all bounding boxes as a single trace
            fig.add_trace(
                go.Scatter3d(
                    x=box_x,
                    y=box_y,
                    z=box_z,
                    mode="lines",
                    line={"color": "#00ff66", "width": 3},
                    name="Features",
                    hoverinfo="skip",
                )
            )

    def _reset_view(self):
        """Reset view to full data extent."""
        self.state.reset_view()
        self.update()

    def _update_breadcrumb(self):
        """Update the breadcrumb trail."""
        if self.breadcrumb_label is None:
            return

        # Check if at full view
        is_full_view = (
            abs(self.state.view_rt_min - self.state.rt_min) < 0.01
            and abs(self.state.view_rt_max - self.state.rt_max) < 0.01
            and abs(self.state.view_mz_min - self.state.mz_min) < 0.01
            and abs(self.state.view_mz_max - self.state.mz_max) < 0.01
        )

        if is_full_view:
            self.breadcrumb_label.set_text("Full view")
        else:
            # Format the current view range
            if self.state.rt_in_minutes:
                rt_text = f"{self.state.view_rt_min / 60:.1f}-{self.state.view_rt_max / 60:.1f} min"
            else:
                rt_text = f"{self.state.view_rt_min:.0f}-{self.state.view_rt_max:.0f} s"

            mz_text = f"{self.state.view_mz_min:.1f}-{self.state.view_mz_max:.1f} m/z"
            self.breadcrumb_label.set_text(f"Full view → {rt_text}, {mz_text}")

    def set_on_update_callback(self, callback: Callable):
        """Set callback to be called when panel triggers external updates.

        Args:
            callback: Function to call when other panels should update
        """
        self._on_update_callback = callback
