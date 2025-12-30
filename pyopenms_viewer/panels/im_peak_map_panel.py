"""Ion Mobility peak map panel component.

This panel displays the 2D ion mobility vs m/z visualization
for TIMS, drift tube, and other ion mobility data.
"""

from typing import Optional

from nicegui import ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel
from pyopenms_viewer.rendering import IMPeakMapRenderer


class IMPeakMapPanel(BasePanel):
    """Ion Mobility peak map panel.

    Features:
    - 2D m/z vs ion mobility visualization
    - Mobilogram display (summed intensity profile)
    - Drag-to-zoom selection
    - Mouse wheel zoom
    - Link to spectrum m/z zoom
    """

    def __init__(self, state: ViewerState):
        """Initialize IM peak map panel.

        Args:
            state: ViewerState instance (shared reference)
        """
        super().__init__(state, "im_peakmap", "Ion Mobility Map", "blur_on")

        # UI elements
        self.im_image_element: Optional[ui.interactive_image] = None
        self.info_label: Optional[ui.label] = None
        self.range_label: Optional[ui.label] = None
        self.link_checkbox = None
        self.mobilogram_checkbox = None

        # Renderer
        self.im_renderer = IMPeakMapRenderer(
            plot_width=state.plot_width,
            plot_height=state.plot_height,
            margin_left=state.margin_left,
            margin_right=state.margin_right,
            margin_top=state.margin_top,
            margin_bottom=state.margin_bottom,
        )

        # Drag state
        self._drag_state = {
            "dragging": False,
            "start_x": 0,
            "start_y": 0,
            "in_mobilogram": False,  # Track if drag started in mobilogram area
        }

    def build(self, container: ui.element) -> ui.expansion:
        """Build the IM peak map panel UI.

        Args:
            container: Parent element to build panel in

        Returns:
            The expansion element created
        """
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                self._build_controls_row()
                self._build_range_row()
                self._build_image_area()

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)
        self.state.on_view_changed(self._on_view_changed)

        self._is_built = True
        return self.expansion

    def _build_controls_row(self):
        """Build the controls row."""
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            self.info_label = ui.label("No ion mobility data").classes("text-sm text-gray-400")
            ui.element("div").classes("flex-grow")

            # Link to spectrum m/z checkbox
            self.link_checkbox = (
                ui.checkbox(
                    "Link to Spectrum m/z", value=self.state.link_spectrum_mz_to_im, on_change=self._on_link_change
                )
                .props("dense")
                .tooltip("Sync m/z range with spectrum plot zoom")
            )

            # Show mobilogram checkbox
            self.mobilogram_checkbox = (
                ui.checkbox("Show Mobilogram", value=self.state.show_mobilogram, on_change=self._on_mobilogram_change)
                .props("dense")
                .tooltip("Show summed intensity profile vs ion mobility")
            )

            ui.button("Reset View", icon="home", on_click=self._reset_view).props("dense outline size=sm").tooltip(
                "Reset to full IM range"
            )

            ui.button(icon="download", on_click=self._save_png).props("dense flat size=sm").tooltip(
                "Save ion mobility map as PNG"
            )

    def _build_range_row(self):
        """Build the range display row."""
        with ui.row().classes("w-full"):
            self.range_label = ui.label("IM: --").classes("text-xs text-gray-500")

        # Store reference in state
        self.state.im_range_label = self.range_label

    def _build_image_area(self):
        """Build the IM image display area."""
        mobilogram_space = self.state.mobilogram_plot_width + 20 if self.state.show_mobilogram else 0
        im_canvas_width = self.state.canvas_width + mobilogram_space

        with ui.column().classes("w-full items-center"):
            self.im_image_element = (
                ui.interactive_image(
                    on_mouse=self._on_im_mouse,
                    events=["mousedown", "mousemove", "mouseup"],
                    cross=False,
                )
                .style(
                    f"width: {im_canvas_width}px; height: {self.state.canvas_height}px; "
                    f"background: transparent; cursor: crosshair;"
                )
                .classes("border border-gray-600")
            )

            # Store reference in state
            self.state.im_image_element = self.im_image_element

            # Additional event handlers (wheel and dblclick are separate)
            self.im_image_element.on("dblclick", lambda e: self._reset_view())
            self.im_image_element.on("wheel.prevent", self._on_wheel)

    def update(self) -> None:
        """Update the IM peak map display."""
        if not self.state.has_ion_mobility or self.im_image_element is None:
            return

        base64_img = self.im_renderer.render(self.state)
        if base64_img:
            self.im_image_element.set_source(f"data:image/png;base64,{base64_img}")

        # Update range label
        self._update_range_label()

        # Update info label
        if self.info_label is not None and self.state.has_ion_mobility:
            n_peaks = len(self.state.im_df) if self.state.im_df is not None else 0
            self.info_label.set_text(f"Ion mobility data: {n_peaks:,} peaks | {self.state.im_type or 'Unknown type'}")

    def _has_data(self) -> bool:
        """Check if panel has data to display."""
        return self.state.has_ion_mobility

    def _update_range_label(self):
        """Update the range display label."""
        if self.range_label is None:
            return

        if self.state.view_im_min is not None and self.state.view_im_max is not None:
            self.range_label.set_text(
                f"m/z: {self.state.view_mz_min:.2f} - {self.state.view_mz_max:.2f} | "
                f"IM: {self.state.view_im_min:.3f} - {self.state.view_im_max:.3f} {self.state.im_unit}"
            )
        else:
            self.range_label.set_text("IM: --")

    # === Event handlers ===

    def _on_data_loaded(self, data_type: str):
        """Handle data loaded event."""
        if data_type == "mzml":
            # Update visibility based on whether IM data is present
            self.update_visibility()
            if self.state.has_ion_mobility:
                self.update()
                # Auto-expand if IM data present
                if self.expansion:
                    self.expansion.value = True
            else:
                # Clear display and collapse when loading non-IM data
                self._clear_display()
                if self.expansion:
                    self.expansion.value = False

    def _clear_display(self):
        """Clear the IM display when no IM data is available."""
        if self.im_image_element is not None:
            # Set to transparent/empty image
            self.im_image_element.set_source("")
            self.im_image_element.content = ""
        if self.info_label is not None:
            self.info_label.set_text("No ion mobility data")
        if self.range_label is not None:
            self.range_label.set_text("IM: --")

    def _on_view_changed(self):
        """Handle view changed event."""
        if self.state.has_ion_mobility:
            self.update()

    def _on_link_change(self, e):
        """Handle link to spectrum change."""
        self.state.link_spectrum_mz_to_im = e.value
        if e.value and self.state.spectrum_zoom_range:
            xmin, xmax = self.state.spectrum_zoom_range
            self.state.view_mz_min = max(self.state.mz_min, xmin)
            self.state.view_mz_max = min(self.state.mz_max, xmax)
            self.update()

    def _on_mobilogram_change(self, e):
        """Handle mobilogram toggle."""
        self.state.show_mobilogram = e.value

        # Update image width
        mobilogram_space = self.state.mobilogram_plot_width + 20 if self.state.show_mobilogram else 0
        new_width = self.state.canvas_width + mobilogram_space

        if self.im_image_element:
            self.im_image_element.style(
                f"width: {new_width}px; height: {self.state.canvas_height}px; "
                f"background: transparent; cursor: crosshair;"
            )

        self.update()

    def _reset_view(self):
        """Reset to full IM view."""
        self.state.view_mz_min = self.state.mz_min
        self.state.view_mz_max = self.state.mz_max
        self.state.view_im_min = self.state.im_min
        self.state.view_im_max = self.state.im_max
        self.update()

    def _save_png(self):
        """Save ion mobility map as PNG file."""
        if self.im_image_element is None:
            ui.notify("No image to save", type="warning")
            return

        # Get current image source (base64 data URL)
        src = self.im_image_element._props.get("src", "")
        if not src or not src.startswith("data:image/png;base64,"):
            ui.notify("No image data available", type="warning")
            return

        # Trigger download via JavaScript
        ui.run_javascript(f'''
            const link = document.createElement("a");
            link.href = "{src}";
            link.download = "ion_mobility_map.png";
            link.click();
        ''')
        ui.notify("Downloading ion_mobility_map.png", type="positive")

    # === Mouse handlers ===

    def _is_in_mobilogram(self, x: float, y: float) -> bool:
        """Check if coordinates are in the mobilogram area."""
        if not self.state.show_mobilogram:
            return False

        mob_left = self.state.margin_left + self.state.plot_width + 10
        mob_right = mob_left + self.state.mobilogram_plot_width
        mob_top = self.state.margin_top
        mob_bottom = self.state.margin_top + self.state.plot_height

        return mob_left <= x <= mob_right and mob_top <= y <= mob_bottom

    def _get_mobilogram_bounds(self) -> tuple:
        """Get mobilogram area pixel bounds."""
        mob_left = self.state.margin_left + self.state.plot_width + 10
        mob_right = mob_left + self.state.mobilogram_plot_width
        mob_top = self.state.margin_top
        mob_bottom = self.state.margin_top + self.state.plot_height
        return mob_left, mob_right, mob_top, mob_bottom

    def _on_im_mouse(self, e):
        """Unified mouse handler for interactive_image on_mouse callback."""
        try:
            event_type = e.type
            offset_x = e.image_x
            offset_y = e.image_y

            if event_type == "mousedown":
                self._drag_state["dragging"] = True
                self._drag_state["start_x"] = offset_x
                self._drag_state["start_y"] = offset_y
                self._drag_state["in_mobilogram"] = self._is_in_mobilogram(offset_x, offset_y)

            elif event_type == "mouseup":
                if not self._drag_state["dragging"]:
                    return
                self._drag_state["dragging"] = False

                start_x = self._drag_state["start_x"]
                start_y = self._drag_state["start_y"]
                in_mobilogram = self._drag_state["in_mobilogram"]

                # Check if significant drag (only Y matters for mobilogram)
                dx = abs(offset_x - start_x)
                dy = abs(offset_y - start_y)
                min_drag = 5 if not in_mobilogram else 3  # Lower threshold for mobilogram Y-only
                if (not in_mobilogram and dx < min_drag and dy < min_drag) or (in_mobilogram and dy < min_drag):
                    if self.im_image_element:
                        self.im_image_element.content = ""
                    return

                if in_mobilogram:
                    # Mobilogram: only zoom IM axis (Y)
                    mob_left, mob_right, mob_top, mob_bottom = self._get_mobilogram_bounds()
                    y1_frac = (min(start_y, offset_y) - mob_top) / self.state.plot_height
                    y2_frac = (max(start_y, offset_y) - mob_top) / self.state.plot_height
                    y1_frac = max(0, min(1, y1_frac))
                    y2_frac = max(0, min(1, y2_frac))

                    im_range = self.state.view_im_max - self.state.view_im_min
                    new_im_max = self.state.view_im_max - y1_frac * im_range
                    new_im_min = self.state.view_im_max - y2_frac * im_range

                    # Only update IM bounds (m/z stays the same)
                    self.state.view_im_min = new_im_min
                    self.state.view_im_max = new_im_max
                else:
                    # Main plot: zoom both m/z and IM
                    x1_frac = (min(start_x, offset_x) - self.state.margin_left) / self.state.plot_width
                    x2_frac = (max(start_x, offset_x) - self.state.margin_left) / self.state.plot_width
                    y1_frac = (min(start_y, offset_y) - self.state.margin_top) / self.state.plot_height
                    y2_frac = (max(start_y, offset_y) - self.state.margin_top) / self.state.plot_height

                    x1_frac = max(0, min(1, x1_frac))
                    x2_frac = max(0, min(1, x2_frac))
                    y1_frac = max(0, min(1, y1_frac))
                    y2_frac = max(0, min(1, y2_frac))

                    mz_range = self.state.view_mz_max - self.state.view_mz_min
                    im_range = self.state.view_im_max - self.state.view_im_min

                    new_mz_min = self.state.view_mz_min + x1_frac * mz_range
                    new_mz_max = self.state.view_mz_min + x2_frac * mz_range
                    new_im_max = self.state.view_im_max - y1_frac * im_range
                    new_im_min = self.state.view_im_max - y2_frac * im_range

                    self.state.view_mz_min = new_mz_min
                    self.state.view_mz_max = new_mz_max
                    self.state.view_im_min = new_im_min
                    self.state.view_im_max = new_im_max

                if self.im_image_element:
                    self.im_image_element.content = ""

                self.update()

            elif event_type == "mousemove":
                if self._drag_state["dragging"] and self.im_image_element:
                    start_x = self._drag_state["start_x"]
                    start_y = self._drag_state["start_y"]
                    in_mobilogram = self._drag_state["in_mobilogram"]

                    if in_mobilogram:
                        # Mobilogram: full-width selection rectangle (only Y varies)
                        mob_left, mob_right, mob_top, mob_bottom = self._get_mobilogram_bounds()
                        rect_x = mob_left
                        rect_y = min(start_y, offset_y)
                        rect_w = mob_right - mob_left
                        rect_h = abs(offset_y - start_y)
                        # Cyan color for mobilogram selection
                        self.im_image_element.content = (
                            f'<rect x="{rect_x}" y="{rect_y}" width="{rect_w}" height="{rect_h}" '
                            f'fill="rgba(0,200,255,0.2)" stroke="rgba(0,200,255,0.7)" stroke-width="1"/>'
                        )
                    else:
                        # Main plot: normal rectangle
                        rect_x = min(start_x, offset_x)
                        rect_y = min(start_y, offset_y)
                        rect_w = abs(offset_x - start_x)
                        rect_h = abs(offset_y - start_y)
                        self.im_image_element.content = (
                            f'<rect x="{rect_x}" y="{rect_y}" width="{rect_w}" height="{rect_h}" '
                            f'fill="rgba(255,255,0,0.15)" stroke="rgba(255,255,0,0.5)" stroke-width="1"/>'
                        )

        except Exception:
            pass

    def _on_wheel(self, e):
        """Handle mouse wheel zoom."""
        try:
            offset_x = e.args.get("offsetX", 0)
            offset_y = e.args.get("offsetY", 0)
            delta_y = e.args.get("deltaY", 0)

            # Check if in mobilogram area
            if self._is_in_mobilogram(offset_x, offset_y):
                # Mobilogram: only zoom IM axis (Y)
                mob_left, mob_right, mob_top, mob_bottom = self._get_mobilogram_bounds()
                y_frac = (offset_y - mob_top) / self.state.plot_height
                zoom_in = delta_y < 0

                factor = 0.8 if zoom_in else 1.25
                im_range = self.state.view_im_max - self.state.view_im_min
                cursor_im = self.state.view_im_max - y_frac * im_range

                new_im_range = im_range * factor
                self.state.view_im_min = cursor_im - (1 - y_frac) * new_im_range
                self.state.view_im_max = cursor_im + y_frac * new_im_range

                # Clamp to data bounds
                self.state.view_im_min = max(self.state.im_min, self.state.view_im_min)
                self.state.view_im_max = min(self.state.im_max, self.state.view_im_max)

                self.update()
                return

            # Check if in main plot area
            x_in_plot = self.state.margin_left <= offset_x <= self.state.margin_left + self.state.plot_width
            y_in_plot = self.state.margin_top <= offset_y <= self.state.margin_top + self.state.plot_height

            if x_in_plot and y_in_plot:
                x_frac = (offset_x - self.state.margin_left) / self.state.plot_width
                y_frac = (offset_y - self.state.margin_top) / self.state.plot_height
                zoom_in = delta_y < 0

                factor = 0.8 if zoom_in else 1.25
                mz_range = self.state.view_mz_max - self.state.view_mz_min
                im_range = self.state.view_im_max - self.state.view_im_min

                cursor_mz = self.state.view_mz_min + x_frac * mz_range
                cursor_im = self.state.view_im_max - y_frac * im_range

                new_mz_range = mz_range * factor
                new_im_range = im_range * factor

                self.state.view_mz_min = cursor_mz - x_frac * new_mz_range
                self.state.view_mz_max = cursor_mz + (1 - x_frac) * new_mz_range
                self.state.view_im_min = cursor_im - (1 - y_frac) * new_im_range
                self.state.view_im_max = cursor_im + y_frac * new_im_range

                # Clamp to data bounds
                self.state.view_mz_min = max(self.state.mz_min, self.state.view_mz_min)
                self.state.view_mz_max = min(self.state.mz_max, self.state.view_mz_max)
                self.state.view_im_min = max(self.state.im_min, self.state.view_im_min)
                self.state.view_im_max = min(self.state.im_max, self.state.view_im_max)

                self.update()

        except Exception:
            pass
