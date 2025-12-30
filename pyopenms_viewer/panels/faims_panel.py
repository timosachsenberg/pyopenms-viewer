"""FAIMS multi-CV peak map panel."""

from typing import Optional

from nicegui import ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer


class FAIMSPanel:
    """FAIMS multi-CV peak map panel.

    Displays separate peak maps for each FAIMS compensation voltage (CV) value.
    This is NOT an expansion panel but a card that can be shown/hidden via
    the FAIMS toggle in the toolbar.

    Features:
    - One peak map per detected CV value
    - Synchronized zoom/pan with main peak map
    - Dynamically creates images when FAIMS data is detected
    """

    def __init__(self, state: ViewerState):
        """Initialize FAIMS panel.

        Args:
            state: ViewerState instance (shared reference)
        """
        self.state = state
        self.card: Optional[ui.card] = None
        self.images_row: Optional[ui.row] = None
        self.cv_images: dict[float, ui.image] = {}  # CV -> image element
        self.renderer: Optional[PeakMapRenderer] = None

    def build(self, container: ui.element) -> ui.card:
        """Build the FAIMS panel UI.

        Args:
            container: Parent element to build panel in

        Returns:
            The card element created
        """
        with container:
            self.card = ui.card().classes("w-full max-w-6xl mt-2 p-2")
            self.card.set_visibility(False)

            with self.card:
                with ui.row().classes("w-full items-center mb-2"):
                    ui.label("FAIMS Compensation Voltage Peak Maps").classes("text-lg font-semibold text-purple-300")
                    ui.element("div").classes("flex-grow")
                    ui.button(icon="download", on_click=self._save_all_png).props("dense flat size=sm").tooltip(
                        "Save all FAIMS peak maps as PNG"
                    )

                ui.label("Separate peak maps for each CV value - zoom/pan is synchronized").classes(
                    "text-xs text-gray-500 mb-2"
                )

                # Container for dynamic FAIMS images
                self.images_row = ui.row().classes("w-full gap-1 flex-wrap justify-center")

        # Store reference in state for visibility toggling
        self.state.faims_container = self.card

        # Initialize renderer
        self.renderer = PeakMapRenderer(
            plot_width=self.state.canvas_width,
            plot_height=self.state.canvas_height,
        )

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)
        self.state.on_view_changed(self._on_view_changed)

        # Register the update method with state
        self.state.update_faims_plots = self.update

        return self.card

    def update(self) -> None:
        """Update all FAIMS CV peak map images."""
        if not self.state.has_faims or not self.state.show_faims_view:
            return

        if self.renderer is None:
            return

        for cv in self.state.faims_cvs:
            if cv in self.cv_images and self.cv_images[cv] is not None:
                img_data = self.renderer.render_faims(self.state, cv)
                if img_data:
                    self.cv_images[cv].set_source(f"data:image/png;base64,{img_data}")

    def _create_faims_images(self) -> None:
        """Create FAIMS image elements dynamically based on detected CVs."""
        if self.images_row is None:
            return

        self.images_row.clear()
        self.cv_images = {}

        if not self.state.has_faims:
            return

        n_cvs = len(self.state.faims_cvs)
        # Calculate width for each panel (max 4 per row)
        panel_width = self.state.canvas_width // max(1, min(n_cvs, 4))
        panel_height = self.state.canvas_height

        with self.images_row:
            for cv in self.state.faims_cvs:
                with ui.column().classes("flex-none"):
                    # CV label
                    ui.label(f"CV: {cv:.1f}V").classes("text-sm text-purple-400 mb-1")
                    # Image element
                    img = ui.image().style(
                        f"width: {panel_width}px; height: {panel_height}px; background: rgba(30,30,30,0.8);"
                    )
                    self.cv_images[cv] = img

        # Initial render
        self.update()

    def _on_data_loaded(self, data_type: str) -> None:
        """Handle data loaded event."""
        if data_type == "mzml":
            # Create FAIMS images when mzML data is loaded
            if self.state.has_faims:
                self._create_faims_images()
                # Update FAIMS info label
                if hasattr(self.state, "faims_info_label") and self.state.faims_info_label:
                    cv_str = ", ".join([f"{cv:.1f}V" for cv in self.state.faims_cvs])
                    self.state.faims_info_label.set_text(f"FAIMS CVs detected: {cv_str}")
                    self.state.faims_info_label.set_visibility(True)
            else:
                # Hide FAIMS UI elements if no FAIMS data
                if hasattr(self.state, "faims_info_label") and self.state.faims_info_label:
                    self.state.faims_info_label.set_visibility(False)
                self.state.show_faims_view = False
                if self.card:
                    self.card.set_visibility(False)

    def _on_view_changed(self) -> None:
        """Handle view changed event - update FAIMS plots."""
        if self.state.show_faims_view:
            self.update()

    def _save_all_png(self) -> None:
        """Save all FAIMS peak map images as PNG files."""
        if not self.cv_images:
            ui.notify("No FAIMS images to save", type="warning")
            return

        saved_count = 0
        for cv, img in self.cv_images.items():
            if img is None:
                continue
            src = img._props.get("src", "")
            if not src or not src.startswith("data:image/png;base64,"):
                continue

            # Create safe filename with CV value
            cv_str = f"{cv:.1f}".replace(".", "_").replace("-", "neg")
            filename = f"faims_cv_{cv_str}V.png"

            ui.run_javascript(f'''
                const link = document.createElement("a");
                link.href = "{src}";
                link.download = "{filename}";
                link.click();
            ''')
            saved_count += 1

        if saved_count > 0:
            ui.notify(f"Downloading {saved_count} FAIMS peak map(s)", type="positive")
        else:
            ui.notify("No image data available", type="warning")
