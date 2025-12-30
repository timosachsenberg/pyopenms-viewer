"""Features table panel component.

This panel displays a table of detected features from featureXML files
with filtering and zoom-to-feature functionality.
"""

from typing import Callable, Optional

from nicegui import ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel


class FeaturesTablePanel(BasePanel):
    """Features table panel.

    Features:
    - Table of detected features with metadata
    - Filtering by intensity, quality, charge
    - Click to zoom to feature location
    - Hover highlighting
    """

    def __init__(self, state: ViewerState):
        """Initialize features table panel.

        Args:
            state: ViewerState instance (shared reference)
        """
        super().__init__(state, "features_table", "Features", "scatter_plot")

        # UI elements
        self.feature_table = None
        self.min_intensity_input = None
        self.min_quality_input = None
        self.charge_select = None

        # Callback for feature selection
        self._on_feature_selected: Optional[Callable] = None

    def build(self, container: ui.element) -> ui.expansion:
        """Build the features table panel UI.

        Args:
            container: Parent element to build panel in

        Returns:
            The expansion element created
        """
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                self._build_help_text()
                self._build_filter_row()
                self._build_table()

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)
        self.state.on_selection_changed(self._on_selection_changed)

        self._is_built = True
        return self.expansion

    def _build_help_text(self):
        """Build the help text."""
        ui.label("Click a row to zoom to that feature").classes("text-sm text-gray-400 mb-2")

    def _build_filter_row(self):
        """Build the filter controls row."""
        with ui.row().classes("w-full items-end gap-2 mb-2 flex-wrap"):
            ui.label("Filter:").classes("text-xs text-gray-400")

            self.min_intensity_input = (
                ui.number(label="Min Intensity", format="%.0f").props("dense outlined").classes("w-28")
            )

            self.min_quality_input = (
                ui.number(label="Min Quality", format="%.2f").props("dense outlined").classes("w-24")
            )

            self.charge_select = (
                ui.select(["All", "1", "2", "3", "4", "5+"], value="All", label="Charge")
                .props("dense outlined")
                .classes("w-20")
            )

            ui.button("Apply", on_click=self._apply_filter).props("dense size=sm color=primary")
            ui.button("Reset", on_click=self._reset_filter).props("dense size=sm color=grey")

            ui.element("div").classes("flex-grow")  # Spacer

            # Export button
            ui.button("Export TSV", icon="download", on_click=self._export_tsv).props(
                "dense outline size=sm color=grey"
            ).tooltip("Export filtered table data as TSV")

    def _build_table(self):
        """Build the features table."""
        columns = [
            {"name": "idx", "label": "#", "field": "idx", "sortable": True, "align": "left"},
            {"name": "rt", "label": "RT (s)", "field": "rt", "sortable": True, "align": "right"},
            {"name": "mz", "label": "m/z", "field": "mz", "sortable": True, "align": "right"},
            {"name": "intensity", "label": "Intensity", "field": "intensity", "sortable": True, "align": "right"},
            {"name": "charge", "label": "Z", "field": "charge", "sortable": True, "align": "center"},
            {"name": "quality", "label": "Quality", "field": "quality", "sortable": True, "align": "right"},
        ]

        self.feature_table = (
            ui.table(
                columns=columns,
                rows=[],
                row_key="idx",
                pagination={"rowsPerPage": 8, "sortBy": "intensity", "descending": True},
                selection="single",
                on_select=self._on_table_select,
            )
            .classes("w-full hover-highlight")
            .props("flat bordered dense")
        )
        # Note: Removed rowClick handler - on_select already handles selection
        # and rowClick can send large amounts of data via websocket

        # Store reference in state
        self.state.feature_table = self.feature_table

    def update(self) -> None:
        """Update the table display."""
        if self.feature_table is not None:
            self.feature_table.rows = self.state.feature_data
            self.feature_table.update()

    def _has_data(self) -> bool:
        """Check if panel has data to display."""
        return len(self.state.feature_data) > 0

    def _get_filtered_data(self) -> list:
        """Get filtered feature data based on current filters."""
        data = self.state.feature_data

        # Filter by intensity
        if self.min_intensity_input and self.min_intensity_input.value is not None:
            data = [f for f in data if f.get("intensity", 0) >= self.min_intensity_input.value]

        # Filter by quality
        if self.min_quality_input and self.min_quality_input.value is not None:
            data = [f for f in data if f.get("quality", 0) >= self.min_quality_input.value]

        # Filter by charge
        if self.charge_select and self.charge_select.value and self.charge_select.value != "All":
            if self.charge_select.value == "5+":
                data = [f for f in data if f.get("charge", 0) >= 5]
            else:
                charge_val = int(self.charge_select.value)
                data = [f for f in data if f.get("charge", 0) == charge_val]

        return data

    # === Event handlers ===

    def _on_data_loaded(self, data_type: str):
        """Handle data loaded event."""
        if data_type == "features":
            # Update visibility based on whether feature data is present
            self.update_visibility()
            self.update()
            # Auto-expand if features present
            if self._has_data() and self.expansion:
                self.expansion.value = True

    def _on_selection_changed(self, selection_type: str, index: int | None):
        """Handle selection changed event from other panels."""
        if selection_type == "feature" and self.feature_table is not None:
            # Check if we're already selecting this row to avoid loops
            current_selection = self.feature_table.selected
            current_idx = current_selection[0].get("idx") if current_selection else None

            if index != current_idx:
                if index is not None:
                    # Find the row in the current table data that matches this index
                    for row in self.feature_table.rows:
                        if row.get("idx") == index:
                            # Select this row in the table
                            self.feature_table.selected = [row]
                            # Navigate to the page containing this row
                            self._navigate_to_row(index)
                            break
                else:
                    # Clear selection
                    self.feature_table.selected = []

    def _navigate_to_row(self, feature_idx: int):
        """Navigate table pagination to show the row with given feature index.

        Args:
            feature_idx: The feature index to navigate to
        """
        if self.feature_table is None:
            return

        # Get current pagination settings
        pagination = self.feature_table._props.get("pagination", {})
        rows_per_page = pagination.get("rowsPerPage", 8)
        sort_by = pagination.get("sortBy", "intensity")
        descending = pagination.get("descending", True)

        # Get the current rows (which may be filtered)
        rows = list(self.feature_table.rows)
        if not rows:
            return

        # Sort rows the same way the table is sorted to find position
        if sort_by:
            rows = sorted(rows, key=lambda r: r.get(sort_by, 0) or 0, reverse=descending)

        # Find the position of the feature in the sorted list
        row_position = None
        for i, row in enumerate(rows):
            if row.get("idx") == feature_idx:
                row_position = i
                break

        if row_position is None:
            return

        # Calculate which page this row is on (1-indexed)
        page = (row_position // rows_per_page) + 1

        # Update pagination to show that page
        self.feature_table._props["pagination"]["page"] = page
        self.feature_table.update()

    def _on_table_select(self, e):
        """Handle row selection."""
        if e.selection:
            row = e.selection[0]
            if row and "idx" in row:
                self._zoom_to_feature(row["idx"])

    def _apply_filter(self):
        """Apply current filter settings."""
        filtered = self._get_filtered_data()
        if self.feature_table:
            self.feature_table.rows = filtered
        ui.notify(f"Showing {len(filtered)} features", type="info")

    def _reset_filter(self):
        """Reset all filters."""
        if self.min_intensity_input:
            self.min_intensity_input.value = None
        if self.min_quality_input:
            self.min_quality_input.value = None
        if self.charge_select:
            self.charge_select.value = "All"
        self.update()

    def _zoom_to_feature(self, feature_idx: int):
        """Zoom the view to a specific feature.

        Args:
            feature_idx: Index of the feature to zoom to
        """
        if self.state.feature_map is None:
            return

        try:
            feature = self.state.feature_map[feature_idx]
            rt = feature.getRT()
            mz = feature.getMZ()

            # Get feature bounds from ALL convex hulls or use defaults
            hulls = feature.getConvexHulls()
            if hulls:
                all_points = []
                for hull in hulls:
                    points = hull.getHullPoints()
                    all_points.extend([(p[0], p[1]) for p in points])

                if all_points:
                    rts = [p[0] for p in all_points]
                    mzs = [p[1] for p in all_points]
                    rt_min, rt_max = min(rts), max(rts)
                    mz_min, mz_max = min(mzs), max(mzs)
                else:
                    # Default zoom
                    rt_min, rt_max = rt - 10, rt + 10
                    mz_min, mz_max = mz - 2, mz + 2
            else:
                # Default zoom
                rt_min, rt_max = rt - 10, rt + 10
                mz_min, mz_max = mz - 2, mz + 2

            # Ensure minimum zoom range and add padding
            rt_range = max(rt_max - rt_min, 20)
            mz_range = max(mz_max - mz_min, 4)
            rt_pad = rt_range * 0.2
            mz_pad = mz_range * 0.2
            rt_min = rt_min - rt_pad
            rt_max = rt_max + rt_pad
            mz_min = mz_min - mz_pad
            mz_max = mz_max + mz_pad

            # Update view bounds
            self.state.view_rt_min = max(self.state.rt_min, rt_min)
            self.state.view_rt_max = min(self.state.rt_max, rt_max)
            self.state.view_mz_min = max(self.state.mz_min, mz_min)
            self.state.view_mz_max = min(self.state.mz_max, mz_max)

            # Select the feature (this will emit selection_changed event)
            self.state.select_feature(feature_idx)

            # Emit view changed
            self.state.emit_view_changed()

        except Exception as e:
            ui.notify(f"Error zooming to feature: {e}", type="negative")

    def _export_tsv(self):
        """Export filtered table data as TSV file."""
        data = self._get_filtered_data()
        if not data:
            ui.notify("No data to export", type="warning")
            return

        # Column definitions matching the table
        columns = [
            {"field": "idx", "label": "#"},
            {"field": "rt", "label": "RT (s)"},
            {"field": "mz", "label": "m/z"},
            {"field": "intensity", "label": "Intensity"},
            {"field": "charge", "label": "Z"},
            {"field": "quality", "label": "Quality"},
        ]
        column_fields = [col["field"] for col in columns]
        column_labels = [col["label"] for col in columns]

        # Build TSV content
        lines = ["\t".join(column_labels)]  # Header row
        for row in data:
            values = []
            for field in column_fields:
                val = row.get(field, "")
                # Convert None to empty string, format numbers
                if val is None:
                    val = ""
                elif isinstance(val, float):
                    val = f"{val:.4f}" if abs(val) < 1000 else f"{val:.2e}"
                else:
                    val = str(val)
                values.append(val)
            lines.append("\t".join(values))

        tsv_content = "\n".join(lines)

        # Escape backticks for JavaScript template literal
        escaped_content = tsv_content.replace("`", "\\`")

        # Trigger download using JavaScript
        ui.run_javascript(f"""
            const blob = new Blob([`{escaped_content}`], {{type: "text/tab-separated-values"}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "features_table.tsv";
            a.click();
            URL.revokeObjectURL(url);
        """)
        ui.notify(f"Exported {len(data)} rows", type="positive")

    def set_on_feature_selected(self, callback: Callable):
        """Set callback for when a feature is selected.

        Args:
            callback: Function to call with feature index
        """
        self._on_feature_selected = callback
