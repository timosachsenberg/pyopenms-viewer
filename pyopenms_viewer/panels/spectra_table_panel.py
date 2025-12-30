"""Spectra table panel component.

This panel displays a table of all spectra with their metadata,
including identification information when available.
"""

from typing import Callable, Optional

from nicegui import ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel


class SpectraTablePanel(BasePanel):
    """Spectra table panel.

    Features:
    - Table of all spectra with metadata
    - View mode filters (All, MS2, Identified)
    - RT and sequence filtering
    - Click to view spectrum
    - Integration with ID data showing sequences and scores
    """

    def __init__(self, state: ViewerState):
        """Initialize spectra table panel.

        Args:
            state: ViewerState instance (shared reference)
        """
        super().__init__(state, "spectra_table", "Spectra", "list")

        # UI elements
        self.spectrum_table = None
        self.view_mode_toggle = None
        self.show_advanced_cb = None
        self.show_meta_values_cb = None
        self.show_all_hits_cb = None
        self.rt_min_input = None
        self.rt_max_input = None
        self.seq_pattern_input = None
        self.min_score_input = None
        self.annotate_peaks_cb = None
        self.tolerance_input = None
        self.mirror_view_cb = None

        # Callback for spectrum selection
        self._on_spectrum_selected: Optional[Callable] = None

    def build(self, container: ui.element) -> ui.expansion:
        """Build the spectra table panel UI.

        Args:
            container: Parent element to build panel in

        Returns:
            The expansion element created
        """
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                self._build_help_text()
                self._build_view_controls()
                self._build_filter_controls()
                self._build_table()

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)

        self._is_built = True
        return self.expansion

    def _build_help_text(self):
        """Build the help text."""
        ui.label("Click a row to view spectrum. Identified spectra show sequence and score.").classes(
            "text-sm text-gray-400 mb-2"
        )

    def _build_view_controls(self):
        """Build the view mode and column toggles."""
        with ui.row().classes("w-full items-center gap-4 mb-2"):
            # View filter
            self.view_mode_toggle = (
                ui.toggle(["All", "MS2", "Identified"], value="All", on_change=self._update_table)
                .props("dense size=sm")
                .classes("text-xs")
            )

            # Advanced columns toggle
            self.show_advanced_cb = (
                ui.checkbox("Advanced", value=False, on_change=self._rebuild_columns)
                .props("dense")
                .classes("text-xs text-gray-400")
            )
            ui.tooltip("Show additional columns: Peaks, TIC, BPI, m/z Range")

            # Meta values toggle
            self.show_meta_values_cb = (
                ui.checkbox("Meta Values", value=False, on_change=self._rebuild_columns)
                .props("dense")
                .classes("text-xs text-gray-400")
            )
            ui.tooltip("Show PeptideIdentification meta values")

            # All hits toggle
            self.show_all_hits_cb = (
                ui.checkbox("All Hits", value=False, on_change=self._update_table)
                .props("dense")
                .classes("text-xs text-gray-400")
            )
            ui.tooltip("Show all peptide hits for each spectrum")

            ui.element("div").classes("flex-grow")  # Spacer

            # Export button
            ui.button("Export TSV", icon="download", on_click=self._export_tsv).props(
                "dense outline size=sm color=grey"
            ).tooltip("Export filtered table data as TSV")

    def _build_filter_controls(self):
        """Build the filter controls row."""
        with ui.row().classes("w-full items-end gap-2 mb-2 flex-wrap"):
            ui.label("Filter:").classes("text-xs text-gray-400")

            self.rt_min_input = (
                ui.number(label="RT Min", format="%.0f", on_change=self._update_table)
                .props("dense outlined")
                .classes("w-20")
            )

            self.rt_max_input = (
                ui.number(label="RT Max", format="%.0f", on_change=self._update_table)
                .props("dense outlined")
                .classes("w-20")
            )

            self.seq_pattern_input = (
                ui.input(label="Sequence", placeholder="e.g. PEPTIDE", on_change=self._update_table)
                .props("dense outlined")
                .classes("w-28")
            )

            self.min_score_input = (
                ui.number(label="Min Score", format="%.2f", on_change=self._update_table)
                .props("dense outlined")
                .classes("w-24")
            )

            # Annotation settings
            self.annotate_peaks_cb = (
                ui.checkbox("Annotate", value=self.state.annotate_peaks, on_change=self._toggle_annotate)
                .props("dense")
                .classes("text-blue-400")
            )

            self.tolerance_input = (
                ui.number(
                    label="Tol (Da)",
                    value=self.state.annotation_tolerance_da,
                    format="%.2f",
                    on_change=self._update_tolerance,
                )
                .props("dense outlined")
                .classes("w-20")
            )
            ui.tooltip("Mass tolerance for matching peaks to theoretical ions (Da)")

            self.mirror_view_cb = (
                ui.checkbox("Mirror", value=self.state.mirror_annotation_view, on_change=self._toggle_mirror)
                .props("dense")
                .classes("text-blue-400")
            )
            ui.tooltip("Mirror view: flip annotated peaks downward for comparison")

    def _build_table(self):
        """Build the spectrum table."""
        columns = self._build_columns()

        self.spectrum_table = (
            ui.table(
                columns=columns,
                rows=self._get_filtered_data(),
                row_key="idx",
                pagination={"rowsPerPage": 20, "sortBy": "idx"},
                selection="single",
                on_select=self._on_table_select,
            )
            .classes("w-full")
            .props("dense flat bordered virtual-scroll")
        )

        # Store reference in state
        self.state.spectrum_table = self.spectrum_table

    def _build_columns(self) -> list:
        """Build column definitions based on current toggles."""
        # Start with basic columns (without CV)
        cols = [
            {"name": "idx", "label": "#", "field": "idx", "sortable": True, "align": "left"},
            {"name": "rt", "label": "RT (s)", "field": "rt", "sortable": True, "align": "right"},
            {"name": "ms_level", "label": "MS", "field": "ms_level", "sortable": True, "align": "center"},
        ]

        # Add CV column only if FAIMS data is present
        if self.state.has_faims:
            cols.append({"name": "cv", "label": "CV", "field": "cv", "sortable": True, "align": "right"})

        advanced_columns = [
            {"name": "n_peaks", "label": "Peaks", "field": "n_peaks", "sortable": True, "align": "right"},
            {"name": "tic", "label": "TIC", "field": "tic", "sortable": True, "align": "right"},
            {"name": "bpi", "label": "BPI", "field": "bpi", "sortable": True, "align": "right"},
            {"name": "mz_range", "label": "m/z Range", "field": "mz_range", "sortable": False, "align": "center"},
        ]

        if self.show_advanced_cb and self.show_advanced_cb.value:
            cols = cols + advanced_columns

        # Add precursor columns
        cols.append(
            {"name": "precursor_mz", "label": "Prec m/z", "field": "precursor_mz", "sortable": True, "align": "right"}
        )
        cols.append({"name": "precursor_z", "label": "Z", "field": "precursor_z", "sortable": True, "align": "center"})

        # Add rank column if showing all hits
        if self.show_all_hits_cb and self.show_all_hits_cb.value:
            cols.append({"name": "hit_rank", "label": "Rank", "field": "hit_rank", "sortable": True, "align": "center"})

        # Add ID columns
        cols.append({"name": "sequence", "label": "Sequence", "field": "sequence", "sortable": True, "align": "left"})
        cols.append({"name": "score", "label": "Score", "field": "score", "sortable": True, "align": "right"})

        # Add meta value columns if enabled
        if self.show_meta_values_cb and self.show_meta_values_cb.value:
            for key in self.state.id_meta_keys:
                prefix, name = key.split(":", 1) if ":" in key else ("", key)
                label_prefix = "PID" if prefix == "pid" else "Hit" if prefix == "hit" else prefix.upper()
                label_name = name.replace("_", " ").title()
                if len(label_name) > 15:
                    label_name = label_name[:13] + ".."
                label = f"{label_name} ({label_prefix})"
                cols.append(
                    {
                        "name": key,
                        "label": label,
                        "field": key,
                        "sortable": True,
                        "align": "left",
                    }
                )

        return cols

    def _get_filtered_data(self) -> list:
        """Get filtered spectrum data based on current filters."""
        data = self.state.spectrum_data

        # Apply view mode filter
        mode = self.view_mode_toggle.value if self.view_mode_toggle else "All"
        if mode == "MS2":
            data = [s for s in data if s.get("ms_level") == 2]
        elif mode == "Identified":
            data = [s for s in data if s.get("id_idx") is not None]

        # Apply RT filter
        if self.rt_min_input and self.rt_min_input.value is not None:
            data = [s for s in data if s.get("rt", 0) >= self.rt_min_input.value]
        if self.rt_max_input and self.rt_max_input.value is not None:
            data = [s for s in data if s.get("rt", 0) <= self.rt_max_input.value]

        # Apply sequence filter
        if self.seq_pattern_input and self.seq_pattern_input.value:
            pattern = self.seq_pattern_input.value.upper()
            data = [s for s in data if pattern in s.get("sequence", "").upper()]

        # Apply score filter
        if self.min_score_input and self.min_score_input.value is not None:
            data = [s for s in data if (s.get("score") or 0) >= self.min_score_input.value]

        return data

    def update(self) -> None:
        """Update the table display."""
        if self.spectrum_table is not None:
            self.spectrum_table.rows = self._get_filtered_data()
            self.spectrum_table.update()

    def _has_data(self) -> bool:
        """Check if panel has data to display."""
        return len(self.state.spectrum_data) > 0

    # === Event handlers ===

    def _on_data_loaded(self, data_type: str):
        """Handle data loaded event."""
        if data_type in ("mzml", "ids"):
            # Rebuild columns when mzML loaded (CV column depends on has_faims)
            if data_type == "mzml" and self.spectrum_table is not None:
                self.spectrum_table.columns = self._build_columns()
            if self._has_data():
                self.update()
                # Auto-expand when data is loaded
                if data_type == "mzml" and self.expansion:
                    self.expansion.value = True
            else:
                # Clear table when data is removed
                if self.spectrum_table is not None:
                    self.spectrum_table.rows = []
                    self.spectrum_table.update()
                if self.expansion:
                    self.expansion.value = False

    def _on_table_select(self, e):
        """Handle row selection in table."""
        if e.selection:
            selected_row = e.selection[0] if isinstance(e.selection, list) else e.selection
            if isinstance(selected_row, dict) and "idx" in selected_row:
                spectrum_idx = selected_row["idx"]
                self.state.select_spectrum(spectrum_idx)

    def _update_table(self, e=None):
        """Update table after filter change."""
        self.update()

    def _rebuild_columns(self, e=None):
        """Rebuild columns after toggle change."""
        if self.spectrum_table is not None:
            self.spectrum_table.columns = self._build_columns()
            self.spectrum_table.update()
        self.update()

    def _toggle_annotate(self, e):
        """Toggle peak annotation."""
        self.state.annotate_peaks = e.value
        if self.state.selected_spectrum_idx is not None:
            self.state.emit_selection_changed("spectrum", self.state.selected_spectrum_idx)

    def _update_tolerance(self, e):
        """Update annotation tolerance."""
        if self.tolerance_input and self.tolerance_input.value is not None and self.tolerance_input.value > 0:
            self.state.annotation_tolerance_da = self.tolerance_input.value
            if self.state.selected_spectrum_idx is not None and self.state.annotate_peaks:
                self.state.emit_selection_changed("spectrum", self.state.selected_spectrum_idx)

    def _toggle_mirror(self, e):
        """Toggle mirror view."""
        self.state.mirror_annotation_view = e.value
        if self.state.selected_spectrum_idx is not None:
            self.state.emit_selection_changed("spectrum", self.state.selected_spectrum_idx)

    def _export_tsv(self):
        """Export filtered table data as TSV file."""
        data = self._get_filtered_data()
        if not data:
            ui.notify("No data to export", type="warning")
            return

        # Get column definitions
        columns = self._build_columns()
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
            a.download = "spectra_table.tsv";
            a.click();
            URL.revokeObjectURL(url);
        """)
        ui.notify(f"Exported {len(data)} rows", type="positive")

    def set_on_spectrum_selected(self, callback: Callable):
        """Set callback for when a spectrum is selected.

        Args:
            callback: Function to call with spectrum index
        """
        self._on_spectrum_selected = callback
