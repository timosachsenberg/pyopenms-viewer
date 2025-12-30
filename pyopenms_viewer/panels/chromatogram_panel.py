"""Chromatogram panel component.

This panel displays extracted ion chromatograms (XICs) and other
chromatogram data from mzML files.
"""

from typing import Optional

import plotly.graph_objects as go
from nicegui import ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel


class ChromatogramPanel(BasePanel):
    """Chromatogram viewer panel.

    Features:
    - Display multiple chromatograms simultaneously
    - Table for chromatogram selection (multi-select supported)
    - View range indicator showing current RT window
    - RT unit toggle (seconds/minutes)
    """

    def __init__(self, state: ViewerState):
        """Initialize chromatogram panel.

        Args:
            state: ViewerState instance (shared reference)
        """
        super().__init__(state, "chromatograms", "Chromatograms", "timeline")

        # UI elements
        self.chromatogram_plot: Optional[ui.plotly] = None
        self.chromatogram_table = None
        self.info_label: Optional[ui.label] = None

        # Color palette for multiple chromatograms
        self.colors = [
            "#00d4ff",
            "#ff6b6b",
            "#4ecdc4",
            "#ffe66d",
            "#95e1d3",
            "#f38181",
            "#aa96da",
            "#fcbad3",
            "#a8d8ea",
            "#ffb6b9",
        ]

        # Plotly config (must be included in figure dict)
        self._plotly_config = {
            "modeBarButtonsToRemove": ["autoScale2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "chromatogram",
                "width": 1200,
                "height": 400,
                "scale": 1,
            },
        }

    def _figure_with_config(self, fig: go.Figure) -> dict:
        """Convert go.Figure to dict and add config for modebar customization."""
        fig_dict = fig.to_plotly_json()
        fig_dict["config"] = self._plotly_config
        return fig_dict

    def build(self, container: ui.element) -> ui.expansion:
        """Build the chromatogram panel UI.

        Args:
            container: Parent element to build panel in

        Returns:
            The expansion element created
        """
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                self._build_header_row()
                self._build_plot()
                self._build_table()

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)
        self.state.on_view_changed(self._on_view_changed)
        self.state.on_display_options_changed(self._on_display_options_changed)

        self._is_built = True
        return self.expansion

    def _build_header_row(self):
        """Build the header row with info and controls."""
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            self.info_label = ui.label("No chromatograms loaded").classes("text-sm text-gray-400")
            ui.element("div").classes("flex-grow")

            ui.button("Export TSV", icon="download", on_click=self._export_tsv).props(
                "dense outline size=sm color=grey"
            ).tooltip("Export chromatogram table as TSV")

            ui.button("Clear Selection", on_click=self._clear_selection).props("dense outline size=sm color=grey")

    def _build_plot(self):
        """Build the chromatogram plot."""
        self.chromatogram_plot = ui.plotly(self._figure_with_config(self._create_figure())).classes("w-full")

        # Store reference in state
        self.state.chromatogram_plot = self.chromatogram_plot

    def _build_table(self):
        """Build the chromatogram selection table."""
        ui.label("Chromatogram Table (click to select, Ctrl+click to multi-select)").classes(
            "text-xs text-gray-500 mt-2"
        )

        columns = [
            {"name": "idx", "label": "#", "field": "idx", "sortable": True, "align": "left"},
            {"name": "type", "label": "Type", "field": "type", "sortable": True, "align": "left"},
            {"name": "native_id", "label": "Native ID", "field": "native_id", "sortable": True, "align": "left"},
            {"name": "precursor_mz", "label": "Q1 (m/z)", "field": "precursor_mz", "sortable": True, "align": "right"},
            {"name": "product_mz", "label": "Q3 (m/z)", "field": "product_mz", "sortable": True, "align": "right"},
            {"name": "rt_min", "label": "RT Start", "field": "rt_min", "sortable": True, "align": "right"},
            {"name": "rt_max", "label": "RT End", "field": "rt_max", "sortable": True, "align": "right"},
            {"name": "n_points", "label": "Points", "field": "n_points", "sortable": True, "align": "right"},
            {"name": "max_int", "label": "Max Int", "field": "max_int", "sortable": True, "align": "right"},
        ]

        self.chromatogram_table = (
            ui.table(
                columns=columns,
                rows=self.state.chromatograms,
                row_key="idx",
                pagination={"rowsPerPage": 15, "sortBy": "idx"},
                selection="multiple",
                on_select=self._on_table_select,
            )
            .classes("w-full")
            .props("dense flat bordered virtual-scroll")
        )

        # Store reference in state
        self.state.chromatogram_table = self.chromatogram_table

    def update(self) -> None:
        """Update the chromatogram display."""
        if self.chromatogram_plot is not None:
            fig = self._create_figure()
            self.chromatogram_plot.update_figure(self._figure_with_config(fig))

        # Update info label
        if self.info_label is not None:
            if self.state.has_chromatograms:
                n_chroms = len(self.state.chromatograms)
                n_selected = len(self.state.selected_chromatogram_indices)
                self.info_label.set_text(f"{n_chroms} chromatograms available, {n_selected} selected")
            else:
                self.info_label.set_text("No chromatograms loaded")

        # Update table rows
        if self.chromatogram_table is not None:
            self.chromatogram_table.rows = self.state.chromatograms
            self.chromatogram_table.update()

    def _has_data(self) -> bool:
        """Check if panel has data to display."""
        return self.state.has_chromatograms

    def _create_figure(self) -> go.Figure:
        """Create a Plotly figure showing selected chromatograms.

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        if not self.state.has_chromatograms or not self.state.selected_chromatogram_indices:
            fig.update_layout(
                title={"text": "Chromatograms - Select from table below", "font": {"color": "#888"}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#888"},
                height=250,
            )
            # Hide axes completely when nothing selected
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        # Convert RT to display units
        rt_divisor = 60.0 if self.state.rt_in_minutes else 1.0
        rt_unit = "min" if self.state.rt_in_minutes else "s"

        # Plot each selected chromatogram
        for i, chrom_idx in enumerate(self.state.selected_chromatogram_indices):
            if chrom_idx not in self.state.chromatogram_data:
                continue

            rt_array, int_array = self.state.chromatogram_data[chrom_idx]
            display_rt = rt_array / rt_divisor

            # Find metadata for label
            chrom_meta = next((c for c in self.state.chromatograms if c["idx"] == chrom_idx), None)
            if chrom_meta:
                native_id = chrom_meta["native_id"]
                label = native_id[:27] + "..." if len(native_id) > 30 else native_id
            else:
                label = f"Chrom {chrom_idx}"

            color = self.colors[i % len(self.colors)]

            fig.add_trace(
                go.Scatter(
                    x=display_rt,
                    y=int_array,
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 1.5},
                    hovertemplate=f"{label}<br>RT: %{{x:.2f}}{rt_unit}<br>Intensity: %{{y:.2e}}<extra></extra>",
                )
            )

        # Add view range indicator if data is loaded
        if self.state.view_rt_min is not None and self.state.view_rt_max is not None:
            fig.add_vrect(
                x0=self.state.view_rt_min / rt_divisor,
                x1=self.state.view_rt_max / rt_divisor,
                fillcolor="rgba(255, 255, 0, 0.1)",
                layer="below",
                line_width=1,
                line_color="rgba(255, 255, 0, 0.3)",
            )

        n_selected = len(self.state.selected_chromatogram_indices)
        title_text = f"Chromatograms ({n_selected} selected)"

        fig.update_layout(
            title={"text": title_text, "font": {"size": 14, "color": "#888"}},
            xaxis_title=f"RT ({rt_unit})",
            yaxis_title="Intensity",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#888"},
            height=250,
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "font": {"size": 9},
            },
            hovermode="x unified",
        )

        # Style axes
        fig.update_xaxes(showgrid=False, linecolor="#888", tickcolor="#888")
        fig.update_yaxes(showgrid=False, linecolor="#888", tickcolor="#888")

        return fig

    # === Event handlers ===

    def _on_data_loaded(self, data_type: str):
        """Handle data loaded event."""
        if data_type == "mzml":
            # Update visibility based on whether chromatogram data is present
            self.update_visibility()
            self.update()
            # Auto-expand if chromatograms are present
            if self.state.has_chromatograms and self.expansion:
                self.expansion.value = True

    def _on_view_changed(self):
        """Handle view changed event."""
        # Update to show new view range indicator
        self.update()

    def _on_display_options_changed(self, option_name: str, value):
        """Handle display options changed event."""
        if option_name == "rt_in_minutes":
            self.update()

    def _on_table_select(self, e):
        """Handle chromatogram selection from table."""
        if e.selection:
            self.state.selected_chromatogram_indices = [row["idx"] for row in e.selection if "idx" in row]
            self.update()

    def _clear_selection(self):
        """Clear all selected chromatograms."""
        self.state.selected_chromatogram_indices = []
        if self.chromatogram_table:
            self.chromatogram_table.selected = []
            self.chromatogram_table.update()
        self.update()

    def _export_tsv(self):
        """Export chromatogram table data as TSV file."""
        data = self.state.chromatograms
        if not data:
            ui.notify("No data to export", type="warning")
            return

        # Column definitions matching the table
        columns = [
            {"field": "idx", "label": "#"},
            {"field": "type", "label": "Type"},
            {"field": "native_id", "label": "Native ID"},
            {"field": "precursor_mz", "label": "Q1 (m/z)"},
            {"field": "product_mz", "label": "Q3 (m/z)"},
            {"field": "rt_min", "label": "RT Start"},
            {"field": "rt_max", "label": "RT End"},
            {"field": "n_points", "label": "Points"},
            {"field": "max_int", "label": "Max Int"},
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
            a.download = "chromatograms_table.tsv";
            a.click();
            URL.revokeObjectURL(url);
        """)
        ui.notify(f"Exported {len(data)} rows", type="positive")
