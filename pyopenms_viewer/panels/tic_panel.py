"""TIC (Total Ion Chromatogram) panel implementation."""

from typing import Optional

import plotly.graph_objects as go
from nicegui import ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel


class TICPanel(BasePanel):
    """Total Ion Chromatogram panel.

    Displays the TIC or BPC trace with:
    - Interactive Plotly chart
    - Click to select spectrum at RT
    - Drag to zoom RT range
    - Visual indicator of current view range
    """

    def __init__(self, state: ViewerState):
        super().__init__(state, "tic", "TIC", "show_chart")
        self.plot: Optional[ui.plotly] = None
        self._updating_from_tic: bool = False

        # Plotly config (must be included in figure dict)
        self._plotly_config = {
            "modeBarButtonsToRemove": ["autoScale2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "tic",
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
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                ui.label("Click to view spectrum, drag to zoom RT range").classes("text-xs text-gray-500 mb-1")
                self.plot = ui.plotly(self._figure_with_config(self._create_figure())).classes("w-full")
                # Only register events we actually need to avoid large websocket messages
                self.plot.on("plotly_click", self._on_click)
                # Note: plotly_selected removed - use plotly_relayout for zooming instead
                # plotly_selected can send large amounts of point data
                self.plot.on("plotly_relayout", self._on_relayout)

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)
        self.state.on_view_changed(self._on_view_changed)
        self.state.on_selection_changed(self._on_selection_changed)

        return self.expansion

    def update(self) -> None:
        if self.plot and not self._updating_from_tic:
            self.plot.update_figure(self._figure_with_config(self._create_figure()))

    def _create_figure(self) -> go.Figure:
        """Create the TIC Plotly figure."""
        fig = go.Figure()

        if self.state.tic_rt is None or len(self.state.tic_rt) == 0:
            fig.update_layout(
                title={"text": "TIC - No data loaded", "font": {"color": "#888"}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200,
            )
            return fig

        # RT conversion
        rt_divisor = 60.0 if self.state.rt_in_minutes else 1.0
        display_rt = self.state.tic_rt / rt_divisor

        # Main TIC trace
        fig.add_trace(
            go.Scatter(
                x=display_rt,
                y=self.state.tic_intensity,
                mode="lines",
                name=self.state.tic_source,
                line={"color": "#00d4ff", "width": 1},
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.2)",
            )
        )

        # Add view range indicator
        view_rt_min = self.state.view_rt_min if self.state.view_rt_min is not None else self.state.rt_min
        view_rt_max = self.state.view_rt_max if self.state.view_rt_max is not None else self.state.rt_max

        fig.add_vrect(
            x0=view_rt_min / rt_divisor,
            x1=view_rt_max / rt_divisor,
            fillcolor="rgba(255, 255, 0, 0.15)",
            line_width=1,
            line_color="rgba(255, 255, 0, 0.5)",
        )

        # Add selected spectrum marker
        if self.state.selected_spectrum_idx is not None and self.state.exp is not None:
            spec = self.state.exp[self.state.selected_spectrum_idx]
            marker_rt = spec.getRT() / rt_divisor
            fig.add_vline(
                x=marker_rt,
                line_color="#ff6b6b",
                line_width=2,
                line_dash="dash",
            )

        # Layout
        rt_unit = "min" if self.state.rt_in_minutes else "s"
        fig.update_layout(
            title={"text": self.state.tic_source, "font": {"color": "#888", "size": 14}},
            xaxis_title=f"RT ({rt_unit})",
            yaxis_title="Intensity",
            height=200,
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#888"},
            xaxis={
                "gridcolor": "rgba(128,128,128,0.2)",
                "linecolor": "#888",
                "tickcolor": "#888",
            },
            yaxis={
                "gridcolor": "rgba(128,128,128,0.2)",
                "linecolor": "#888",
                "tickcolor": "#888",
                "exponentformat": "e",
            },
            dragmode="zoom",  # Use zoom instead of select to avoid large websocket messages
        )

        return fig

    def _on_data_loaded(self, data_type: str) -> None:
        if data_type == "mzml":
            if self._has_data():
                self.update()
                # Auto-expand when data is loaded
                if self.expansion:
                    self.expansion.value = True
            else:
                # Clear display when data is removed
                self._clear_display()
                if self.expansion:
                    self.expansion.value = False

    def _clear_display(self) -> None:
        """Clear the TIC display."""
        if self.plot is not None:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis_title="RT",
                yaxis_title="Intensity",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            self.plot.update_figure(empty_fig)

    def _on_view_changed(self) -> None:
        if not self._updating_from_tic:
            self.update()

    def _on_selection_changed(self, selection_type: str, index: Optional[int]) -> None:
        if selection_type == "spectrum":
            self.update()

    def _on_click(self, e) -> None:
        """Handle TIC click to show spectrum at clicked RT and center peak map."""
        if not e.args or "points" not in e.args:
            return

        points = e.args.get("points", [])
        if not points:
            return

        clicked_rt = points[0].get("x", 0)
        if self.state.rt_in_minutes:
            clicked_rt *= 60.0

        # Find closest spectrum
        if self.state.exp is not None:
            best_idx = 0
            best_diff = float("inf")
            for i in range(len(self.state.exp)):
                diff = abs(self.state.exp[i].getRT() - clicked_rt)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            # Select spectrum (triggers spectrum panel update)
            self.state.select_spectrum(best_idx)

            # Also center the peak map on this RT (matching original behavior)
            rt_range = self.state.view_rt_max - self.state.view_rt_min
            new_rt_min = max(self.state.rt_min, clicked_rt - rt_range / 2)
            new_rt_max = min(self.state.rt_max, clicked_rt + rt_range / 2)
            self.state.view_rt_min = new_rt_min
            self.state.view_rt_max = new_rt_max
            self.state.emit_view_changed()

    def _on_relayout(self, e) -> None:
        """Handle zoom/pan events from Plotly - sync RT range to peak map."""
        try:
            if not e.args:
                return

            # Check for x-axis range changes (zoom or pan)
            if "xaxis.range[0]" in e.args and "xaxis.range[1]" in e.args:
                rt_min = float(e.args["xaxis.range[0]"])
                rt_max = float(e.args["xaxis.range[1]"])

                # Convert from display units
                if self.state.rt_in_minutes:
                    rt_min *= 60.0
                    rt_max *= 60.0

                # Clamp to data bounds and update view
                self._updating_from_tic = True
                self.state.set_view(rt_min=max(self.state.rt_min, rt_min), rt_max=min(self.state.rt_max, rt_max))
                self._updating_from_tic = False

            elif e.args.get("xaxis.autorange"):
                # Reset to full range
                self._updating_from_tic = True
                self.state.set_view(rt_min=self.state.rt_min, rt_max=self.state.rt_max)
                self._updating_from_tic = False

        except Exception:
            self._updating_from_tic = False

    def _has_data(self) -> bool:
        return self.state.tic_rt is not None and len(self.state.tic_rt) > 0
