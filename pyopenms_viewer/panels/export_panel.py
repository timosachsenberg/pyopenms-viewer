"""Data export panel component.

This panel allows users to filter and export subsets of loaded mzML data
by RT range, m/z range, and MS level to a new mzML file.
"""

import tempfile
from pathlib import Path

from nicegui import run, ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel


def _export_filtered_mzml(exp, output_path, rt_min, rt_max, mz_min, mz_max, selected_levels):
    """Export a filtered subset of an MSExperiment to a new mzML file.

    Iterates spectra, filters by MS level and RT range, trims peaks by m/z range,
    and writes to a new file. The original experiment is not modified since
    getSpectrum() returns a copy.

    Args:
        exp: pyOpenMS MSExperiment to filter
        output_path: Path for the output mzML file
        rt_min: Minimum retention time (seconds)
        rt_max: Maximum retention time (seconds)
        mz_min: Minimum m/z value
        mz_max: Maximum m/z value
        selected_levels: Set/list of MS levels to include (e.g. {1, 2})

    Returns:
        Number of spectra written to the output file
    """
    import numpy as np
    from pyopenms import MSExperiment, MzMLFile

    output_exp = MSExperiment()
    count = 0

    for i in range(exp.getNrSpectra()):
        spec = exp.getSpectrum(i)

        # Filter by MS level
        if spec.getMSLevel() not in selected_levels:
            continue

        # Filter by RT range
        rt = spec.getRT()
        if rt < rt_min or rt > rt_max:
            continue

        # Trim peaks by m/z range
        mzs, intensities = spec.get_peaks()
        if len(mzs) > 0:
            mask = (mzs >= mz_min) & (mzs <= mz_max)
            spec.set_peaks((mzs[mask].astype(np.float64), intensities[mask].astype(np.float64)))

        output_exp.addSpectrum(spec)
        count += 1

    MzMLFile().store(str(output_path), output_exp)
    return count


class ExportPanel(BasePanel):
    """Data export panel.

    Allows users to filter loaded mzML data by RT range, m/z range,
    and MS level, then export the filtered subset to a new mzML file.
    """

    def __init__(self, state: ViewerState):
        super().__init__(state, "export", "Data Export", "file_download")

        # UI elements
        self._rt_min_input = None
        self._rt_max_input = None
        self._mz_min_input = None
        self._mz_max_input = None
        self._level_checkboxes = {}
        self._output_input = None
        self._preview_label = None
        self._export_btn = None

    def build(self, container: ui.element) -> ui.expansion:
        with container:
            self._make_expansion()

            with self.expansion:
                # RT range row
                with ui.row().classes("w-full items-end gap-2 flex-wrap"):
                    ui.label("RT range (s):").classes("text-xs text-gray-400")
                    self._rt_min_input = (
                        ui.number(label="RT min", format="%.1f", on_change=self._update_preview)
                        .props("dense outlined")
                        .classes("w-28")
                    )
                    self._rt_max_input = (
                        ui.number(label="RT max", format="%.1f", on_change=self._update_preview)
                        .props("dense outlined")
                        .classes("w-28")
                    )

                    ui.label("m/z range:").classes("text-xs text-gray-400 ml-4")
                    self._mz_min_input = (
                        ui.number(label="m/z min", format="%.2f", on_change=self._update_preview)
                        .props("dense outlined")
                        .classes("w-28")
                    )
                    self._mz_max_input = (
                        ui.number(label="m/z max", format="%.2f", on_change=self._update_preview)
                        .props("dense outlined")
                        .classes("w-28")
                    )

                # Buttons row
                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    ui.button("Use current view", icon="crop_free", on_click=self._use_current_view).props(
                        "flat dense size=sm"
                    )
                    ui.button("Reset to full", icon="zoom_out_map", on_click=self._reset_to_full).props(
                        "flat dense size=sm"
                    )

                # MS level checkboxes
                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    ui.label("MS levels:").classes("text-xs text-gray-400")
                    self._levels_container = ui.row().classes("items-center gap-2")

                # Output path
                with ui.row().classes("w-full items-end gap-2 mt-1"):
                    self._output_input = (
                        ui.input(label="Output file", placeholder="output.mzML")
                        .props("dense outlined")
                        .classes("flex-grow")
                    )

                # Preview + buttons
                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    self._preview_label = ui.label("").classes("text-xs text-gray-400 flex-grow")
                    self._apply_btn = ui.button(
                        "Apply filter", icon="filter_alt", on_click=self._do_apply
                    ).props("flat").tooltip("Replace current data with filtered subset")
                    self._export_btn = ui.button("Download", icon="file_download", on_click=self._do_export).props(
                        "color=primary"
                    )

        # Subscribe to events
        self.state.on_data_loaded(self._on_data_loaded)

        return self.expansion

    def update(self) -> None:
        self._populate_from_state()

    def _has_data(self) -> bool:
        return self.state.exp is not None

    def _on_data_loaded(self, data_type: str) -> None:
        if data_type == "mzml":
            self._populate_from_state()
            self.update_visibility()

    def _populate_from_state(self) -> None:
        if self.state.exp is None:
            return

        # Set range inputs to full data bounds
        self._rt_min_input.value = self.state.rt_min
        self._rt_max_input.value = self.state.rt_max
        self._mz_min_input.value = self.state.mz_min
        self._mz_max_input.value = self.state.mz_max

        # Discover MS levels from spectrum_data
        levels = sorted({s["ms_level"] for s in self.state.spectrum_data if "ms_level" in s})
        if not levels:
            levels = [1]

        # Rebuild level checkboxes
        self._levels_container.clear()
        self._level_checkboxes = {}
        with self._levels_container:
            for lvl in levels:
                cb = ui.checkbox(f"MS{lvl}", value=True, on_change=self._update_preview).props("dense")
                self._level_checkboxes[lvl] = cb

        # Default output path
        if self.state.current_file:
            stem = Path(self.state.current_file).stem
            self._output_input.value = f"{stem}_filtered.mzML"

        self._update_preview()

    def _use_current_view(self) -> None:
        vrt_min = self.state.view_rt_min if self.state.view_rt_min is not None else self.state.rt_min
        vrt_max = self.state.view_rt_max if self.state.view_rt_max is not None else self.state.rt_max
        vmz_min = self.state.view_mz_min if self.state.view_mz_min is not None else self.state.mz_min
        vmz_max = self.state.view_mz_max if self.state.view_mz_max is not None else self.state.mz_max
        self._rt_min_input.value = round(vrt_min, 1)
        self._rt_max_input.value = round(vrt_max, 1)
        self._mz_min_input.value = round(vmz_min, 2)
        self._mz_max_input.value = round(vmz_max, 2)
        self._update_preview()

    def _reset_to_full(self) -> None:
        self._rt_min_input.value = self.state.rt_min
        self._rt_max_input.value = self.state.rt_max
        self._mz_min_input.value = self.state.mz_min
        self._mz_max_input.value = self.state.mz_max
        self._update_preview()

    def _get_selected_levels(self) -> set[int]:
        return {lvl for lvl, cb in self._level_checkboxes.items() if cb.value}

    def _update_preview(self, _=None) -> None:
        if self.state.exp is None or not self.state.spectrum_data:
            self._preview_label.set_text("")
            return

        rt_min = self._rt_min_input.value or 0
        rt_max = self._rt_max_input.value or 0
        selected_levels = self._get_selected_levels()

        count = sum(
            1
            for s in self.state.spectrum_data
            if s.get("ms_level") in selected_levels and rt_min <= s.get("rt", 0) <= rt_max
        )
        self._preview_label.set_text(f"~{count} spectra match filters")

    def _get_filter_params(self):
        """Validate and return filter parameters, or None if invalid."""
        if self.state.exp is None:
            ui.notify("No data loaded", type="warning")
            return None

        selected_levels = self._get_selected_levels()
        if not selected_levels:
            ui.notify("Please select at least one MS level", type="warning")
            return None

        return {
            "rt_min": self._rt_min_input.value or 0,
            "rt_max": self._rt_max_input.value or 0,
            "mz_min": self._mz_min_input.value or 0,
            "mz_max": self._mz_max_input.value or 0,
            "selected_levels": selected_levels,
        }

    async def _do_export(self) -> None:
        params = self._get_filter_params()
        if params is None:
            return

        output_name = (self._output_input.value or "").strip()
        if not output_name:
            ui.notify("Please specify an output file name", type="warning")
            return

        # Write to a temp file so the download works even in web/remote mode
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mzML")
        import os

        os.close(tmp_fd)

        self._export_btn.props("loading")
        try:
            count = await run.io_bound(
                _export_filtered_mzml, self.state.exp, tmp_path, **params
            )
            ui.notify(f"Exported {count} spectra", type="positive")
            ui.download(tmp_path, filename=output_name)
        except Exception as e:
            ui.notify(f"Export failed: {e}", type="negative")
        finally:
            self._export_btn.props(remove="loading")

    async def _do_apply(self) -> None:
        """Export filtered data to a temp file, then reload it as the current file."""
        params = self._get_filter_params()
        if params is None:
            return

        if not hasattr(self.state, "_load_mzml"):
            ui.notify("Loader not available", type="warning")
            return

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mzML")
        import os

        os.close(tmp_fd)

        self._apply_btn.props("loading")
        try:
            count = await run.io_bound(
                _export_filtered_mzml, self.state.exp, tmp_path, **params
            )
            if count == 0:
                ui.notify("No spectra match filters", type="warning")
                return

            # Derive a display name from current file
            name = "filtered.mzML"
            if self.state.current_file:
                stem = Path(self.state.current_file).stem
                name = f"{stem}_filtered.mzML"

            # Clear current file path so load_mzml doesn't skip as "already loaded"
            self.state.current_file = None
            await self.state._load_mzml(tmp_path, name)
            ui.notify(f"Applied filter: {count} spectra loaded", type="positive")
        except Exception as e:
            ui.notify(f"Apply filter failed: {e}", type="negative")
        finally:
            self._apply_btn.props(remove="loading")
