"""Python scripting panel component.

Provides a code editor for running custom pyOpenMS scripts against loaded data.
"""

import io
import sys
import traceback

from nicegui import run, ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel

_DEFAULT_CODE = (
    "# Variables: exp, feature_map, peptide_ids, protein_ids, state\n"
    "# Selection: selected_spectrum_idx, selected_feature_idx, selected_id_idx\n"
    "# Chromatograms: chromatogram_data, selected_chromatogram_indices\n"
    "# Modules: pyopenms (also as oms), np (numpy), pd (pandas)\n"
    "# Note: exp[i] returns a copy — modify and write back: exp[i] = spec\n"
    "s = selected_spectrum_idx\n"
    "print(f'Selected spectrum: {s}, total: {exp.getNrSpectra()}')\n"
)


class ScriptingPanel(BasePanel):
    """Python scripting panel.

    Provides a code editor with Python syntax highlighting for running
    custom pyOpenMS code against the loaded MSExperiment, FeatureMap,
    and peptide IDs.
    """

    def __init__(self, state: ViewerState):
        super().__init__(state, "scripting", "Python Scripting", "code")
        self._editor = None
        self._output_log = None
        self._force_refresh = None

    def build(self, container: ui.element) -> ui.expansion:
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                self._editor = ui.codemirror(value=_DEFAULT_CODE, language="Python").classes("w-full").style(
                    "max-height: 400px;"
                )

                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    ui.button("Run", icon="play_arrow", on_click=self._run_script).props("color=primary dense")
                    ui.button("Clear Output", icon="delete_outline", on_click=self._clear_output).props(
                        "flat dense size=sm"
                    )
                    self._force_refresh = ui.checkbox("Force refresh after run", value=True).props("dense").classes(
                        "ml-4 text-sm"
                    )

                self._output_log = ui.log(max_lines=500).classes("w-full h-48 mt-1")

        self.state.on_data_loaded(self._on_data_loaded)
        self.update_visibility()
        self._is_built = True
        return self.expansion

    def update(self) -> None:
        pass

    def _has_data(self) -> bool:
        return self.state.exp is not None

    def _on_data_loaded(self, data_type: str) -> None:
        self.update_visibility()

    def _clear_output(self) -> None:
        if self._output_log is not None:
            self._output_log.clear()

    async def _run_script(self) -> None:
        if self._editor is None or self._output_log is None:
            return

        code = self._editor.value
        if not code or not code.strip():
            self._output_log.push("[No code to execute]")
            return

        # Capture references before execution for identity checks
        orig_exp = self.state.exp
        orig_feature_map = self.state.feature_map
        orig_peptide_ids = self.state.peptide_ids
        orig_protein_ids = self.state.protein_ids

        self._output_log.push(">>> Running script...")

        try:
            result = await run.io_bound(self._exec_code, code)
        except Exception as e:
            self._output_log.push(f"[Internal error: {e}]")
            return

        stdout_text, error_text, namespace = result

        if stdout_text:
            for line in stdout_text.splitlines():
                self._output_log.push(line)

        if error_text:
            self._output_log.push("--- ERROR ---")
            for line in error_text.splitlines():
                self._output_log.push(line)
            self._output_log.push(">>> Done (with errors).")
            return

        # Check for state changes and refresh
        force = self._force_refresh.value if self._force_refresh else False
        exp_changed = namespace.get("exp") is not orig_exp
        feature_map_changed = namespace.get("feature_map") is not orig_feature_map
        ids_changed = namespace.get("peptide_ids") is not orig_peptide_ids

        if exp_changed or force:
            if exp_changed:
                self.state.exp = namespace["exp"]
            await self._refresh_mzml()
            self._output_log.push("[State refreshed: mzML data re-derived]")

        if feature_map_changed or force:
            if feature_map_changed:
                self.state.feature_map = namespace["feature_map"]
            self._refresh_features()
            self._output_log.push("[State refreshed: feature data re-derived]")

        if ids_changed or force:
            if ids_changed:
                self.state.peptide_ids = namespace["peptide_ids"]
            if namespace.get("protein_ids") is not orig_protein_ids:
                self.state.protein_ids = namespace["protein_ids"]
            self._refresh_ids()
            self._output_log.push("[State refreshed: ID data re-derived]")

        self._output_log.push(">>> Done.")

    def _exec_code(self, code: str) -> tuple[str, str, dict]:
        """Execute code in a namespace with loaded data. Runs in io_bound thread."""
        import numpy as np
        import pandas as pd
        import pyopenms

        namespace = {
            "state": self.state,
            "exp": self.state.exp,
            "feature_map": self.state.feature_map,
            "peptide_ids": self.state.peptide_ids,
            "protein_ids": self.state.protein_ids,
            "spectrum_data": self.state.spectrum_data,
            "selected_spectrum_idx": self.state.selected_spectrum_idx,
            "selected_feature_idx": self.state.selected_feature_idx,
            "selected_id_idx": self.state.selected_id_idx,
            "chromatogram_data": self.state.chromatogram_data,
            "selected_chromatogram_indices": self.state.selected_chromatogram_indices,
            "oms": pyopenms,
            "np": np,
            "pd": pd,
        }
        namespace.update(pyopenms.__dict__)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr
            exec(code, namespace)  # noqa: S102
            return captured_stdout.getvalue(), "", namespace
        except Exception:
            tb = traceback.format_exc()
            return captured_stdout.getvalue(), tb, namespace
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    async def _refresh_mzml(self) -> None:
        """Re-derive all mzML data from state.exp and emit event."""
        from pyopenms_viewer.loaders import MzMLLoader

        loader = MzMLLoader(self.state)
        filepath = self.state.current_file or "scripting"
        await run.io_bound(loader.process, filepath)
        self.state.cached_minimap_raster = None
        self.state.faims_minimap_rasters = {}
        self.state.emit_data_loaded("mzml")

    def _refresh_features(self) -> None:
        """Re-derive feature data and emit event."""
        from pyopenms_viewer.loaders.feature_loader import extract_feature_data

        self.state.feature_data = extract_feature_data(self.state)
        self.state.emit_data_loaded("features")

    def _refresh_ids(self) -> None:
        """Re-derive ID data and emit event."""
        from pyopenms_viewer.loaders.id_loader import extract_id_data, link_ids_to_spectra

        self.state.id_data = extract_id_data(self.state)
        link_ids_to_spectra(self.state)
        self.state.emit_data_loaded("ids")
