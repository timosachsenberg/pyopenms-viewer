"""Algorithm plugin system for running pyOpenMS algorithms on loaded data.

Provides a registry of pyOpenMS algorithms (feature finders, peak pickers) with
auto-generated parameter UIs and background execution.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from nicegui import run, ui

if TYPE_CHECKING:
    from pyopenms_viewer.core.state import ViewerState


def _decode(val: Any) -> str:
    """Decode bytes to str, pass through str unchanged."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


@dataclass
class AlgorithmDescriptor:
    """Describes a runnable algorithm."""

    name: str
    category: str  # "Feature Finder" or "Peak Picker"
    description: str
    get_params_fn: Callable[[], Any]
    run_fn: Callable


# ========== Algorithm Registry ==========


def _build_registry() -> list[AlgorithmDescriptor]:
    """Build list of available algorithms, skipping unavailable ones."""
    registry: list[AlgorithmDescriptor] = []

    try:
        from pyopenms import Biosaur2Algorithm

        registry.append(
            AlgorithmDescriptor(
                name="Biosaur2",
                category="Feature Finder",
                description="Feature detection using the Biosaur2 algorithm for LC-MS data",
                get_params_fn=lambda: Biosaur2Algorithm().getParameters(),
                run_fn=_run_biosaur2,
            )
        )
    except ImportError:
        pass

    try:
        from pyopenms import FeatureFinderMultiplexAlgorithm

        registry.append(
            AlgorithmDescriptor(
                name="FeatureFinder Multiplex",
                category="Feature Finder",
                description="Detect peptide features with multiplex labeling support",
                get_params_fn=lambda: FeatureFinderMultiplexAlgorithm().getDefaults(),
                run_fn=_run_ff_multiplex,
            )
        )
    except ImportError:
        pass

    try:
        from pyopenms import FeatureFinderMetabo

        registry.append(
            AlgorithmDescriptor(
                name="FeatureFinder Metabo",
                category="Feature Finder",
                description="Metabolomics feature detection (mass traces + elution peaks + feature finding)",
                get_params_fn=lambda: FeatureFinderMetabo().getParameters(),
                run_fn=_run_ff_metabo,
            )
        )
    except ImportError:
        pass

    try:
        from pyopenms import PeakPickerHiRes

        registry.append(
            AlgorithmDescriptor(
                name="PeakPicker HiRes",
                category="Peak Picker",
                description="High-resolution peak picking using wavelet-based approach",
                get_params_fn=lambda: PeakPickerHiRes().getParameters(),
                run_fn=_run_peakpicker_hires,
            )
        )
    except ImportError:
        pass

    try:
        from pyopenms import PeakPickerIM

        def _get_ppim_params():
            try:
                return PeakPickerIM().getParameters()
            except Exception:
                from pyopenms import Param

                return Param()

        registry.append(
            AlgorithmDescriptor(
                name="PeakPicker IM",
                category="Peak Picker",
                description="Peak picking optimized for ion mobility data",
                get_params_fn=_get_ppim_params,
                run_fn=_run_peakpicker_im,
            )
        )
    except ImportError:
        pass

    return registry


# ========== Algorithm Run Functions ==========


def _run_biosaur2(state: ViewerState, params: Any) -> Any:
    from pyopenms import Biosaur2Algorithm, FeatureMap

    algo = Biosaur2Algorithm()
    algo.setParameters(params)
    algo.setMSData(state.exp)
    fm = FeatureMap()
    algo.run(fm)
    return fm


def _run_ff_multiplex(state: ViewerState, params: Any) -> Any:
    from pyopenms import FeatureFinderMultiplexAlgorithm

    algo = FeatureFinderMultiplexAlgorithm()
    algo.setParameters(params)
    algo.run(state.exp, True)
    return algo.getFeatureMap()


def _run_ff_metabo(state: ViewerState, params: Any) -> Any:
    from pyopenms import ElutionPeakDetection, FeatureFinderMetabo, FeatureMap, MassTraceDetection

    mtd = MassTraceDetection()
    mass_traces = []
    mtd.run(state.exp, mass_traces, 0)

    epd = ElutionPeakDetection()
    mass_traces_split = []
    epd.detectPeaks(mass_traces, mass_traces_split)

    ffm = FeatureFinderMetabo()
    ffm.setParameters(params)
    fm = FeatureMap()
    chromatograms = []
    ffm.run(mass_traces_split, fm, chromatograms)
    return fm


def _run_peakpicker_hires(state: ViewerState, params: Any) -> Any:
    from pyopenms import MSExperiment, PeakPickerHiRes

    pp = PeakPickerHiRes()
    pp.setParameters(params)
    output = MSExperiment()
    pp.pickExperiment(state.exp, output, True)
    return output


def _run_peakpicker_im(state: ViewerState, params: Any) -> Any:
    from pyopenms import MSExperiment, MSSpectrum, PeakPickerIM

    pp = PeakPickerIM()
    try:
        pp.setParameters(params)
    except Exception:
        pass
    output = MSExperiment()
    for i in range(len(state.exp)):
        picked = MSSpectrum()
        try:
            pp.pick(state.exp[i], picked)
        except Exception:
            picked = state.exp[i]
        output.addSpectrum(picked)
    return output


# ========== Parameter UI Helpers ==========


def _get_param_entry_info(param, key, value):
    """Extract bounds and valid strings from a Param entry."""
    min_val = None
    max_val = None
    valid_strings: list[str] = []
    try:
        entry = param.getEntry(key)
        if isinstance(value, float):
            if entry.min_float is not None and math.isfinite(entry.min_float) and entry.min_float > -1e200:
                min_val = entry.min_float
            if entry.max_float is not None and math.isfinite(entry.max_float) and entry.max_float < 1e200:
                max_val = entry.max_float
        elif isinstance(value, int):
            if entry.min_int is not None and entry.min_int > -(2**31):
                min_val = entry.min_int
            if entry.max_int is not None and entry.max_int < 2**31 - 1:
                max_val = entry.max_int
        if entry.valid_strings:
            valid_strings = [_decode(s) for s in entry.valid_strings if s]
    except Exception:
        pass
    return min_val, max_val, valid_strings


def _build_param_widgets(param, container, show_advanced: bool) -> dict:
    """Build NiceGUI widgets for Param entries inside container.

    Uses param.to_dict() which returns str keys with proper Python values.
    Returns dict mapping str key -> widget.
    """
    widgets = {}
    container.clear()

    try:
        param_dict = param.to_dict()
    except Exception:
        param_dict = {}

    if not param_dict:
        with container:
            ui.label("No configurable parameters").classes("text-xs text-gray-400 p-2")
        return widgets

    with container:
        for key, value in param_dict.items():
            # Tags are bytes (e.g. b"advanced")
            try:
                tags = param.getTags(key)
            except Exception:
                tags = []

            if not show_advanced and b"advanced" in tags:
                continue

            try:
                desc = param.getDescription(key)
            except Exception:
                desc = ""

            display_name = key.rsplit(":", 1)[-1] if ":" in key else key
            min_val, max_val, valid_strings = _get_param_entry_info(param, key, value)

            with ui.row().classes("w-full items-center gap-2 py-0.5"):
                lbl = ui.label(display_name).classes("w-48 text-xs truncate")
                if desc:
                    lbl.tooltip(desc)

                if isinstance(value, float):
                    kwargs: dict[str, Any] = {"value": value, "step": 0.001}
                    if min_val is not None:
                        kwargs["min"] = min_val
                    if max_val is not None:
                        kwargs["max"] = max_val
                    w = ui.number(**kwargs).classes("flex-grow").props("dense")
                    widgets[key] = w

                elif isinstance(value, int):
                    kwargs = {"value": value, "step": 1}
                    if min_val is not None:
                        kwargs["min"] = min_val
                    if max_val is not None:
                        kwargs["max"] = max_val
                    w = ui.number(**kwargs).classes("flex-grow").props("dense")
                    widgets[key] = w

                elif isinstance(value, str):
                    if valid_strings and value in valid_strings:
                        w = ui.select(options=valid_strings, value=value).classes("flex-grow").props("dense")
                    else:
                        w = ui.input(value=value).classes("flex-grow").props("dense")
                    widgets[key] = w

                elif isinstance(value, (list, tuple)):
                    csv_val = ", ".join(str(v) for v in value)
                    w = ui.input(value=csv_val).classes("flex-grow").props("dense")
                    widgets[key] = w

                else:
                    w = ui.input(value=str(value)).classes("flex-grow").props("dense")
                    widgets[key] = w

    return widgets


def _apply_params_from_widgets(param, widgets: dict) -> None:
    """Copy widget values back into Param object using str keys."""
    for key, widget in widgets.items():
        try:
            original = param.getValue(key)
            new_val = widget.value
            if new_val is None:
                continue

            if isinstance(original, float):
                param.setValue(key, float(new_val))
            elif isinstance(original, int):
                param.setValue(key, int(new_val))
            elif isinstance(original, (str, bytes)):
                param.setValue(key, str(new_val))
            elif isinstance(original, (list, tuple)):
                parts = [p.strip() for p in str(new_val).split(",") if p.strip()]
                param.setValue(key, parts)
        except Exception:
            pass


# ========== Result Handling ==========


async def _execute_algorithm(
    state: ViewerState,
    algo: AlgorithmDescriptor,
    params: Any,
    feature_info_label=None,
) -> None:
    """Execute algorithm in background and handle results."""
    if state.exp is None:
        ui.notify("No data loaded. Load an mzML file first.", type="warning")
        return

    ui.notify(f"Running {algo.name}...", type="info")

    try:
        result = await run.io_bound(algo.run_fn, state, params)
    except Exception as e:
        ui.notify(f"{algo.name} failed: {e}", type="negative")
        import traceback

        traceback.print_exc()
        return

    if result is None:
        ui.notify(f"{algo.name} returned no result", type="warning")
        return

    try:
        if algo.category == "Feature Finder":
            state.feature_map = result
            from pyopenms_viewer.loaders import extract_feature_data

            state.feature_data = extract_feature_data(state)
            state.emit_data_loaded("features")
            count = result.size()
            ui.notify(f"Found {count:,} features", type="positive")
            if feature_info_label:
                feature_info_label.set_text(f"Features: {count:,}")

        elif algo.category == "Peak Picker":
            state.exp = result
            from pyopenms_viewer.loaders import MzMLLoader

            loader = MzMLLoader(state)
            filepath = state.current_file or ""
            success = await run.io_bound(loader.process, filepath)
            if success:
                state.emit_data_loaded("mzml")
                ui.notify(f"Peak picking complete ({len(result):,} spectra)", type="positive")
            else:
                ui.notify("Peak picking succeeded but reprocessing failed", type="warning")
    except Exception as e:
        ui.notify(f"Error processing {algo.name} results: {e}", type="negative")
        import traceback

        traceback.print_exc()


# ========== Main Dialog ==========


async def show_algorithm_dialog(state: ViewerState, feature_info_label=None) -> None:
    """Show the algorithm selection and parameter configuration dialog."""
    registry = _build_registry()
    if not registry:
        ui.notify("No pyOpenMS algorithms available", type="warning")
        return

    current_algo = [registry[0]]
    current_params = [registry[0].get_params_fn()]
    param_widgets: list[dict] = [{}]

    with ui.dialog() as dialog, ui.card().classes("min-w-[500px] max-w-[700px]"):
        ui.label("Run Algorithm").classes("text-lg font-bold mb-2")

        algo_select = ui.select(
            options=[a.name for a in registry],
            value=registry[0].name,
        ).classes("w-full")

        desc_label = ui.label(registry[0].description).classes("text-xs text-gray-400 mb-2")

        show_adv = ui.checkbox("Show advanced parameters", value=False)

        param_container = ui.scroll_area().classes("w-full border rounded").style("max-height: 400px;")

        def _rebuild_params():
            param_widgets[0] = _build_param_widgets(current_params[0], param_container, show_adv.value)

        def on_algo_change(e):
            for a in registry:
                if a.name == e.value:
                    current_algo[0] = a
                    current_params[0] = a.get_params_fn()
                    desc_label.set_text(a.description)
                    _rebuild_params()
                    break

        algo_select.on_value_change(on_algo_change)
        show_adv.on_value_change(lambda _: _rebuild_params())

        _rebuild_params()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):

            def reset_defaults():
                current_params[0] = current_algo[0].get_params_fn()
                _rebuild_params()

            def on_run():
                _apply_params_from_widgets(current_params[0], param_widgets[0])
                algo = current_algo[0]
                params = current_params[0]
                dialog.close()
                # Fire-and-forget: decouple from dialog lifecycle so dialog.close()
                # does not cancel the long-running algorithm execution.
                asyncio.create_task(_execute_algorithm(state, algo, params, feature_info_label))

            ui.button("Reset Defaults", on_click=reset_defaults).props("flat")
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Run", on_click=on_run).props("color=primary")

    dialog.open()
