"""Algorithm plugin system for running pyOpenMS algorithms on loaded data.

Provides a registry of pyOpenMS algorithms (feature finders, peak pickers) with
auto-generated parameter UIs and background execution.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
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
        from pyopenms import ElutionPeakDetection, FeatureFindingMetabo, MassTraceDetection, Param

        registry.append(
            AlgorithmDescriptor(
                name="Mass Trace Detection",
                category="Feature Finder",
                description="Detect mass traces in LC-MS data (first step of metabolomics feature finding)",
                get_params_fn=lambda: MassTraceDetection().getParameters(),
                run_fn=_run_mass_trace_detection,
            )
        )

        def _get_ff_metabo_params():
            combined = Param()
            mtd_params = MassTraceDetection().getParameters()
            epd_params = ElutionPeakDetection().getParameters()
            ffm_params = FeatureFindingMetabo().getParameters()
            combined.insert("mtd:", mtd_params)
            combined.insert("epd:", epd_params)
            combined.insert("ffm:", ffm_params)
            combined.setSectionDescription("mtd", "Mass Trace Detection")
            combined.setSectionDescription("epd", "Elution Peak Detection")
            combined.setSectionDescription("ffm", "Feature Finding Metabo")
            return combined

        registry.append(
            AlgorithmDescriptor(
                name="FeatureFinder Metabo",
                category="Feature Finder",
                description="Metabolomics feature detection (mass traces + elution peaks + feature finding)",
                get_params_fn=_get_ff_metabo_params,
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
        from pyopenms import PeakPickerIM  # noqa: F401

        def _get_ppim_params():
            from pyopenms import Param

            p = Param()
            p.setValue("method", "Cluster", "Peak picking method to use")
            p.setValidStrings("method", [b"Cluster", b"ElutionProfiles", b"Traces"])
            return p

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
    from pyopenms import FeatureFinderMultiplexAlgorithm, MSExperiment

    algo = FeatureFinderMultiplexAlgorithm()
    algo.setParameters(params)
    exp_copy = MSExperiment(state.exp)
    algo.run(exp_copy, True)
    return algo.getFeatureMap()


def _run_mass_trace_detection(state: ViewerState, params: Any) -> Any:
    from pyopenms import Feature, FeatureMap, MassTraceDetection

    mtd = MassTraceDetection()
    mtd.setParameters(params)
    mass_traces = []
    mtd.run(state.exp, mass_traces, 0)

    fm = FeatureMap()
    for trace in mass_traces:
        f = Feature()
        f.setRT(trace.getCentroidRT())
        f.setMZ(trace.getCentroidMZ())
        f.setIntensity(trace.computePeakArea())
        f.getConvexHulls().append(trace.getConvexhull())
        fm.push_back(f)
    return fm


def _run_ff_metabo(state: ViewerState, params: Any) -> Any:
    from pyopenms import ElutionPeakDetection, FeatureFindingMetabo, FeatureMap, MassTraceDetection

    mtd_params = params.copy("mtd:", True)
    epd_params = params.copy("epd:", True)
    ffm_params = params.copy("ffm:", True)

    mtd = MassTraceDetection()
    mtd.setParameters(mtd_params)
    mass_traces = []
    mtd.run(state.exp, mass_traces, 0)

    epd = ElutionPeakDetection()
    epd.setParameters(epd_params)
    mass_traces_split = []
    epd.detectPeaks(mass_traces, mass_traces_split)

    ffm = FeatureFindingMetabo()
    ffm.setParameters(ffm_params)
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
    from pyopenms import MSExperiment, PeakPickerIM

    method = str(params.getValue("method")) if params.size() > 0 else "Cluster"
    pp = PeakPickerIM()
    pick_fn = {"Cluster": pp.pickIMCluster, "ElutionProfiles": pp.pickIMElutionProfiles, "Traces": pp.pickIMTraces}[
        method
    ]
    output = MSExperiment()
    for i in range(state.exp.getNrSpectra()):
        spec = state.exp.getSpectrum(i)
        try:
            pick_fn(spec)
        except Exception:
            pass
        output.addSpectrum(spec)
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

    # Pre-compute which section prefixes have visible params (to avoid empty headers)
    visible_sections: set[str] = set()
    for key in param_dict:
        try:
            tags = param.getTags(key)
        except Exception:
            tags = []
        if not show_advanced and b"advanced" in tags:
            continue
        # Section prefix is the first colon-delimited part (e.g. "mtd" from "mtd:param_name")
        if ":" in key:
            visible_sections.add(key.split(":")[0])

    current_section = None

    with container:
        for key, value in param_dict.items():
            # Tags are bytes (e.g. b"advanced")
            try:
                tags = param.getTags(key)
            except Exception:
                tags = []

            if not show_advanced and b"advanced" in tags:
                continue

            # Emit section header when prefix changes
            section = key.split(":")[0] if ":" in key else ""
            if section and section in visible_sections and section != current_section:
                current_section = section
                try:
                    section_desc = param.getSectionDescription(section)
                except Exception:
                    section_desc = section
                if current_section is not None:
                    ui.separator().classes("my-1")
                ui.label(section_desc or section).classes("text-sm font-bold text-gray-600 mt-1 mb-0.5")

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
                # Use pyOpenMS list types to preserve the correct type
                type_name = type(original).__name__
                try:
                    if "Int" in type_name:
                        from pyopenms import IntList

                        param.setValue(key, IntList([int(p) for p in parts]))
                    elif "Double" in type_name or "Float" in type_name:
                        from pyopenms import DoubleList

                        param.setValue(key, DoubleList([float(p) for p in parts]))
                    else:
                        from pyopenms import StringList

                        param.setValue(key, StringList(parts))
                except (ValueError, TypeError):
                    param.setValue(key, parts)
        except Exception:
            pass


# ========== C++ stdout/stderr Capture ==========

# Process-wide lock: fd redirection affects fds 1/2 globally, so only one
# capture can be active at a time.
_capture_lock = threading.Lock()


def _capture_fd_output(func, *args):
    """Run *func(*args)* while capturing C-level stdout and stderr.

    pyOpenMS algorithms write diagnostics at the C++ level which bypasses
    Python's sys.stdout. We redirect file descriptors 1 and 2 into a pipe,
    run the function, then restore the original fds.

    A background reader thread drains the pipe to prevent deadlock when the
    output exceeds the OS pipe buffer (~64 KB).

    Returns:
        (result, captured_text) tuple. If func raises, the exception is
        re-raised wrapped in CapturedOutputError so the caller can access
        both the error and any diagnostic output.
    """
    with _capture_lock:
        read_fd, write_fd = os.pipe()

        # Save original fds
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)

        captured_chunks: list[bytes] = []

        def _reader():
            """Drain the read end of the pipe in a background thread."""
            try:
                while True:
                    chunk = os.read(read_fd, 4096)
                    if not chunk:
                        break
                    captured_chunks.append(chunk)
            except OSError:
                pass  # read_fd closed externally after join timeout

        reader_thread = threading.Thread(target=_reader, daemon=True)
        exc_to_raise = None
        result = None

        try:
            # Redirect stdout and stderr to the write end of the pipe
            os.dup2(write_fd, 1)
            os.dup2(write_fd, 2)
            reader_thread.start()

            result = func(*args)
        except Exception as e:
            exc_to_raise = e
        finally:
            # Restore original fds
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)

            # Close write end so the reader sees EOF
            os.close(write_fd)
            if reader_thread.is_alive():
                reader_thread.join(timeout=5)
            os.close(read_fd)

        captured_text = b"".join(captured_chunks).decode("utf-8", errors="replace")

        if exc_to_raise is not None:
            raise CapturedOutputError(captured_text) from exc_to_raise

    return result, captured_text


class CapturedOutputError(Exception):
    """Wraps an algorithm exception while preserving captured fd output."""

    def __init__(self, captured_output: str):
        self.captured_output = captured_output
        super().__init__()


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
    if feature_info_label and algo.category == "Feature Finder":
        feature_info_label.set_text(f"Running {algo.name}...")
    await asyncio.sleep(0)  # Let UI update before heavy computation

    try:
        result, captured_output = await run.io_bound(_capture_fd_output, algo.run_fn, state, params)
    except CapturedOutputError as coe:
        # Algorithm failed but we captured its output — emit before reporting error
        if coe.captured_output and coe.captured_output.strip():
            state.emit_algorithm_log(algo.name, coe.captured_output.strip())
        original = coe.__cause__
        ui.notify(f"{algo.name} failed: {original}", type="negative")
        import traceback

        traceback.print_exc()
        return
    except Exception as e:
        ui.notify(f"{algo.name} failed: {e}", type="negative")
        import traceback

        traceback.print_exc()
        return

    # Emit captured C++ output to the log panel
    if captured_output and captured_output.strip():
        state.emit_algorithm_log(algo.name, captured_output.strip())

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
                dialog.submit({"algo": current_algo[0], "params": current_params[0]})

            ui.button("Reset Defaults", on_click=reset_defaults).props("flat")
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
            ui.button("Run", on_click=on_run).props("color=primary")

    # await dialog result — dialog is fully closed before we continue
    result = await dialog
    if result is None:
        return

    # Fire-and-forget: capture client context so the independent task can
    # update the UI, but return immediately so the click handler completes
    # and doesn't block the WebSocket connection.
    from nicegui import context

    client = context.client
    algo = result["algo"]
    params = result["params"]

    async def _run_with_context():
        with client:
            await _execute_algorithm(state, algo, params, feature_info_label)

    asyncio.create_task(_run_with_context())
