"""Main application module for pyopenms-viewer.

This module creates the NiceGUI interface and wires all components together
using the modular panel architecture.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from nicegui import app, run, ui

from pyopenms_viewer.components.local_file_picker import LocalFilePicker
from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders import FeatureLoader, IDLoader, MzMLLoader
from pyopenms_viewer.panels import (
    ChromatogramPanel,
    ExportPanel,
    FAIMSPanel,
    FeaturesTablePanel,
    IMPeakMapPanel,
    LogPanel,
    PanelManager,
    PeakMapPanel,
    SpectraTablePanel,
    SpectrumPanel,
    TICPanel,
)

# Map of filepath -> asyncio.Event used to signal completion of an in-progress load
_LOAD_EVENTS: dict[str, asyncio.Event] = {}
_LOAD_EVENTS_LOCK = asyncio.Lock()


async def create_ui():
    """Create the main NiceGUI interface.

    This is called by NiceGUI as the root page handler.
    """
    # Get CLI options for initialization
    from pyopenms_viewer.cli import get_cli_options

    cli_options = get_cli_options()

    # Helper: safe UI interactions that may fail if client disconnected
    def safe_notify(message: str, type: str = "info") -> None:
        try:
            ui.notify(message, type=type)
        except Exception:
            # Client may have disconnected; ignore UI errors
            pass

    def safe_set_label(label, text: str) -> None:
        try:
            if label:
                label.set_text(text)
        except Exception:
            pass

    # Create or reuse a shared state (persist across page reloads to avoid
    # reloading large mzML/DFs when the frontend reconnects or times out).
    # We keep a module-level singleton so repeated NiceGUI page handler
    # invocations reuse the already-loaded data.
    global _GLOBAL_VIEWER_STATE

    try:
        _GLOBAL_VIEWER_STATE  # noqa: B018
    except NameError:
        _GLOBAL_VIEWER_STATE = None

    if _GLOBAL_VIEWER_STATE is None:
        state = ViewerState()
        _GLOBAL_VIEWER_STATE = state
    else:
        state = _GLOBAL_VIEWER_STATE

    # Initialize or reconfigure data manager with CLI options (safe to call repeatedly)
    cache_dir = Path(cli_options["cache_dir"]) if cli_options["cache_dir"] else None
    state.init_data_manager(out_of_core=cli_options["out_of_core"], cache_dir=cache_dir)

    # Setup dark mode
    dark_mode = os.environ.get("PYOPENMS_VIEWER_DARK_MODE", "1") == "1"
    dark = ui.dark_mode()
    if dark_mode:
        dark.enable()
    else:
        dark.disable()

    # Store reference in state for potential toggling
    state.dark = dark

    # Top-right corner buttons (dark mode toggle + fullscreen)
    with ui.element("div").classes("fixed top-2 right-2 z-50 flex gap-1"):

        def toggle_dark_mode():
            dark.toggle()
            dark_btn.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")
            dark_btn._props["icon"] = "light_mode" if dark.value else "dark_mode"
            dark_btn.update()

        dark_btn = (
            ui.button(icon="light_mode", on_click=toggle_dark_mode)
            .props("flat round dense color=grey")
            .tooltip("Toggle dark/light mode")
        )
        ui.button(
            icon="fullscreen",
            on_click=lambda: ui.run_javascript("""
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        """),
        ).props("flat round dense color=grey").tooltip("Toggle fullscreen (F11)")

    with ui.column().classes("w-full items-center p-2"):
        # Compact toolbar for file operations and info display
        with (
            ui.row()
            .classes("w-full max-w-[1700px] items-center gap-2 px-2 py-1 rounded")
            .style("background: rgba(128,128,128,0.1);")
        ):
            # Setup helper functions for file loading

            def _relink_ids_and_update_label():
                """Re-link peptide IDs to spectra and update the ID info label."""
                if state.peptide_ids:
                    from pyopenms_viewer.loaders import link_ids_to_spectra

                    link_ids_to_spectra(state)
                    n_linked = sum(1 for s in state.spectrum_data if s.get("id_idx") is not None)
                    safe_set_label(id_info_label, f"IDs: {len(state.peptide_ids):,} ({n_linked} linked)")

            def _update_info_label(name: str) -> int:
                """Build info text from state, update the info label, and return peak count."""
                if state.data_manager is not None:
                    peak_count = state.data_manager.get_peak_count()
                elif state.df is not None:
                    peak_count = len(state.df)
                else:
                    peak_count = 0
                info_text = f"Loaded: {name} | Spectra: {len(state.exp):,} | Peaks: {peak_count:,}"
                if state.has_faims:
                    info_text += f" | FAIMS: {len(state.faims_cvs)} CVs"
                if state.out_of_core:
                    info_text += " | Out-of-core"
                safe_set_label(info_label, info_text)
                return peak_count

            async def load_mzml(filepath: str, original_name: str = None):
                """Load mzML file in background.

                If the same file is already loaded into `state` we skip reloading
                to avoid expensive repeated parses when the UI reconnects or
                times out. If another coroutine is already loading the same
                filepath, wait for it to complete and then reuse the loaded
                data.
                """
                loader = MzMLLoader(state)
                name = original_name or Path(filepath).name

                # Normalize paths for comparison
                try:
                    new_fp = str(Path(filepath).resolve())
                except Exception:
                    new_fp = str(filepath)

                cur_fp = None
                if state.current_file:
                    try:
                        cur_fp = str(Path(state.current_file).resolve())
                    except Exception:
                        cur_fp = str(state.current_file)

                # If same file already loaded and data present, skip reload
                if cur_fp is not None and cur_fp == new_fp and (state.df is not None or state.data_manager is not None):
                    _relink_ids_and_update_label()
                    state.emit_data_loaded("mzml")
                    _update_info_label(name)
                    safe_notify(f"File already loaded: {name}", type="info")
                    return

                # Atomically check/create an Event for this filepath so only one
                # coroutine will perform the actual load. Others will wait on the
                # same event.
                async with _LOAD_EVENTS_LOCK:
                    existing_event = _LOAD_EVENTS.get(new_fp)
                    if existing_event is None:
                        event = asyncio.Event()
                        _LOAD_EVENTS[new_fp] = event
                        is_loader = True
                    else:
                        event = existing_event
                        is_loader = False

                if not is_loader:
                    # Wait for the original loader to finish
                    await event.wait()
                    # Recompute cur_fp — state.current_file was updated by the loader
                    if state.current_file:
                        try:
                            cur_fp = str(Path(state.current_file).resolve())
                        except Exception:
                            cur_fp = str(state.current_file)
                    # After wait, data should be available (or failed). Reuse if available.
                    if state.current_file and cur_fp == new_fp and (state.df is not None or state.data_manager is not None):
                        _relink_ids_and_update_label()
                        state.emit_data_loaded("mzml")
                        _update_info_label(name)
                        safe_notify(f"File already loaded: {name}", type="info")
                        return

                # Mark file as loading in state to prevent other entrypoints
                try:
                    state._loading_files.add(new_fp)
                except Exception:
                    pass

                # Progress callback updates shared state.load_progress so the UI
                # can poll it and avoid pushing messages to disconnected clients.
                def _progress_cb(message: str, progress: float) -> None:
                    try:
                        state.load_progress[new_fp] = (message, float(progress))
                    except Exception:
                        pass

                safe_notify(f"Loading {name}...", type="info")
                try:
                    success = await run.io_bound(loader.load_sync, filepath, _progress_cb)
                finally:
                    # Signal any waiters that loading is finished
                    ev = _LOAD_EVENTS.pop(new_fp, None)
                    if ev is not None:
                        ev.set()
                    try:
                        state._loading_files.discard(new_fp)
                    except Exception:
                        pass
                    try:
                        state.load_progress.pop(new_fp, None)
                    except Exception:
                        pass
                if success:
                    _relink_ids_and_update_label()
                    progress_label.set_text("Rendering...")
                    await asyncio.sleep(0)  # Let UI update before heavy renders
                    state.emit_data_loaded("mzml")
                    progress_label.set_text("")
                    peak_count = _update_info_label(name)
                    safe_notify(f"Loaded {peak_count:,} peaks from {name}", type="positive")
                else:
                    safe_notify(f"Failed to load {name}", type="negative")

            # Store loaders in state for use by panels
            state._load_mzml = load_mzml

            async def handle_upload(e):
                """Handle uploaded file - detect type and load appropriately.

                NiceGUI 3.x: UploadEventArguments has .file attribute (FileUpload object)
                FileUpload has: .name, .content_type, async .read(), async .save(path)
                """
                file = e.file
                filename = file.name.lower()
                original_name = file.name

                # Save to temp file using FileUpload.save()
                suffix = Path(filename).suffix
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                await file.save(tmp_path)

                try:
                    if filename.endswith(".mzml"):
                        await load_mzml(tmp_path, original_name)

                    elif filename.endswith(".featurexml") or (filename.endswith(".xml") and "feature" in filename):
                        loader = FeatureLoader(state)
                        success = await run.io_bound(loader.load_sync, tmp_path)
                        if success:
                            state.emit_data_loaded("features")
                            if feature_info_label:
                                feature_info_label.set_text(f"Features: {state.feature_map.size():,}")
                            ui.notify(
                                f"Loaded {state.feature_map.size():,} features from {original_name}", type="positive"
                            )

                    elif filename.endswith(".idxml") or (filename.endswith(".xml") and "id" in filename):
                        loader = IDLoader(state)
                        success = await run.io_bound(loader.load_sync, tmp_path)
                        if success:
                            state.emit_data_loaded("ids")
                            n_linked = sum(1 for s in state.spectrum_data if s.get("id_idx") is not None)
                            if id_info_label:
                                id_info_label.set_text(f"IDs: {len(state.peptide_ids):,} ({n_linked} linked)")
                            ui.notify(
                                f"Loaded {len(state.peptide_ids):,} IDs ({n_linked} linked) from {original_name}",
                                type="positive",
                            )

                    else:
                        ui.notify(
                            f"Unknown file type: {original_name}. Supported: .mzML, .featureXML, .idXML", type="warning"
                        )

                except Exception as ex:
                    ui.notify(f"Error loading {original_name}: {ex}", type="negative")

                finally:
                    # Clean up temp file
                    try:
                        Path(tmp_path).unlink()
                    except Exception:
                        pass

            async def open_native_file_dialog():
                """Open native file dialog to select files directly from filesystem."""
                if not app.native.main_window:
                    ui.notify("Native file dialog is only available in native mode (--native)", type="warning")
                    return

                try:
                    files = await app.native.main_window.create_file_dialog(
                        allow_multiple=True,
                        file_types=(
                            "Mass Spec Files (*.mzML;*.featureXML;*.idXML)",
                            "mzML Files (*.mzML)",
                            "Feature Files (*.featureXML)",
                            "ID Files (*.idXML)",
                            "All Files (*.*)",
                        ),
                    )

                    if not files:
                        return

                    for filepath in files:
                        ext = Path(filepath).suffix.lower()
                        name = Path(filepath).name

                        try:
                            if ext == ".mzml":
                                await load_mzml(filepath, name)
                            elif ext == ".featurexml":
                                loader = FeatureLoader(state)
                                success = await run.io_bound(loader.load_sync, filepath)
                                if success:
                                    state.emit_data_loaded("features")
                                    ui.notify(
                                        f"Loaded {state.feature_map.size():,} features from {name}", type="positive"
                                    )
                            elif ext == ".idxml":
                                loader = IDLoader(state)
                                success = await run.io_bound(loader.load_sync, filepath)
                                if success:
                                    state.emit_data_loaded("ids")
                                    ui.notify(f"Loaded IDs from {name}", type="positive")
                            else:
                                ui.notify(f"Unknown file type: {name}", type="warning")
                        except Exception as ex:
                            ui.notify(f"Error loading {name}: {ex}", type="negative")

                except Exception as ex:
                    ui.notify(f"File dialog error: {ex}", type="negative")

            async def open_local_file_picker():
                """Open local file picker dialog to browse server filesystem."""
                picker = LocalFilePicker(directory=str(Path.cwd()), upper_limit=None, multiple=True)
                picker.open()
                result = await picker
                if not result:
                    return

                for filepath in result:
                    ext = Path(filepath).suffix.lower()
                    name = Path(filepath).name

                    try:
                        if ext == ".mzml":
                            await load_mzml(filepath, name)
                        elif ext == ".featurexml":
                            loader = FeatureLoader(state)
                            success = await run.io_bound(loader.load_sync, filepath)
                            if success:
                                state.emit_data_loaded("features")
                                ui.notify(f"Loaded {state.feature_map.size():,} features from {name}", type="positive")
                        elif ext == ".idxml":
                            loader = IDLoader(state)
                            success = await run.io_bound(loader.load_sync, filepath)
                            if success:
                                state.emit_data_loaded("ids")
                                ui.notify(f"Loaded IDs from {name}", type="positive")
                        else:
                            ui.notify(f"Unknown file type: {name}", type="warning")
                    except Exception as ex:
                        ui.notify(f"Error loading {name}: {ex}", type="negative")

            async def open_files():
                """Open files using native dialog (native mode) or local file picker (web mode)."""
                if app.native.main_window:
                    await open_native_file_dialog()
                else:
                    await open_local_file_picker()

            # Open button - uses native dialog in native mode, local picker in web mode
            ui.button(
                icon="folder_open",
                on_click=open_files,
            ).props("flat dense").tooltip("Open files (fast, direct access)")

            # Collapsible drop zone for drag-and-drop uploads
            upload_container = ui.row().classes("items-center")
            upload_container.set_visibility(False)

            def toggle_upload():
                upload_container.set_visibility(not upload_container.visible)

            ui.button(
                icon="upload",
                on_click=toggle_upload,
            ).props("flat dense").tooltip("Toggle drag & drop zone (slower, uploads file)")

            with upload_container:
                ui.upload(on_upload=handle_upload, auto_upload=True, multiple=True).props(
                    'hide-upload-btn no-thumbnails accept=".mzML,.mzml,.featureXML,.featurexml,.idXML,.idxml,.xml"'
                ).classes("w-full border rounded")

            ui.separator().props("vertical").classes("h-6")

            # Clear menu (dropdown)
            def clear_features():
                state.clear_features()
                state.emit_data_loaded("features")  # Trigger UI updates
                state.update_panel_visibility()
                state.emit_view_changed()
                if feature_info_label:
                    feature_info_label.set_text("")

            def clear_ids():
                state.clear_ids()
                state.emit_data_loaded("ids")  # Trigger UI updates (spectra table, etc.)
                state.update_panel_visibility()
                state.emit_view_changed()
                if id_info_label:
                    id_info_label.set_text("")

            def clear_mzml():
                state.clear_mzml_data()
                state.emit_data_loaded("mzml")  # Trigger UI updates
                state.update_panel_visibility()
                state.emit_view_changed()
                if info_label:
                    info_label.set_text("No file loaded")

            def clear_all():
                state.clear_all()
                if info_label:
                    info_label.set_text("No file loaded")
                if feature_info_label:
                    feature_info_label.set_text("")
                if id_info_label:
                    id_info_label.set_text("")
                # Emit events to update all panels
                state.emit_data_loaded("mzml")
                state.emit_data_loaded("features")
                state.emit_data_loaded("ids")
                state.update_panel_visibility()
                state.emit_view_changed()

            with ui.button(icon="delete_outline").props("flat dense size=sm").tooltip("Clear data"):
                with ui.menu().props("auto-close"):
                    ui.menu_item("Clear mzML", on_click=clear_mzml).classes("text-blue-400")
                    ui.menu_item("Clear Features", on_click=clear_features).classes("text-cyan-400")
                    ui.menu_item("Clear IDs", on_click=clear_ids).classes("text-orange-400")
                    ui.separator()
                    ui.menu_item("Clear All", on_click=clear_all).classes("text-red-400")

            ui.separator().props("vertical").classes("h-6")

            # Settings button for panel order
            def show_panel_settings():
                with ui.dialog() as dialog, ui.card().classes("min-w-[400px]"):
                    ui.label("Panel Configuration").classes("text-lg font-bold mb-2")

                    # Panel visibility section
                    ui.label("Visibility").classes("text-sm font-semibold text-gray-400 mt-2")
                    ui.label("Toggle panels on/off. 'Auto' hides when no data.").classes("text-xs text-gray-500 mb-2")

                    visibility_container = ui.column().classes("w-full gap-1 mb-4")

                    # Panels that support "auto" visibility
                    auto_panels = {"chromatograms", "im_peakmap", "features_table", "log"}

                    def refresh_visibility():
                        visibility_container.clear()
                        with visibility_container:
                            for panel_id in state.panel_order:
                                panel_def = state.panel_definitions.get(panel_id, {})
                                current_vis = state.panel_visibility.get(panel_id, True)

                                with ui.row().classes("w-full items-center gap-2"):
                                    ui.icon(panel_def.get("icon", "widgets")).classes("text-gray-400 text-sm")
                                    ui.label(panel_def.get("name", panel_id)).classes("flex-grow text-sm")

                                    if panel_id in auto_panels:
                                        options = ["Hide", "Auto", "Show"]
                                        if current_vis is False:
                                            current_val = "Hide"
                                        elif current_vis == "auto":
                                            current_val = "Auto"
                                        else:
                                            current_val = "Show"

                                        def make_toggle_handler(pid=panel_id):
                                            def handler(e):
                                                if e.value == "Hide":
                                                    state.panel_visibility[pid] = False
                                                elif e.value == "Auto":
                                                    state.panel_visibility[pid] = "auto"
                                                else:
                                                    state.panel_visibility[pid] = True

                                            return handler

                                        ui.toggle(options, value=current_val, on_change=make_toggle_handler()).props(
                                            "dense size=sm"
                                        ).classes("text-xs")
                                    else:

                                        def make_checkbox_handler(pid=panel_id):
                                            def handler(e):
                                                state.panel_visibility[pid] = e.value

                                            return handler

                                        ui.checkbox(
                                            "", value=current_vis is True, on_change=make_checkbox_handler()
                                        ).props("dense")

                    refresh_visibility()

                    ui.separator().classes("my-2")

                    # Panel order section
                    ui.label("Order").classes("text-sm font-semibold text-gray-400")
                    ui.label("Reorder panels using arrows").classes("text-xs text-gray-500 mb-2")

                    panel_list = ui.column().classes("w-full gap-1")

                    def refresh_list():
                        panel_list.clear()
                        with panel_list:
                            for idx, panel_id in enumerate(state.panel_order):
                                panel_def = state.panel_definitions.get(panel_id, {})
                                with (
                                    ui.row()
                                    .classes("w-full items-center gap-2 p-1 rounded")
                                    .style("background: rgba(128,128,128,0.15);")
                                ):
                                    ui.icon(panel_def.get("icon", "widgets")).classes("text-gray-400 text-sm")
                                    ui.label(panel_def.get("name", panel_id)).classes("flex-grow text-sm")

                                    def move_up(i=idx):
                                        if i > 0:
                                            state.panel_order[i], state.panel_order[i - 1] = (
                                                state.panel_order[i - 1],
                                                state.panel_order[i],
                                            )
                                            refresh_list()
                                            refresh_visibility()

                                    def move_down(i=idx):
                                        if i < len(state.panel_order) - 1:
                                            state.panel_order[i], state.panel_order[i + 1] = (
                                                state.panel_order[i + 1],
                                                state.panel_order[i],
                                            )
                                            refresh_list()
                                            refresh_visibility()

                                    ui.button(icon="keyboard_arrow_up", on_click=move_up).props(
                                        "flat dense size=sm"
                                    ).set_enabled(idx > 0)
                                    ui.button(icon="keyboard_arrow_down", on_click=move_down).props(
                                        "flat dense size=sm"
                                    ).set_enabled(idx < len(state.panel_order) - 1)

                    refresh_list()

                    ui.separator().classes("my-2")

                    # Performance section
                    ui.label("Performance").classes("text-sm font-semibold text-gray-400 mt-2")
                    ui.label("Memory optimization for large datasets").classes("text-xs text-gray-500 mb-2")

                    # Out-of-core toggle
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.icon("storage").classes("text-gray-400 text-sm")
                        ui.label("Out-of-core mode").classes("flex-grow text-sm")
                        ooc_switch = ui.switch(value=state.out_of_core).props("dense")

                    ui.label("Caches data to disk, reducing RAM usage").classes("text-xs text-gray-500 ml-6")

                    # Downsampling toggle (affects peak map, minimap, and spectrum)
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.icon("speed").classes("text-gray-400 text-sm")
                        ui.label("Downsample Peaks").classes("flex-grow text-sm")
                        downsample_switch = ui.switch(value=state.peakmap_downsampling).props("dense")

                    ui.label("Faster rendering for large datasets").classes("text-xs text-gray-500 ml-6")

                    # Cache status
                    def get_cache_status():
                        if state.out_of_core:
                            size = state.get_cache_size_mb()
                            return f"Cache: {size:.1f} MB"
                        return "Cache: inactive"

                    cache_label = ui.label(get_cache_status()).classes("text-xs text-gray-400 ml-6")

                    def on_ooc_change():
                        # Read current switch value
                        if ooc_switch.value:
                            cache_label.set_text("Cache: will activate on apply")
                        else:
                            cache_label.set_text("Cache: inactive")

                    ooc_switch.on_value_change(on_ooc_change)

                    with ui.row().classes("w-full justify-end gap-2 mt-4"):

                        def apply_settings():
                            # Reorder panels using move()
                            if state.panels_container:
                                for idx, panel_id in enumerate(state.panel_order):
                                    if panel_id in state.panel_elements:
                                        state.panel_elements[panel_id].move(target_index=idx)
                            # Apply visibility
                            state.update_panel_visibility()

                            # Apply out-of-core setting - read directly from switch widget
                            new_ooc = ooc_switch.value
                            if new_ooc != state.out_of_core:
                                state.init_data_manager(out_of_core=new_ooc)
                                mode_str = "enabled" if new_ooc else "disabled"

                                if state.current_file:
                                    # Data already loaded - inform user to reload
                                    ui.notify(
                                        f"Out-of-core mode {mode_str}. Re-upload your file to apply.",
                                        type="warning",
                                    )
                                else:
                                    ui.notify(f"Out-of-core mode {mode_str}.", type="info")

                            # Apply downsampling setting (affects peak map, minimap, and spectrum)
                            new_downsample = downsample_switch.value
                            if new_downsample != state.peakmap_downsampling:
                                state.peakmap_downsampling = new_downsample
                                # Trigger redraw if data is loaded
                                if state.df is not None or state.data_manager is not None:
                                    state.emit_view_changed()
                                # Re-render spectrum if one is selected
                                if state.selected_spectrum_idx is not None:
                                    state.emit_selection_changed("spectrum", state.selected_spectrum_idx)

                            dialog.close()
                            ui.notify("Panel settings updated", type="positive")

                        ui.button("Cancel", on_click=dialog.close).props("flat")
                        ui.button("Apply", on_click=apply_settings).props("color=primary")

                dialog.open()

            ui.button(icon="tune", on_click=show_panel_settings).props("flat dense").tooltip("Panel Settings")

            ui.separator().props("vertical").classes("h-6")

            # Algorithm button
            from pyopenms_viewer.algorithms import show_algorithm_dialog

            async def on_algorithm_click():
                await show_algorithm_dialog(state, feature_info_label)

            ui.button(icon="science", on_click=on_algorithm_click).props("flat dense").tooltip("Run Algorithm")

            # Spacer to push info to the right
            ui.space()

            # Inline info labels
            info_label = ui.label("No file loaded").classes("text-xs text-gray-400")
            feature_info_label = ui.label("").classes("text-xs text-cyan-400")
            id_info_label = ui.label("").classes("text-xs text-orange-400")
            faims_info_label = ui.label("").classes("text-xs text-purple-400")
            faims_info_label.set_visibility(False)
            progress_label = ui.label("").classes("text-xs text-gray-400 ml-4")

            # Poll load progress from the shared state without pushing from background threads
            def _poll_progress():
                try:
                    # Prefer current_file progress, otherwise any active load
                    fp = None
                    if state.current_file and state.current_file in state.load_progress:
                        fp = state.current_file
                    elif state.load_progress:
                        fp = next(iter(state.load_progress))

                    if fp is None:
                        text = ""
                    else:
                        msg, pr = state.load_progress.get(fp, ("", 0.0))
                        pct = int(pr * 100)
                        text = f"{msg} ({pct}%)" if msg else ""
                    progress_label.set_text(text)
                except Exception:
                    pass

            ui.timer(0.5, _poll_progress)

            # Store labels in state for updates
            state.info_label = info_label
            state.feature_info_label = feature_info_label
            state.id_info_label = id_info_label
            state.faims_info_label = faims_info_label

            # Note: FAIMS toggle moved to peak map panel's minimap area

        # Panels container - holds all reorderable expansion panels
        panels_container = ui.column().classes("w-full items-center gap-2")
        state.panels_container = panels_container

        # Create panel manager
        panel_manager = PanelManager(state, panels_container)

        # Create and register all panels
        peak_map_panel = PeakMapPanel(state)
        panels = [
            TICPanel(state),
            ChromatogramPanel(state),
            peak_map_panel,
            IMPeakMapPanel(state),
            SpectrumPanel(state),
            SpectraTablePanel(state),
            FeaturesTablePanel(state),
            ExportPanel(state),
            LogPanel(state),
        ]

        for panel in panels:
            panel.build(panels_container)
            panel_manager.register(panel)

        # Store peak_map_panel reference for keyboard shortcut
        state.peak_map_panel = peak_map_panel

        # FAIMS panel (not managed by PanelManager since it's a card, not expansion)
        faims_panel = FAIMSPanel(state)
        faims_panel.build(panels_container)

        # Help panel
        with panels_container:
            with ui.expansion("Help", icon="help", value=False).classes("w-full max-w-[1700px]") as legend_exp:
                with ui.row().classes("gap-6 flex-wrap w-full"):
                    # Keyboard Shortcuts
                    with ui.card().classes("p-3").style("min-width: 280px;"):
                        ui.label("Keyboard Shortcuts").classes("font-bold text-lg mb-2")
                        ui.markdown("""
| Key | Action |
|-----|--------|
| `+` or `=` | Zoom in |
| `-` | Zoom out |
| `Left` `Right` | Pan left/right (RT) |
| `Up` `Down` | Pan up/down (m/z) |
| `G` | Go to exact range |
| `Home` | Reset to full view |
| `Delete` | Delete selected measurement |
| `F11` | Toggle fullscreen |
""").classes("text-sm")

                    # Mouse Controls
                    with ui.card().classes("p-3").style("min-width: 280px;"):
                        ui.label("Mouse Controls").classes("font-bold text-lg mb-2")
                        ui.markdown("""
| Action | Effect |
|--------|--------|
| **Scroll wheel** | Zoom in/out at cursor |
| **Drag** | Select region to zoom |
| **Shift + Drag** | Measure distance |
| **Ctrl + Drag** | Pan (grab & move) |
| **Double-click** | Reset to full view |
| **Click TIC** | Jump to spectrum at RT |
| **Click table row** | Select spectrum/feature |
""").classes("text-sm")

                    # 1D Spectrum Controls
                    with ui.card().classes("p-3").style("min-width: 280px;"):
                        ui.label("1D Spectrum Tools").classes("font-bold text-lg mb-2")
                        ui.markdown("""
| Tool | Usage |
|------|-------|
| **Measure** | Click two peaks to measure m/z |
| **Label** | Click peak to add annotation |
| **m/z Labels** | Toggle to show all peak m/z |
| **Auto Y** | Auto-scale Y-axis to visible |
| **Navigation** | < > prev/next, MS1/MS2 by level |
| **3D View** | Toggle 3D surface visualization |
""").classes("text-sm")

                    # Overlay Colors
                    with ui.card().classes("p-3").style("min-width: 220px;"):
                        ui.label("Overlay Colors").classes("font-bold text-lg mb-2")
                        with ui.column().classes("gap-1"):
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:#00ff64;border-radius:50%;'
                                    'border:1px solid white;"></div>',
                                    sanitize=False,
                                )
                                ui.label("Feature Centroid").classes("text-sm")
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;border:2px solid #ffff00;"></div>',
                                    sanitize=False,
                                )
                                ui.label("Feature Bounding Box").classes("text-sm")
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:rgba(0,200,255,0.5);'
                                    'border:1px solid #00c8ff;"></div>',
                                    sanitize=False,
                                )
                                ui.label("Feature Convex Hull").classes("text-sm")
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:#ff9632;'
                                    'transform:rotate(45deg);"></div>',
                                    sanitize=False,
                                )
                                ui.label("ID Precursor").classes("text-sm")

                    # Ion Colors
                    with ui.card().classes("p-3").style("min-width: 180px;"):
                        ui.label("Ion Annotations").classes("font-bold text-lg mb-2")
                        with ui.column().classes("gap-1"):
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:#1f77b4;"></div>', sanitize=False
                                )
                                ui.label("b-ions").classes("text-sm")
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:#d62728;"></div>', sanitize=False
                                )
                                ui.label("y-ions").classes("text-sm")
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:#2ca02c;"></div>', sanitize=False
                                )
                                ui.label("a-ions").classes("text-sm")
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:#ff7f0e;"></div>', sanitize=False
                                )
                                ui.label("Precursor").classes("text-sm")
                            with ui.row().classes("items-center gap-2"):
                                ui.html(
                                    '<div style="width:14px;height:14px;background:#7f7f7f;"></div>', sanitize=False
                                )
                                ui.label("Unmatched").classes("text-sm")

                    # File Types
                    with ui.card().classes("p-3").style("min-width: 200px;"):
                        ui.label("Supported Files").classes("font-bold text-lg mb-2")
                        ui.markdown("""
| Extension | Content |
|-----------|---------|
| `.mzML` | MS peak data |
| `.mzML` | MS peak data |
| `.featureXML` | Detected features |
| `.featureXML` | Detected features |
| `.idXML` | Peptide IDs |
""").classes("text-sm")
                        ui.label("Tips:").classes("font-semibold mt-2 text-sm")
                        ui.markdown("""
- Drag & drop files to load
- Use `--native` for file dialog
- Load multiple files at once
""").classes("text-xs text-gray-400")

                state.panel_elements["legend"] = legend_exp

        # Keyboard handlers (NiceGUI 3.x API)
        def on_global_key(e):
            if not e.action.keydown:
                return
            if e.key in ["+", "="]:
                state.zoom_in()
            elif e.key == "-":
                state.zoom_out()
            elif e.key.arrow_left:
                state.pan(rt_frac=-0.1)
            elif e.key.arrow_right:
                state.pan(rt_frac=0.1)
            elif e.key.arrow_up:
                state.pan(mz_frac=0.1)
            elif e.key.arrow_down:
                state.pan(mz_frac=-0.1)
            elif e.key == "Home":
                state.reset_view()
            elif e.key in ["Delete", "Backspace"]:
                # Delete selected measurement in spectrum browser
                if (
                    hasattr(state, "spectrum_selected_measurement_idx")
                    and state.spectrum_selected_measurement_idx is not None
                ):
                    state.delete_selected_measurement()
            elif str(e.key).lower() == "g":
                # Open go-to range dialog
                if hasattr(state, "peak_map_panel") and state.peak_map_panel:
                    state.peak_map_panel._open_range_popover()

        ui.keyboard(on_key=on_global_key)

    # Load CLI files after UI is ready
    async def load_cli_files():
        from pyopenms_viewer.cli import get_cli_files

        cli_files = get_cli_files()

        if cli_files["mzml"]:
            await load_mzml(cli_files["mzml"])

        if cli_files["featurexml"]:
            loader = FeatureLoader(state)
            if loader.load_sync(cli_files["featurexml"]):
                state.emit_data_loaded("features")
                ui.notify("Loaded features", type="positive")

        if cli_files["idxml"]:
            loader = IDLoader(state)
            if loader.load_sync(cli_files["idxml"]):
                state.emit_data_loaded("ids")
                ui.notify("Loaded IDs", type="positive")

    # Start loading CLI files in background so the page handler returns
    # immediately and avoids NiceGUI's "Response for / not ready after 3.0 seconds".
    try:
        asyncio.create_task(load_cli_files())
    except Exception:
        # Fallback to await if background task creation fails
        await load_cli_files()


# Register the page
@ui.page("/")
async def index():
    await create_ui()
