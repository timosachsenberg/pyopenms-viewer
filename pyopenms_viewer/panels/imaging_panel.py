"""Ion Image panel for MSI (imzML) data.

Implements the interactive browsing loop from the reference notebook
``ion_image_viewer_marimo.py``:

- **Ion image** — 2-D heatmap. Defaults to TIC; switches to the extracted
  ion image (m/z ± ppm) as soon as the user clicks a peak in the aggregate
  spectrum below, or types an m/z and hits *Extract*.
- **Aggregate spectrum** — one-pass mean or skyline over every pixel on a
  log-spaced ppm grid (following ``compute_aggregate`` in the reference).
  **Click a peak → instant ion image** at that m/z ± ppm (fast browse loop).
- **Pixel click** on the heatmap selects that pixel's spectrum in
  SpectrumPanel; if the imzML carries per-peak ion mobility, IMPeakMapPanel
  auto-activates via the loader.
- **Multi-ion overlay** — *Add to overlay* accumulates each extracted ion
  image into an IHC-style RGB composite, each ion in its own hue.

All heavy work uses OpenMS-native APIs:
- ``oms.ImzMLFile().load()`` (via ImzMLLoader)
- ``mie.extractIonImage(mz, ppm)`` → ``oms.IonImage.get_data()``
- ``mie.getGeometry().get_pixels_struct()``
- ``spec.calculateTIC()``

State: ``state.msi_experiment`` (MSImagingExperiment, set by ImzMLLoader);
panel is only visible when ``state.has_imzml`` is True.
"""

import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps as mpl_colormaps
from nicegui import ui

from pyopenms_viewer.core.config import NEUTRAL_GRAY_HEX
from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel

_COLORMAPS = ["viridis", "magma", "plasma", "hot", "inferno"]
_AGGREGATE_MODES = {"mean": "Mean", "skyline": "Skyline (max)"}

# Distinct hues for the multi-ion overlay ("IHC stain" analogy in the
# user story). Rotated in order of overlay additions.
_OVERLAY_HUES: list[tuple[int, int, int]] = [
    (255, 60, 60),    # red
    (60, 255, 60),    # green
    (80, 140, 255),   # blue
    (255, 200, 50),   # amber
    (255, 80, 220),   # magenta
    (80, 240, 240),   # cyan
    (255, 140, 60),   # orange
    (180, 100, 255),  # violet
]

# Cap the number of aggregate-spectrum bars sent to the browser — the
# figure is serialised to JSON on every update, and centroid imzML
# files can have >100k non-zero bins. Same cap as the reference.
_TOP_N_PEAKS = 400

# For large MSI grids, server-side color mapping + go.Image avoids expensive
# Heatmap serialization and improves interaction latency.
_FAST_RENDER_PIXELS = 200_000


class ImagingPanel(BasePanel):
    """Interactive Ion Image panel for MSI (imzML) data.

    Two co-registered views over the same MSImagingGeometry:
    - **Ion image heatmap** on top (TIC by default; ion image after any
      peak-click or Extract).
    - **Aggregate spectrum** below (mean or skyline over all pixels);
      click a peak → the heatmap replaces its content with the ion image
      at that m/z ± ppm.

    Deliberate composition: *Add to overlay* accumulates ion images into
    an IHC-style multi-channel RGB composite (each ion in its own hue).
    """

    def __init__(self, state: ViewerState) -> None:
        super().__init__(state, "imaging", "Ion Image", "image")

        # UI handles
        self.plot: ui.plotly | None = None
        self.spectrum_plot: ui.plotly | None = None
        self.overlay_plot: ui.plotly | None = None
        self._mode_select = None
        self._agg_select = None
        self._bin_ppm_input = None
        self._mz_input = None
        self._ppm_input = None
        self._status_label = None
        self._overlay_container = None
        self._overlay_legend = None

        # Display state
        self._colorscale: str = "viridis"
        self._mode: str = "tic"                     # "tic" | "ion"
        self._agg_mode: str = "mean"                # "mean" | "skyline"
        self._current_mz: float | None = None
        self._current_ppm: float = 10.0

        # Cached aggregate spectrum (recomputed on load / bin-ppm change).
        # Stored full-resolution; the top-N pruning happens at render time.
        self._agg_bin_ppm: float = 5.0
        self._agg_centers: np.ndarray | None = None
        self._agg_mean: np.ndarray | None = None
        self._agg_skyline: np.ndarray | None = None

        # Overlay: list of {"mz", "ppm", "hue", "img"} dicts. Rendered
        # additively into an RGB composite.
        self._overlay_entries: list[dict] = []

        # Lightweight caches to keep browse interactions snappy.
        self._tic_cache: np.ndarray | None = None
        self._ion_cache_key: tuple[float, float] | None = None
        self._ion_cache_img: np.ndarray | None = None

        self._plotly_config = {
            "modeBarButtonsToRemove": ["autoScale2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "ion_image",
                "width": 900,
                "height": 700,
                "scale": 1,
            },
        }

    # ------------------------------------------------------------------
    # BasePanel interface
    # ------------------------------------------------------------------

    def build(self, container: ui.element) -> ui.expansion:
        with container:
            self._make_expansion()
            with self.expansion:
                self._build_controls()
                self._build_image_plot()
                self._build_spectrum_plot()
                self._build_overlay()

        # Re-render all figures once the expansion finishes opening.
        # Quasar fires ``after-show`` after the open animation completes,
        # at which point the container has its final nonzero dimensions and
        # Plotly's axis-scaling math succeeds.  This handles the case where
        # update_figure fires while the panel is still mid-animation (the
        # initial auto-open on data load), which would otherwise leave the
        # aggregate spectrum stuck on its empty placeholder.
        self.expansion.on('after-show', lambda _: self._after_show_render())

        self.state.on_data_loaded(self._on_data_loaded)
        return self.expansion

    def update(self) -> None:
        """Re-render the current image (called on data_loaded / view refresh)."""
        if self.plot is None:
            return
        if not self._has_data():
            self.plot.update_figure(self._figure_with_config(self._empty_figure()))
            if self.spectrum_plot is not None:
                self.spectrum_plot.update_figure(
                    self._figure_with_config(self._empty_spectrum_figure())
                )
            return
        if self._mode == "tic":
            self._render_tic()
        else:
            if self._current_mz is not None:
                self._render_ion_image(self._current_mz, self._current_ppm)
            else:
                self._render_tic()
        self._render_aggregate_spectrum()

    def _after_show_render(self) -> None:
        """Re-render after the expansion open animation completes.

        Called via the Quasar ``after-show`` event, which fires in a proper
        NiceGUI WebSocket-event context (slot stack is valid, client is
        connected).  We use ``ui.run_javascript`` to call ``Plotly.react``
        directly, bypassing Vue prop-equality short-circuits that suppress a
        second render when the figure data hasn't changed.
        """
        if not self._has_data():
            return
        self.update()   # refresh stored figure state
        # Force re-render via JS for scatter-based plots (heatmaps are fine).
        import json as _json
        for plot in [self.spectrum_plot, self.overlay_plot]:
            if plot is None:
                continue
            stored = plot._props.get("options")
            if not stored or not stored.get("data"):
                continue
            eid = plot.id
            try:
                ui.run_javascript(
                    f"(function(){{"
                    f"var el=document.getElementById('c{eid}');"
                    f"if(!el||!window.Plotly)return;"
                    f"var f={_json.dumps(stored)};"
                    f"Plotly.react(el,f.data,f.layout,f.config||{{}});"
                    f"}})();"
                )
            except Exception as _e:
                print(f"[ImagingPanel] after-show JS error c{eid}: {_e!r}", flush=True)

    def _has_data(self) -> bool:
        return self.state.has_imzml and self.state.msi_experiment is not None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        with ui.row().classes("items-center gap-3 w-full flex-wrap py-1"):
            ui.label("Display:").classes("text-sm text-gray-400")

            self._mode_select = ui.select(
                options={"tic": "TIC Image", "ion": "Ion Image (m/z)"},
                value="tic",
                label="",
                on_change=lambda e: self._on_mode_change(e.value),
            ).classes("w-40 text-sm").props("dense outlined")

            self._mz_input = ui.number(
                label="m/z",
                value=500.0,
                step=0.001,
                precision=4,
                format="%.4f",
            ).classes("w-32 text-sm").props("dense outlined")

            self._ppm_input = ui.number(
                label="ppm tol",
                value=10.0,
                step=1.0,
                min=0.1,
                on_change=lambda e: self._on_ppm_change(e.value),
            ).classes("w-24 text-sm").props("dense outlined")

            ui.button(
                "Extract",
                on_click=self._on_extract,
                icon="search",
            ).props("dense flat").classes("text-sm text-primary")

            ui.separator().props("vertical")

            ui.label("Aggregate:").classes("text-sm text-gray-400")
            self._agg_select = ui.select(
                options=_AGGREGATE_MODES,
                value="mean",
                label="",
                on_change=lambda e: self._on_agg_mode_change(e.value),
            ).classes("w-36 text-sm").props("dense outlined")

            self._bin_ppm_input = ui.number(
                label="bin (ppm)",
                value=5.0,
                step=0.5,
                min=0.5,
                on_change=lambda e: self._on_bin_ppm_change(e.value),
            ).classes("w-24 text-sm").props("dense outlined")

            ui.separator().props("vertical")

            ui.select(
                options={c: c.capitalize() for c in _COLORMAPS},
                value="viridis",
                label="Colormap",
                on_change=lambda e: self._on_colormap_change(e.value),
            ).classes("w-28 text-sm").props("dense outlined")

            ui.separator().props("vertical")

            ui.button(
                "Add to overlay",
                on_click=self._on_add_overlay,
                icon="add",
            ).props("dense flat").classes("text-sm").tooltip(
                "Add the current ion image as a colour channel of the overlay below"
            )
            ui.button(
                "Clear overlay",
                on_click=self._on_clear_overlay,
                icon="close",
            ).props("dense flat").classes("text-sm text-gray-400")

            self._status_label = ui.label("").classes(
                "text-xs text-gray-400 ml-2 self-center"
            )

    def _build_image_plot(self) -> None:
        ui.label(
            "Click a pixel to jump to its spectrum. If the file carries ion "
            "mobility, the IM frame panel activates automatically."
        ).classes("text-xs text-gray-500 mb-1")
        self.plot = ui.plotly(
            self._figure_with_config(self._empty_figure())
        ).classes("w-full")
        self.plot.on("plotly_click", self._on_image_click)

    def _build_spectrum_plot(self) -> None:
        ui.label(
            "Aggregate spectrum over every pixel — click a peak to browse "
            "ion images at that m/z (\u00b1 ppm tolerance above)."
        ).classes("text-xs text-gray-500 mt-2 mb-1")
        self.spectrum_plot = ui.plotly(
            self._figure_with_config(self._empty_spectrum_figure())
        ).classes("w-full")
        self.spectrum_plot.on("plotly_click", self._on_spectrum_click)

    def _build_overlay(self) -> None:
        self._overlay_container = ui.column().classes("w-full mt-3 gap-1")
        with self._overlay_container:
            ui.label(
                "Multi-ion overlay — each ion is one colour channel; images "
                "are pixel-aligned by construction."
            ).classes("text-xs text-gray-500")
            self.overlay_plot = ui.plotly(
                self._figure_with_config(self._empty_overlay_figure())
            ).classes("w-full")
            self._overlay_legend = ui.row().classes(
                "items-center gap-3 flex-wrap text-xs text-gray-400"
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_data_loaded(self, data_type: str) -> None:
        """Rebuild the aggregate cache and render TIC when a new imzML loads."""
        if data_type != "mzml":
            return
        # Update visibility first so update_figure calls reach a visible element.
        self.update_visibility()
        if not self._has_data():
            if self.expansion:
                self.expansion.value = False
            return
        if self.expansion:
            self.expansion.value = True
        self._mode = "tic"
        self._current_mz = None
        self._overlay_entries.clear()
        self._tic_cache = None
        self._ion_cache_key = None
        self._ion_cache_img = None
        if self._mode_select is not None:
            self._mode_select.set_value("tic")
        # Rebuild the aggregate — a one-time O(N_spectra × N_peaks) pass.
        try:
            self._recompute_aggregate()
        except Exception as exc:
            print(f"[ImagingPanel] aggregate error: {exc}")
        try:
            self._render_tic()
        except Exception as exc:
            print(f"[ImagingPanel] TIC render error: {exc}")
        # Scatter-based plots (aggregate spectrum, overlay) are deferred to
        # ``_after_show_render`` which runs from the Quasar ``after-show`` event
        # — a proper NiceGUI WebSocket context — so Plotly.react fires after the
        # expansion animation completes and the container has nonzero dimensions.


    def _on_mode_change(self, mode: str) -> None:
        self._mode = mode
        if mode == "tic":
            self._render_tic()
        elif self._current_mz is not None:
            self._render_ion_image(self._current_mz, self._current_ppm)

    def _on_agg_mode_change(self, agg_mode: str) -> None:
        self._agg_mode = agg_mode
        self._render_aggregate_spectrum()

    def _on_bin_ppm_change(self, value) -> None:
        try:
            new_ppm = float(value)
        except (TypeError, ValueError):
            return
        if new_ppm <= 0 or abs(new_ppm - self._agg_bin_ppm) < 1e-9:
            return
        self._agg_bin_ppm = new_ppm
        if self._has_data():
            self._recompute_aggregate()
            self._render_aggregate_spectrum()

    def _on_ppm_change(self, value) -> None:
        try:
            self._current_ppm = float(value)
        except (TypeError, ValueError):
            pass

    def _on_colormap_change(self, colorscale: str) -> None:
        self._colorscale = colorscale
        self.update()

    def _on_extract(self) -> None:
        """Explicit *Extract* button — read the m/z / ppm inputs."""
        if not self._has_data():
            return
        try:
            mz = float(self._mz_input.value)
            ppm = float(self._ppm_input.value)
        except (TypeError, ValueError):
            return
        if mz <= 0 or ppm <= 0:
            return
        self._current_mz = mz
        self._current_ppm = ppm
        self._mode = "ion"
        if self._mode_select is not None:
            self._mode_select.set_value("ion")
        self._render_ion_image(mz, ppm)

    def _on_spectrum_click(self, e) -> None:
        """Click on aggregate spectrum peak → instant ion image (browse loop)."""
        if not self._has_data():
            return
        pts = (e.args or {}).get("points") or []
        if not pts:
            return
        try:
            mz = float(pts[0].get("x"))
        except (TypeError, ValueError):
            return
        if mz <= 0:
            return
        self._current_mz = mz
        self._mode = "ion"
        if self._mode_select is not None:
            self._mode_select.set_value("ion")
        if self._mz_input is not None:
            self._mz_input.set_value(mz)
        self._render_ion_image(mz, self._current_ppm)

    def _on_image_click(self, e) -> None:
        """Click on a heatmap pixel → select that pixel's spectrum."""
        pts = (e.args or {}).get("points") or []
        if not pts:
            return
        try:
            px_x = int(round(float(pts[0].get("x", -1))))
            px_y = int(round(float(pts[0].get("y", -1))))
        except (TypeError, ValueError):
            return
        spec_idx = self._pixel_to_spectrum_idx(px_x, px_y)
        if spec_idx is not None:
            self.state.select_spectrum(spec_idx)
            if self._status_label is not None:
                self._status_label.set_text(
                    f"Pixel ({px_x}, {px_y})  \u2192  spectrum #{spec_idx}"
                )

    def _on_add_overlay(self) -> None:
        """Add the current ion image as a colour channel of the overlay."""
        if not self._has_data():
            return
        if self._mode != "ion" or self._current_mz is None:
            if self._status_label is not None:
                self._status_label.set_text(
                    "Extract an ion image first (click a peak or the Extract button)."
                )
            return
        img = self._compute_ion_image(self._current_mz, self._current_ppm)
        if img is None or img.size == 0:
            return
        hue = _OVERLAY_HUES[len(self._overlay_entries) % len(_OVERLAY_HUES)]
        self._overlay_entries.append({
            "mz": float(self._current_mz),
            "ppm": float(self._current_ppm),
            "hue": hue,
            "img": img,
        })
        self._render_overlay()

    def _on_clear_overlay(self) -> None:
        self._overlay_entries.clear()
        self._render_overlay()

    # ------------------------------------------------------------------
    # Image computation (OpenMS-native)
    # ------------------------------------------------------------------

    def _compute_tic_image(self) -> np.ndarray:
        """TIC image (H, W) with NaN for unmeasured pixels (reference: ``tic_image``)."""
        if self._tic_cache is not None:
            return self._tic_cache
        mie = self.state.msi_experiment
        geom = mie.getGeometry()
        H, W = geom.getHeight(), geom.getWidth()
        img = np.full((H, W), np.nan, dtype=np.float64)
        spectra = mie.getMSExperiment().getSpectra()
        px = geom.get_pixels_struct()
        ys = px["y"].astype(np.intp)
        xs = px["x"].astype(np.intp)
        tic_vals = np.fromiter(
            (spectra[int(si)].calculateTIC() for si in px["spectrum_index"]),
            dtype=np.float64,
            count=len(px),
        )
        img[ys, xs] = tic_vals
        self._tic_cache = img
        return img

    def _compute_ion_image(self, mz: float, ppm: float) -> np.ndarray:
        """Ion image via native ``MSImagingExperiment.extractIonImage()``."""
        key = (float(mz), float(ppm))
        if self._ion_cache_key == key and self._ion_cache_img is not None:
            return self._ion_cache_img
        img = self.state.msi_experiment.extractIonImage(mz, ppm).get_data()
        self._ion_cache_key = key
        self._ion_cache_img = img
        return img

    def _pixel_to_spectrum_idx(self, x: int, y: int) -> int | None:
        """Grid (x, y) → spectrum index (bounds-checked)."""
        if not self._has_data():
            return None
        geom = self.state.msi_experiment.getGeometry()
        if not geom.hasPixel(x, y):
            return None
        try:
            return int(geom.getSpectrumIndex(x, y))
        except Exception:
            # Fallback: linear scan (older bindings may not expose getSpectrumIndex(x,y)).
            px = geom.get_pixels_struct()
            mask = (px["x"] == x) & (px["y"] == y)
            hits = px["spectrum_index"][mask]
            return int(hits[0]) if len(hits) > 0 else None

    # ------------------------------------------------------------------
    # Aggregate spectrum (reference: compute_aggregate + build_spectrum)
    # ------------------------------------------------------------------

    def _recompute_aggregate(self) -> None:
        """Single pass over every pixel → mean & skyline on a ppm-spaced grid.

        Mirrors the reference notebook's ``compute_aggregate`` cell verbatim:
        log-spaced m/z edges spaced by ``bin_ppm`` ppm, ``np.searchsorted`` +
        ``np.add.at`` / ``np.maximum.at`` for the sum / max accumulators.
        """
        mie = self.state.msi_experiment
        if mie is None:
            self._agg_centers = self._agg_mean = self._agg_skyline = None
            return
        msexp = mie.getMSExperiment()
        try:
            msexp.updateRanges()
        except Exception:
            pass
        min_mz = float(msexp.getMinMZ())
        max_mz = float(msexp.getMaxMZ())
        if not (max_mz > min_mz > 0):
            self._agg_centers = self._agg_mean = self._agg_skyline = None
            return

        bin_ppm = float(self._agg_bin_ppm)
        step = float(np.log1p(bin_ppm * 1e-6))
        n_edges = int(np.ceil((np.log(max_mz) - np.log(min_mz)) / step)) + 1
        edges = min_mz * np.exp(np.arange(n_edges) * step)
        centers = 0.5 * (edges[:-1] + edges[1:])
        n_bins = centers.size

        sum_ = np.zeros(n_bins, dtype=np.float64)
        max_ = np.zeros(n_bins, dtype=np.float64)
        count = np.zeros(n_bins, dtype=np.int64)

        geom = mie.getGeometry()
        spectra = msexp.getSpectra()
        for px in geom.get_pixels_struct():
            mzs, ints = spectra[int(px["spectrum_index"])].get_peaks()
            if mzs.size == 0:
                continue
            idx = np.searchsorted(edges, mzs, side="right") - 1
            np.clip(idx, 0, n_bins - 1, out=idx)
            np.add.at(sum_, idx, ints)
            np.maximum.at(max_, idx, ints)
            count[np.unique(idx)] += 1

        mean = np.divide(sum_, count, out=np.zeros_like(sum_), where=count > 0)

        self._agg_centers = centers
        self._agg_mean = mean
        self._agg_skyline = max_

    def _get_aggregate_top_peaks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mz, intensity) for the top-N non-zero aggregate peaks."""
        if self._agg_centers is None or self._agg_mean is None:
            return np.array([]), np.array([])
        intensity = self._agg_skyline if self._agg_mode == "skyline" else self._agg_mean
        nz = np.flatnonzero(intensity > 0)
        if nz.size == 0:
            return np.array([]), np.array([])
        if nz.size > _TOP_N_PEAKS:
            top = nz[np.argpartition(intensity[nz], -_TOP_N_PEAKS)[-_TOP_N_PEAKS:]]
        else:
            top = nz
        top = top[np.argsort(self._agg_centers[top])]
        return self._agg_centers[top], intensity[top]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _push_figure(self, plot_el, fig: go.Figure) -> None:
        """Deliver a Plotly figure to a ui.plotly element.

        Wraps ``update_figure`` so all renders go through one path. The
        expansion's ``after-show`` handler calls ``update()`` once the
        container has its final dimensions, so figures that land here while
        the panel is mid-animation will be re-applied correctly on open.
        """
        plot_el.update_figure(self._figure_with_config(fig))

    def _render_tic(self) -> None:
        if not self._has_data() or self.plot is None:
            return
        img = self._compute_tic_image()
        fig = self._create_heatmap_figure(img, "TIC Image", self._colorscale)
        self._push_figure(self.plot, fig)
        if self._status_label is not None:
            self._status_label.set_text(
                "TIC — click a pixel for its spectrum, or a peak below for an ion image"
            )

    def _render_ion_image(self, mz: float, ppm: float) -> None:
        if not self._has_data() or self.plot is None:
            return
        img = self._compute_ion_image(mz, ppm)
        title = f"Ion Image  m/z {mz:.4f} \u00b1 {ppm:.1f} ppm"
        fig = self._create_heatmap_figure(img, title, self._colorscale)
        self._push_figure(self.plot, fig)
        if self._status_label is not None:
            self._status_label.set_text(
                f"m/z {mz:.4f} \u00b1 {ppm:.1f} ppm  "
                "\u2014 click *Add to overlay* to keep this ion as a channel"
            )

    def _render_aggregate_spectrum(self) -> None:
        if self.spectrum_plot is None:
            return
        mz, intensity = self._get_aggregate_top_peaks()
        if mz.size == 0:
            self.spectrum_plot.update_figure(
                self._figure_with_config(self._empty_spectrum_figure())
            )
            return
        # Native-Python coordinates only: NiceGUI serialises figures with orjson,
        # which rejects numpy scalars (np.float64). Passing raw numpy values in
        # stick coordinates silently aborts the update, leaving the plot blank.
        mz_list = [float(v) for v in mz]
        int_list = [float(v) for v in intensity]

        # Draw the sticks as ONE line trace with None gaps between peaks (far
        # lighter than one shape per peak), plus a tip-marker trace so clicks
        # land accurately and drive the ion-image browse loop.
        stick_x: list[float | None] = []
        stick_y: list[float | None] = []
        for m, y in zip(mz_list, int_list):
            stick_x += [m, m, None]
            stick_y += [0.0, y, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stick_x,
            y=stick_y,
            mode="lines",
            line=dict(color="steelblue", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=mz_list,
            y=int_list,
            mode="markers",
            marker=dict(size=6, color="steelblue"),
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.3e}<extra></extra>",
            name=self._agg_mode,
        ))
        # Mark the currently-selected m/z with a red vertical line.
        if self._current_mz is not None:
            fig.add_vline(
                x=float(self._current_mz),
                line=dict(color="crimson", width=1.5, dash="dot"),
            )
        title = (
            f"{_AGGREGATE_MODES[self._agg_mode]} spectrum "
            f"({mz.size:,} peaks, bin = {self._agg_bin_ppm:g} ppm) "
            "\u2014 click a peak to browse the ion image"
        )
        fig.update_layout(
            title={"text": title, "font": {"color": NEUTRAL_GRAY_HEX, "size": 12},
                   "x": 0.01, "xanchor": "left"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            showlegend=False,
            xaxis=dict(title="m/z", color=NEUTRAL_GRAY_HEX, gridcolor="rgba(128,128,128,0.15)"),
            yaxis=dict(
                title="intensity",
                color=NEUTRAL_GRAY_HEX,
                gridcolor="rgba(128,128,128,0.15)",
                autorange=True,       # y rescales automatically when x is zoomed
            ),
            margin=dict(l=60, r=20, t=40, b=45),
            dragmode="zoom",
        )
        self._push_figure(self.spectrum_plot, fig)

    def _render_overlay(self) -> None:
        if self.overlay_plot is None:
            return

        # Refresh the legend row every time — safe because it's just labels.
        if self._overlay_legend is not None:
            self._overlay_legend.clear()
            with self._overlay_legend:
                if not self._overlay_entries:
                    ui.label("No ions in overlay yet.")
                else:
                    for entry in self._overlay_entries:
                        r, g, b = entry["hue"]
                        css = f"#{r:02x}{g:02x}{b:02x}"
                        ui.html(
                            f'<div style="width:12px;height:12px;background:{css};'
                            'border:1px solid #666;border-radius:2px;"></div>',
                            sanitize=False,
                        )
                        ui.label(f"m/z {entry['mz']:.4f} \u00b1 {entry['ppm']:.1f} ppm")

        if not self._overlay_entries:
            self._push_figure(self.overlay_plot, self._empty_overlay_figure())
            return

        composite = self._compose_overlay_rgb()
        if composite is None:
            return
        H, W = composite.shape[:2]

        fig = go.Figure(go.Image(z=composite))
        fig.update_layout(
            title={"text": f"Overlay ({len(self._overlay_entries)} ions)",
                   "font": {"color": NEUTRAL_GRAY_HEX, "size": 13},
                   "x": 0.01, "xanchor": "left"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            # Origin-lower to match the single-ion heatmaps above (the overlay
            # is composed from the same [y, x]-indexed ion images).
            xaxis=dict(visible=False, range=[-0.5, W - 0.5]),
            yaxis=dict(visible=False, range=[-0.5, H - 0.5], scaleanchor="x"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        self._push_figure(self.overlay_plot, fig)

    def _compose_overlay_rgb(self) -> np.ndarray | None:
        """Combine every overlay entry into a single H×W×3 uint8 image.

        Each ion image is 99th-percentile normalised, tinted with its hue, and
        additively blended (max-clipped at 255). Yields the "IHC stain"
        composite described in the user story.
        """
        first = self._overlay_entries[0]["img"]
        if first is None or first.size == 0:
            return None
        H, W = first.shape
        rgb = np.zeros((H, W, 3), dtype=np.float64)

        for entry in self._overlay_entries:
            img = entry["img"]
            if img.shape != (H, W):
                # Skip stale images from a previous experiment.
                continue
            arr = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.max() <= 0:
                continue
            top = np.percentile(arr, 99)
            if top <= 0:
                top = arr.max()
            norm = np.clip(arr / top, 0.0, 1.0)
            r, g, b = entry["hue"]
            rgb[..., 0] += norm * r
            rgb[..., 1] += norm * g
            rgb[..., 2] += norm * b

        return np.clip(rgb, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Figure builders
    # ------------------------------------------------------------------

    def _create_heatmap_figure(
        self, img: np.ndarray, title: str, colorscale: str
    ) -> go.Figure:
        """Build a Plotly Heatmap from an (H, W) intensity array."""
        H, W = img.shape

        # 99th-percentile clip for contrast — matches the reference notebook.
        display_img = img
        if not np.all(np.isnan(img)):
            threshold = float(np.nanpercentile(img, 99))
            if threshold > 0:
                display_img = np.clip(img, a_min=0, a_max=threshold)

        # Fast path for large images: pre-color server-side and ship uint8 RGB.
        if display_img.size >= _FAST_RENDER_PIXELS:
            finite = np.nan_to_num(display_img, nan=0.0, posinf=0.0, neginf=0.0)
            vmax = float(np.max(finite))
            if vmax > 0:
                norm = np.clip(finite / vmax, 0.0, 1.0)
            else:
                norm = np.zeros_like(finite, dtype=np.float64)
            cmap = mpl_colormaps.get_cmap(colorscale)
            rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)

            fig = go.Figure(go.Image(z=rgb))
            fig.update_layout(
                title={"text": title, "font": {"color": NEUTRAL_GRAY_HEX, "size": 13},
                       "x": 0.01, "xanchor": "left"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=440,
                xaxis=dict(title="pixel x", color=NEUTRAL_GRAY_HEX,
                           showgrid=False, zeroline=False, range=[-0.5, W - 0.5]),
                # Non-reversed y-axis => array row 0 (geometry y=0) at the
                # bottom (origin="lower"), matching the Heatmap path and the
                # reference notebook's imshow(..., origin="lower").
                yaxis=dict(title="pixel y", color=NEUTRAL_GRAY_HEX,
                           showgrid=False, zeroline=False,
                           range=[-0.5, H - 0.5], scaleanchor="x"),
                margin=dict(l=60, r=20, t=50, b=50),
            )
            return fig

        fig = go.Figure(go.Heatmap(
            z=display_img,
            x=list(range(W)),
            y=list(range(H)),
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate=(
                "x: %{x}<br>y: %{y}<br>intensity: %{z:.3e}<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text="Intensity", font=dict(color=NEUTRAL_GRAY_HEX, size=11)),
                tickfont=dict(color=NEUTRAL_GRAY_HEX, size=11),
                thickness=12,
                len=0.85,
            ),
        ))
        fig.update_layout(
            title={"text": title, "font": {"color": NEUTRAL_GRAY_HEX, "size": 13},
                   "x": 0.01, "xanchor": "left"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=440,
            xaxis=dict(title="pixel x", color=NEUTRAL_GRAY_HEX,
                       showgrid=False, zeroline=False),
            yaxis=dict(title="pixel y", color=NEUTRAL_GRAY_HEX,
                       showgrid=False, zeroline=False,
                       scaleanchor="x"),
            margin=dict(l=60, r=20, t=50, b=50),
        )
        return fig

    def _empty_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title={"text": "Ion Image \u2014 load an imzML file to display",
                   "font": {"color": NEUTRAL_GRAY_HEX}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
        )
        return fig

    def _empty_spectrum_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title={
                "text": "Aggregate spectrum \u2014 load an imzML file to display",
                "font": {"color": NEUTRAL_GRAY_HEX, "size": 12},
                "x": 0.01, "xanchor": "left",
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=140,
            xaxis=dict(color=NEUTRAL_GRAY_HEX),
            yaxis=dict(color=NEUTRAL_GRAY_HEX),
            margin=dict(l=60, r=20, t=40, b=30),
        )
        return fig

    def _empty_overlay_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title={
                "text": "Overlay \u2014 extract an ion image, then click *Add to overlay*",
                "font": {"color": NEUTRAL_GRAY_HEX, "size": 12},
                "x": 0.01, "xanchor": "left",
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=140,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig
