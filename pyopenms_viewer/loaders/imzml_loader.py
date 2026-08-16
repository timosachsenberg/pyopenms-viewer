"""imzML file loading using the OpenMS native MSImagingExperiment API.

Uses oms.ImzMLFile().load() (OpenMS PRs #9454, #9654, #9908) — pyimzml is NOT used.

Strategy:
- A single ImzMLFile.load() call reads the .ibd, builds spectra, and populates
  MSImagingGeometry directly from per-spectrum coordinates.
- Ion mobility arrives automatically on each spectrum via the OpenMS contract
  (``containsIMData()`` / ``getIMData()``, PR #9908); no imzML-specific IM parsing.
- Pseudo-RT (spectrum index) is applied so downstream tools have stable ordering.
- Zero-intensity centroid artefacts are dropped with ``MSSpectrum.select`` so
  FloatDataArrays (including IM) stay length-aligned with peaks.
- state.msi_experiment stores the full MSImagingExperiment for ion-image
  extraction (ImagingPanel) and pixel→spectrum lookup.
- state.exp stores the underlying MSExperiment for backward-compatible panels
  (SpectrumPanel, SpectraTablePanel, TICPanel, PeakMapPanel, etc.).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pyopenms_viewer.core.state import ViewerState

# ---------------------------------------------------------------------------
# Module-level helpers (follow reference imzml_import.py exactly)
# ---------------------------------------------------------------------------


def _apply_pseudo_rt(mie) -> None:
    """Set each spectrum's RT to its index (pseudo-RT) and ensure MS level = 1.

    imzML files carry no meaningful retention time; the parser leaves RT at -1.
    A monotonic pseudo-RT is required by sortSpectra and mass-trace pipelines.

    The imzML parser also leaves MS level at 0 (MSI acquisitions are typically
    single-stage MS, not tandem). Downstream rasterization / TIC / peak-map
    panels filter by ``ms_level == 1``, so we promote the level here — otherwise
    every non-imaging panel silently renders empty.
    """
    for idx, spec in enumerate(mie.getMSExperiment().getSpectra()):
        spec.setRT(float(idx))
        if spec.getMSLevel() < 1:
            spec.setMSLevel(1)


def _select_positive_intensity_peaks(spec) -> None:
    """Keep only intensity>0 peaks in ``spec``, in place.

    Uses ``MSSpectrum.select`` (not ``set_peaks``) so that peaks AND all
    associated FloatDataArrays — e.g. per-peak ion mobility — are filtered
    together and stay index-aligned. ``set_peaks`` shortens only the m/z /
    intensity arrays, leaving the ion-mobility array at its original length,
    which silently breaks IM detection downstream.
    """
    _, ints = spec.get_peaks()
    if len(ints) == 0:
        return
    keep = ints > 0
    if keep.all():
        return
    spec.select(np.flatnonzero(keep).tolist())


def _drop_zero_intensity_peaks(mie) -> None:
    """Drop intensity==0 peaks from every spectrum, keeping FloatDataArrays in sync.

    Centroiders sometimes leave placeholder peaks with zero intensity;
    these inflate peak counts and poison aggregate spectra.

    Must use ``MSSpectrum.select`` (not ``set_peaks``) so per-peak arrays such as
    ion mobility stay length-aligned with the peaks — OpenMS cannot fix a
    desync introduced here.
    """
    for spec in mie.getMSExperiment().getSpectra():
        _select_positive_intensity_peaks(spec)


def _extract_im_data(spec) -> tuple[str, np.ndarray, Any] | None:
    """Return ``(array_name, im_values, DriftTimeUnit)`` for a spectrum, or None.

    Ion mobility is read purely through the OpenMS spectrum contract, so any
    file format OpenMS can populate IM for works here unchanged. None is
    returned when the spectrum carries no IM array, or when the array length
    disagrees with the peak count (such a spectrum cannot be plotted).
    """
    if not spec.containsIMData():
        return None
    fda_index, unit = spec.getIMData()
    fda = spec.getFloatDataArrays()[fda_index]
    values = np.asarray(fda.get_data(), dtype=np.float32)
    if len(values) != spec.size():
        return None
    return fda.getName() or "ion mobility array", values, unit


def _im_state_from_unit(unit: Any) -> tuple[str, str] | None:
    """Map an OpenMS ``DriftTimeUnit`` to the viewer's ``(im_type, im_unit)`` pair.

    Preferred over FloatDataArray name heuristics: OpenMS resolves the unit from
    the PSI-MS ontology, so mean 1/K0 arrays are reported as VSSC rather than
    falling back to milliseconds. None means the unit is one the panels have no
    axis labelling for, and existing state should be left alone.
    """
    import pyopenms as oms

    if unit == oms.DriftTimeUnit.VSSC:
        return "inverse_k0", "Vs/cm\u00b2"
    if unit == oms.DriftTimeUnit.MILLISECOND:
        return "drift_time", "ms"
    return None


class ImzMLLoader:
    """Loads imzML/ibd file pairs into ViewerState via the OpenMS native parser.

    The .ibd binary must reside alongside the .imzML file.
    pyimzml is NOT used — loading is done via oms.ImzMLFile().load().

    After a successful load:
    - state.msi_experiment  — MSImagingExperiment (ion image extraction,
                               geometry, pixel→spectrum lookup)
    - state.exp             — underlying MSExperiment (all existing panels)
    - state.df              — None (peak map uses ``rasterizeRTMZ``, like mzML)
    - state.has_imzml       — True

    Example::

        state = ViewerState()
        loader = ImzMLLoader(state)
        if loader.load_sync("sample.imzML"):
            print(f"Loaded {state.msi_experiment.getNumberOfPixels()} pixels")
    """

    def __init__(self, state: ViewerState) -> None:
        self.state = state

    def load_sync(
        self,
        filepath: str,
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> bool:
        """Load an imzML file synchronously (blocking).

        Args:
            filepath: Absolute path to the ``.imzML`` file.
                      The ``.ibd`` binary must be alongside it.
            progress_cb: Optional ``(message, fraction)`` callback.

        Returns:
            True on success, False on failure.
            On failure ``state._last_load_error`` is set to an error string.
        """

        def _prog(msg: str, frac: float) -> None:
            if progress_cb:
                try:
                    progress_cb(msg, frac)
                except Exception:
                    pass

        try:
            import pyopenms as oms

            path = Path(filepath)
            _prog("Opening imzML file…", 0.0)

            mie = oms.MSImagingExperiment()
            oms.ImzMLFile().load(str(path), mie)

            geom = mie.getGeometry()
            n_spectra = mie.getNumberOfPixels()

            if n_spectra == 0:
                self.state._last_load_error = "No spectra found in imzML file"
                return False

            _prog("Applying pseudo-RT…", 0.15)
            _apply_pseudo_rt(mie)

            _prog("Dropping zero-intensity peaks…", 0.20)
            _drop_zero_intensity_peaks(mie)

            _prog("Updating ranges…", 0.25)
            msexp = mie.getMSExperiment()
            msexp.updateRanges()
            min_mz = msexp.getMinMZ()
            max_mz = msexp.getMaxMZ()

            # Expose real pixel size when present; sentinel -1 when absent.
            # The parser only writes imzml:pixel_size_x/y MetaValues when the
            # file supplies IMS:1000046/47, so their absence means truly unknown.
            if msexp.metaValueExists("imzml:pixel_size_x") and msexp.metaValueExists(
                "imzml:pixel_size_y"
            ):
                px_size_x = float(msexp.getMetaValue("imzml:pixel_size_x"))
                px_size_y = float(msexp.getMetaValue("imzml:pixel_size_y"))
                geom.setPixelSize(px_size_x, px_size_y, geom.getPixelSizeUnit())
            else:
                geom.setPixelSize(-1.0, -1.0, "unknown")
            mie.setGeometry(geom)

            # Spectrum stats + IM frame index only — no peak DataFrame.
            # Peak map uses rasterizeRTMZ on state.exp (same as mzML).
            _prog("Scanning spectra…", 0.30)

            spectra = msexp.getSpectra()
            n_total_spectra = len(spectra)
            spectrum_stats: list[dict] = []
            total_peaks = 0

            # Ion mobility via the OpenMS spectrum contract (ImzMLFile.load /
            # OnDisc getSpectrum populate FloatDataArrays automatically):
            #   spec.containsIMData() → True
            #   getIMData() → (fda_index, DriftTimeUnit)  # VSSC for mean 1/K0
            # Stream min/max only — do not retain per-peak IM arrays.
            detected_im_name: str | None = None
            im_frame_indices: list[int] = []
            im_unit = None
            im_mz_min_acc = np.inf
            im_mz_max_acc = -np.inf
            im_min_acc = np.inf
            im_max_acc = -np.inf

            for si in range(n_total_spectra):
                spec = spectra[si]
                mzs, ints = spec.get_peaks()
                mzs = np.asarray(mzs, dtype=np.float64)
                ints = np.asarray(ints, dtype=np.float64)
                n = len(mzs)
                total_peaks += n
                if n > 0:
                    tic_v = float(np.sum(ints))
                    bpi = float(np.max(ints))
                    mz_lo = float(mzs.min())
                    mz_hi = float(mzs.max())
                else:
                    tic_v = bpi = mz_lo = mz_hi = 0.0
                spectrum_stats.append(
                    {
                        "tic": tic_v,
                        "bpi": bpi,
                        "mz_min": mz_lo,
                        "mz_max": mz_hi,
                        "cv": None,
                    }
                )

                im_data = _extract_im_data(spec) if n > 0 else None
                if im_data is not None:
                    array_name, im_array, unit = im_data
                    if detected_im_name is None:
                        detected_im_name = array_name
                        im_unit = unit
                    im_mz_min_acc = min(im_mz_min_acc, mz_lo)
                    im_mz_max_acc = max(im_mz_max_acc, mz_hi)
                    im_min_acc = min(im_min_acc, float(im_array.min()))
                    im_max_acc = max(im_max_acc, float(im_array.max()))
                    im_frame_indices.append(si)

                if si % 500 == 0:
                    _prog(
                        f"Scanning spectra… {si + 1}/{n_total_spectra}",
                        0.30 + 0.40 * (si + 1) / n_total_spectra,
                    )

            _prog("Extracting spectrum metadata…", 0.80)

            # Extract against the local msexp WITHOUT mutating shared state, so a
            # failure here leaves the previous experiment fully intact and the UI
            # never sees new spectra combined with old geometry/df.
            from pyopenms_viewer.loaders.spectrum_extractor import extract_spectrum_data

            spectrum_data = extract_spectrum_data(
                self.state, spectrum_stats=spectrum_stats, exp=msexp
            )

            _prog("Updating viewer state…", 0.92)

            rt_data_min = 0.0
            rt_data_max = float(max(n_total_spectra - 1, 0))
            if min_mz > 0 and max_mz > 0:
                mz_data_min, mz_data_max = float(min_mz), float(max_mz)
            else:
                mz_data_min, mz_data_max = 0.0, 1.0

            # TIC trace: one entry per spectrum (pseudo-RT = spectrum index)
            tic_rt = np.arange(n_total_spectra, dtype=np.float64)
            tic_int = np.array([s["tic"] for s in spectrum_stats], dtype=np.float64)

            # Clear any previous data first (resets exp, msi_experiment, df, etc.)
            self.state.clear_mzml_data()

            # Core data — no peak DataFrame (rasterizeRTMZ / extractIonImage)
            self.state.exp = msexp
            self.state.msi_experiment = mie  # full MSImagingExperiment
            self.state.df = None
            self.state.total_peaks = total_peaks
            self.state.spectrum_data = spectrum_data
            self.state.current_file = str(path.resolve())

            # Bounds (RT = pixel index, not real time)
            self.state.rt_min = rt_data_min
            self.state.rt_max = rt_data_max
            self.state.mz_min = mz_data_min
            self.state.mz_max = mz_data_max
            self.state.view_rt_min = rt_data_min
            self.state.view_rt_max = rt_data_max
            self.state.view_mz_min = mz_data_min
            self.state.view_mz_max = mz_data_max

            # TIC
            self.state.tic_rt = tic_rt
            self.state.tic_intensity = tic_int
            self.state.tic_source = "MSI Pixel TIC"

            # Flags
            self.state.has_imzml = True
            self.state.rt_in_minutes = False  # pixel indices are not time

            # If per-peak ion mobility was detected, populate IM state so the
            # IMPeakMapPanel auto-activates (bounds-only; no peak-array concat).
            # Best-effort: the core imaging load above is already committed and
            # useful, so an IM-processing failure disables IM rather than
            # discarding the whole (valid) load.
            if detected_im_name is not None and im_frame_indices:
                try:
                    from pyopenms_viewer.loaders.mzml_loader import MzMLLoader

                    _im_loader = MzMLLoader(self.state)
                    _im_loader._process_ion_mobility_data(
                        im_mz_list=[],
                        im_im_list=[],
                        im_int_list=[],
                        detected_im_name=detected_im_name,
                        filepath=str(path),
                        im_frame_indices=im_frame_indices,
                        im_frame_ms_levels=[1] * len(im_frame_indices),
                        im_mz_min=float(im_mz_min_acc),
                        im_mz_max=float(im_mz_max_acc),
                        im_min=float(im_min_acc),
                        im_max=float(im_max_acc),
                    )
                    # Prefer OpenMS DriftTimeUnit over name heuristics when available.
                    labelling = (
                        _im_state_from_unit(im_unit) if im_unit is not None else None
                    )
                    if labelling is not None:
                        self.state.im_type, self.state.im_unit = labelling
                except Exception as im_ex:
                    self.state.has_ion_mobility = False
                    self.state.im_df = None
                    print(f"[ImzMLLoader] ion-mobility processing failed, disabled: {im_ex}")

            # Signal a new imaging dataset. Bumped LAST so panels that key off
            # this token only react once the full commit above is in place.
            self.state.msi_load_token += 1

            print(
                f"Loaded {n_spectra} pixels; geometry "
                f"{geom.getWidth()}x{geom.getHeight()}, "
                f"m/z {mz_data_min:.4f}\u2013{mz_data_max:.4f}"
                + (
                    f", ion mobility: {detected_im_name} ({len(im_frame_indices)} pixels)"
                    if im_frame_indices
                    else ""
                ),
                flush=True,  # loads run in a worker thread; stdout is block-buffered when piped
            )

            _prog("Done", 1.0)
            return True

        except Exception as ex:
            self.state._last_load_error = str(ex)
            return False
