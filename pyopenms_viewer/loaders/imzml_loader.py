"""imzML file loading using the OpenMS native MSImagingExperiment API.

Uses oms.ImzMLFile().load() (OpenMS PRs #9454, #9654) — pyimzml is NOT used.

Strategy:
- A single ImzMLFile.load() call reads the .ibd, builds spectra, and populates
  MSImagingGeometry directly from per-spectrum coordinates.
- Pseudo-RT (spectrum index) is applied so downstream tools have stable ordering.
- Zero-intensity centroid artefacts are dropped before building the DataFrame.
- state.msi_experiment stores the full MSImagingExperiment for ion-image
  extraction (ImagingPanel) and pixel→spectrum lookup.
- state.exp stores the underlying MSExperiment for backward-compatible panels
  (SpectrumPanel, SpectraTablePanel, TICPanel, PeakMapPanel, etc.).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

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


def _drop_zero_intensity_peaks(mie) -> None:
    """Drop intensity==0 peaks from every spectrum in the MSImagingExperiment, in place.

    Centroiders sometimes leave placeholder peaks with zero intensity;
    these inflate peak counts and poison aggregate spectra.
    """
    for spec in mie.getMSExperiment().getSpectra():
        mzs, ints = spec.get_peaks()
        if len(ints) == 0:
            continue
        keep = ints > 0
        if keep.all():
            continue
        spec.set_peaks((
            np.array(mzs[keep], dtype=np.float64, copy=True),
            np.array(ints[keep], dtype=np.float32, copy=True),
        ))


class ImzMLLoader:
    """Loads imzML/ibd file pairs into ViewerState via the OpenMS native parser.

    The .ibd binary must reside alongside the .imzML file.
    pyimzml is NOT used — loading is done via oms.ImzMLFile().load().

    After a successful load:
    - state.msi_experiment  — MSImagingExperiment (ion image extraction,
                               geometry, pixel→spectrum lookup)
    - state.exp             — underlying MSExperiment (all existing panels)
    - state.df              — peak DataFrame (rt = pseudo-RT index)
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
            if (msexp.metaValueExists("imzml:pixel_size_x") and
                    msexp.metaValueExists("imzml:pixel_size_y")):
                px_size_x = float(msexp.getMetaValue("imzml:pixel_size_x"))
                px_size_y = float(msexp.getMetaValue("imzml:pixel_size_y"))
                geom.setPixelSize(px_size_x, px_size_y, geom.getPixelSizeUnit())
            else:
                geom.setPixelSize(-1.0, -1.0, "unknown")
            mie.setGeometry(geom)

            _prog("Building peak DataFrame…", 0.30)

            spectra = msexp.getSpectra()
            n_total_spectra = len(spectra)

            all_rt: list[np.ndarray] = []
            all_mz: list[np.ndarray] = []
            all_int: list[np.ndarray] = []
            spectrum_stats: list[dict] = []

            # Ion-mobility detection: some imzML files carry IM per-peak in
            # <spectrum>/binaryDataArrayList; the parser exposes those as
            # FloatDataArrays on each MSSpectrum. Same detection contract as
            # MzMLLoader — mirrors state so IMPeakMapPanel activates.
            im_array_names = (
                "ion mobility",
                "inverse reduced ion mobility",
                "drift time",
                "1/k0",
            )
            detected_im_name: str | None = None
            im_mz_list: list[np.ndarray] = []
            im_im_list: list[np.ndarray] = []
            im_int_list: list[np.ndarray] = []
            im_frame_indices: list[int] = []

            for si in range(n_total_spectra):
                spec = spectra[si]
                rt = spec.getRT()  # pseudo-RT = si (set by _apply_pseudo_rt)
                mzs, ints = spec.get_peaks()
                mzs = np.asarray(mzs, dtype=np.float64)
                ints = np.asarray(ints, dtype=np.float64)
                n = len(mzs)
                if n > 0:
                    all_rt.append(np.full(n, rt, dtype=np.float32))
                    all_mz.append(mzs.astype(np.float32))
                    all_int.append(ints.astype(np.float32))
                tic_v = float(np.sum(ints)) if n > 0 else 0.0
                bpi = float(np.max(ints)) if n > 0 else 0.0
                spectrum_stats.append({
                    "tic": tic_v,
                    "bpi": bpi,
                    "mz_min": float(mzs.min()) if n > 0 else 0.0,
                    "mz_max": float(mzs.max()) if n > 0 else 0.0,
                    "cv": None,
                })

                # Detect and extract per-peak ion mobility (same shape as MzMLLoader).
                if n > 0:
                    float_arrays = spec.getFloatDataArrays()
                    if detected_im_name is None:
                        for fda in float_arrays:
                            name = fda.getName().lower() if fda.getName() else ""
                            if any(im_name in name for im_name in im_array_names):
                                detected_im_name = fda.getName()
                                break
                    if detected_im_name is not None:
                        for fda in float_arrays:
                            if fda.getName() == detected_im_name:
                                im_array = np.asarray(fda.get_data(), dtype=np.float32)
                                if len(im_array) == n:
                                    im_mz_list.append(mzs.astype(np.float32))
                                    im_im_list.append(im_array)
                                    im_int_list.append(ints.astype(np.float32))
                                    im_frame_indices.append(si)
                                break

                if si % 500 == 0:
                    _prog(
                        f"Building peak DataFrame… {si + 1}/{n_total_spectra}",
                        0.30 + 0.40 * (si + 1) / n_total_spectra,
                    )

            _prog("Finalizing DataFrame…", 0.72)

            if all_rt:
                df = pd.DataFrame({
                    "rt": np.concatenate(all_rt),
                    "mz": np.concatenate(all_mz),
                    "intensity": np.concatenate(all_int),
                })
            else:
                df = pd.DataFrame({"rt": [], "mz": [], "intensity": []})
            df["log_intensity"] = np.log1p(df["intensity"])

            _prog("Extracting spectrum metadata…", 0.80)

            # Temporarily assign msexp so extract_spectrum_data can iterate it
            self.state.exp = msexp
            from pyopenms_viewer.loaders.spectrum_extractor import extract_spectrum_data
            spectrum_data = extract_spectrum_data(self.state, spectrum_stats=spectrum_stats)

            _prog("Updating viewer state…", 0.92)

            rt_data_min = 0.0
            rt_data_max = float(n_total_spectra - 1)
            if len(df) > 0:
                mz_data_min = float(df["mz"].min())
                mz_data_max = float(df["mz"].max())
            elif min_mz > 0 and max_mz > 0:
                mz_data_min, mz_data_max = float(min_mz), float(max_mz)
            else:
                mz_data_min, mz_data_max = 0.0, 1.0

            # TIC trace: one entry per spectrum (pseudo-RT = spectrum index)
            tic_rt = np.arange(n_total_spectra, dtype=np.float64)
            tic_int = np.array([s["tic"] for s in spectrum_stats], dtype=np.float64)

            # Clear any previous data first (resets exp, msi_experiment, df, etc.)
            self.state.clear_mzml_data()

            # Core data
            self.state.exp = msexp
            self.state.msi_experiment = mie          # full MSImagingExperiment
            self.state.df = df
            self.state.total_peaks = len(df)
            self.state.spectrum_data = spectrum_data
            self.state.current_file = str(path)

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
            # IMPeakMapPanel auto-activates. Reuses MzMLLoader's proven pipeline.
            if detected_im_name is not None and im_mz_list:
                from pyopenms_viewer.loaders.mzml_loader import MzMLLoader
                _im_loader = MzMLLoader(self.state)
                _im_loader._process_ion_mobility_data(
                    im_mz_list=im_mz_list,
                    im_im_list=im_im_list,
                    im_int_list=im_int_list,
                    detected_im_name=detected_im_name,
                    filepath=str(path),
                    im_frame_indices=im_frame_indices,
                    im_frame_ms_levels=[1] * len(im_frame_indices),
                )

            print(
                f"Loaded {n_spectra} pixels; geometry "
                f"{geom.getWidth()}x{geom.getHeight()}, "
                f"m/z {mz_data_min:.4f}\u2013{mz_data_max:.4f}"
                + (f", ion mobility: {detected_im_name}" if detected_im_name else "")
            )

            _prog("Done", 1.0)
            return True

        except Exception as ex:
            self.state._last_load_error = str(ex)
            return False
