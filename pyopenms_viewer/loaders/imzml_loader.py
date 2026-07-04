"""imzML file loading via the OpenMS native parser.

Uses ``oms.ImzMLFile().load()`` (OpenMS PRs #9454, #9654) to read imzML/.ibd
pairs. pyimzml is NOT used.

This loader is scoped to TIC / generic-viewer integration:
- ``ImzMLFile.load()`` populates an ``MSImagingExperiment`` from which the
  underlying ``MSExperiment`` is extracted for downstream panels.
- Pseudo-RT (spectrum index) is applied so panels have a monotonic axis.
- MS level is promoted from 0 to 1 so panels that filter ``ms_level == 1``
  (TIC, peak map, rasterization) render correctly.
- ``state.exp`` stores the underlying MSExperiment so existing panels
  (SpectrumPanel, SpectraTablePanel, TICPanel, PeakMapPanel, ExportPanel)
  work without modification.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pyopenms_viewer.core.state import ViewerState

# ---------------------------------------------------------------------------
# Module-level helpers
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


class ImzMLLoader:
    """Loads imzML/ibd file pairs into ViewerState via the OpenMS native parser.

    The .ibd binary must reside alongside the .imzML file.
    pyimzml is NOT used — loading is done via ``oms.ImzMLFile().load()``.

    After a successful load:
    - ``state.exp``        — MSExperiment (all existing panels)
    - ``state.df``         — peak DataFrame (rt = pseudo-RT index)
    - ``state.tic_rt`` /
      ``state.tic_intensity`` — per-spectrum TIC trace
    - ``state.has_imzml``  — True

    Example::

        state = ViewerState()
        loader = ImzMLLoader(state)
        if loader.load_sync("sample.imzML"):
            print(f"Loaded {len(state.exp)} spectra")
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

            msexp = mie.getMSExperiment()
            n_spectra = len(msexp.getSpectra())

            if n_spectra == 0:
                self.state._last_load_error = "No spectra found in imzML file"
                return False

            _prog("Applying pseudo-RT…", 0.15)
            _apply_pseudo_rt(mie)

            _prog("Updating ranges…", 0.30)
            msexp.updateRanges()
            min_mz = msexp.getMinMZ()
            max_mz = msexp.getMaxMZ()

            _prog("Building peak DataFrame…", 0.40)

            spectra = msexp.getSpectra()
            all_rt: list[np.ndarray] = []
            all_mz: list[np.ndarray] = []
            all_int: list[np.ndarray] = []
            spectrum_stats: list[dict] = []

            for si in range(n_spectra):
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
                spectrum_stats.append({
                    "tic": float(np.sum(ints)) if n > 0 else 0.0,
                    "bpi": float(np.max(ints)) if n > 0 else 0.0,
                    "mz_min": float(mzs.min()) if n > 0 else 0.0,
                    "mz_max": float(mzs.max()) if n > 0 else 0.0,
                    "cv": None,
                })

                if si % 500 == 0:
                    _prog(
                        f"Building peak DataFrame… {si + 1}/{n_spectra}",
                        0.40 + 0.35 * (si + 1) / n_spectra,
                    )

            _prog("Finalizing DataFrame…", 0.78)

            if all_rt:
                df = pd.DataFrame({
                    "rt": np.concatenate(all_rt),
                    "mz": np.concatenate(all_mz),
                    "intensity": np.concatenate(all_int),
                })
            else:
                df = pd.DataFrame({"rt": [], "mz": [], "intensity": []})
            df["log_intensity"] = np.log1p(df["intensity"])

            _prog("Extracting spectrum metadata…", 0.85)

            # Clear previous data, then assign the new MSExperiment once so
            # extract_spectrum_data (which iterates state.exp) sees the current
            # data.
            self.state.clear_mzml_data()
            self.state.exp = msexp
            from pyopenms_viewer.loaders.spectrum_extractor import extract_spectrum_data
            spectrum_data = extract_spectrum_data(self.state, spectrum_stats=spectrum_stats)

            _prog("Updating viewer state…", 0.92)

            rt_data_min = 0.0
            rt_data_max = float(n_spectra - 1)
            if len(df) > 0:
                mz_data_min = float(df["mz"].min())
                mz_data_max = float(df["mz"].max())
            elif min_mz > 0 and max_mz > 0:
                mz_data_min, mz_data_max = float(min_mz), float(max_mz)
            else:
                mz_data_min, mz_data_max = 0.0, 1.0

            # TIC trace: one entry per spectrum (pseudo-RT = spectrum index)
            tic_rt = np.arange(n_spectra, dtype=np.float64)
            tic_int = np.array([s["tic"] for s in spectrum_stats], dtype=np.float64)

            # Core data (state.exp already set above for extract_spectrum_data)
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

            print(
                f"Loaded {n_spectra} spectra from {path.name}, "
                f"m/z {mz_data_min:.4f}\u2013{mz_data_max:.4f}"
            )

            _prog("Done", 1.0)
            return True

        except Exception as ex:
            self.state._last_load_error = str(ex)
            return False
