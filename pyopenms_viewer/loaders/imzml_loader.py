"""imzML file loading and processing.

This module handles loading imzML/ibd file pairs using pyimzml and populating
the ViewerState so that all existing panels (TIC, peak map, spectrum browser,
spectra table, export) work without modification.

Strategy:
- Build an MSExperiment where each pixel (x, y, z) becomes one MSSpectrum
  with RT set to pixel-X.  This lets spectrum_panel, spectra_table, etc.
  operate exactly as for regular mzML.
- Build a peak DataFrame (rt=pixel-X, mz, intensity) for the peak-map renderer.
- Compute a per-pixel-X TIC for the TIC panel.
- Force rt_in_minutes=False so pixel coordinates are not divided by 60.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pyopenms_viewer.core.state import ViewerState


class ImzMLLoader:
    """Loads imzML/ibd file pairs and populates ViewerState.

    The paired .ibd file must reside in the same directory as the .imzML file.

    Example::

        state = ViewerState()
        loader = ImzMLLoader(state)
        if loader.load_sync("sample.imzML"):
            print(f"Loaded {len(state.df):,} peaks from {len(state.exp)} pixels")
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
            progress_cb: Optional ``(message, fraction)`` callback for progress
                         reporting (fraction in ``[0.0, 1.0]``).

        Returns:
            ``True`` on success, ``False`` on failure.
            On failure ``state._last_load_error`` is set to an error string.
        """
        try:
            from pyimzml.ImzMLParser import ImzMLParser
        except ImportError:
            self.state._last_load_error = (
                "pyimzml is not installed. Install with: pip install pyimzml"
            )
            return False

        def _prog(msg: str, frac: float) -> None:
            if progress_cb:
                try:
                    progress_cb(msg, frac)
                except Exception:
                    pass

        try:
            path = Path(filepath)
            _prog("Opening imzML file…", 0.0)
            parser = ImzMLParser(str(path))

            n_spectra = len(parser.coordinates)
            if n_spectra == 0:
                self.state._last_load_error = "No spectra found in imzML file"
                return False

            from pyopenms import MSExperiment, MSSpectrum

            exp = MSExperiment()

            all_rt: list[float] = []
            all_mz: list[np.ndarray] = []
            all_int: list[np.ndarray] = []

            # Accumulate per-x TIC (summed over all y, z at the same x)
            tic_by_x: dict[int, float] = {}

            spectrum_stats: list[dict] = []

            _prog("Loading pixel spectra…", 0.05)
            for i, (x, y, z) in enumerate(parser.coordinates):
                mz_arr, int_arr = parser.getspectrum(i)
                mz_arr = np.asarray(mz_arr, dtype=np.float64)
                int_arr = np.asarray(int_arr, dtype=np.float64)

                rt = float(x)  # Use pixel X as the "retention time" axis

                # Build MSSpectrum so existing panels can access raw data
                spec = MSSpectrum()
                spec.setRT(rt)
                spec.setMSLevel(1)
                spec.set_peaks((mz_arr, int_arr))
                spec.setNativeID(f"pixel x={x} y={y} z={z}")
                exp.addSpectrum(spec)

                n = len(mz_arr)
                if n > 0:
                    all_rt.append(np.full(n, rt, dtype=np.float32))
                    all_mz.append(mz_arr.astype(np.float32))
                    all_int.append(int_arr.astype(np.float32))

                tic = float(np.sum(int_arr)) if n > 0 else 0.0
                bpi = float(np.max(int_arr)) if n > 0 else 0.0
                mz_min_v = float(mz_arr.min()) if n > 0 else 0.0
                mz_max_v = float(mz_arr.max()) if n > 0 else 0.0

                tic_by_x[x] = tic_by_x.get(x, 0.0) + tic

                spectrum_stats.append(
                    {
                        "tic": tic,
                        "bpi": bpi,
                        "mz_min": mz_min_v,
                        "mz_max": mz_max_v,
                        "cv": None,
                    }
                )

                if i % 200 == 0:
                    _prog(
                        f"Loading pixel spectra… {i + 1}/{n_spectra}",
                        0.05 + 0.65 * (i + 1) / n_spectra,
                    )

            _prog("Building peak DataFrame…", 0.72)

            # Ensure pyOpenMS knows the RT/mz extents (needed by rasterizeRTMZ)
            exp.updateRanges()

            if all_rt:
                rt_col = np.concatenate(all_rt)
                mz_col = np.concatenate(all_mz)
                int_col = np.concatenate(all_int)
            else:
                rt_col = np.array([], dtype=np.float32)
                mz_col = np.array([], dtype=np.float32)
                int_col = np.array([], dtype=np.float32)

            df = pd.DataFrame(
                {
                    "rt": rt_col,
                    "mz": mz_col,
                    "intensity": int_col,
                }
            )
            df["log_intensity"] = np.log1p(df["intensity"])


            _prog("Extracting spectrum metadata…", 0.85)
            from pyopenms_viewer.loaders.spectrum_extractor import extract_spectrum_data

            # Temporarily assign exp so extract_spectrum_data can iterate it
            self.state.exp = exp
            spectrum_data = extract_spectrum_data(self.state, spectrum_stats=spectrum_stats)

            _prog("Updating viewer state…", 0.92)

            coords_arr = np.array(parser.coordinates)
            x_vals = coords_arr[:, 0].astype(float)
            rt_data_min = float(x_vals.min())
            rt_data_max = float(x_vals.max())

            if len(df) > 0:
                mz_data_min = float(df["mz"].min())
                mz_data_max = float(df["mz"].max())
            else:
                mz_data_min, mz_data_max = 0.0, 1.0

            # TIC: sort by x, one entry per unique x (sum of all y/z pixels)
            sorted_x = sorted(tic_by_x.keys())
            tic_rt = np.array(sorted_x, dtype=np.float64)
            tic_int = np.array([tic_by_x[x] for x in sorted_x], dtype=np.float64)

            # Clear any previous mzML data first
            self.state.clear_mzml_data()

            # Core data
            self.state.exp = exp
            self.state.df = df
            self.state.total_peaks = len(df)
            self.state.spectrum_data = spectrum_data
            self.state.current_file = str(path)

            # Bounds
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
            # Pixel coordinates must not be divided by 60 (not time)
            self.state.rt_in_minutes = False

            _prog("Done", 1.0)
            return True

        except Exception as ex:
            self.state._last_load_error = str(ex)
            return False
