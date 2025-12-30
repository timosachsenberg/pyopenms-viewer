"""mzML file loading and processing.

This module handles loading mzML files using pyOpenMS and extracting peak data
into the ViewerState. All data is written directly to the state reference.

Two-phase loading:
1. parse() - Blocking pyOpenMS C++ call to load the file
2. process() - Extract peaks, TIC, chromatograms, ion mobility data
"""

import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pyopenms import DriftTimeUnit, MSExperiment, MzMLFile

from pyopenms_viewer.core.state import ViewerState

# Regex to extract CV from filter string (e.g., "cv=-45.00" or "cv=0.00")
_CV_FILTER_PATTERN = re.compile(r"\bcv=(-?\d+(?:\.\d+)?)\b", re.IGNORECASE)


def get_cv_from_spectrum(spec) -> Optional[float]:
    """Extract FAIMS compensation voltage from spectrum metadata.

    Uses getDriftTimeUnit() to check if spectrum has FAIMS CV data, then
    retrieves the value via getDriftTime(). Falls back to metadata keys
    and filter string parsing for compatibility.

    Args:
        spec: MSSpectrum object

    Returns:
        Compensation voltage value, or None if not found

    Note: pyOpenMS 3.5 includes performance improvements (7-25% faster mzML loading
    via SIMD ASCII conversion, 20-40% faster ion mobility data loading).
    """
    # Primary method: use DriftTimeUnit (most reliable for properly annotated mzML)
    try:
        if spec.getDriftTimeUnit() == DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE:
            cv = spec.getDriftTime()
            if cv != 0.0 or spec.getDriftTime() == 0.0:  # 0.0 can be valid CV
                return float(cv)
    except Exception:
        pass

    # Fallback: Try common CV metadata names
    cv_names = [
        "FAIMS compensation voltage",
        "ion mobility drift time",
        "MS:1001581",  # CV accession for FAIMS CV
    ]
    for name in cv_names:
        if spec.metaValueExists(name):
            try:
                return float(spec.getMetaValue(name))
            except (ValueError, TypeError):
                pass

    # Fallback: Check in acquisition info
    try:
        acq = spec.getAcquisitionInfo()
        if acq:
            for a in acq:
                for name in cv_names:
                    if a.metaValueExists(name):
                        return float(a.getMetaValue(name))
    except Exception:
        pass

    # Fallback: Parse CV from filter string (Thermo format: "cv=-45.00")
    if spec.metaValueExists("filter string"):
        try:
            filter_str = spec.getMetaValue("filter string")
            if isinstance(filter_str, bytes):
                filter_str = filter_str.decode()
            match = _CV_FILTER_PATTERN.search(filter_str)
            if match:
                return float(match.group(1))
        except Exception:
            pass

    return None


class MzMLLoader:
    """Loads and processes mzML files.

    Two-phase loading:
    1. parse() - Blocking pyOpenMS C++ call
    2. process() - Extract peaks and build DataFrame with progress callbacks

    All data is written directly to the ViewerState reference.

    Example:
        state = ViewerState()
        loader = MzMLLoader(state)
        if loader.load_sync("data.mzML"):
            print(f"Loaded {len(state.df)} peaks")
    """

    def __init__(self, state: ViewerState):
        """Initialize loader with state reference.

        Args:
            state: ViewerState instance to populate with data
        """
        self.state = state

    def parse(self, filepath: str) -> bool:
        """Parse mzML file using pyOpenMS (blocking C++ call).

        This is the first phase of loading - just parses the file.

        Args:
            filepath: Path to the mzML file

        Returns:
            True if successful and file has spectra
        """
        try:
            filename = Path(filepath).name
            print(f"Reading {filename} with MzMLFile (this may take a while)...")
            self.state.exp = MSExperiment()
            MzMLFile().load(filepath, self.state.exp)
            print(f"Loaded {len(self.state.exp)} spectra from {filename}")
            return len(self.state.exp) > 0
        except Exception as e:
            print(f"Error parsing mzML: {e}")
            return False

    def process(
        self,
        filepath: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Process parsed mzML data to extract peaks and create DataFrame.

        This is the second phase of loading - processes spectra with progress updates.
        All data is written directly to self.state.

        Args:
            filepath: Path to the mzML file (for storing reference)
            progress_callback: Optional callback(message, progress) for progress updates

        Returns:
            True if successful
        """
        try:
            if self.state.exp is None:
                return False

            total_peaks = sum(len(spec) for spec in self.state.exp)

            if total_peaks == 0:
                return False

            # First pass: detect FAIMS CVs
            if progress_callback:
                progress_callback("Detecting FAIMS CVs...", 0.05)

            cv_set = set()
            for spec in self.state.exp:
                if spec.getMSLevel() == 1:
                    cv = get_cv_from_spectrum(spec)
                    if cv is not None:
                        cv_set.add(cv)

            self.state.has_faims = len(cv_set) > 1
            self.state.faims_cvs = sorted(cv_set) if self.state.has_faims else []

            # Data structures for peak extraction
            rts = np.empty(total_peaks, dtype=np.float32)
            mzs = np.empty(total_peaks, dtype=np.float32)
            intensities = np.empty(total_peaks, dtype=np.float32)
            cvs = np.empty(total_peaks, dtype=np.float32) if self.state.has_faims else None

            # TIC computation
            tic_rts = []
            tic_intensities = []
            faims_tic_data = {cv: {"rt": [], "int": []} for cv in self.state.faims_cvs} if self.state.has_faims else {}

            if progress_callback:
                progress_callback("Extracting peaks...", 0.1)

            idx = 0
            ms1_count = 0
            total_ms1 = sum(1 for spec in self.state.exp if spec.getMSLevel() == 1)

            # Determine TIC source: MS1 TIC or fallback to MS2+ BPC
            if total_ms1 > 0:
                tic_ms_level = 1
                self.state.tic_source = "MS1 TIC"
            else:
                ms_levels = {spec.getMSLevel() for spec in self.state.exp}
                tic_ms_level = min(lv for lv in ms_levels if lv > 1) if ms_levels else 2
                self.state.tic_source = f"MS{tic_ms_level} BPC"

            total_tic_spectra = sum(1 for spec in self.state.exp if spec.getMSLevel() == tic_ms_level)

            for spec in self.state.exp:
                if spec.getMSLevel() != tic_ms_level:
                    if tic_ms_level == 1 or spec.getMSLevel() != 1:
                        continue

                ms1_count += 1
                if progress_callback and ms1_count % 100 == 0:
                    progress = 0.1 + 0.6 * (ms1_count / max(total_tic_spectra, 1))
                    progress_callback(f"Extracting peaks... {ms1_count:,}/{total_tic_spectra:,}", progress)

                rt = spec.getRT()
                mz_array, int_array = spec.get_peaks()
                n = len(mz_array)

                cv = get_cv_from_spectrum(spec) if self.state.has_faims else None

                if n > 0:
                    # Only add to peak map DataFrame if MS1
                    if spec.getMSLevel() == 1:
                        rts[idx : idx + n] = rt
                        mzs[idx : idx + n] = mz_array
                        intensities[idx : idx + n] = int_array
                        if self.state.has_faims and cv is not None:
                            cvs[idx : idx + n] = cv
                        idx += n

                    # TIC/BPC calculation
                    if tic_ms_level == 1:
                        tic_value = float(np.sum(int_array))
                    else:
                        tic_value = float(np.max(int_array))

                    tic_rts.append(rt)
                    tic_intensities.append(tic_value)

                    # Per-CV TIC
                    if self.state.has_faims and cv is not None:
                        faims_tic_data[cv]["rt"].append(rt)
                        faims_tic_data[cv]["int"].append(tic_value)

            # Trim arrays
            rts = rts[:idx]
            mzs = mzs[:idx]
            intensities = intensities[:idx]
            if self.state.has_faims:
                cvs = cvs[:idx]

            if progress_callback:
                progress_callback("Building TIC...", 0.75)

            # Store TIC data (sorted by RT)
            tic_rt_arr = np.array(tic_rts, dtype=np.float32)
            tic_int_arr = np.array(tic_intensities, dtype=np.float32)
            sort_idx = np.argsort(tic_rt_arr)
            self.state.tic_rt = tic_rt_arr[sort_idx]
            self.state.tic_intensity = tic_int_arr[sort_idx]

            # Store per-CV TIC data
            self.state.faims_tic = {}
            for cv in self.state.faims_cvs:
                cv_rt = np.array(faims_tic_data[cv]["rt"], dtype=np.float32)
                cv_int = np.array(faims_tic_data[cv]["int"], dtype=np.float32)
                cv_sort_idx = np.argsort(cv_rt)
                self.state.faims_tic[cv] = (cv_rt[cv_sort_idx], cv_int[cv_sort_idx])

            if progress_callback:
                progress_callback("Extracting chromatograms...", 0.77)

            # Extract chromatograms
            from pyopenms_viewer.loaders.chromatogram_loader import extract_chromatograms

            extract_chromatograms(self.state)

            if progress_callback:
                progress_callback("Extracting ion mobility data...", 0.78)

            # Extract ion mobility data
            from pyopenms_viewer.loaders.ion_mobility_loader import extract_ion_mobility_data

            extract_ion_mobility_data(self.state)

            if progress_callback:
                progress_callback("Extracting spectrum metadata...", 0.8)

            # Extract spectrum metadata
            from pyopenms_viewer.loaders.spectrum_extractor import extract_spectrum_data

            self.state.spectrum_data = extract_spectrum_data(self.state)

            if progress_callback:
                progress_callback("Creating DataFrame...", 0.85)

            # Create main DataFrame
            df = pd.DataFrame({"rt": rts, "mz": mzs, "intensity": intensities})
            if self.state.has_faims:
                df["cv"] = cvs
            df["log_intensity"] = np.log1p(df["intensity"])

            if progress_callback:
                progress_callback("Registering with data manager...", 0.88)

            # Register DataFrame with data manager (handles both in-memory and out-of-core)
            if self.state.data_manager is not None:
                # data_manager.register_peaks returns DataFrame for in-memory, None for out-of-core
                self.state.df = self.state.data_manager.register_peaks(df, filepath)

                # Get bounds from data manager (works for both modes)
                bounds = self.state.data_manager.get_bounds()
                self.state.rt_min = bounds["rt_min"]
                self.state.rt_max = bounds["rt_max"]
                self.state.mz_min = bounds["mz_min"]
                self.state.mz_max = bounds["mz_max"]
            else:
                # Legacy: no data manager, keep DataFrame in state
                self.state.df = df

            if progress_callback:
                progress_callback("Finalizing...", 0.95)

            # Create per-CV DataFrames for FAIMS view (only in-memory mode)
            self.state.faims_data = {}
            if self.state.has_faims and self.state.df is not None:
                for cv in self.state.faims_cvs:
                    cv_df = self.state.df[self.state.df["cv"] == cv].copy()
                    self.state.faims_data[cv] = cv_df

            # Set bounds from peak data (fallback if data_manager not used)
            if self.state.data_manager is None and self.state.df is not None and len(self.state.df) > 0:
                self.state.rt_min = float(self.state.df["rt"].min())
                self.state.rt_max = float(self.state.df["rt"].max())
                self.state.mz_min = float(self.state.df["mz"].min())
                self.state.mz_max = float(self.state.df["mz"].max())
            elif self.state.data_manager is None:
                # Fall back to IM data or spectrum metadata
                if self.state.has_ion_mobility and self.state.im_df is not None and len(self.state.im_df) > 0:
                    self.state.mz_min = float(self.state.im_df["mz"].min())
                    self.state.mz_max = float(self.state.im_df["mz"].max())
                if self.state.spectrum_data:
                    rts_meta = [
                        s["rt"] for s in self.state.spectrum_data if isinstance(s["rt"], (int, float)) and s["rt"] > 0
                    ]
                    if rts_meta:
                        self.state.rt_min = min(rts_meta)
                        self.state.rt_max = max(rts_meta)

            # Ensure valid ranges
            if self.state.rt_max <= self.state.rt_min:
                self.state.rt_max = self.state.rt_min + 1.0
            if self.state.mz_max <= self.state.mz_min:
                self.state.mz_max = self.state.mz_min + 1.0

            # Set initial view to full extent
            self.state.view_rt_min = self.state.rt_min
            self.state.view_rt_max = self.state.rt_max
            self.state.view_mz_min = self.state.mz_min
            self.state.view_mz_max = self.state.mz_max

            # Auto-enable downsampling if any spectrum has more than 10000 peaks
            max_peaks_per_spectrum = max((len(spec) for spec in self.state.exp), default=0)
            if max_peaks_per_spectrum > 10000:
                self.state.peakmap_downsampling = True

            self.state.current_file = filepath
            return True

        except Exception as e:
            import traceback

            print(f"Error processing mzML: {e}")
            traceback.print_exc()
            return False

    def load_sync(self, filepath: str) -> bool:
        """Load mzML file synchronously (for background thread).

        Convenience method that calls both parse and process phases.

        Args:
            filepath: Path to the mzML file

        Returns:
            True if successful
        """
        if not self.parse(filepath):
            return False
        return self.process(filepath)
