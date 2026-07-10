"""mzML file loading and processing.

This module handles loading mzML files using pyOpenMS and extracting peak data
into the ViewerState. All data is written directly to the state reference.

Two-phase loading:
1. parse() - Blocking pyOpenMS C++ call to load the file
2. process() - Extract peaks, TIC, chromatograms, ion mobility data
"""

import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pyopenms import DriftTimeUnit, MSExperiment, MzMLFile

from pyopenms_viewer.core.state import ViewerState

# Regex to extract CV from filter string (e.g., "cv=-45.00" or "cv=0.00")
_CV_FILTER_PATTERN = re.compile(r"\bcv=(-?\d+(?:\.\d+)?)\b", re.IGNORECASE)


def get_cv_from_spectrum(spec) -> float | None:
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
        filename = Path(filepath).name
        # Ensure filepath is a plain str (not Path) so pyopenms C++ binding
        # receives a native string on all platforms including Windows.
        fp_str = str(filepath)
        try:
            print(f"[PID:{os.getpid()} TID:{threading.get_ident()}] Reading {filename} with MzMLFile (this may take a while)...")
            self.state.exp = MSExperiment()
            MzMLFile().load(fp_str, self.state.exp)
            print(f"[PID:{os.getpid()} TID:{threading.get_ident()}] Loaded {len(self.state.exp)} spectra from {filename}")
            if len(self.state.exp) == 0:
                self.state.last_load_error = f"The file '{filename}' was parsed but contains no spectra."
                return False
            return True
        except Exception as e:
            import traceback as _tb
            details = _tb.format_exc()
            print(f"Error parsing mzML: {e}\n{details}")
            self.state.last_load_error = str(e)
            return False

    def process(
        self,
        filepath: str,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> bool:
        """Process parsed mzML data to extract peaks and create DataFrame.

        This is the second phase of loading - processes spectra with progress updates.
        Uses a SINGLE PASS through all spectra to avoid redundant copies (pyOpenMS
        creates new Python objects for each exp[i] access).

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

            if progress_callback:
                progress_callback("Processing spectra...", 0.05)

            total_spectra = len(self.state.exp)
            print(f"[PID:{os.getpid()} TID:{threading.get_ident()}] Starting processing of parsed data ({total_spectra} spectra)...")
            if total_spectra == 0:
                return False

            # =========================================================
            # SINGLE PASS: Extract everything we need in one iteration
            # This avoids multiple iterations that create redundant copies
            # =========================================================

            # Pre-allocate with estimated size (will trim later)
            # Estimate: average 1000 peaks per MS1 spectrum, ~20% are MS1
            estimated_ms1_peaks = total_spectra * 200
            rts = np.empty(estimated_ms1_peaks, dtype=np.float32)
            mzs = np.empty(estimated_ms1_peaks, dtype=np.float32)
            intensities = np.empty(estimated_ms1_peaks, dtype=np.float32)
            cvs_arr = np.empty(estimated_ms1_peaks, dtype=np.float32)

            # Ion mobility data arrays
            im_mz_list = []
            im_im_list = []
            im_int_list = []
            im_frame_indices_list = []  # All IM-bearing frame indices (any MS level)
            im_frame_ms_levels_list = []  # Parallel MS levels
            im_frame_precursor_mz_list = []  # Parallel precursor target m/z (nan for MS1)
            im_frame_precursor_lower_list = []  # Parallel precursor window lower (nan for MS1)
            im_frame_precursor_upper_list = []  # Parallel precursor window upper (nan for MS1)
            detected_im_name = None
            im_array_names = [
                "ion mobility",
                "inverse reduced ion mobility",
                "drift time",
                "ion mobility drift time",
            ]

            # Counters and collectors
            peak_idx = 0
            total_peaks = 0
            ms1_count = 0
            max_peaks_per_spectrum = 0
            cv_set = set()
            ms_levels = set()

            # TIC data
            tic_rts = []
            tic_intensities = []

            # Spectrum stats cache (for extract_spectrum_data)
            spectrum_stats = []

            # FAIMS per-CV TIC (will populate after we know all CVs)
            cv_tic_data = {}  # cv -> {"rt": [], "int": []}

            for spec_idx in range(total_spectra):
                if progress_callback and spec_idx % 100 == 0:
                    progress = 0.05 + 0.65 * (spec_idx / total_spectra)
                    progress_callback(f"Processing spectra... {spec_idx:,}/{total_spectra:,}", progress)

                spec = self.state.exp[spec_idx]
                rt = spec.getRT()
                ms_level = spec.getMSLevel()
                ms_levels.add(ms_level)

                # Get peaks ONCE per spectrum
                mz_array, int_array = spec.get_peaks()
                n = len(mz_array)
                total_peaks += n

                if n > max_peaks_per_spectrum:
                    max_peaks_per_spectrum = n

                # Compute spectrum stats
                if n > 0:
                    tic_value = float(np.sum(int_array))
                    bpi_value = float(np.max(int_array))
                    mz_min_value = float(mz_array.min())
                    mz_max_value = float(mz_array.max())
                else:
                    tic_value = 0.0
                    bpi_value = 0.0
                    mz_min_value = 0.0
                    mz_max_value = 0.0

                # Get FAIMS CV (for MS1)
                cv = None
                if ms_level == 1:
                    cv = get_cv_from_spectrum(spec)
                    if cv is not None:
                        cv_set.add(cv)

                # Cache stats for extract_spectrum_data
                spectrum_stats.append(
                    {
                        "tic": tic_value,
                        "bpi": bpi_value,
                        "mz_min": mz_min_value,
                        "mz_max": mz_max_value,
                        "cv": cv,
                    }
                )

                # Process MS1 spectra
                if ms_level == 1:
                    ms1_count += 1

                    if n > 0:
                        # Grow arrays if needed
                        if peak_idx + n > len(rts):
                            new_size = max(len(rts) * 2, peak_idx + n)
                            rts = np.resize(rts, new_size)
                            mzs = np.resize(mzs, new_size)
                            intensities = np.resize(intensities, new_size)
                            cvs_arr = np.resize(cvs_arr, new_size)

                        # Store peak data
                        rts[peak_idx : peak_idx + n] = rt
                        mzs[peak_idx : peak_idx + n] = mz_array
                        intensities[peak_idx : peak_idx + n] = int_array
                        if cv is not None:
                            cvs_arr[peak_idx : peak_idx + n] = cv
                        else:
                            cvs_arr[peak_idx : peak_idx + n] = np.nan
                        peak_idx += n

                        # TIC data
                        tic_rts.append(rt)
                        tic_intensities.append(tic_value)

                        # Per-CV TIC (store for now, organize later)
                        if cv is not None:
                            if cv not in cv_tic_data:
                                cv_tic_data[cv] = {"rt": [], "int": []}
                            cv_tic_data[cv]["rt"].append(rt)
                            cv_tic_data[cv]["int"].append(tic_value)

                # Ion mobility detection and extraction (all MS levels)
                if n > 0:
                    float_arrays = spec.getFloatDataArrays()

                    # Ion mobility detection (all MS levels)
                    if detected_im_name is None:
                        for fda in float_arrays:
                            name = fda.getName().lower() if fda.getName() else ""
                            for im_name in im_array_names:
                                if im_name in name:
                                    detected_im_name = fda.getName()
                                    break
                            if detected_im_name:
                                break

                    # Extract IM data if available (all MS levels)
                    if detected_im_name is not None:
                        for fda in float_arrays:
                            if fda.getName() == detected_im_name:
                                im_array = np.array(fda.get_data(), dtype=np.float32)
                                if len(im_array) == n:
                                    im_mz_list.append(mz_array)
                                    im_im_list.append(im_array)
                                    im_int_list.append(int_array)
                                    im_frame_indices_list.append(spec_idx)
                                    im_frame_ms_levels_list.append(ms_level)
                                    if ms_level >= 2:
                                        # For MS3+ we record precs[0] (first-stage isolation); MS3 IM is rare and out of scope.
                                        precs = spec.getPrecursors()
                                        if precs:
                                            prec = precs[0]
                                            pmz = float(prec.getMZ())
                                            lo_off = float(prec.getIsolationWindowLowerOffset())
                                            hi_off = float(prec.getIsolationWindowUpperOffset())
                                            im_frame_precursor_mz_list.append(pmz)
                                            im_frame_precursor_lower_list.append(pmz - lo_off)
                                            im_frame_precursor_upper_list.append(pmz + hi_off)
                                        else:
                                            im_frame_precursor_mz_list.append(np.nan)
                                            im_frame_precursor_lower_list.append(np.nan)
                                            im_frame_precursor_upper_list.append(np.nan)
                                    else:
                                        im_frame_precursor_mz_list.append(np.nan)
                                        im_frame_precursor_lower_list.append(np.nan)
                                        im_frame_precursor_upper_list.append(np.nan)
                                break

            # =========================================================
            # Post-processing
            # =========================================================

            if total_peaks == 0:
                return False

            self.state.total_peaks = total_peaks

            if progress_callback:
                progress_callback("Building data structures...", 0.72)

            # Trim peak arrays
            rts = rts[:peak_idx]
            mzs = mzs[:peak_idx]
            intensities = intensities[:peak_idx]
            cvs_arr = cvs_arr[:peak_idx]

            # Determine FAIMS status
            self.state.has_faims = len(cv_set) > 1
            self.state.faims_cvs = sorted(cv_set) if self.state.has_faims else []

            # Determine TIC source
            if ms1_count > 0:
                self.state.tic_source = "MS1 TIC"
            else:
                tic_ms_level = min(lv for lv in ms_levels if lv > 1) if ms_levels else 2
                self.state.tic_source = f"MS{tic_ms_level} BPC"

            # Store TIC data (sorted by RT)
            if tic_rts:
                tic_rt_arr = np.array(tic_rts, dtype=np.float32)
                tic_int_arr = np.array(tic_intensities, dtype=np.float32)
                sort_idx = np.argsort(tic_rt_arr)
                self.state.tic_rt = tic_rt_arr[sort_idx]
                self.state.tic_intensity = tic_int_arr[sort_idx]
            else:
                self.state.tic_rt = np.array([], dtype=np.float32)
                self.state.tic_intensity = np.array([], dtype=np.float32)

            # Store per-CV TIC data
            self.state.faims_tic = {}
            for cv in self.state.faims_cvs:
                if cv in cv_tic_data:
                    cv_rt = np.array(cv_tic_data[cv]["rt"], dtype=np.float32)
                    cv_int = np.array(cv_tic_data[cv]["int"], dtype=np.float32)
                    cv_sort_idx = np.argsort(cv_rt)
                    self.state.faims_tic[cv] = (cv_rt[cv_sort_idx], cv_int[cv_sort_idx])

            if progress_callback:
                progress_callback("Extracting chromatograms...", 0.75)

            # Extract chromatograms (iterates over chromatograms, not spectra)
            from pyopenms_viewer.loaders.chromatogram_loader import (
                extract_chromatograms,
            )

            extract_chromatograms(self.state)

            if progress_callback:
                progress_callback("Processing ion mobility data...", 0.77)

            # Process ion mobility data (already extracted in main loop)
            self._process_ion_mobility_data(
                im_mz_list,
                im_im_list,
                im_int_list,
                detected_im_name,
                filepath,
                im_frame_indices_list,
                im_frame_ms_levels_list,
                im_frame_precursor_mz_list,
                im_frame_precursor_lower_list,
                im_frame_precursor_upper_list,
            )

            if progress_callback:
                progress_callback("Extracting spectrum metadata...", 0.8)

            # Extract spectrum metadata (using cached stats)
            from pyopenms_viewer.loaders.spectrum_extractor import extract_spectrum_data

            self.state.spectrum_data = extract_spectrum_data(self.state, spectrum_stats)

            if progress_callback:
                progress_callback("Finalizing...", 0.95)
            print(f"[PID:{os.getpid()} TID:{threading.get_ident()}] Finished processing data...")

            # Compute bounds from MS1 peaks only (rts/mzs arrays contain only MS1 data)
            if peak_idx > 0:
                self.state.rt_min = float(rts.min())
                self.state.rt_max = float(rts.max())
                self.state.mz_min = float(mzs.min())
                self.state.mz_max = float(mzs.max())
            else:
                self.state.exp.updateRanges()
                self.state.rt_min = float(self.state.exp.getMinRT())
                self.state.rt_max = float(self.state.exp.getMaxRT())
                self.state.mz_min = float(self.state.exp.getMinMZ())
                self.state.mz_max = float(self.state.exp.getMaxMZ())

            # No DataFrame needed — rasterizeRTMZ renders directly
            self.state.df = None

            # Build per-CV MSExperiment objects for FAIMS rasterization
            # Each per-CV experiment contains only MS1 spectra for that CV,
            # allowing rasterizeRTMZ to work natively per-CV.
            self.state.faims_data = {}
            self.state.faims_experiments = {}
            if self.state.has_faims:
                for cv in self.state.faims_cvs:
                    cv_exp = MSExperiment()
                    for i, stats in enumerate(spectrum_stats):
                        if stats["cv"] is not None and abs(stats["cv"] - cv) < 0.01:
                            cv_exp.addSpectrum(self.state.exp[i])
                    cv_exp.updateRanges()
                    self.state.faims_experiments[cv] = cv_exp

            # Ensure valid ranges (apply to both paths)
            if self.state.rt_max <= self.state.rt_min:
                self.state.rt_max = self.state.rt_min + 1.0
            if self.state.mz_max <= self.state.mz_min:
                self.state.mz_max = self.state.mz_min + 1.0

            # Set initial view to full extent
            self.state.view_rt_min = self.state.rt_min
            self.state.view_rt_max = self.state.rt_max
            self.state.view_mz_min = self.state.mz_min
            self.state.view_mz_max = self.state.mz_max

            # Auto-enable downsampling for high-res spectra
            if max_peaks_per_spectrum > 10000:
                self.state.peakmap_downsampling = True

            self.state.current_file = filepath

            # Clear any stale MSI state from a previously-loaded imzML so the
            # Ion Image panel does not keep rendering the old experiment beside
            # this plain-mzML data (MzMLLoader does not go through
            # clear_mzml_data). msi_load_token is intentionally left untouched.
            self.state.has_imzml = False
            self.state.msi_experiment = None

            # Invalidate minimap cache when new file is loaded
            self.state.invalidate_minimap_cache()

            return True

        except Exception as e:
            import traceback

            print(f"Error processing mzML: {e}")
            traceback.print_exc()
            return False

    def _process_ion_mobility_data(
        self,
        im_mz_list: list,
        im_im_list: list,
        im_int_list: list,
        detected_im_name: str | None,
        filepath: str,
        im_frame_indices: list | None = None,
        im_frame_ms_levels: list | None = None,
        im_frame_precursor_mz: list | None = None,
        im_frame_precursor_lower: list | None = None,
        im_frame_precursor_upper: list | None = None,
    ) -> None:
        """Process pre-extracted ion mobility data and populate state."""
        if not im_mz_list or detected_im_name is None:
            self.state.has_ion_mobility = False
            self.state.im_df = None
            return

        name_lower = detected_im_name.lower()
        if "inverse" in name_lower or "1/k0" in name_lower:
            self.state.im_type = "inverse_k0"
            self.state.im_unit = "Vs/cm\u00b2"
        elif "drift" in name_lower:
            self.state.im_type = "drift_time"
            self.state.im_unit = "ms"
        else:
            self.state.im_type = "ion_mobility"
            self.state.im_unit = ""

        mz_concat = np.concatenate(im_mz_list)
        im_concat = np.concatenate(im_im_list)

        im_mz_min = float(mz_concat.min())
        im_mz_max = float(mz_concat.max())
        im_min_val = float(im_concat.min())
        im_max_val = float(im_concat.max())

        if im_frame_indices:
            n_frames = len(im_frame_indices)
            # Backward-compat defaults when caller omits new lists (e.g., tests).
            if im_frame_ms_levels is None:
                im_frame_ms_levels = [1] * n_frames
            if im_frame_precursor_mz is None:
                im_frame_precursor_mz = [float("nan")] * n_frames
            if im_frame_precursor_lower is None:
                im_frame_precursor_lower = [float("nan")] * n_frames
            if im_frame_precursor_upper is None:
                im_frame_precursor_upper = [float("nan")] * n_frames

            frame_rts = np.array(
                [self.state.exp[idx].getRT() for idx in im_frame_indices], dtype=np.float64
            )
            # Sort by (rt, spec_idx) for stable, deterministic tie-breaking.
            order = sorted(range(len(im_frame_indices)),
                           key=lambda i: (frame_rts[i], im_frame_indices[i]))

            sorted_indices = [im_frame_indices[i] for i in order]
            sorted_rts = frame_rts[order]
            sorted_levels = np.array(
                [im_frame_ms_levels[i] for i in order], dtype=np.int32
            )
            sorted_pmz = np.array(
                [im_frame_precursor_mz[i] for i in order], dtype=np.float64
            )
            sorted_plo = np.array(
                [im_frame_precursor_lower[i] for i in order], dtype=np.float64
            )
            sorted_phi = np.array(
                [im_frame_precursor_upper[i] for i in order], dtype=np.float64
            )

            self.state.im_frame_indices = sorted_indices
            self.state.im_frame_rts = sorted_rts
            self.state.im_frame_ms_levels = sorted_levels
            self.state.im_frame_precursor_mz = sorted_pmz
            self.state.im_frame_precursor_lower = sorted_plo
            self.state.im_frame_precursor_upper = sorted_phi
            self.state.im_frame_position_by_index = {
                spec_idx: pos for pos, spec_idx in enumerate(sorted_indices)
            }

            ms1_mask = sorted_levels == 1
            self.state.ms1_im_frame_indices = [
                sorted_indices[i] for i in range(len(sorted_indices)) if ms1_mask[i]
            ]
            self.state.ms1_im_frame_rts = sorted_rts[ms1_mask]

            self.state.im_min = im_min_val
            self.state.im_max = im_max_val
            self.state.im_df = None

            # Default selection: prefer the first MS1 IM frame; fall back to first IM frame.
            if self.state.ms1_im_frame_indices:
                self.state.selected_im_frame_idx = self.state.ms1_im_frame_indices[0]
            else:
                self.state.selected_im_frame_idx = sorted_indices[0]

        # Ensure valid IM range
        if self.state.im_max <= self.state.im_min:
            self.state.im_max = self.state.im_min + 1.0
        self.state.view_im_min = self.state.im_min
        self.state.view_im_max = self.state.im_max

        # Update mz bounds
        if self.state.mz_min == 0 or im_mz_min < self.state.mz_min:
            self.state.mz_min = im_mz_min
        if self.state.mz_max == 0 or im_mz_max > self.state.mz_max:
            self.state.mz_max = im_mz_max
        if self.state.view_mz_min is None or self.state.view_mz_min < self.state.mz_min:
            self.state.view_mz_min = self.state.mz_min
        if self.state.view_mz_max is None or self.state.view_mz_max > self.state.mz_max:
            self.state.view_mz_max = self.state.mz_max

        self.state.has_ion_mobility = True

    def load_sync(self, filepath: str, progress_callback: Callable[[str, float], None] | None = None) -> bool:
        """Load mzML file synchronously (for background thread).

        Convenience method that calls both parse and process phases.

        Args:
            filepath: Path to the mzML file

        Returns:
            True if successful
        """
        # Normalize filepath
        try:
            fp = str(Path(filepath).resolve())
        except Exception:
            fp = str(filepath)

        # Use a thread-safe per-state event map so concurrent background
        # threads wait for a single load instead of starting duplicate parses.
        state = self.state
        # First, acquire lock and check/create event
        with state._loading_events_lock:
            existing = state._loading_events.get(fp)
            if existing is not None:
                # Another thread is loading this file; wait for it to finish
                wait_event = existing
                should_load = False
            else:
                wait_event = threading.Event()
                state._loading_events[fp] = wait_event
                should_load = True

        if not should_load:
            # Wait for the other loader to finish (avoid busy spin; timeout occasionally)
            wait_event.wait(timeout=600)
            # After wait, determine if data is present
            if state.current_file and state.current_file == fp and (state.df is not None or state.data_manager is not None):
                return True
            # If waited and still nothing, fall through to attempt load

        try:
            # Pass the progress_callback through to `process` so it can update UI via shared state
            if not self.parse(filepath):
                return False
            return self.process(filepath, progress_callback=progress_callback)
        finally:
            # Signal and clean up event
            with state._loading_events_lock:
                ev = state._loading_events.pop(fp, None)
                if ev is not None:
                    try:
                        ev.set()
                    except Exception:
                        pass
