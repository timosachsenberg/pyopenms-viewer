"""Vendor file loading using AlphaRaw.

This module handles loading proprietary vendor files (Thermo .raw, Bruker .d, Sciex .wiff)
using the alpharaw library and converting them to the format expected by pyopenms-viewer.

Two-phase loading:
1. parse() - Load vendor file using AlphaRaw
2. process() - Extract peaks, TIC, and metadata into ViewerState
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from pyopenms_viewer.core.state import ViewerState

# Import AlphaRaw with fallback for better error messages
try:
    import alpharaw
    from alpharaw.sciex import SciexWiffData
    from alpharaw.thermo import ThermoRawData
    # BrukerDDAData not available in AlphaRaw 0.5.0
    BrukerDDAData = None
    ALPHARAW_AVAILABLE = True
    print("[AlphaRaw] AlphaRaw loaded successfully")
except (ImportError, Exception) as e:
    ALPHARAW_AVAILABLE = False
    alpharaw = None
    ThermoRawData = None
    SciexWiffData = None
    BrukerDDAData = None
    print(f"[AlphaRaw] Error: AlphaRaw not available: {e}")


class _VendorPrecursor:
    """Minimal shim matching pyOpenMS Precursor.getMZ() / getCharge() interface."""

    def __init__(self, mz: float, charge: int):
        self._mz = mz
        self._charge = charge

    def getMZ(self) -> float:
        return self._mz

    def getCharge(self) -> int:
        return self._charge


class VendorSpectrumAdapter:
    """Wraps an AlphaRaw spectrum so it looks like a pyOpenMS MSSpectrum.

    Implements only the subset of the pyOpenMS MSSpectrum interface that is
    used by SpectrumPanel, TICPanel, and PeakMapPanel:
        get_peaks()      -> (mz_array, intensity_array)
        getRT()          -> float  (seconds)
        getMSLevel()     -> int
        getPrecursors()  -> list[_VendorPrecursor]

    Args:
        raw_reader: AlphaRaw reader object (has .spectrum_df and .get_peaks())
        spec_idx: Row index into spectrum_df (and argument to get_peaks())
    """

    def __init__(self, raw_reader: object, spec_idx: int):
        self._reader = raw_reader
        self._spec_idx = spec_idx
        row = raw_reader.spectrum_df.iloc[spec_idx]
        self._rt = float(row["rt"]) * 60.0  # minutes → seconds
        self._ms_level = int(row.get("ms_level", 1))
        # Build precursor list for MS2
        self._precursors: list[_VendorPrecursor] = []
        if self._ms_level == 2:
            raw_mz = float(row.get("precursor_mz", -1.0))
            if raw_mz > 0.0:
                raw_z = int(row.get("precursor_charge", 0))
                charge = raw_z if raw_z > 0 else 0
                self._precursors.append(_VendorPrecursor(raw_mz, charge))

    def get_peaks(self) -> tuple[np.ndarray, np.ndarray]:  
        return self._reader.get_peaks(self._spec_idx)

    def getRT(self) -> float:  
        return self._rt

    def getMSLevel(self) -> int:  # noqa: N802
        return self._ms_level

    def getPrecursors(self) -> list[_VendorPrecursor]:  # noqa: N802
        return self._precursors


class AlphaRawLoader:
    """Load vendor MS files using AlphaRaw.

    Supports:
    - Thermo RAW files (.raw)
    - Bruker files (.d folders)
    - Sciex WIFF files (.wiff)
    """

    def __init__(self, state: ViewerState):
        """Initialize loader with state reference.

        Args:
            state: ViewerState to populate with loaded data
        """
        self.state = state
        self.raw_reader: Optional[object] = None  # AlphaRaw reader object
        self.filepath: Optional[str] = None

    def load_sync(self, filepath: str, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Load vendor file synchronously.

        Args:
            filepath: Path to vendor file (.raw, .d, .wiff)
            progress_callback: Optional callback for progress updates

        Returns:
            True if loading succeeded, False otherwise
        """
        if not ALPHARAW_AVAILABLE:
            raise ImportError(
                "AlphaRaw is not installed. Install it with: pip install alpharaw\n"
                "Note: AlphaRaw requires platform-specific dependencies for vendor file support."
            )

        self.filepath = filepath

        try:
            if progress_callback:
                progress_callback("Opening vendor file with AlphaRaw...")

            # Parse vendor file
            self.raw_reader = self._parse(filepath)

            if progress_callback:
                progress_callback("Extracting MS data...")

            # Process and populate state
            self._process(progress_callback)

            return True

        except Exception as e:
            print(f"Error loading vendor file with AlphaRaw: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _parse(self, filepath: str) -> object:
        """Load vendor file using AlphaRaw.

        Args:
            filepath: Path to vendor file

        Returns:
            AlphaRaw reader object with loaded data
        """
        # AlphaRaw 0.5.0+ API - uses factory pattern
        # Import the appropriate reader based on file type
        path_lower = filepath.lower()

        if path_lower.endswith('.raw'):
            # Thermo RAW file
            # AlphaRaw API: construct with config, then call import_raw(path)
            try:
                from alpharaw.thermo import ThermoRawData
                reader = ThermoRawData()
                reader.import_raw(filepath)
                return reader
            except (ImportError, Exception) as e:
                raise ImportError(
                    f"Thermo RAW file support not available: {e}\n"
                    "Thermo files require .NET dependencies. See AlphaRaw documentation."
                ) from e
        elif path_lower.endswith('.d') and not Path(filepath).is_dir():
            # .d suffix but NOT a directory — was uploaded via browser (drag-and-drop saves as file)
            raise ValueError(
                "Bruker .d files are folders and cannot be uploaded via drag-and-drop.\n"
                "Use the folder-open button (native mode: --native) or the local file picker\n"
                "to load .d folders directly from disk."
            )
        elif path_lower.endswith('.d') and Path(filepath).is_dir():
            # Bruker .d folder
            # Note: Bruker support may not be available in all AlphaRaw versions
            if BrukerDDAData is None:
                raise ImportError(
                    "Bruker .d file support not available in this AlphaRaw version."
                )
            try:
                reader = BrukerDDAData()
                reader.import_raw(filepath)
                return reader
            except Exception as e:
                raise ImportError(
                    f"Failed to load Bruker .d file: {e}\n"
                    "Bruker files may require specific dependencies or conversion to mzML."
                ) from e
        elif path_lower.endswith(('.wiff', '.wiff2')):
            # Sciex WIFF file - supported on Windows only
            # AlphaRaw API: construct with config, then call import_raw(path)
            try:
                from alpharaw.sciex import SciexWiffData
                reader = SciexWiffData()
                reader.import_raw(filepath)
                return reader
            except (ImportError, Exception) as e:
                raise ImportError(
                    f"Sciex WIFF file support not available: {e}\n"
                    "Sciex files require Windows .NET Framework dependencies."
                ) from e
        else:
            raise ValueError(f"Unsupported vendor file format: {filepath}")

    def _process(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Extract peak data and metadata from AlphaRaw reader into ViewerState.

        Args:
            progress_callback: Optional callback for progress updates
        """
        if self.raw_reader is None:
            return

        # AlphaRaw MSData_Base API: data lives in .spectrum_df and .peak_df
        # spectrum_df columns: spec_idx, rt (minutes), ms_level, peak_start_idx,
        #                      peak_stop_idx, precursor_mz, precursor_charge, ...
        # peak_df columns: mz, intensity
        # get_peaks(spec_idx) -> (mz_array, intensity_array)
        spectrum_df = self.raw_reader.spectrum_df
        num_spectra = len(spectrum_df)

        if num_spectra == 0:
            # Check if this is a Sciex file without companion .scan file
            error_msg = f"No spectra found in vendor file: {self.filepath}\n\n"
            if str(self.filepath).lower().endswith(('.wiff', '.wiff2')):
                scan_file = str(self.filepath) + '.scan'
                error_msg += (
                    "Sciex .wiff files require a companion .wiff.scan file.\n"
                    f"Expected file: {scan_file}\n\n"
                    "Please ensure both files are present:\n"
                    f"  - {Path(self.filepath).name}\n"
                    f"  - {Path(self.filepath).name}.scan\n"
                )
            else:
                error_msg += "The file may be empty or corrupted."
            raise ValueError(error_msg)

        if progress_callback:
            progress_callback(f"Processing {num_spectra:,} spectra...")

        peaks_rt_list = []    # list of numpy arrays (one per MS1 spectrum)
        peaks_mz_list = []
        peaks_int_list = []
        tic_data = []
        spectrum_data = []

        # Extract data from each spectrum using AlphaRaw's get_peaks() API
        for spec_idx in range(num_spectra):
            try:
                row = spectrum_df.iloc[spec_idx]

                # RT is stored in minutes in AlphaRaw — convert to seconds
                rt = float(row['rt']) * 60.0

                ms_level = int(row.get('ms_level', 1))

                # get_peaks returns (mz_array, intensity_array) sliced from peak_df
                mz_array, intensity_array = self.raw_reader.get_peaks(spec_idx)

                if len(mz_array) == 0:
                    continue

                # Add to TIC
                total_intensity = float(np.sum(intensity_array))
                tic_data.append({
                    'rt': rt,
                    'intensity': total_intensity,
                    'ms_level': ms_level
                })

                # Get precursor info for MS2
                precursor_mz = 0.0
                precursor_charge = 0
                if ms_level == 2:
                    raw_precursor_mz = float(row.get('precursor_mz', -1.0))
                    # AlphaRaw uses -1.0 as sentinel for "no precursor"
                    precursor_mz = raw_precursor_mz if raw_precursor_mz > 0.0 else 0.0
                    raw_charge = int(row.get('precursor_charge', 0))
                    precursor_charge = raw_charge if raw_charge > 0 else 0

                # Store spectrum metadata — keys must exactly match spectrum_extractor.py
                prec_mz_display = round(precursor_mz, 4) if precursor_mz > 0.0 else "-"
                prec_z_display = precursor_charge if precursor_charge > 0 else "-"
                bpi_val = float(np.max(intensity_array))
                mz_min_val = float(np.min(mz_array))
                mz_max_val = float(np.max(mz_array))
                spectrum_data.append({
                    'idx': spec_idx,
                    'rt': round(rt, 2),
                    'ms_level': ms_level,
                    'cv': None,
                    'n_peaks': len(mz_array),
                    'tic': f"{total_intensity:.2e}",
                    'bpi': f"{bpi_val:.2e}",
                    'mz_range': f"{mz_min_val:.1f}-{mz_max_val:.1f}" if len(mz_array) > 0 else "-",
                    'precursor_mz': prec_mz_display,
                    'precursor_z': prec_z_display,
                    # ID fields — populated later by link_ids_to_spectra()
                    'sequence': '-',
                    'full_sequence': '',
                    'score': '-',
                    'id_idx': None,
                })

                # Collect MS1 peaks as numpy arrays (vectorized — avoids per-peak Python loop)
                if ms_level == 1:
                    peaks_rt_list.append(np.full(len(mz_array), rt, dtype=np.float64))
                    peaks_mz_list.append(mz_array.astype(np.float64))
                    peaks_int_list.append(intensity_array.astype(np.float64))

            except Exception as e:
                print(f"Warning: Failed to process spectrum {spec_idx}: {e}")
                continue

            # Progress update every 100 spectra
            if progress_callback and spec_idx % 100 == 0:
                progress_callback(f"Processed {spec_idx:,} / {num_spectra:,} spectra...")

        # Build peaks DataFrame from collected numpy arrays (vectorized — no per-peak Python loop)
        if peaks_rt_list:
            all_rt = np.concatenate(peaks_rt_list)
            all_mz = np.concatenate(peaks_mz_list)
            all_int = np.concatenate(peaks_int_list)
            total_ms1_peaks = len(all_rt)

            peaks_df = pd.DataFrame({
                'rt': all_rt,
                'mz': all_mz,
                'intensity': all_int,
            })
            peaks_df['log_intensity'] = np.log1p(peaks_df['intensity'])

            # Register with DataManager
            if self.state.data_manager:
                registered_df = self.state.data_manager.register_peaks(peaks_df, self.filepath)
                if registered_df is not None:
                    self.state.df = registered_df
                else:
                    self.state.df = None  # Data on disk
            else:
                self.state.df = peaks_df

            # Set bounds
            self.state.rt_min = float(all_rt.min())
            self.state.rt_max = float(all_rt.max())
            self.state.mz_min = float(all_mz.min())
            self.state.mz_max = float(all_mz.max())

            # Initialize view bounds to full data range
            self.state.view_rt_min = self.state.rt_min
            self.state.view_rt_max = self.state.rt_max
            self.state.view_mz_min = self.state.mz_min
            self.state.view_mz_max = self.state.mz_max

            # Track total MS1 peaks (used by app.py info label)
            self.state.total_peaks = total_ms1_peaks
        else:
            total_ms1_peaks = 0

        # Set TIC data — MS1 only, sorted by RT, as numpy arrays (matches state.tic_rt / state.tic_intensity)
        if tic_data:
            tic_df = pd.DataFrame(tic_data)
            ms1_tic = tic_df[tic_df['ms_level'] == 1].sort_values('rt')
            if not ms1_tic.empty:
                self.state.tic_rt = ms1_tic['rt'].to_numpy(dtype=np.float32)
                self.state.tic_intensity = ms1_tic['intensity'].to_numpy(dtype=np.float32)
            else:
                # Fallback: all spectra if no MS1 found
                tic_all = tic_df.sort_values('rt')
                self.state.tic_rt = tic_all['rt'].to_numpy(dtype=np.float32)
                self.state.tic_intensity = tic_all['intensity'].to_numpy(dtype=np.float32)
            self.state.tic_source = "MS1 TIC"

        # Set spectrum data
        if spectrum_data:
            self.state.spectrum_data = spectrum_data

        # Set current file path — used by export panel, scripting panel, and OOC mode check
        self.state.current_file = self.filepath

        # Store this loader so panels can retrieve spectra via VendorSpectrumAdapter
        # (state.exp is None for vendor files — panels use state.get_spectrum() instead)
        self.state.vendor_reader = self

        # Invalidate minimap cache so stale rasters from a previously loaded file are cleared
        self.state.invalidate_minimap_cache()

        if progress_callback:
            progress_callback(f"Loaded {total_ms1_peaks:,} MS1 peaks from vendor file")


def is_vendor_file(filepath: str) -> bool:
    """Check if file is a supported vendor format.

    Args:
        filepath: Path to check

    Returns:
        True if file is a supported vendor format
    """
    path = Path(filepath)

    # Thermo RAW files
    if path.suffix.lower() == '.raw':
        return True

    # Bruker .d folders
    if path.suffix.lower() == '.d' and path.is_dir():
        return True

    # Sciex WIFF files
    if path.suffix.lower() in ['.wiff', '.wiff2']:
        return True

    return False
