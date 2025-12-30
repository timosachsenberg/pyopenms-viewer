"""Ion mobility data extraction from mzML experiments."""

import numpy as np
import pandas as pd

from pyopenms_viewer.core.state import ViewerState


def extract_ion_mobility_data(state: ViewerState) -> None:
    """Extract ion mobility data from spectra that contain IM arrays.

    For IM data, each spectrum stores a whole frame with concatenated peaks
    from multiple IM scans. The IM value for each peak is stored in a parallel
    float data array.

    Creates state.im_df with columns: mz, im, intensity, log_intensity

    Args:
        state: ViewerState with exp (MSExperiment) already loaded
    """
    if state.exp is None:
        state.has_ion_mobility = False
        state.im_df = None
        return

    # Known IM array names (check in order of preference)
    im_array_names = [
        "ion mobility",
        "inverse reduced ion mobility",  # 1/K0 from TIMS (Vs/cm²)
        "drift time",  # Drift tube (ms)
        "ion mobility drift time",
    ]

    # First pass: detect IM data and determine array name
    detected_im_name = None
    for spec in state.exp:
        if spec.getMSLevel() != 1:
            continue
        float_arrays = spec.getFloatDataArrays()
        for fda in float_arrays:
            name = fda.getName().lower() if fda.getName() else ""
            for im_name in im_array_names:
                if im_name in name:
                    detected_im_name = fda.getName()
                    break
            if detected_im_name:
                break
        if detected_im_name:
            break

    if not detected_im_name:
        state.has_ion_mobility = False
        state.im_df = None
        return

    # Determine IM type and unit for display
    name_lower = detected_im_name.lower()
    if "inverse" in name_lower or "1/k0" in name_lower:
        state.im_type = "inverse_k0"
        state.im_unit = "Vs/cm²"
    elif "drift" in name_lower:
        state.im_type = "drift_time"
        state.im_unit = "ms"
    else:
        state.im_type = "ion_mobility"
        state.im_unit = ""

    # Second pass: extract all IM data
    all_mz = []
    all_im = []
    all_int = []

    for spec in state.exp:
        if spec.getMSLevel() != 1:
            continue

        mz_array, int_array = spec.get_peaks()
        if len(mz_array) == 0:
            continue

        # Find the IM array
        im_array = None
        float_arrays = spec.getFloatDataArrays()
        for fda in float_arrays:
            if fda.getName() == detected_im_name:
                im_array = np.array(fda.get_data(), dtype=np.float32)
                break

        if im_array is None or len(im_array) != len(mz_array):
            continue

        all_mz.append(mz_array)
        all_im.append(im_array)
        all_int.append(int_array)

    if not all_mz:
        state.has_ion_mobility = False
        state.im_df = None
        return

    # Concatenate all arrays
    mz_concat = np.concatenate(all_mz)
    im_concat = np.concatenate(all_im)
    int_concat = np.concatenate(all_int)

    # Create DataFrame
    im_df = pd.DataFrame(
        {
            "mz": mz_concat,
            "im": im_concat,
            "intensity": int_concat,
        }
    )
    im_df["log_intensity"] = np.log1p(im_df["intensity"])

    # Register with data manager if available (handles both in-memory and out-of-core)
    if state.data_manager is not None and state.current_file:
        # data_manager.register_im_peaks returns DataFrame for in-memory, None for out-of-core
        state.im_df = state.data_manager.register_im_peaks(im_df, state.current_file)

        # Get bounds from data manager
        im_bounds = state.data_manager.get_im_bounds()
        state.im_min = im_bounds["im_min"]
        state.im_max = im_bounds["im_max"]
        im_mz_min = im_bounds["mz_min"]
        im_mz_max = im_bounds["mz_max"]
    else:
        # Legacy: no data manager, keep DataFrame in state
        state.im_df = im_df
        state.im_min = float(im_df["im"].min())
        state.im_max = float(im_df["im"].max())
        im_mz_min = float(im_df["mz"].min())
        im_mz_max = float(im_df["mz"].max())

    # Ensure valid IM range
    if state.im_max <= state.im_min:
        state.im_max = state.im_min + 1.0
    state.view_im_min = state.im_min
    state.view_im_max = state.im_max

    # Update mz bounds from IM data if not already set
    if state.mz_min == 0 or im_mz_min < state.mz_min:
        state.mz_min = im_mz_min
    if state.mz_max == 0 or im_mz_max > state.mz_max:
        state.mz_max = im_mz_max
    if state.view_mz_min is None or state.view_mz_min < state.mz_min:
        state.view_mz_min = state.mz_min
    if state.view_mz_max is None or state.view_mz_max > state.mz_max:
        state.view_mz_max = state.mz_max

    state.has_ion_mobility = True
