"""Spectrum metadata extraction from mzML experiments."""

from typing import Any

import numpy as np

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders.mzml_loader import get_cv_from_spectrum


def extract_spectrum_data(state: ViewerState) -> list[dict[str, Any]]:
    """Extract spectrum metadata for the unified spectrum table.

    Includes fields for ID info (sequence, score) which are populated
    when IDs are loaded via link_ids_to_spectra().

    Args:
        state: ViewerState with exp (MSExperiment) already loaded

    Returns:
        List of spectrum metadata dictionaries
    """
    if state.exp is None:
        return []

    data = []
    for idx in range(len(state.exp)):
        spec = state.exp[idx]
        rt = spec.getRT()
        ms_level = spec.getMSLevel()
        n_peaks = len(spec)

        # Get peaks for TIC and BPI calculation
        mz_array, int_array = spec.get_peaks()
        tic = float(np.sum(int_array)) if len(int_array) > 0 else 0
        bpi = float(np.max(int_array)) if len(int_array) > 0 else 0

        # Get m/z range
        mz_min = float(mz_array.min()) if len(mz_array) > 0 else 0
        mz_max = float(mz_array.max()) if len(mz_array) > 0 else 0

        # Get precursor info for MS2+
        precursor_mz = "-"
        precursor_charge = "-"
        if ms_level > 1:
            precursors = spec.getPrecursors()
            if precursors:
                precursor_mz = round(precursors[0].getMZ(), 4)
                charge = precursors[0].getCharge()
                precursor_charge = charge if charge > 0 else "-"

        # Get FAIMS CV if available (stored as float, None if not available)
        cv = get_cv_from_spectrum(spec)

        data.append(
            {
                "idx": idx,
                "rt": round(rt, 2),
                "ms_level": ms_level,
                "cv": cv,
                "n_peaks": n_peaks,
                "tic": f"{tic:.2e}",
                "bpi": f"{bpi:.2e}",
                "mz_range": f"{mz_min:.1f}-{mz_max:.1f}" if n_peaks > 0 else "-",
                "precursor_mz": precursor_mz,
                "precursor_z": precursor_charge,
                # ID fields - populated by link_ids_to_spectra()
                "sequence": "-",
                "full_sequence": "",
                "score": "-",
                "id_idx": None,
            }
        )

    return data
