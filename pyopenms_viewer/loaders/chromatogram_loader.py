"""Chromatogram extraction from mzML experiments."""

import numpy as np

from pyopenms_viewer.core.state import ViewerState


def extract_chromatograms(state: ViewerState) -> None:
    """Extract chromatogram data from the experiment.

    Extracts all chromatograms (including TIC if stored in file) and stores
    metadata in state.chromatograms and data in state.chromatogram_data.
    TIC chromatograms are marked with is_tic=True in metadata.

    Args:
        state: ViewerState with exp (MSExperiment) already loaded
    """
    if state.exp is None:
        state.chromatograms = []
        state.chromatogram_data = {}
        state.has_chromatograms = False
        return

    chroms = state.exp.getChromatograms()
    if len(chroms) == 0:
        state.chromatograms = []
        state.chromatogram_data = {}
        state.has_chromatograms = False
        return

    state.chromatograms = []
    state.chromatogram_data = {}

    for idx, chrom in enumerate(chroms):
        native_id = chrom.getNativeID()

        # Check if this is a TIC chromatogram
        is_tic = "TIC" in native_id.upper() or "total ion" in native_id.lower()

        # Get RT and intensity arrays
        rt_array, int_array = chrom.get_peaks()
        if len(rt_array) == 0:
            continue

        # Get precursor info (Q1 for DIA/SRM)
        precursor = chrom.getPrecursor()
        precursor_mz = precursor.getMZ() if precursor else 0.0
        precursor_charge = precursor.getCharge() if precursor else 0

        # Get product info (Q3 for SRM/MRM)
        product = chrom.getProduct()
        product_mz = product.getMZ() if product else 0.0

        # Calculate summary statistics
        rt_min = float(rt_array.min())
        rt_max = float(rt_array.max())
        max_intensity = float(int_array.max()) if len(int_array) > 0 else 0
        total_intensity = float(int_array.sum()) if len(int_array) > 0 else 0

        # Store metadata
        state.chromatograms.append(
            {
                "idx": idx,
                "native_id": native_id,
                "is_tic": is_tic,
                "type": "TIC" if is_tic else "",
                "precursor_mz": round(precursor_mz, 4) if precursor_mz > 0 else "-",
                "precursor_z": precursor_charge if precursor_charge > 0 else "-",
                "product_mz": round(product_mz, 4) if product_mz > 0 else "-",
                "rt_min": round(rt_min, 2),
                "rt_max": round(rt_max, 2),
                "n_points": len(rt_array),
                "max_int": f"{max_intensity:.2e}",
                "total_int": f"{total_intensity:.2e}",
            }
        )

        # Store data arrays
        state.chromatogram_data[idx] = (
            np.array(rt_array, dtype=np.float32),
            np.array(int_array, dtype=np.float32),
        )

    state.has_chromatograms = len(state.chromatograms) > 0
    state.selected_chromatogram_indices = []
