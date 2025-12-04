#!/usr/bin/env python3
"""
pyopenms-viewer: Fast mzML viewer using NiceGUI + Datashader + pyOpenMS

Designed to handle 50+ million peaks with smooth zooming and panning.
Uses datashader for server-side rendering of massive datasets.
Supports FeatureMap overlay with centroids, bounding boxes, and convex hulls.
Supports idXML overlay showing peptide identification precursor positions.
Includes annotated MS2 spectrum viewer for peptide identifications.
Displays Total Ion Chromatogram (TIC) with clickable MS1 spectrum viewer.

Usage:
    pyopenms-viewer                           # Start with empty viewer
    pyopenms-viewer sample.mzML               # Load mzML file
    pyopenms-viewer sample.mzML features.featureXML  # Load mzML + features
    pyopenms-viewer sample.mzML ids.idXML     # Load mzML + identifications
    pyopenms-viewer sample.mzML features.featureXML ids.idXML  # All three
"""

import base64
import io
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

# Fix for PyInstaller windowed mode: sys.stdout/stderr are None which breaks uvicorn's
# logging formatter that calls .isatty(). Redirect to devnull before importing uvicorn.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Set OpenMP threads for pyOpenMS to use all available cores
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))

import click
import colorcet as cc

# Datashader for fast rendering
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib
import plotly.graph_objects as go

# NiceGUI for the web interface
from nicegui import app, run, ui
from nicegui.events import MouseEventArguments

# PIL for drawing overlays and axes
from PIL import Image, ImageDraw, ImageFont

# pyOpenMS for file loading and spectrum annotation
from pyopenms import (
    AASequence,
    FeatureMap,
    FeatureXMLFile,
    IdXMLFile,
    MSExperiment,
    MSSpectrum,
    MzMLFile,
    SpectrumAlignment,
    SpectrumAnnotator,
    TheoreticalSpectrumGenerator,
)

# CLI files to load on startup
_cli_files = {"mzml": None, "featurexml": None, "idxml": None}


# Ion type colors for spectrum annotation
ION_COLORS = {
    "b": "#1f77b4",  # Blue
    "y": "#d62728",  # Red
    "a": "#2ca02c",  # Green
    "c": "#9467bd",  # Purple
    "x": "#8c564b",  # Brown
    "z": "#e377c2",  # Pink
    "precursor": "#ff7f0e",  # Orange
    "unknown": "#7f7f7f",  # Gray
}


def calculate_nice_ticks(vmin: float, vmax: float, num_ticks: int = 6) -> list[float]:
    """Calculate nice round tick values for an axis."""
    if vmin >= vmax:
        return [vmin]

    range_val = vmax - vmin
    rough_step = range_val / (num_ticks - 1)

    mag = math.floor(math.log10(rough_step))
    pow10 = 10**mag
    norm_step = rough_step / pow10

    if norm_step < 1.5:
        nice_step = 1
    elif norm_step < 3:
        nice_step = 2
    elif norm_step < 7:
        nice_step = 5
    else:
        nice_step = 10

    step = nice_step * pow10
    first_tick = math.ceil(vmin / step) * step
    ticks = []
    tick = first_tick
    while tick <= vmax + step * 0.001:
        ticks.append(tick)
        tick += step

    return ticks


def format_tick_label(value: float, range_val: float) -> str:
    """Format a tick label based on the value and range."""
    if range_val >= 1000:
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.1f}"
    elif range_val >= 10:
        return f"{value:.1f}"
    elif range_val >= 1:
        return f"{value:.2f}"
    else:
        return f"{value:.3f}"


def generate_theoretical_spectrum(sequence: AASequence, charge: int) -> dict[str, list[tuple[float, str]]]:
    """Generate theoretical b/y ion spectrum for annotation."""
    tsg = TheoreticalSpectrumGenerator()
    spec = MSSpectrum()

    # Configure for b and y ions
    params = tsg.getParameters()
    params.setValue("add_b_ions", "true")
    params.setValue("add_y_ions", "true")
    params.setValue("add_a_ions", "false")
    params.setValue("add_c_ions", "false")
    params.setValue("add_x_ions", "false")
    params.setValue("add_z_ions", "false")
    params.setValue("add_metainfo", "true")
    tsg.setParameters(params)

    tsg.getSpectrum(spec, sequence, 1, min(charge, 2))

    ions = {"b": [], "y": [], "other": []}

    for i in range(spec.size()):
        mz = spec[i].getMZ()
        spec[i].getIntensity()

        # Get ion annotation from metadata
        ion_name = ""
        if spec[i].metaValueExists("IonName"):
            ion_name = spec[i].getMetaValue("IonName")

        if ion_name.startswith("b"):
            ions["b"].append((mz, ion_name))
        elif ion_name.startswith("y"):
            ions["y"].append((mz, ion_name))
        else:
            ions["other"].append((mz, ion_name))

    return ions


def annotate_spectrum_with_id(
    spectrum: MSSpectrum, peptide_hit, tolerance_da: float = 0.05
) -> list[tuple[int, str, str]]:
    """Annotate a spectrum using SpectrumAnnotator.

    Returns list of (peak_index, ion_name, ion_type) tuples for matched peaks.
    Uses TheoreticalSpectrumGenerator and SpectrumAnnotator to generate annotations.
    Annotations are stored in the spectrum's string data array "IonNames".

    Args:
        spectrum: The experimental MS2 spectrum
        peptide_hit: PeptideHit with sequence information
        tolerance_da: Mass tolerance in Da for matching (default 0.05 Da)

    Returns:
        List of (peak_index, ion_name, ion_type) for annotated peaks
    """
    annotations = []

    try:
        # Create copy to avoid modifying original
        spec_copy = MSSpectrum(spectrum)

        # Setup TheoreticalSpectrumGenerator
        tsg = TheoreticalSpectrumGenerator()
        params = tsg.getParameters()
        params.setValue("add_b_ions", "true")
        params.setValue("add_y_ions", "true")
        params.setValue("add_a_ions", "true")
        params.setValue("add_c_ions", "false")
        params.setValue("add_x_ions", "false")
        params.setValue("add_z_ions", "false")
        params.setValue("add_metainfo", "true")
        tsg.setParameters(params)

        # Setup SpectrumAlignment with absolute tolerance in Da
        sa = SpectrumAlignment()
        sa_params = sa.getParameters()
        sa_params.setValue("tolerance", tolerance_da)
        sa_params.setValue("is_relative_tolerance", "false")
        sa.setParameters(sa_params)

        # Setup SpectrumAnnotator
        annotator = SpectrumAnnotator()

        # Annotate the spectrum - this adds "IonNames" string data array
        annotator.annotateMatches(spec_copy, peptide_hit, tsg, sa)

        # Read annotations from spectrum's string data arrays
        sda = spec_copy.getStringDataArrays()
        for arr in sda:
            arr_name = arr.getName()
            # Handle both bytes and string for array name
            if arr_name == "IonNames" or arr_name == b"IonNames":
                for peak_idx, ann_str in enumerate(arr):
                    if ann_str:
                        # Handle bytes or string annotation
                        if isinstance(ann_str, bytes):
                            ann_str = ann_str.decode("utf-8", errors="ignore")

                        # Clean up the annotation
                        ion_name = ann_str.strip("'\"")

                        # Determine ion type from the cleaned name
                        if ion_name.startswith("y"):
                            ion_type = "y"
                        elif ion_name.startswith("b"):
                            ion_type = "b"
                        elif ion_name.startswith("a"):
                            ion_type = "a"
                        elif ion_name.startswith("c"):
                            ion_type = "c"
                        elif ion_name.startswith("x"):
                            ion_type = "x"
                        elif ion_name.startswith("z"):
                            ion_type = "z"
                        else:
                            ion_type = "unknown"

                        annotations.append((peak_idx, ion_name, ion_type))
                break

    except Exception as e:
        # If annotation fails, return empty list
        print(f"SpectrumAnnotator error: {e}")

    return annotations


def get_external_peak_annotations(
    peptide_hit,
    exp_mz: np.ndarray,
    tolerance_da: float = 0.05,
) -> list[tuple[int, str, str]]:
    """Get external peak annotations from a PeptideHit using getPeakAnnotations() API.

    This uses the pyOpenMS PeptideHit.getPeakAnnotations() method which returns
    pre-parsed PeakAnnotation objects from idXML fragment_annotation data.

    Args:
        peptide_hit: PeptideHit object with peak annotations
        exp_mz: Experimental m/z array to match annotations to peak indices
        tolerance_da: Mass tolerance in Da for matching annotations to peaks

    Returns:
        List of (peak_index, ion_name, ion_type) for matched annotations
    """
    annotations = []

    if len(exp_mz) == 0:
        return annotations

    try:
        peak_annotations = peptide_hit.getPeakAnnotations()

        if not peak_annotations:
            return annotations

        for peak_ann in peak_annotations:
            ann_mz = peak_ann.mz
            ion_name = peak_ann.annotation

            # Handle bytes if needed
            if isinstance(ion_name, bytes):
                ion_name = ion_name.decode("utf-8", errors="ignore")

            # Find closest experimental peak within tolerance
            diffs = np.abs(exp_mz - ann_mz)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= tolerance_da:
                # Determine ion type from name
                ion_name_lower = ion_name.lower()
                if ion_name_lower.startswith("y"):
                    ion_type = "y"
                elif ion_name_lower.startswith("b"):
                    ion_type = "b"
                elif ion_name_lower.startswith("a"):
                    ion_type = "a"
                elif ion_name_lower.startswith("c"):
                    ion_type = "c"
                elif ion_name_lower.startswith("x"):
                    ion_type = "x"
                elif ion_name_lower.startswith("z"):
                    ion_type = "z"
                elif "mi:" in ion_name_lower or ion_name_lower.startswith("i"):
                    ion_type = "unknown"  # Immonium ions
                elif "[m" in ion_name_lower:
                    ion_type = "precursor"  # Precursor-related ions
                else:
                    ion_type = "unknown"

                annotations.append((int(min_idx), ion_name, ion_type))

    except Exception as e:
        print(f"Error getting external peak annotations: {e}")

    return annotations


def parse_external_fragment_annotations(
    fragment_annotation_str: str,
    exp_mz: np.ndarray,
    tolerance_da: float = 0.05,
) -> list[tuple[int, str, str]]:
    """Parse external fragment annotations from idXML fragment_annotation UserParam string.

    This is a fallback for when getPeakAnnotations() is not available.
    The format is pipe-separated: m/z,intensity,charge,"ion_name"|...
    Example: '201.087,1.0,1,"b2+"|712.362,0.394,1,"y5+U'-H2O+"'

    Args:
        fragment_annotation_str: The fragment_annotation string from idXML
        exp_mz: Experimental m/z array to match annotations to peak indices
        tolerance_da: Mass tolerance in Da for matching annotations to peaks

    Returns:
        List of (peak_index, ion_name, ion_type) for matched annotations
    """
    annotations = []

    if not fragment_annotation_str or len(exp_mz) == 0:
        return annotations

    try:
        # Split by pipe to get individual annotations
        parts = fragment_annotation_str.split("|")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            try:
                # Find opening quote for ion name
                quote_start = part.find('"')
                if quote_start == -1:
                    # Try without quotes
                    fields = part.split(",")
                    if len(fields) >= 4:
                        ann_mz = float(fields[0])
                        ion_name = fields[3].strip("'\"")
                    else:
                        continue
                else:
                    # Extract the prefix (m/z,intensity,charge,) and quoted ion name
                    prefix = part[:quote_start].rstrip(",")
                    fields = prefix.split(",")
                    if len(fields) < 3:
                        continue
                    ann_mz = float(fields[0])

                    # Extract ion name from quotes
                    quote_end = part.rfind('"')
                    if quote_end > quote_start:
                        ion_name = part[quote_start + 1 : quote_end]
                    else:
                        ion_name = part[quote_start + 1 :]

                # Clean up ion name
                ion_name = ion_name.replace("&quot;", '"').replace("&apos;", "'").strip()

                # Find closest experimental peak within tolerance
                diffs = np.abs(exp_mz - ann_mz)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] <= tolerance_da:
                    # Determine ion type from name
                    ion_name_lower = ion_name.lower()
                    if ion_name_lower.startswith("y"):
                        ion_type = "y"
                    elif ion_name_lower.startswith("b"):
                        ion_type = "b"
                    elif ion_name_lower.startswith("a"):
                        ion_type = "a"
                    elif ion_name_lower.startswith("c"):
                        ion_type = "c"
                    elif ion_name_lower.startswith("x"):
                        ion_type = "x"
                    elif ion_name_lower.startswith("z"):
                        ion_type = "z"
                    elif "mi:" in ion_name_lower or ion_name_lower.startswith("i"):
                        ion_type = "unknown"
                    elif "[m" in ion_name_lower:
                        ion_type = "precursor"
                    else:
                        ion_type = "unknown"

                    annotations.append((int(min_idx), ion_name, ion_type))

            except (ValueError, IndexError):
                continue

    except Exception as e:
        print(f"Error parsing external fragment annotations: {e}")

    return annotations


def create_annotated_spectrum_plot(
    exp_mz: np.ndarray,
    exp_int: np.ndarray,
    sequence_str: str,
    charge: int,
    precursor_mz: float,
    tolerance_da: float = 0.5,
    peak_annotations: Optional[list[tuple[int, str, str]]] = None,
    annotate: bool = True,
    mirror_mode: bool = False,
) -> go.Figure:
    """Create an annotated spectrum plot using Plotly.

    Args:
        exp_mz: Experimental m/z values
        exp_int: Experimental intensity values
        sequence_str: Peptide sequence string
        charge: Precursor charge
        precursor_mz: Precursor m/z value
        tolerance_da: Mass tolerance in Da for matching (used if no peak_annotations)
        peak_annotations: Optional list of (peak_index, ion_name, ion_type) from SpectrumAnnotator
        annotate: Whether to show annotations (if False, shows raw spectrum)
        mirror_mode: If True, flip annotated peaks downward for comparison view
    """

    # Normalize intensities to percentage
    max_int = exp_int.max() if len(exp_int) > 0 else 1
    exp_int_norm = (exp_int / max_int) * 100

    # Create figure
    fig = go.Figure()

    # Add experimental spectrum as vertical lines (stem plot) - doesn't get thicker when zooming
    x_stems = []
    y_stems = []
    for mz, intensity in zip(exp_mz, exp_int_norm):
        x_stems.extend([mz, mz, None])
        y_stems.extend([0, intensity, None])

    fig.add_trace(
        go.Scatter(
            x=x_stems,
            y=y_stems,
            mode="lines",
            line={"color": "gray", "width": 1},
            name="Experimental",
            hoverinfo="skip",
            opacity=0.6,
        )
    )

    # Add invisible hover points for experimental peaks (opacity 0 hides markers but keeps hover)
    fig.add_trace(
        go.Scatter(
            x=exp_mz,
            y=exp_int_norm,
            mode="markers",
            marker={"color": "gray", "size": 8, "opacity": 0},
            showlegend=False,
            hovertemplate="m/z: %{x:.4f}<br>Intensity: %{y:.1f}%<extra></extra>",
        )
    )

    # Add annotations if enabled
    if annotate:
        matched_peaks = {"b": [], "y": [], "a": [], "c": [], "x": [], "z": [], "precursor": [], "unknown": []}

        if peak_annotations:
            # Use provided peak annotations from SpectrumAnnotator
            for peak_idx, ion_name, ion_type in peak_annotations:
                if peak_idx < len(exp_mz):
                    matched_peaks[ion_type].append(
                        {"mz": exp_mz[peak_idx], "intensity": exp_int_norm[peak_idx], "label": ion_name}
                    )
        else:
            # Fall back to generating theoretical spectrum for annotation
            try:
                seq = AASequence.fromString(sequence_str)
                theo_ions = generate_theoretical_spectrum(seq, charge)

                for ion_type, ions in [("b", theo_ions["b"]), ("y", theo_ions["y"])]:
                    for theo_mz, ion_name in ions:
                        # Find closest experimental peak
                        if len(exp_mz) > 0:
                            diffs = np.abs(exp_mz - theo_mz)
                            min_idx = np.argmin(diffs)
                            if diffs[min_idx] <= tolerance_da:
                                matched_peaks[ion_type].append(
                                    {"mz": exp_mz[min_idx], "intensity": exp_int_norm[min_idx], "label": ion_name}
                                )
            except Exception:
                pass

        # Add matched peaks as colored lines grouped by ion type
        # In mirror mode, flip annotated peaks downward (negative y values)
        for ion_type, peaks in matched_peaks.items():
            if not peaks:
                continue
            color = ION_COLORS[ion_type]

            # Create stem plot for this ion type
            x_ions = []
            y_ions = []
            for peak in peaks:
                x_ions.extend([peak["mz"], peak["mz"], None])
                if mirror_mode:
                    y_ions.extend([0, -peak["intensity"], None])
                else:
                    y_ions.extend([0, peak["intensity"], None])

            fig.add_trace(
                go.Scatter(
                    x=x_ions,
                    y=y_ions,
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=f"{ion_type}-ions",
                    hoverinfo="skip",
                )
            )

            # Add hover points and annotations for matched peaks
            for peak in peaks:
                y_val = -peak["intensity"] if mirror_mode else peak["intensity"]
                fig.add_trace(
                    go.Scatter(
                        x=[peak["mz"]],
                        y=[y_val],
                        mode="markers",
                        marker={"color": color, "size": 4},
                        showlegend=False,
                        hovertemplate=f"{peak['label']}<br>m/z: {peak['mz']:.4f}<br>Intensity: {peak['intensity']:.1f}%<extra></extra>",
                    )
                )

                # Add text annotation (below peak in mirror mode)
                if mirror_mode:
                    text_y = y_val - 3
                    text_angle = 45  # Flip angle for readability
                else:
                    text_y = peak["intensity"] + 3
                    text_angle = -45
                fig.add_annotation(
                    x=peak["mz"],
                    y=text_y,
                    text=peak["label"],
                    showarrow=False,
                    font={"size": 9, "color": color},
                    textangle=text_angle,
                )

    # Add precursor marker
    fig.add_vline(
        x=precursor_mz, line_dash="dash", line_color="orange", annotation_text=f"Precursor ({precursor_mz:.2f})"
    )

    # Update layout - use transparent backgrounds for light/dark mode compatibility
    fig.update_layout(
        title={"text": f"MS2 Spectrum: {sequence_str} (z={charge}+)", "font": {"size": 14, "color": "#888"}},
        xaxis_title="m/z",
        yaxis_title="Relative Intensity (%)",
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent outer background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
        height=400,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"size": 10}},
        modebar={"remove": ["lasso2d", "select2d"]},  # Remove lasso and box select tools
        font={"color": "#888"},  # Gray text works on both light and dark
    )

    fig.update_xaxes(
        range=[0, max(exp_mz) * 1.05] if len(exp_mz) > 0 else [0, 2000],
        showgrid=False,
        linecolor="#888",
        tickcolor="#888",
    )

    if mirror_mode:
        # Symmetric y-axis for mirror view with zero line
        fig.update_yaxes(
            range=[-110, 110],
            showgrid=False,
            fixedrange=True,
            linecolor="#888",
            tickcolor="#888",
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=1,
            # Show absolute values on tick labels
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=["100", "50", "0", "50", "100"],
        )
    else:
        fig.update_yaxes(
            range=[0, 110],
            showgrid=False,
            fixedrange=True,
            linecolor="#888",
            tickcolor="#888",
        )

    return fig


# Available colormaps for peak map visualization
COLORMAPS = {
    "jet": matplotlib.colormaps["jet"],
    "hot": matplotlib.colormaps["hot"],
    "fire": cc.fire,
    "viridis": matplotlib.colormaps["viridis"],
    "plasma": matplotlib.colormaps["plasma"],
    "inferno": matplotlib.colormaps["inferno"],
    "magma": matplotlib.colormaps["magma"],
}


def get_colormap_background(colormap_name: str) -> str:
    """Get the lowest color from a colormap as a hex string for background."""
    cmap = COLORMAPS.get(colormap_name)
    if cmap is None:
        return "black"

    # Check if it's a matplotlib colormap (has __call__ method)
    if callable(cmap):
        # Matplotlib colormap - get color at 0
        rgba = cmap(0)
        # Convert to hex
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    elif isinstance(cmap, list) and len(cmap) > 0:
        # Colorcet list - first element is the lowest color
        return cmap[0]
    else:
        return "black"


class Viewer:
    """High-performance mzML peak map viewer using datashader with feature and ID overlay."""

    def __init__(self):
        self.exp = None
        self.df = None
        self.current_file = None

        # FeatureMap data
        self.feature_map = None
        self.features_file = None
        self.feature_data = []

        # Identification data
        self.peptide_ids = []
        self.protein_ids = []
        self.id_file = None
        self.id_data = []
        self.id_meta_keys = []  # Discovered meta value keys from PeptideIdentification/PeptideHit

        # TIC data
        self.tic_rt = None
        self.tic_intensity = None
        self.tic_source = "MS1 TIC"  # Description of TIC source (e.g., "MS1 TIC", "MS2 BPC")

        # Chromatogram data (all chromatograms from mzML, including stored TIC)
        self.chromatograms = []  # List of chromatogram metadata dicts
        self.chromatogram_data = {}  # Dict: chrom_idx -> (rt_array, intensity_array)
        self.selected_chromatogram_indices = []  # List of selected chromatogram indices to display
        self.has_chromatograms = False

        # Ion Mobility data (for TIMS, drift tube, etc.)
        self.has_ion_mobility = False
        self.im_type = None  # "ion mobility", "inverse reduced ion mobility", "drift time"
        self.im_unit = ""  # Unit string for display
        self.im_df = None  # DataFrame with columns: mz, im, intensity, log_intensity
        self.im_min = 0
        self.im_max = 1
        self.view_im_min = None
        self.view_im_max = None

        # FAIMS data
        self.faims_cvs = []  # List of unique CV values
        self.faims_data = {}  # Dict: CV -> DataFrame of peaks
        self.faims_tic = {}  # Dict: CV -> (rt_array, intensity_array)
        self.has_faims = False
        self.show_faims_view = False  # Toggle for FAIMS multi-panel view

        # Spectrum browser data
        self.spectrum_data = []  # List of spectrum metadata for table
        self.selected_spectrum_idx = None

        # View bounds
        self.rt_min = 0
        self.rt_max = 1
        self.mz_min = 0
        self.mz_max = 1

        # Current view
        self.view_rt_min = None
        self.view_rt_max = None
        self.view_mz_min = None
        self.view_mz_max = None

        # Selected indices
        self.selected_feature_idx = None
        self.selected_id_idx = None

        # Image dimensions
        self.plot_width = 1100
        self.plot_height = 550

        # Margins
        self.margin_left = 80
        self.margin_right = 20
        self.margin_top = 20
        self.margin_bottom = 50

        self.canvas_width = self.plot_width + self.margin_left + self.margin_right
        self.canvas_height = self.plot_height + self.margin_top + self.margin_bottom

        # Display options
        self.show_centroids = True
        self.show_bounding_boxes = False  # Disabled by default for faster rendering
        self.swap_axes = True  # Default: m/z on x-axis, RT on y-axis; when False: RT on x, m/z on y
        self.show_spectrum_marker = True  # Show spectrum marker (crosshair) on 2D peakmap
        self.show_convex_hulls = False  # Disabled by default for faster rendering
        self.show_ids = True
        self.show_id_sequences = False  # Show peptide sequences on 2D peakmap (off by default)
        self.colormap = "jet"  # Default colormap
        self.rt_in_minutes = False  # Display RT in minutes instead of seconds
        self.spectrum_intensity_percent = True  # Display spectrum intensity as percentage (vs absolute)
        self.spectrum_auto_scale = False  # Auto-scale y-axis to fit visible peaks
        self.annotate_peaks = True  # Annotate peaks in spectrum view when ID is selected
        self.annotation_tolerance_da = 0.05  # Mass tolerance for peak annotation in Da
        self.mirror_annotation_view = False  # Mirror mode: flip annotated peaks downward for comparison
        self.show_all_hits = False  # Show all peptide hits, not just the best hit

        # Colors
        self.centroid_color = (0, 255, 100, 255)
        self.bbox_color = (255, 255, 0, 200)
        self.hull_color = (0, 200, 255, 150)
        self.selected_color = (255, 100, 255, 255)
        self.id_color = (255, 150, 50, 255)
        self.id_selected_color = (255, 50, 50, 255)

        # Neutral gray colors that work on both light and dark backgrounds
        self.axis_color = (136, 136, 136, 255)  # #888
        self.tick_color = (136, 136, 136, 255)  # #888
        self.label_color = (136, 136, 136, 255)  # #888
        self.grid_color = (60, 60, 60, 255)

        # UI elements
        self.image_element = None
        self.status_label = None
        self.info_label = None
        self.feature_info_label = None
        self.id_info_label = None
        self.rt_range_label = None
        self.mz_range_label = None
        self.feature_table = None
        self.tic_plot = None
        self.tic_expansion = None  # Collapsible panel for TIC

        # Chromatogram UI elements
        self.chromatogram_plot = None
        self.chromatogram_table = None
        self.chromatogram_expansion = None  # Collapsible panel for chromatograms
        self.chromatogram_info_label = None

        # Ion Mobility UI elements
        self.im_expansion = None  # Collapsible panel for IM peak map
        self.im_image_element = None  # Interactive image for IM peak map
        self.im_info_label = None
        self.im_range_label = None  # Label showing current IM range
        self.link_spectrum_mz_to_im = False  # Link spectrum m/z zoom to IM peakmap m/z range
        self.link_spectrum_mz_checkbox = None  # UI checkbox reference
        self.show_mobilogram = True  # Show mobilogram plot on the side of IM peakmap
        self.mobilogram_plot_width = 150  # Width of mobilogram plot in pixels

        # Spectrum browser UI elements
        self.spectrum_table = None
        self.spectrum_table_expansion = None  # Collapsible panel for Spectrum Table
        self.spectrum_browser_plot = None
        self.spectrum_browser_info = None
        self.spectrum_nav_label = None
        self.spectrum_expansion = None  # Collapsible panel for 1D Spectrum

        # Spectrum measurement tool state
        self.spectrum_measure_mode = False  # Whether measurement mode is active
        self.spectrum_measure_start = None  # (mz, intensity) of first measurement point, or None
        self.spectrum_measurements = {}  # Dict: spectrum_idx -> list of (mz1, int1, mz2, int2) tuples
        self.spectrum_hover_peak = None  # Currently highlighted peak (mz, intensity) during hover
        self.spectrum_selected_measurement_idx = None  # Index of selected measurement for deletion/repositioning
        self.spectrum_dragging = False  # Whether we're in drag mode for creating/repositioning
        self.spectrum_zoom_range = None  # (xmin, xmax) to preserve zoom during measurement

        # Peak annotation state
        self.peak_annotations = {}  # Dict: spectrum_idx -> list of {"mz": float, "int": float, "label": str}
        self.peak_annotation_mode = False  # Whether annotation mode is active (click to add/edit labels)
        self.show_mz_labels = False  # Show m/z values as labels on all peaks

        # FAIMS UI elements
        self.faims_container = None  # Container for multiple peak maps
        self.faims_images = {}  # Dict: CV -> image element
        self.faims_toggle = None
        self.faims_info_label = None

        # Navigation UI elements
        self.minimap_image = None  # Overview image
        self.minimap_width = 400
        self.minimap_height = 200
        self.breadcrumb_label = None  # Zoom history display
        self.coord_label = None  # RT/m/z coordinate display

        # Zoom history for breadcrumb trail
        self.zoom_history = []  # List of (rt_min, rt_max, mz_min, mz_max, label) tuples
        self.max_zoom_history = 10  # Max history entries

        # UI update flags
        self._updating_from_tic = False  # Prevent circular TIC updates

        # Hover state for visual feedback
        self.hover_feature_idx = None  # Feature being hovered in table
        self.hover_id_idx = None  # ID being hovered in table
        self._hover_update_pending = False  # Debounce hover updates

        # 3D visualization
        self.show_3d_view = False
        self.plot_3d = None  # Plotly 3D plot element
        self.scene_3d_container = None
        self.view_3d_status = None  # Status label for 3D view

        # Dark mode reference
        self.dark = None  # Set by create_ui() after viewer creation
        self.max_3d_peaks = 5000  # Limit peaks for 3D performance
        self.rt_threshold_3d = 120.0  # Max RT range for 3D (seconds)
        self.mz_threshold_3d = 50.0  # Max m/z range for 3D

        # Panel order configuration for reorderable panels
        self.panel_definitions = {
            "tic": {"name": "TIC", "icon": "show_chart"},
            "chromatograms": {"name": "Chromatograms", "icon": "timeline"},
            "peakmap": {"name": "2D Peak Map", "icon": "grid_on"},
            "im_peakmap": {"name": "Ion Mobility Map", "icon": "blur_on"},
            "spectrum": {"name": "1D Spectrum", "icon": "ssid_chart"},
            "spectra_table": {"name": "Spectra", "icon": "list"},
            "features_table": {"name": "Features", "icon": "scatter_plot"},
            "custom_range": {"name": "Custom Range", "icon": "tune"},
            "legend": {"name": "Help", "icon": "help"},
        }
        self.panel_order = [
            "tic", "chromatograms", "peakmap", "im_peakmap", "spectrum", "spectra_table",
            "features_table", "custom_range", "legend"
        ]
        self.panel_elements = {}  # Dict: panel_id -> expansion element
        self.panels_container = None  # Column container holding all panels
        # Panel visibility: True = always show, False = always hide, "auto" = show only when data exists
        self.panel_visibility = {
            "tic": True,  # Always show
            "chromatograms": "auto",  # Show only when chromatograms exist
            "peakmap": True,  # Always show
            "im_peakmap": "auto",  # Show only when ion mobility data exists
            "spectrum": True,  # Always show
            "spectra_table": True,  # Always show
            "features_table": "auto",  # Show only when features loaded
            "custom_range": True,  # Always show
            "legend": True,  # Always show
        }

        # NiceGUI 3.x: Event callbacks for state management
        # These allow UI components to subscribe to state changes
        self._on_data_loaded_callbacks: list[callable] = []
        self._on_view_changed_callbacks: list[callable] = []
        self._on_selection_changed_callbacks: list[callable] = []

    def on_data_loaded(self, callback: callable) -> None:
        """Register a callback for when data is loaded (mzML, features, or IDs)."""
        self._on_data_loaded_callbacks.append(callback)

    def on_view_changed(self, callback: callable) -> None:
        """Register a callback for when the view (zoom/pan) changes."""
        self._on_view_changed_callbacks.append(callback)

    def on_selection_changed(self, callback: callable) -> None:
        """Register a callback for when selection (spectrum, feature, ID) changes."""
        self._on_selection_changed_callbacks.append(callback)

    def _emit_data_loaded(self, data_type: str) -> None:
        """Emit data loaded event to all registered callbacks."""
        for callback in self._on_data_loaded_callbacks:
            try:
                callback(data_type, self)
            except Exception:
                pass

    def _emit_view_changed(self) -> None:
        """Emit view changed event to all registered callbacks."""
        for callback in self._on_view_changed_callbacks:
            try:
                callback(self)
            except Exception:
                pass

    def _emit_selection_changed(self, selection_type: str, index: Optional[int]) -> None:
        """Emit selection changed event to all registered callbacks."""
        for callback in self._on_selection_changed_callbacks:
            try:
                callback(selection_type, index, self)
            except Exception:
                pass

    def should_panel_be_visible(self, panel_id: str) -> bool:
        """Determine if a panel should be visible based on visibility setting and data availability.

        Args:
            panel_id: The panel identifier (e.g., "tic", "im_peakmap", "chromatograms")

        Returns:
            True if panel should be visible, False otherwise
        """
        visibility = self.panel_visibility.get(panel_id, True)

        if visibility is True:
            return True
        elif visibility is False:
            return False
        elif visibility == "auto":
            # Auto-visibility based on data availability
            if panel_id == "im_peakmap":
                return self.has_ion_mobility
            elif panel_id == "chromatograms":
                return self.has_chromatograms
            elif panel_id == "features_table":
                return self.feature_data is not None and len(self.feature_data) > 0
            else:
                return True
        return True

    def update_panel_visibility(self) -> None:
        """Update visibility of all panels based on current visibility settings."""
        for panel_id, element in self.panel_elements.items():
            if element is not None:
                should_show = self.should_panel_be_visible(panel_id)
                element.set_visibility(should_show)

    def set_panel_visibility(self, panel_id: str, visibility: bool | str) -> None:
        """Set visibility for a specific panel.

        Args:
            panel_id: The panel identifier
            visibility: True (always show), False (always hide), or "auto" (data-dependent)
        """
        self.panel_visibility[panel_id] = visibility
        if panel_id in self.panel_elements and self.panel_elements[panel_id] is not None:
            should_show = self.should_panel_be_visible(panel_id)
            self.panel_elements[panel_id].set_visibility(should_show)

    def set_loading(self, is_loading: bool, message: str = "") -> None:
        """Set the loading state and optionally display a message.

        Args:
            is_loading: Whether loading is in progress
            message: Optional message to display in status label
        """
        if self.status_label:
            if is_loading and message:
                self.status_label.set_text(message)
            elif not is_loading:
                self.status_label.set_text("Ready")

    def update_loading_progress(self, message: str, progress: float = 0.0) -> None:
        """Update the loading progress message.

        Args:
            message: Progress message to display
            progress: Optional progress value (0.0 to 1.0), currently unused
        """
        if self.status_label:
            self.status_label.set_text(message)

    def _get_cv_from_spectrum(self, spec) -> Optional[float]:
        """Extract FAIMS compensation voltage from spectrum metadata."""
        # Try common CV metadata names
        cv_names = [
            "FAIMS compensation voltage",
            "ion mobility drift time",  # Alternative
            "MS:1001581",  # CV accession for FAIMS CV
        ]
        for name in cv_names:
            if spec.metaValueExists(name):
                try:
                    return float(spec.getMetaValue(name))
                except (ValueError, TypeError):
                    pass

        # Check in acquisition info / scan windows
        try:
            # Try to get from instrument settings or other metadata
            acq = spec.getAcquisitionInfo()
            if acq:
                for a in acq:
                    for name in cv_names:
                        if a.metaValueExists(name):
                            return float(a.getMetaValue(name))
        except Exception:
            pass

        return None

    def parse_mzml_file(self, filepath: str) -> bool:
        """Parse mzML file using pyOpenMS (blocking C++ call).

        This is the first phase of loading - just parses the file.
        Returns True if successful and file has spectra.
        """
        try:
            filename = Path(filepath).name
            print(f"Reading {filename} with MzMLFile (this may take a while)...")
            self.exp = MSExperiment()
            MzMLFile().load(filepath, self.exp)
            print(f"Loaded {self.exp.size()} spectra from {filename}")
            return self.exp.size() > 0
        except Exception as e:
            print(f"Error parsing mzML: {e}")
            return False

    def process_mzml_data(self, filepath: str, progress_callback=None) -> bool:
        """Process parsed mzML data to extract peaks and create DataFrame.

        This is the second phase of loading - processes spectra with progress updates.
        Args:
            filepath: Path to the mzML file (for storing reference)
            progress_callback: Optional callback(message, progress) for progress updates
        """
        try:
            if self.exp is None:
                return False

            self.exp.size()
            total_peaks = sum(spec.size() for spec in self.exp)

            if total_peaks == 0:
                return False

            # First pass: detect FAIMS CVs
            if progress_callback:
                progress_callback("Detecting FAIMS CVs...", 0.05)

            cv_set = set()
            for spec in self.exp:
                if spec.getMSLevel() == 1:
                    cv = self._get_cv_from_spectrum(spec)
                    if cv is not None:
                        cv_set.add(cv)

            self.has_faims = len(cv_set) > 1
            self.faims_cvs = sorted(cv_set) if self.has_faims else []

            # Data structures for peak extraction
            rts = np.empty(total_peaks, dtype=np.float32)
            mzs = np.empty(total_peaks, dtype=np.float32)
            intensities = np.empty(total_peaks, dtype=np.float32)
            cvs = np.empty(total_peaks, dtype=np.float32) if self.has_faims else None

            # Also compute TIC (overall and per-CV)
            tic_rts = []
            tic_intensities = []
            faims_tic_data = {cv: {"rt": [], "int": []} for cv in self.faims_cvs} if self.has_faims else {}

            if progress_callback:
                progress_callback("Extracting peaks...", 0.1)

            idx = 0
            ms1_count = 0
            total_ms1 = sum(1 for spec in self.exp if spec.getMSLevel() == 1)

            # Determine TIC source: MS1 TIC or fallback to MS2+ BPC
            if total_ms1 > 0:
                tic_ms_level = 1
                self.tic_source = "MS1 TIC"
            else:
                # No MS1 spectra - find the lowest available MS level > 1
                ms_levels = set(spec.getMSLevel() for spec in self.exp)
                tic_ms_level = min(lv for lv in ms_levels if lv > 1) if ms_levels else 2
                self.tic_source = f"MS{tic_ms_level} BPC"

            total_tic_spectra = sum(1 for spec in self.exp if spec.getMSLevel() == tic_ms_level)

            for spec in self.exp:
                if spec.getMSLevel() != tic_ms_level:
                    # Only extract peaks from MS1 for the peak map (if available)
                    if tic_ms_level == 1 or spec.getMSLevel() != 1:
                        continue

                ms1_count += 1
                # Update progress every 100 spectra
                if progress_callback and ms1_count % 100 == 0:
                    progress = 0.1 + 0.6 * (ms1_count / max(total_tic_spectra, 1))
                    progress_callback(f"Extracting peaks... {ms1_count:,}/{total_tic_spectra:,}", progress)

                rt = spec.getRT()
                mz_array, int_array = spec.get_peaks()
                n = len(mz_array)

                cv = self._get_cv_from_spectrum(spec) if self.has_faims else None

                if n > 0:
                    # Only add to peak map DataFrame if MS1
                    if spec.getMSLevel() == 1:
                        rts[idx : idx + n] = rt
                        mzs[idx : idx + n] = mz_array
                        intensities[idx : idx + n] = int_array
                        if self.has_faims and cv is not None:
                            cvs[idx : idx + n] = cv
                        idx += n

                    # TIC/BPC calculation
                    if tic_ms_level == 1:
                        # MS1: use sum (TIC)
                        tic_value = float(np.sum(int_array))
                    else:
                        # MS2+: use base peak (BPC)
                        tic_value = float(np.max(int_array))

                    tic_rts.append(rt)
                    tic_intensities.append(tic_value)

                    # Per-CV TIC
                    if self.has_faims and cv is not None:
                        faims_tic_data[cv]["rt"].append(rt)
                        faims_tic_data[cv]["int"].append(tic_value)

            rts = rts[:idx]
            mzs = mzs[:idx]
            intensities = intensities[:idx]
            if self.has_faims:
                cvs = cvs[:idx]

            if progress_callback:
                progress_callback("Building TIC...", 0.75)

            # Store TIC data (sorted by RT to avoid zig-zag line plots)
            tic_rt_arr = np.array(tic_rts, dtype=np.float32)
            tic_int_arr = np.array(tic_intensities, dtype=np.float32)
            sort_idx = np.argsort(tic_rt_arr)
            self.tic_rt = tic_rt_arr[sort_idx]
            self.tic_intensity = tic_int_arr[sort_idx]

            # Store per-CV TIC data (also sorted by RT)
            self.faims_tic = {}
            for cv in self.faims_cvs:
                cv_rt = np.array(faims_tic_data[cv]["rt"], dtype=np.float32)
                cv_int = np.array(faims_tic_data[cv]["int"], dtype=np.float32)
                cv_sort_idx = np.argsort(cv_rt)
                self.faims_tic[cv] = (cv_rt[cv_sort_idx], cv_int[cv_sort_idx])

            if progress_callback:
                progress_callback("Extracting chromatograms...", 0.77)

            # Extract chromatograms (all chromatograms including stored TIC)
            self._extract_chromatograms()

            if progress_callback:
                progress_callback("Extracting ion mobility data...", 0.78)

            # Extract ion mobility data if present (TIMS, drift tube, etc.)
            self._extract_ion_mobility_data()

            if progress_callback:
                progress_callback("Extracting spectrum metadata...", 0.8)

            # Extract spectrum metadata for browser
            self.spectrum_data = self._extract_spectrum_data()

            if progress_callback:
                progress_callback("Creating DataFrame...", 0.85)

            # Create main DataFrame
            self.df = pd.DataFrame({"rt": rts, "mz": mzs, "intensity": intensities})
            if self.has_faims:
                self.df["cv"] = cvs
            self.df["log_intensity"] = np.log1p(self.df["intensity"])

            if progress_callback:
                progress_callback("Finalizing...", 0.95)

            # Create per-CV DataFrames for FAIMS view
            self.faims_data = {}
            if self.has_faims:
                for cv in self.faims_cvs:
                    cv_df = self.df[self.df["cv"] == cv].copy()
                    self.faims_data[cv] = cv_df

            # Set bounds from peak data if available
            if len(self.df) > 0:
                self.rt_min = float(self.df["rt"].min())
                self.rt_max = float(self.df["rt"].max())
                self.mz_min = float(self.df["mz"].min())
                self.mz_max = float(self.df["mz"].max())
            else:
                # No regular peak data - use IM data bounds or spectrum metadata
                if self.has_ion_mobility and self.im_df is not None and len(self.im_df) > 0:
                    self.mz_min = float(self.im_df["mz"].min())
                    self.mz_max = float(self.im_df["mz"].max())
                # Get RT from spectrum metadata
                if self.spectrum_data:
                    rts = [s["rt"] for s in self.spectrum_data if isinstance(s["rt"], (int, float)) and s["rt"] > 0]
                    if rts:
                        self.rt_min = min(rts)
                        self.rt_max = max(rts)

            # Ensure valid ranges (avoid zero range which causes division by zero)
            if self.rt_max <= self.rt_min:
                self.rt_max = self.rt_min + 1.0
            if self.mz_max <= self.mz_min:
                self.mz_max = self.mz_min + 1.0

            self.view_rt_min = self.rt_min
            self.view_rt_max = self.rt_max
            self.view_mz_min = self.mz_min
            self.view_mz_max = self.mz_max

            self.current_file = filepath
            return True

        except Exception as e:
            import traceback
            print(f"Error processing mzML: {e}")
            traceback.print_exc()
            return False

    def load_mzml_sync(self, filepath: str) -> bool:
        """Load mzML file synchronously without UI updates (for background thread).

        This is a convenience method that calls both parse and process phases.
        """
        if not self.parse_mzml_file(filepath):
            return False
        return self.process_mzml_data(filepath)

    def load_mzml(self, filepath: str) -> bool:
        """Load mzML file and extract peak data (with UI updates)."""
        try:
            filename = Path(filepath).name
            self.set_loading(True, f"Reading {filename}...")
            if self.status_label:
                self.status_label.set_text(f"Reading {filename}...")
            ui.notify(f"Loading {filepath}...", type="info")

            self.exp = MSExperiment()
            MzMLFile().load(filepath, self.exp)

            n_spectra = self.exp.size()
            self.update_loading_progress(f"Loaded {n_spectra:,} spectra, counting peaks...", 0.05)

            total_peaks = sum(spec.size() for spec in self.exp)

            if total_peaks == 0:
                ui.notify("No peaks found in file!", type="warning")
                return False

            self.update_loading_progress(f"Found {total_peaks:,} peaks, detecting FAIMS...", 0.10)

            # First pass: detect FAIMS CVs
            cv_set = set()
            for spec in self.exp:
                if spec.getMSLevel() == 1:
                    cv = self._get_cv_from_spectrum(spec)
                    if cv is not None:
                        cv_set.add(cv)

            self.has_faims = len(cv_set) > 1
            self.faims_cvs = sorted(cv_set) if self.has_faims else []

            self.update_loading_progress(f"Extracting {total_peaks:,} peaks...")

            # Data structures for peak extraction
            rts = np.empty(total_peaks, dtype=np.float32)
            mzs = np.empty(total_peaks, dtype=np.float32)
            intensities = np.empty(total_peaks, dtype=np.float32)
            cvs = np.empty(total_peaks, dtype=np.float32) if self.has_faims else None

            # Also compute TIC (overall and per-CV)
            tic_rts = []
            tic_intensities = []
            faims_tic_data = {cv: {"rt": [], "int": []} for cv in self.faims_cvs} if self.has_faims else {}

            # Determine TIC source: MS1 TIC or fallback to MS2+ BPC
            total_ms1 = sum(1 for spec in self.exp if spec.getMSLevel() == 1)
            if total_ms1 > 0:
                tic_ms_level = 1
                self.tic_source = "MS1 TIC"
            else:
                # No MS1 spectra - find the lowest available MS level > 1
                ms_levels = set(spec.getMSLevel() for spec in self.exp)
                tic_ms_level = min(lv for lv in ms_levels if lv > 1) if ms_levels else 2
                self.tic_source = f"MS{tic_ms_level} BPC"

            idx = 0
            spec_count = 0
            progress_interval = max(1, n_spectra // 20)  # Update progress ~20 times
            for spec in self.exp:
                spec_count += 1
                if spec_count % progress_interval == 0:
                    pct = int(100 * spec_count / n_spectra)
                    self.update_loading_progress(f"Extracting peaks... {pct}% ({idx:,} peaks)")

                ms_level = spec.getMSLevel()
                # Process spectra for TIC and peak map
                if ms_level != 1 and ms_level != tic_ms_level:
                    continue

                rt = spec.getRT()
                mz_array, int_array = spec.get_peaks()
                n = len(mz_array)

                cv = self._get_cv_from_spectrum(spec) if self.has_faims else None

                if n > 0:
                    # Only add to peak map DataFrame if MS1
                    if ms_level == 1:
                        rts[idx : idx + n] = rt
                        mzs[idx : idx + n] = mz_array
                        intensities[idx : idx + n] = int_array
                        if self.has_faims and cv is not None:
                            cvs[idx : idx + n] = cv
                        idx += n

                    # TIC/BPC calculation for the selected level
                    if ms_level == tic_ms_level:
                        if tic_ms_level == 1:
                            # MS1: use sum (TIC)
                            tic_value = float(np.sum(int_array))
                        else:
                            # MS2+: use base peak (BPC)
                            tic_value = float(np.max(int_array))

                        tic_rts.append(rt)
                        tic_intensities.append(tic_value)

                        # Per-CV TIC
                        if self.has_faims and cv is not None:
                            faims_tic_data[cv]["rt"].append(rt)
                            faims_tic_data[cv]["int"].append(tic_value)

            self.update_loading_progress("Building data structures...")

            rts = rts[:idx]
            mzs = mzs[:idx]
            intensities = intensities[:idx]
            if self.has_faims:
                cvs = cvs[:idx]

            # Store TIC data (sorted by RT to avoid zig-zag line plots)
            tic_rt_arr = np.array(tic_rts, dtype=np.float32)
            tic_int_arr = np.array(tic_intensities, dtype=np.float32)
            sort_idx = np.argsort(tic_rt_arr)
            self.tic_rt = tic_rt_arr[sort_idx]
            self.tic_intensity = tic_int_arr[sort_idx]

            # Store per-CV TIC data (also sorted by RT)
            self.faims_tic = {}
            for cv in self.faims_cvs:
                cv_rt = np.array(faims_tic_data[cv]["rt"], dtype=np.float32)
                cv_int = np.array(faims_tic_data[cv]["int"], dtype=np.float32)
                cv_sort_idx = np.argsort(cv_rt)
                self.faims_tic[cv] = (cv_rt[cv_sort_idx], cv_int[cv_sort_idx])

            self.update_loading_progress("Extracting spectrum metadata...")

            # Extract spectrum metadata for browser
            self.spectrum_data = self._extract_spectrum_data()

            self.update_loading_progress(f"Creating DataFrame ({idx:,} peaks)...")

            # Create main DataFrame
            self.df = pd.DataFrame({"rt": rts, "mz": mzs, "intensity": intensities})
            if self.has_faims:
                self.df["cv"] = cvs
            self.df["log_intensity"] = np.log1p(self.df["intensity"])

            self.update_loading_progress("Finalizing...")

            # Create per-CV DataFrames for FAIMS view
            self.faims_data = {}
            if self.has_faims:
                for cv in self.faims_cvs:
                    cv_df = self.df[self.df["cv"] == cv].copy()
                    self.faims_data[cv] = cv_df

            self.rt_min = float(self.df["rt"].min())
            self.rt_max = float(self.df["rt"].max())
            self.mz_min = float(self.df["mz"].min())
            self.mz_max = float(self.df["mz"].max())

            self.view_rt_min = self.rt_min
            self.view_rt_max = self.rt_max
            self.view_mz_min = self.mz_min
            self.view_mz_max = self.mz_max

            self.current_file = filepath

            # Build info text
            info_text = f"Loaded: {Path(filepath).name} | Spectra: {self.exp.size():,} | Peaks: {len(self.df):,}"
            if self.has_faims:
                info_text += f" | FAIMS: {len(self.faims_cvs)} CVs"

            if self.info_label:
                self.info_label.set_text(info_text)
            if self.status_label:
                self.status_label.set_text("Ready")

            # Update FAIMS UI
            if self.has_faims:
                if self.faims_info_label:
                    cv_str = ", ".join([f"{cv:.1f}V" for cv in self.faims_cvs])
                    self.faims_info_label.set_text(f"FAIMS CVs detected: {cv_str}")
                    self.faims_info_label.set_visibility(True)
                if self.faims_toggle:
                    self.faims_toggle.set_visibility(True)
                # Create FAIMS image elements
                if hasattr(self, "_create_faims_images") and self._create_faims_images:
                    self._create_faims_images()
            else:
                if self.faims_info_label:
                    self.faims_info_label.set_visibility(False)
                if self.faims_toggle:
                    self.faims_toggle.set_visibility(False)
                    self.show_faims_view = False
                if self.faims_container:
                    self.faims_container.set_visibility(False)

            # Update spectrum browser table
            if self.spectrum_table is not None:
                self.spectrum_table.rows = self.spectrum_data

            ui.notify(f"Loaded {len(self.df):,} peaks", type="positive")
            if self.has_faims:
                ui.notify(f"FAIMS data detected: {len(self.faims_cvs)} compensation voltages", type="info")

            self.set_loading(False)
            # NiceGUI 3.x: Emit event for state management
            self._emit_data_loaded("mzml")
            return True

        except Exception as e:
            self.set_loading(False)
            if self.status_label:
                self.status_label.set_text(f"Error: {e}")
            ui.notify(f"Error loading file: {e}", type="negative")
            return False

    def _extract_feature_data(self) -> list[dict[str, Any]]:
        """Extract feature data for table display."""
        if self.feature_map is None:
            return []

        data = []
        for idx, feature in enumerate(self.feature_map):
            rt = feature.getRT()
            mz = feature.getMZ()
            intensity = feature.getIntensity()
            charge = feature.getCharge()
            quality = feature.getOverallQuality()

            hulls = feature.getConvexHulls()
            rt_width = 0
            mz_width = 0
            if hulls:
                all_points = []
                for hull in hulls:
                    points = hull.getHullPoints()
                    all_points.extend([(p[0], p[1]) for p in points])
                if all_points:
                    rt_coords = [p[0] for p in all_points]
                    mz_coords = [p[1] for p in all_points]
                    rt_width = max(rt_coords) - min(rt_coords)
                    mz_width = max(mz_coords) - min(mz_coords)

            data.append(
                {
                    "idx": idx,
                    "rt": round(rt, 2),
                    "mz": round(mz, 4),
                    "intensity": f"{intensity:.2e}",
                    "charge": charge if charge != 0 else "-",
                    "quality": round(quality, 3) if quality > 0 else "-",
                    "rt_width": round(rt_width, 2) if rt_width > 0 else "-",
                    "mz_width": round(mz_width, 4) if mz_width > 0 else "-",
                }
            )

        return data

    def _extract_chromatograms(self) -> None:
        """Extract chromatogram data from the experiment.

        Extracts all chromatograms (including TIC if stored in file) and stores
        metadata in self.chromatograms and data in self.chromatogram_data.
        TIC chromatograms are marked with is_tic=True in metadata.
        """
        if self.exp is None:
            self.chromatograms = []
            self.chromatogram_data = {}
            self.has_chromatograms = False
            return

        chroms = self.exp.getChromatograms()
        if len(chroms) == 0:
            self.chromatograms = []
            self.chromatogram_data = {}
            self.has_chromatograms = False
            return

        self.chromatograms = []
        self.chromatogram_data = {}

        for idx, chrom in enumerate(chroms):
            native_id = chrom.getNativeID()

            # Check if this is a TIC chromatogram (stored in file, not computed)
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
            self.chromatograms.append({
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
            })

            # Store data arrays
            self.chromatogram_data[idx] = (
                np.array(rt_array, dtype=np.float32),
                np.array(int_array, dtype=np.float32),
            )

        self.has_chromatograms = len(self.chromatograms) > 0
        self.selected_chromatogram_indices = []  # Reset selection

    def _extract_ion_mobility_data(self) -> None:
        """Extract ion mobility data from spectra that contain IM arrays.

        For IM data, each spectrum stores a whole frame with concatenated peaks
        from multiple IM scans. The IM value for each peak is stored in a parallel
        float data array.

        Creates self.im_df with columns: mz, im, intensity, log_intensity
        """
        if self.exp is None:
            self.has_ion_mobility = False
            self.im_df = None
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
        for spec in self.exp:
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
            self.has_ion_mobility = False
            self.im_df = None
            return

        # Determine IM type and unit for display
        name_lower = detected_im_name.lower()
        if "inverse" in name_lower or "1/k0" in name_lower:
            self.im_type = "inverse_k0"
            self.im_unit = "Vs/cm²"
        elif "drift" in name_lower:
            self.im_type = "drift_time"
            self.im_unit = "ms"
        else:
            self.im_type = "ion_mobility"
            self.im_unit = ""

        # Second pass: extract all IM data
        all_mz = []
        all_im = []
        all_int = []

        for spec in self.exp:
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
            self.has_ion_mobility = False
            self.im_df = None
            return

        # Concatenate all arrays
        mz_concat = np.concatenate(all_mz)
        im_concat = np.concatenate(all_im)
        int_concat = np.concatenate(all_int)

        # Create DataFrame
        self.im_df = pd.DataFrame({
            "mz": mz_concat,
            "im": im_concat,
            "intensity": int_concat,
        })
        self.im_df["log_intensity"] = np.log1p(self.im_df["intensity"])

        # Set bounds
        self.im_min = float(self.im_df["im"].min())
        self.im_max = float(self.im_df["im"].max())
        # Ensure valid IM range (avoid zero range which causes division by zero)
        if self.im_max <= self.im_min:
            self.im_max = self.im_min + 1.0
        self.view_im_min = self.im_min
        self.view_im_max = self.im_max

        # Also update mz bounds from IM data if not already set
        im_mz_min = float(self.im_df["mz"].min())
        im_mz_max = float(self.im_df["mz"].max())
        if self.mz_min == 0 or im_mz_min < self.mz_min:
            self.mz_min = im_mz_min
        if self.mz_max == 0 or im_mz_max > self.mz_max:
            self.mz_max = im_mz_max
        if self.view_mz_min is None or self.view_mz_min < self.mz_min:
            self.view_mz_min = self.mz_min
        if self.view_mz_max is None or self.view_mz_max > self.mz_max:
            self.view_mz_max = self.mz_max

        self.has_ion_mobility = True

    def _extract_spectrum_data(self) -> list[dict[str, Any]]:
        """Extract spectrum metadata for the unified spectrum table.

        Includes fields for ID info (sequence, score) which are populated
        when IDs are loaded via _link_ids_to_spectra().
        """
        if self.exp is None:
            return []

        data = []
        for idx in range(self.exp.size()):
            spec = self.exp[idx]
            rt = spec.getRT()
            ms_level = spec.getMSLevel()
            n_peaks = spec.size()

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

            data.append(
                {
                    "idx": idx,
                    "rt": round(rt, 2),
                    "ms_level": ms_level,
                    "n_peaks": n_peaks,
                    "tic": f"{tic:.2e}",
                    "bpi": f"{bpi:.2e}",
                    "mz_range": f"{mz_min:.1f}-{mz_max:.1f}" if n_peaks > 0 else "-",
                    "precursor_mz": precursor_mz,
                    "precursor_z": precursor_charge,
                    # ID fields - populated by _link_ids_to_spectra()
                    "sequence": "-",
                    "full_sequence": "",
                    "score": "-",
                    "id_idx": None,  # Index into peptide_ids list
                }
            )

        return data

    def _link_ids_to_spectra(self, rt_tolerance: float = 5.0, mz_tolerance: float = 0.5) -> None:
        """Link peptide IDs to spectra by matching RT and precursor m/z.

        Updates spectrum_data with ID info (sequence, score) for matching MS2 spectra.
        Also collects meta value keys from PeptideIdentification and PeptideHit.
        """
        if not self.peptide_ids or not self.spectrum_data:
            return

        # Collect unique meta value keys from all IDs
        meta_keys_set = set()
        for pep_id in self.peptide_ids:
            # Get PeptideIdentification meta values
            pid_keys = []
            pep_id.getKeys(pid_keys)
            for key in pid_keys:
                meta_keys_set.add(f"pid:{key.decode() if isinstance(key, bytes) else key}")

            # Get PeptideHit meta values
            hits = pep_id.getHits()
            if hits:
                hit_keys = []
                hits[0].getKeys(hit_keys)
                for key in hit_keys:
                    meta_keys_set.add(f"hit:{key.decode() if isinstance(key, bytes) else key}")

        self.id_meta_keys = sorted(meta_keys_set)

        # Build lookup of spectrum indices by approximate RT for faster matching
        for spec_row in self.spectrum_data:
            # Clear any existing ID info
            spec_row["sequence"] = "-"
            spec_row["full_sequence"] = ""
            spec_row["score"] = "-"
            spec_row["id_idx"] = None
            spec_row["hit_rank"] = "-"
            spec_row["all_hits"] = []  # Store all hits for this spectrum
            # Initialize meta value fields
            for meta_key in self.id_meta_keys:
                spec_row[meta_key] = "-"

        # For each ID, find matching spectrum
        for id_idx, pep_id in enumerate(self.peptide_ids):
            id_rt = pep_id.getRT()
            id_mz = pep_id.getMZ()

            hits = pep_id.getHits()
            if not hits:
                continue

            # Collect PeptideIdentification meta values (shared by all hits)
            pid_meta_values = {}
            pid_keys = []
            pep_id.getKeys(pid_keys)
            for key in pid_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                value = pep_id.getMetaValue(key)
                if isinstance(value, bytes):
                    value = value.decode()
                elif isinstance(value, float):
                    value = round(value, 4)
                pid_meta_values[f"pid:{key_str}"] = value

            # Collect data for all hits (rank = position in list, 1-indexed)
            all_hits_data = []
            for hit_idx, hit in enumerate(hits):
                sequence = hit.getSequence().toString()
                score = hit.getScore()
                charge = hit.getCharge()

                # Collect hit-specific meta values
                hit_meta_values = dict(pid_meta_values)  # Start with PID meta values
                hit_keys = []
                hit.getKeys(hit_keys)
                for key in hit_keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    value = hit.getMetaValue(key)
                    if isinstance(value, bytes):
                        value = value.decode()
                    elif isinstance(value, float):
                        value = round(value, 4)
                    hit_meta_values[f"hit:{key_str}"] = value

                all_hits_data.append(
                    {
                        "sequence": sequence[:25] + "..." if len(sequence) > 25 else sequence,
                        "full_sequence": sequence,
                        "score": round(score, 4) if score != 0 else "-",
                        "charge": charge,
                        "hit_rank": hit_idx + 1,  # 1-indexed position in list
                        "id_idx": id_idx,
                        "hit_idx": hit_idx,  # 0-indexed for internal use
                        "meta_values": hit_meta_values,
                    }
                )

            # Find best matching MS2 spectrum
            best_spec_idx = None
            best_rt_diff = float("inf")

            for spec_row in self.spectrum_data:
                if spec_row["ms_level"] != 2:
                    continue

                spec_rt = spec_row["rt"]
                spec_prec_mz = spec_row["precursor_mz"]

                if spec_prec_mz == "-":
                    continue

                if abs(spec_rt - id_rt) <= rt_tolerance and abs(spec_prec_mz - id_mz) <= mz_tolerance:
                    rt_diff = abs(spec_rt - id_rt)
                    if rt_diff < best_rt_diff:
                        best_rt_diff = rt_diff
                        best_spec_idx = spec_row["idx"]

            # Update spectrum row with ID info
            if best_spec_idx is not None:
                for spec_row in self.spectrum_data:
                    if spec_row["idx"] == best_spec_idx:
                        # Only update if this is a better match (closer RT) or no existing match
                        if spec_row["id_idx"] is None or best_rt_diff < spec_row.get("_rt_diff", float("inf")):
                            best_hit_data = all_hits_data[0]
                            spec_row["sequence"] = best_hit_data["sequence"]
                            spec_row["full_sequence"] = best_hit_data["full_sequence"]
                            spec_row["score"] = best_hit_data["score"]
                            spec_row["id_idx"] = id_idx
                            spec_row["hit_rank"] = 1
                            spec_row["all_hits"] = all_hits_data
                            spec_row["_rt_diff"] = best_rt_diff
                            # Add meta values from best hit
                            for key, value in best_hit_data["meta_values"].items():
                                spec_row[key] = value
                            # Also update charge from ID if precursor charge is missing
                            if spec_row["precursor_z"] == "-" and best_hit_data["charge"] > 0:
                                spec_row["precursor_z"] = best_hit_data["charge"]
                        break

    def show_spectrum_in_browser(self, spectrum_idx: int):
        """Display a spectrum in the 1D browser view. Shows annotations if matching ID exists."""
        if self.exp is None or spectrum_idx < 0 or spectrum_idx >= self.exp.size():
            return

        self.selected_spectrum_idx = spectrum_idx
        spec = self.exp[spectrum_idx]

        mz_array, int_array = spec.get_peaks()
        rt = spec.getRT()
        ms_level = spec.getMSLevel()

        if len(mz_array) == 0:
            ui.notify("Spectrum is empty", type="warning")
            return

        # Check if there's a matching peptide ID for annotation
        matching_id_idx = self.find_matching_id_for_spectrum(spectrum_idx)

        if matching_id_idx is not None:
            # Use annotated spectrum display
            pep_id = self.peptide_ids[matching_id_idx]
            hits = pep_id.getHits()
            if hits:
                best_hit = hits[0]
                sequence_str = best_hit.getSequence().toString()
                charge = best_hit.getCharge()
                precursors = spec.getPrecursors()
                prec_mz = precursors[0].getMZ() if precursors else pep_id.getMZ()

                # Get peak annotations if enabled
                peak_annotations = None
                if self.annotate_peaks:
                    # First check for external peak annotations (from specialized tools like OpenNuXL)
                    peak_annotations = get_external_peak_annotations(
                        best_hit, mz_array, tolerance_da=self.annotation_tolerance_da
                    )

                    if not peak_annotations:
                        # Fall back to generating annotations with SpectrumAnnotator
                        peak_annotations = annotate_spectrum_with_id(
                            spec, best_hit, tolerance_da=self.annotation_tolerance_da
                        )

                # Create annotated spectrum plot
                fig = create_annotated_spectrum_plot(
                    mz_array,
                    int_array,
                    sequence_str,
                    charge,
                    prec_mz,
                    peak_annotations=peak_annotations,
                    annotate=self.annotate_peaks,
                    mirror_mode=self.mirror_annotation_view,
                )

                # Update title to include spectrum index
                title = f"Spectrum #{spectrum_idx} | {sequence_str} (z={charge}+) | RT={rt:.2f}s"
                fig.update_layout(
                    title={"text": title, "font": {"size": 14}},
                    height=350,
                    uirevision="spectrum_stable",  # Stable key to preserve zoom/pan state
                )

                # Always apply saved zoom range if available
                if self.spectrum_zoom_range is not None:
                    fig.update_xaxes(range=list(self.spectrum_zoom_range))

                # Update info label with ID info
                if self.spectrum_browser_info is not None:
                    self.spectrum_browser_info.set_text(
                        f"RT: {rt:.2f}s | ID: {sequence_str} | Charge: {charge}+ | Precursor: {prec_mz:.4f}"
                    )
            else:
                # No hits, fall back to regular display
                matching_id_idx = None

        if matching_id_idx is None:
            # Regular spectrum display (no annotation)
            max_int = float(int_array.max()) if len(int_array) > 0 else 1.0

            # Choose intensity values based on display mode
            if self.spectrum_intensity_percent:
                int_display = (int_array / max_int) * 100
                y_title = "Relative Intensity (%)"
                hover_fmt = "m/z: %{x:.4f}<br>Intensity: %{y:.1f}%<extra></extra>"
                y_range = [0, 105]
            else:
                int_display = int_array
                y_title = "Intensity"
                hover_fmt = "m/z: %{x:.4f}<br>Intensity: %{y:.2e}<extra></extra>"
                y_range = [0, max_int * 1.05]

            # Create figure
            fig = go.Figure()

            # Color based on theme (same color for all MS levels)
            # Light mode: black; Dark mode: cyan
            is_dark = self.dark.value if self.dark else True
            color = "#00d4ff" if is_dark else "#000000"

            # Add spectrum as vertical lines (stem plot) - doesn't get thicker when zooming
            # Create x, y arrays for stem plot: each peak is [mz, mz, None], [0, intensity, None]
            x_stems = []
            y_stems = []
            for mz, intensity in zip(mz_array, int_display):
                x_stems.extend([mz, mz, None])
                y_stems.extend([0, intensity, None])

            fig.add_trace(
                go.Scatter(x=x_stems, y=y_stems, mode="lines", line={"color": color, "width": 1}, hoverinfo="skip")
            )

            # Add invisible hover points at peak tops (opacity 0 hides markers but keeps hover)
            fig.add_trace(
                go.Scatter(
                    x=mz_array,
                    y=int_display,
                    mode="markers",
                    marker={"color": color, "size": 8, "opacity": 0},
                    hovertemplate=hover_fmt,
                )
            )

            # Title with spectrum info
            title = f"Spectrum #{spectrum_idx} | MS{ms_level} | RT={rt:.2f}s | {len(mz_array):,} peaks"

            # Add precursor line for MS2+
            if ms_level > 1:
                precursors = spec.getPrecursors()
                if precursors:
                    prec_mz = precursors[0].getMZ()
                    fig.add_vline(
                        x=prec_mz, line_dash="dash", line_color="orange", annotation_text=f"Precursor ({prec_mz:.2f})"
                    )
                    title += f" | Precursor: {prec_mz:.4f}"

            # Use transparent backgrounds for light/dark mode compatibility
            fig.update_layout(
                title={"text": title, "font": {"size": 14, "color": "#888"}},
                xaxis_title="m/z",
                yaxis_title=y_title,
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent outer background
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
                height=350,
                margin={"l": 60, "r": 20, "t": 50, "b": 50},
                showlegend=False,
                modebar={"remove": ["lasso2d", "select2d"]},  # Remove lasso and box select tools
                font={"color": "#888"},  # Gray text works on both light and dark
                uirevision="spectrum_stable",  # Stable key to preserve zoom/pan state
            )

            # Apply saved zoom range if available (for measurement mode or auto-scale)
            if self.spectrum_zoom_range is not None:
                fig.update_xaxes(
                    range=list(self.spectrum_zoom_range), showgrid=False, linecolor="#888", tickcolor="#888"
                )
                # Auto-scale y-axis to visible peaks if enabled
                if self.spectrum_auto_scale:
                    xmin, xmax = self.spectrum_zoom_range
                    visible_mask = (mz_array >= xmin) & (mz_array <= xmax)
                    if np.any(visible_mask):
                        visible_max = float(int_display[visible_mask].max())
                        y_range = [0, visible_max / 0.95]  # Scale so max peak is at 95%
            else:
                fig.update_xaxes(showgrid=False, linecolor="#888", tickcolor="#888")
            fig.update_yaxes(range=y_range, showgrid=False, fixedrange=True, linecolor="#888", tickcolor="#888")

            # Update info label
            if self.spectrum_browser_info is not None:
                tic = float(np.sum(int_array))
                mz_range = f"{mz_array.min():.2f} - {mz_array.max():.2f}" if len(mz_array) > 0 else "-"
                self.spectrum_browser_info.set_text(
                    f"RT: {rt:.2f}s | MS Level: {ms_level} | Peaks: {len(mz_array):,} | TIC: {tic:.2e} | m/z: {mz_range}"
                )

        # Add any stored measurements for this spectrum
        self.add_spectrum_measurements_to_figure(fig, spectrum_idx, mz_array, int_array)

        # Add peak annotations (custom labels and m/z labels)
        self.add_peak_annotations_to_figure(fig, spectrum_idx, mz_array, int_array)

        # Add hover highlights and measurement preview
        self.add_spectrum_highlight_to_figure(fig, mz_array, int_array)

        # Update plot
        if self.spectrum_browser_plot is not None:
            self.spectrum_browser_plot.update_figure(fig)

        # Update navigation label
        if self.spectrum_nav_label is not None:
            self.spectrum_nav_label.set_text(f"Spectrum {spectrum_idx + 1} of {self.exp.size()}")

        # Update spectrum table selection
        if self.spectrum_table is not None:
            # Find the row data for this spectrum
            matching_rows = [row for row in self.spectrum_data if row["idx"] == spectrum_idx]
            if matching_rows:
                self.spectrum_table.selected = matching_rows

        # Update TIC to show spectrum marker
        self.update_tic_plot()

        # Update peak map to show the spectrum marker
        if self.show_spectrum_marker and self.df is not None:
            self.update_plot()

        # NiceGUI 3.x: Emit selection change event
        self._emit_selection_changed("spectrum", spectrum_idx)

    def navigate_spectrum(self, direction: int):
        """Navigate to prev/next spectrum."""
        if self.exp is None or self.exp.size() == 0:
            return

        if self.selected_spectrum_idx is None:
            new_idx = 0
        else:
            new_idx = self.selected_spectrum_idx + direction

        # Clamp to valid range
        new_idx = max(0, min(self.exp.size() - 1, new_idx))

        # Clear zoom range when navigating to different spectrum
        if new_idx != self.selected_spectrum_idx:
            self.spectrum_zoom_range = None

        self.show_spectrum_in_browser(new_idx)

    # ==================== Spectrum Measurement Methods ====================

    def snap_to_peak(
        self, target_mz: float, mz_array: np.ndarray, int_array: np.ndarray, target_int: float | None = None
    ) -> tuple[float, float] | None:
        """Snap to the nearest peak using 2D distance (m/z and intensity). Returns (mz, intensity) or None."""
        if len(mz_array) == 0:
            return None

        # Normalize both dimensions to [0, 1] for fair distance comparison
        mz_min, mz_max = float(mz_array.min()), float(mz_array.max())
        int_min, int_max = float(int_array.min()), float(int_array.max())

        mz_range = mz_max - mz_min if mz_max > mz_min else 1.0
        int_range = int_max - int_min if int_max > int_min else 1.0

        # Normalize arrays
        mz_norm = (mz_array - mz_min) / mz_range
        target_mz_norm = (target_mz - mz_min) / mz_range

        if target_int is not None:
            # Use 2D Euclidean distance (m/z and intensity)
            int_norm = (int_array - int_min) / int_range
            target_int_norm = (target_int - int_min) / int_range
            distances = np.sqrt((mz_norm - target_mz_norm) ** 2 + (int_norm - target_int_norm) ** 2)
        else:
            # Fall back to m/z-only distance
            distances = np.abs(mz_norm - target_mz_norm)

        idx = distances.argmin()
        snapped_mz = float(mz_array[idx])
        snapped_int = float(int_array[idx])

        # Only snap if within a reasonable tolerance (1% of m/z range or 1 Da, whichever is larger)
        tolerance = max(1.0, mz_range * 0.01)

        if abs(snapped_mz - target_mz) > tolerance:
            return None

        return (snapped_mz, snapped_int)

    def add_spectrum_measurements_to_figure(
        self, fig: go.Figure, spectrum_idx: int, mz_array: np.ndarray, int_array: np.ndarray
    ):
        """Add stored measurements for this spectrum to the figure."""
        if spectrum_idx not in self.spectrum_measurements:
            return

        max_int = float(int_array.max()) if len(int_array) > 0 else 1.0

        for mz1, int1, mz2, int2 in self.spectrum_measurements[spectrum_idx]:
            # Convert to display intensities (percentage if enabled)
            if self.spectrum_intensity_percent:
                y1 = (int1 / max_int) * 100
                y2 = (int2 / max_int) * 100
                max_bracket = 90  # Cap at 90% to stay in visible area
            else:
                y1, y2 = int1, int2
                max_bracket = max_int * 0.9

            # Draw horizontal bracket at height slightly above the higher peak, capped at 90%
            bracket_y = min(max(y1, y2) * 1.1, max_bracket)

            # Horizontal line between the two m/z values (orange works in light/dark)
            fig.add_shape(
                type="line",
                x0=mz1,
                y0=bracket_y,
                x1=mz2,
                y1=bracket_y,
                line={"color": "#ff8800", "width": 2},
            )

            # Vertical lines down to each peak
            fig.add_shape(
                type="line",
                x0=mz1,
                y0=y1,
                x1=mz1,
                y1=bracket_y,
                line={"color": "#ff8800", "width": 1, "dash": "dot"},
            )
            fig.add_shape(
                type="line",
                x0=mz2,
                y0=y2,
                x1=mz2,
                y1=bracket_y,
                line={"color": "#ff8800", "width": 1, "dash": "dot"},
            )

            # Calculate delta m/z
            delta_mz = abs(mz2 - mz1)
            mid_mz = (mz1 + mz2) / 2

            # Add annotation with delta m/z
            fig.add_annotation(
                x=mid_mz,
                y=bracket_y,
                text=f"Δ{delta_mz:.4f}",
                showarrow=False,
                yshift=12,
                font={"color": "#ff8800", "size": 11},
                borderpad=2,
            )

    def clear_spectrum_measurement(self, spectrum_idx: int | None = None):
        """Clear measurement(s) for spectrum. If spectrum_idx is None, clear for current spectrum."""
        idx = spectrum_idx if spectrum_idx is not None else self.selected_spectrum_idx
        if idx is not None and idx in self.spectrum_measurements:
            del self.spectrum_measurements[idx]
            # Refresh display
            self.show_spectrum_in_browser(idx)

    def delete_selected_measurement(self):
        """Delete the currently selected measurement."""
        if self.selected_spectrum_idx is None:
            return
        if self.spectrum_selected_measurement_idx is None:
            return
        if self.selected_spectrum_idx not in self.spectrum_measurements:
            return

        measurements = self.spectrum_measurements[self.selected_spectrum_idx]
        if 0 <= self.spectrum_selected_measurement_idx < len(measurements):
            measurements.pop(self.spectrum_selected_measurement_idx)
            if len(measurements) == 0:
                del self.spectrum_measurements[self.selected_spectrum_idx]
            self.spectrum_selected_measurement_idx = None
            self.show_spectrum_in_browser(self.selected_spectrum_idx)
            ui.notify("Measurement deleted", type="info")

    def find_measurement_at_position(self, mz: float, y: float) -> int | None:
        """Find if a click position is near an existing measurement line. Returns measurement index or None."""
        if self.selected_spectrum_idx is None:
            return None
        if self.selected_spectrum_idx not in self.spectrum_measurements:
            return None

        spec = self.exp[self.selected_spectrum_idx]
        mz_array, int_array = spec.get_peaks()
        if len(mz_array) == 0:
            return None

        max_int = float(int_array.max())
        mz_range = float(mz_array.max() - mz_array.min())
        mz_tolerance = mz_range * 0.02  # 2% of m/z range

        measurements = self.spectrum_measurements[self.selected_spectrum_idx]
        for i, (mz1, int1, mz2, int2) in enumerate(measurements):
            # Convert to display intensities
            if self.spectrum_intensity_percent:
                y1 = (int1 / max_int) * 100
                y2 = (int2 / max_int) * 100
            else:
                y1, y2 = int1, int2

            bracket_y = max(y1, y2) * 1.1

            # Check if click is near the horizontal bracket line
            if min(mz1, mz2) - mz_tolerance <= mz <= max(mz1, mz2) + mz_tolerance:
                y_tolerance = bracket_y * 0.1
                if abs(y - bracket_y) < y_tolerance:
                    return i

            # Check if click is near the vertical connector lines
            if abs(mz - mz1) < mz_tolerance and min(y1, bracket_y) <= y <= max(y1, bracket_y):
                return i
            if abs(mz - mz2) < mz_tolerance and min(y2, bracket_y) <= y <= max(y2, bracket_y):
                return i

        return None

    def add_spectrum_highlight_to_figure(self, fig: go.Figure, mz_array: np.ndarray, int_array: np.ndarray):
        """Add hover highlight marker and measurement preview to figure."""
        if len(mz_array) == 0:
            return

        max_int = float(int_array.max()) if len(int_array) > 0 else 1.0

        # Add marker for locked-in start point (distinct from hover)
        if self.spectrum_measure_start is not None and self.spectrum_measure_mode:
            start_mz, start_int = self.spectrum_measure_start
            if self.spectrum_intensity_percent:
                start_y = (start_int / max_int) * 100
            else:
                start_y = start_int

            # Orange circle marker for the locked-in start point (works in light/dark)
            fig.add_trace(
                go.Scatter(
                    x=[start_mz],
                    y=[start_y],
                    mode="markers",
                    marker={"color": "#ff8800", "size": 10, "symbol": "circle", "line": {"width": 1, "color": "#333"}},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Add hover highlight marker to show nearest (snap) peak - works in all modes
        if self.spectrum_hover_peak is not None:
            hover_mz, hover_int = self.spectrum_hover_peak
            if self.spectrum_intensity_percent:
                hover_y = (hover_int / max_int) * 100
            else:
                hover_y = hover_int

            # Add a highlighted ring around the hovered peak
            # Orange in measure mode, dark blue otherwise (works in light/dark)
            highlight_color = "#ff8800" if self.spectrum_measure_mode else "#0077cc"
            fig.add_trace(
                go.Scatter(
                    x=[hover_mz],
                    y=[hover_y],
                    mode="markers",
                    marker={"color": highlight_color, "size": 12, "symbol": "circle-open", "line": {"width": 2}},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Add preview line from start point to hover point
        if self.spectrum_measure_start is not None and self.spectrum_hover_peak is not None:
            start_mz, start_int = self.spectrum_measure_start
            hover_mz, hover_int = self.spectrum_hover_peak

            if self.spectrum_intensity_percent:
                y1 = (start_int / max_int) * 100
                y2 = (hover_int / max_int) * 100
                max_bracket = 90  # Cap at 90% to stay in visible area
            else:
                y1, y2 = start_int, hover_int
                max_bracket = max_int * 0.9

            bracket_y = min(max(y1, y2) * 1.1, max_bracket)

            # Preview horizontal line (dashed, semi-transparent orange)
            fig.add_shape(
                type="line",
                x0=start_mz,
                y0=bracket_y,
                x1=hover_mz,
                y1=bracket_y,
                line={"color": "rgba(255, 136, 0, 0.6)", "width": 2, "dash": "dash"},
            )

            # Preview vertical connectors
            fig.add_shape(
                type="line",
                x0=start_mz,
                y0=y1,
                x1=start_mz,
                y1=bracket_y,
                line={"color": "rgba(255, 136, 0, 0.6)", "width": 1, "dash": "dot"},
            )
            fig.add_shape(
                type="line",
                x0=hover_mz,
                y0=y2,
                x1=hover_mz,
                y1=bracket_y,
                line={"color": "rgba(255, 136, 0, 0.6)", "width": 1, "dash": "dot"},
            )

            # Preview delta annotation
            delta_mz = abs(hover_mz - start_mz)
            mid_mz = (start_mz + hover_mz) / 2
            fig.add_annotation(
                x=mid_mz,
                y=bracket_y,
                text=f"Δ{delta_mz:.4f}",
                showarrow=False,
                yshift=12,
                font={"color": "rgba(255, 136, 0, 0.7)", "size": 11},
                borderpad=2,
            )

        # Highlight selected measurement (for deletion)
        if self.spectrum_selected_measurement_idx is not None:
            if self.selected_spectrum_idx in self.spectrum_measurements:
                measurements = self.spectrum_measurements[self.selected_spectrum_idx]
                if 0 <= self.spectrum_selected_measurement_idx < len(measurements):
                    mz1, int1, mz2, int2 = measurements[self.spectrum_selected_measurement_idx]
                    if self.spectrum_intensity_percent:
                        y1 = (int1 / max_int) * 100
                        y2 = (int2 / max_int) * 100
                    else:
                        y1, y2 = int1, int2

                    bracket_y = max(y1, y2) * 1.1

                    # Add selection highlight (cyan glow effect via thicker line behind)
                    fig.add_shape(
                        type="line",
                        x0=mz1,
                        y0=bracket_y,
                        x1=mz2,
                        y1=bracket_y,
                        line={"color": "cyan", "width": 4},
                    )

    def add_peak_annotations_to_figure(
        self, fig: go.Figure, spectrum_idx: int, mz_array: np.ndarray, int_array: np.ndarray
    ):
        """Add peak annotations (m/z labels and custom labels) to the spectrum figure."""
        if len(mz_array) == 0:
            return

        max_int = float(int_array.max()) if len(int_array) > 0 else 1.0

        # Get custom annotations for this spectrum
        custom_annotations = self.peak_annotations.get(spectrum_idx, [])

        # Convert intensity to display units
        def to_display_y(intensity: float) -> float:
            if self.spectrum_intensity_percent:
                return (intensity / max_int) * 100
            return intensity

        # Add custom peak labels
        for ann in custom_annotations:
            mz, intensity, label = ann["mz"], ann["int"], ann["label"]
            y_val = to_display_y(intensity)

            # Add marker at the annotation position
            fig.add_trace(
                go.Scatter(
                    x=[mz],
                    y=[y_val],
                    mode="markers",
                    marker={"color": "#22cc88", "size": 8, "symbol": "diamond"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Add the label text
            display_label = label if label else f"{mz:.4f}"
            fig.add_annotation(
                x=mz,
                y=y_val,
                text=display_label,
                showarrow=True,
                arrowhead=0,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#22cc88",
                ax=0,
                ay=-25,
                font={"color": "#22cc88", "size": 10},
                bgcolor="rgba(0,0,0,0.5)",
                borderpad=2,
            )

        # Add m/z labels to interesting peaks if enabled (but not if we already have custom labels there)
        if self.show_mz_labels:
            # Get m/z values that already have custom annotations
            annotated_mz = {ann["mz"] for ann in custom_annotations}

            # Use scipy.signal.find_peaks to find interesting peaks:
            # - prominence: peak must stand out from surrounding baseline (5% of max intensity)
            # - distance: minimum separation between peaks (in array indices)
            min_prominence = max_int * 0.05  # 5% of max intensity
            peak_indices, properties = find_peaks(
                int_array,
                prominence=min_prominence,
                distance=max(1, len(mz_array) // 50),  # At least 2% spacing
            )

            # If find_peaks returns too many or too few, fall back to top N by prominence
            if len(peak_indices) > 15:
                # Sort by prominence and keep top 15
                prominences = properties.get("prominences", int_array[peak_indices])
                sorted_idx = np.argsort(prominences)[-15:]
                peak_indices = peak_indices[sorted_idx]
            elif len(peak_indices) == 0:
                # Fallback: just use top 10 by intensity
                peak_indices = np.argsort(int_array)[-10:]

            for idx in peak_indices:
                mz = float(mz_array[idx])
                # Skip if this peak already has a custom annotation
                if any(abs(mz - ann_mz) < 0.01 for ann_mz in annotated_mz):
                    continue

                intensity = float(int_array[idx])
                y_val = to_display_y(intensity)

                fig.add_annotation(
                    x=mz,
                    y=y_val,
                    text=f"{mz:.2f}",
                    showarrow=False,
                    yshift=10,
                    font={"color": "#888", "size": 9},
                )

    def add_or_edit_peak_annotation(self, spectrum_idx: int, mz: float, intensity: float, label: str | None = None):
        """Add or edit a peak annotation. If label is None, shows m/z value."""
        if spectrum_idx not in self.peak_annotations:
            self.peak_annotations[spectrum_idx] = []

        annotations = self.peak_annotations[spectrum_idx]

        # Check if annotation already exists at this m/z (within tolerance)
        for ann in annotations:
            if abs(ann["mz"] - mz) < 0.01:
                # Update existing annotation
                ann["label"] = label if label is not None else f"{mz:.4f}"
                return

        # Add new annotation
        annotations.append({"mz": mz, "int": intensity, "label": label if label is not None else f"{mz:.4f}"})

    def remove_peak_annotation(self, spectrum_idx: int, mz: float):
        """Remove a peak annotation at the given m/z."""
        if spectrum_idx not in self.peak_annotations:
            return

        annotations = self.peak_annotations[spectrum_idx]
        self.peak_annotations[spectrum_idx] = [ann for ann in annotations if abs(ann["mz"] - mz) >= 0.01]

    def clear_peak_annotations(self, spectrum_idx: int | None = None):
        """Clear peak annotations. If spectrum_idx is None, clears all annotations."""
        if spectrum_idx is None:
            self.peak_annotations.clear()
        elif spectrum_idx in self.peak_annotations:
            del self.peak_annotations[spectrum_idx]

    def navigate_spectrum_by_ms_level(self, direction: int, ms_level: int):
        """Navigate to prev/next spectrum of specific MS level."""
        if self.exp is None or self.exp.size() == 0:
            return

        start_idx = self.selected_spectrum_idx if self.selected_spectrum_idx is not None else 0

        if direction > 0:
            # Search forward
            for i in range(start_idx + 1, self.exp.size()):
                if self.exp[i].getMSLevel() == ms_level:
                    self.show_spectrum_in_browser(i)
                    return
        else:
            # Search backward
            for i in range(start_idx - 1, -1, -1):
                if self.exp[i].getMSLevel() == ms_level:
                    self.show_spectrum_in_browser(i)
                    return

        ui.notify(f"No more MS{ms_level} spectra in that direction", type="info")

    def load_featuremap_sync(self, filepath: str) -> bool:
        """Load featureXML file synchronously without UI updates."""
        try:
            self.feature_map = FeatureMap()
            FeatureXMLFile().load(filepath, self.feature_map)
            self.features_file = filepath
            self.selected_feature_idx = None
            self.feature_data = self._extract_feature_data()
            return True
        except Exception as e:
            print(f"Error loading features: {e}")
            return False

    def load_featuremap(self, filepath: str) -> bool:
        """Load featureXML file (with UI updates)."""
        try:
            self.set_loading(True, "Loading features...")
            if self.status_label:
                self.status_label.set_text(f"Loading features from {Path(filepath).name}...")
            ui.notify(f"Loading {filepath}...", type="info")

            self.feature_map = FeatureMap()
            FeatureXMLFile().load(filepath, self.feature_map)

            self.features_file = filepath
            self.selected_feature_idx = None
            self.feature_data = self._extract_feature_data()

            n_features = self.feature_map.size()
            if self.feature_info_label:
                self.feature_info_label.set_text(f"Features: {n_features:,}")
            if self.status_label:
                self.status_label.set_text("Ready")
            ui.notify(f"Loaded {n_features:,} features", type="positive")

            if self.feature_table is not None:
                self.feature_table.rows = self.feature_data

            self.set_loading(False)
            # NiceGUI 3.x: Emit event for state management
            self._emit_data_loaded("features")
            return True

        except Exception as e:
            self.set_loading(False)
            if self.status_label:
                self.status_label.set_text(f"Error: {e}")
            ui.notify(f"Error loading features: {e}", type="negative")
            return False

    def clear_features(self):
        """Clear loaded feature map."""
        self.feature_map = None
        self.features_file = None
        self.feature_data = []
        self.selected_feature_idx = None
        if self.feature_info_label:
            self.feature_info_label.set_text("Features: None")
        if self.feature_table is not None:
            self.feature_table.rows = []
        ui.notify("Features cleared", type="info")

    def _extract_id_data(self) -> list[dict[str, Any]]:
        """Extract peptide ID data for table display."""
        if not self.peptide_ids:
            return []

        data = []
        idx = 0
        for pep_id in self.peptide_ids:
            rt = pep_id.getRT()
            mz = pep_id.getMZ()
            hits = pep_id.getHits()

            if hits:
                best_hit = hits[0]
                sequence = best_hit.getSequence().toString()
                score = best_hit.getScore()
                charge = best_hit.getCharge()
            else:
                sequence = "-"
                score = 0
                charge = 0

            data.append(
                {
                    "idx": idx,
                    "rt": round(rt, 2),
                    "mz": round(mz, 4),
                    "sequence": sequence[:30] + "..." if len(sequence) > 30 else sequence,
                    "full_sequence": sequence,
                    "charge": charge if charge != 0 else "-",
                    "score": round(score, 4) if score != 0 else "-",
                }
            )
            idx += 1

        return data

    def load_idxml_sync(self, filepath: str) -> bool:
        """Load idXML file synchronously without UI updates."""
        try:
            self.protein_ids = []
            self.peptide_ids = []
            IdXMLFile().load(filepath, self.protein_ids, self.peptide_ids)
            self.id_file = filepath
            self.selected_id_idx = None
            self.id_data = self._extract_id_data()
            # Link IDs to spectra for unified table
            self._link_ids_to_spectra()
            return True
        except Exception as e:
            print(f"Error loading IDs: {e}")
            return False

    def load_idxml(self, filepath: str) -> bool:
        """Load idXML file with peptide identifications (with UI updates)."""
        try:
            self.set_loading(True, "Loading identifications...")
            if self.status_label:
                self.status_label.set_text(f"Loading IDs from {Path(filepath).name}...")
            ui.notify(f"Loading {filepath}...", type="info")

            self.protein_ids = []
            self.peptide_ids = []
            IdXMLFile().load(filepath, self.protein_ids, self.peptide_ids)

            self.id_file = filepath
            self.selected_id_idx = None
            self.id_data = self._extract_id_data()

            # Link IDs to spectra for unified table
            self._link_ids_to_spectra()

            n_ids = len(self.peptide_ids)
            n_linked = sum(1 for s in self.spectrum_data if s.get("id_idx") is not None)
            if self.id_info_label:
                self.id_info_label.set_text(f"IDs: {n_ids:,} ({n_linked} linked)")
            if self.status_label:
                self.status_label.set_text("Ready")
            ui.notify(f"Loaded {n_ids:,} peptide IDs ({n_linked} linked to spectra)", type="positive")

            # Update unified spectrum table
            if self.spectrum_table is not None:
                self.spectrum_table.rows = self.spectrum_data

            self.set_loading(False)
            # NiceGUI 3.x: Emit event for state management
            self._emit_data_loaded("ids")
            return True

        except Exception as e:
            self.set_loading(False)
            if self.status_label:
                self.status_label.set_text(f"Error: {e}")
            ui.notify(f"Error loading IDs: {e}", type="negative")
            return False

    def clear_ids(self):
        """Clear loaded identifications."""
        self.peptide_ids = []
        self.protein_ids = []
        self.id_file = None
        self.id_data = []
        self.selected_id_idx = None
        # Clear ID info from unified spectrum data
        for spec_row in self.spectrum_data:
            spec_row["sequence"] = "-"
            spec_row["full_sequence"] = ""
            spec_row["score"] = "-"
            spec_row["id_idx"] = None
        if self.id_info_label:
            self.id_info_label.set_text("IDs: None")
        # Update unified spectrum table
        if self.spectrum_table is not None:
            self.spectrum_table.rows = self.spectrum_data
        ui.notify("Identifications cleared", type="info")

    def find_ms2_spectrum(
        self, rt: float, precursor_mz: float, rt_tolerance: float = 5.0, mz_tolerance: float = 0.5
    ) -> Optional[MSSpectrum]:
        """Find the MS2 spectrum matching the given RT and precursor m/z."""
        if self.exp is None:
            return None

        best_spec = None
        best_rt_diff = float("inf")

        for spec in self.exp:
            if spec.getMSLevel() != 2:
                continue

            spec_rt = spec.getRT()
            if abs(spec_rt - rt) > rt_tolerance:
                continue

            # Check precursor m/z
            precursors = spec.getPrecursors()
            if precursors:
                prec_mz = precursors[0].getMZ()
                if abs(prec_mz - precursor_mz) <= mz_tolerance:
                    rt_diff = abs(spec_rt - rt)
                    if rt_diff < best_rt_diff:
                        best_rt_diff = rt_diff
                        best_spec = spec

        return best_spec

    def find_matching_id_for_spectrum(
        self, spectrum_idx: int, rt_tolerance: float = 5.0, mz_tolerance: float = 0.5
    ) -> Optional[int]:
        """Find peptide ID matching the given spectrum. Returns ID index or None."""
        if self.exp is None or not self.peptide_ids:
            return None

        if spectrum_idx < 0 or spectrum_idx >= self.exp.size():
            return None

        spec = self.exp[spectrum_idx]

        # Only MS2 spectra can have peptide IDs
        if spec.getMSLevel() != 2:
            return None

        spec_rt = spec.getRT()
        precursors = spec.getPrecursors()
        if not precursors:
            return None

        spec_prec_mz = precursors[0].getMZ()

        # Find best matching ID
        best_id_idx = None
        best_rt_diff = float("inf")

        for i, pep_id in enumerate(self.peptide_ids):
            id_rt = pep_id.getRT()
            id_mz = pep_id.getMZ()

            if abs(id_rt - spec_rt) <= rt_tolerance and abs(id_mz - spec_prec_mz) <= mz_tolerance:
                rt_diff = abs(id_rt - spec_rt)
                if rt_diff < best_rt_diff:
                    best_rt_diff = rt_diff
                    best_id_idx = i

        return best_id_idx

    def find_spectrum_for_id(self, id_idx: int, rt_tolerance: float = 5.0, mz_tolerance: float = 0.5) -> Optional[int]:
        """Find spectrum index matching the given peptide ID. Returns spectrum index or None."""
        if self.exp is None or not self.peptide_ids:
            return None

        if id_idx < 0 or id_idx >= len(self.peptide_ids):
            return None

        pep_id = self.peptide_ids[id_idx]
        id_rt = pep_id.getRT()
        id_mz = pep_id.getMZ()

        best_spec_idx = None
        best_rt_diff = float("inf")

        for i in range(self.exp.size()):
            spec = self.exp[i]
            if spec.getMSLevel() != 2:
                continue

            spec_rt = spec.getRT()
            if abs(spec_rt - id_rt) > rt_tolerance:
                continue

            precursors = spec.getPrecursors()
            if precursors:
                prec_mz = precursors[0].getMZ()
                if abs(prec_mz - id_mz) <= mz_tolerance:
                    rt_diff = abs(spec_rt - id_rt)
                    if rt_diff < best_rt_diff:
                        best_rt_diff = rt_diff
                        best_spec_idx = i

        return best_spec_idx

    def create_tic_plot(self) -> go.Figure:
        """Create TIC (Total Ion Chromatogram) plot."""
        fig = go.Figure()

        if self.tic_rt is None or len(self.tic_rt) == 0:
            fig.update_layout(
                title={"text": "TIC - No data loaded", "font": {"color": "#888"}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#888"},
                height=200,
            )
            return fig

        # Convert RT to display units
        rt_divisor = 60.0 if self.rt_in_minutes else 1.0
        rt_unit = "min" if self.rt_in_minutes else "s"
        display_rt = [rt / rt_divisor for rt in self.tic_rt]

        # Create TIC trace
        fig.add_trace(
            go.Scatter(
                x=display_rt,
                y=self.tic_intensity,
                mode="lines",
                name="TIC",
                line={"color": "#00d4ff", "width": 1},
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.2)",
                hovertemplate=f"RT: %{{x:.2f}}{rt_unit}<br>Intensity: %{{y:.2e}}<extra></extra>",
            )
        )

        # Add view range indicator
        if self.view_rt_min is not None and self.view_rt_max is not None:
            fig.add_vrect(
                x0=self.view_rt_min / rt_divisor,
                x1=self.view_rt_max / rt_divisor,
                fillcolor="rgba(255, 255, 0, 0.15)",
                layer="below",
                line_width=1,
                line_color="rgba(255, 255, 0, 0.5)",
            )

        # Add vertical marker for selected spectrum
        if self.selected_spectrum_idx is not None and self.exp is not None:
            spec = self.exp[self.selected_spectrum_idx]
            selected_rt = spec.getRT() / rt_divisor
            ms_level = spec.getMSLevel()
            marker_color = "#00d4ff" if ms_level == 1 else "#ff6b6b"
            fig.add_vline(
                x=selected_rt,
                line_dash="solid",
                line_color=marker_color,
                line_width=2,
                annotation_text=f"#{self.selected_spectrum_idx}",
                annotation_position="top",
                annotation_font_color=marker_color,
                annotation_font_size=10,
            )

        # Use tic_source for title (e.g., "MS1 TIC" or "MS2 BPC")
        tic_title = f"{self.tic_source} - Click to select spectrum"
        y_label = "Total Intensity" if "TIC" in self.tic_source else "Base Peak Intensity"

        fig.update_layout(
            title={"text": tic_title, "font": {"size": 14, "color": "#888"}},
            xaxis_title=f"RT ({rt_unit})",
            yaxis_title=y_label,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#888"},
            height=200,
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            showlegend=False,
            hovermode="x unified",
        )

        # Style axes for light/dark mode compatibility
        fig.update_xaxes(showgrid=False, linecolor="#888", tickcolor="#888")
        fig.update_yaxes(showgrid=False, linecolor="#888", tickcolor="#888")

        # Set x-axis range to match data
        if len(self.tic_rt) > 0:
            fig.update_xaxes(range=[self.rt_min / rt_divisor, self.rt_max / rt_divisor])

        return fig

    def update_tic_plot(self):
        """Update the TIC plot display."""
        # Skip if we're updating from a TIC interaction to prevent circular updates
        if self._updating_from_tic:
            return
        if self.tic_plot is not None:
            fig = self.create_tic_plot()
            self.tic_plot.update_figure(fig)

    def create_chromatogram_plot(self) -> go.Figure:
        """Create a plot showing selected chromatograms."""
        fig = go.Figure()

        if not self.has_chromatograms or not self.selected_chromatogram_indices:
            fig.update_layout(
                title={"text": "Chromatograms - Select from table below", "font": {"color": "#888"}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#888"},
                height=250,
            )
            return fig

        # Convert RT to display units
        rt_divisor = 60.0 if self.rt_in_minutes else 1.0
        rt_unit = "min" if self.rt_in_minutes else "s"

        # Color palette for multiple chromatograms
        colors = [
            "#00d4ff", "#ff6b6b", "#4ecdc4", "#ffe66d", "#95e1d3",
            "#f38181", "#aa96da", "#fcbad3", "#a8d8ea", "#ffb6b9",
        ]

        # Plot each selected chromatogram
        for i, chrom_idx in enumerate(self.selected_chromatogram_indices):
            if chrom_idx not in self.chromatogram_data:
                continue

            rt_array, int_array = self.chromatogram_data[chrom_idx]
            display_rt = rt_array / rt_divisor

            # Find metadata for label
            chrom_meta = next((c for c in self.chromatograms if c["idx"] == chrom_idx), None)
            if chrom_meta:
                # Create a short label
                native_id = chrom_meta["native_id"]
                if len(native_id) > 30:
                    label = native_id[:27] + "..."
                else:
                    label = native_id
            else:
                label = f"Chrom {chrom_idx}"

            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=display_rt,
                    y=int_array,
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 1.5},
                    hovertemplate=f"{label}<br>RT: %{{x:.2f}}{rt_unit}<br>Intensity: %{{y:.2e}}<extra></extra>",
                )
            )

        # Add view range indicator if data is loaded
        if self.view_rt_min is not None and self.view_rt_max is not None:
            fig.add_vrect(
                x0=self.view_rt_min / rt_divisor,
                x1=self.view_rt_max / rt_divisor,
                fillcolor="rgba(255, 255, 0, 0.1)",
                layer="below",
                line_width=1,
                line_color="rgba(255, 255, 0, 0.3)",
            )

        n_selected = len(self.selected_chromatogram_indices)
        title_text = f"Chromatograms ({n_selected} selected)"

        fig.update_layout(
            title={"text": title_text, "font": {"size": 14, "color": "#888"}},
            xaxis_title=f"RT ({rt_unit})",
            yaxis_title="Intensity",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#888"},
            height=250,
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"size": 9}},
            hovermode="x unified",
        )

        # Style axes
        fig.update_xaxes(showgrid=False, linecolor="#888", tickcolor="#888")
        fig.update_yaxes(showgrid=False, linecolor="#888", tickcolor="#888")

        return fig

    def update_chromatogram_plot(self):
        """Update the chromatogram plot display."""
        if self.chromatogram_plot is not None:
            fig = self.create_chromatogram_plot()
            self.chromatogram_plot.update_figure(fig)

    def select_chromatogram(self, chrom_idx: int, add: bool = False):
        """Select a chromatogram for display.

        Args:
            chrom_idx: Index of chromatogram to select
            add: If True, add to selection; if False, replace selection
        """
        if add:
            if chrom_idx in self.selected_chromatogram_indices:
                self.selected_chromatogram_indices.remove(chrom_idx)
            else:
                self.selected_chromatogram_indices.append(chrom_idx)
        else:
            self.selected_chromatogram_indices = [chrom_idx]
        self.update_chromatogram_plot()

    def clear_chromatogram_selection(self):
        """Clear all selected chromatograms."""
        self.selected_chromatogram_indices = []
        self.update_chromatogram_plot()

    def find_spectrum_idx_at_rt(self, target_rt: float, ms_level: Optional[int] = None) -> Optional[int]:
        """Find the spectrum index closest to the given RT, optionally filtered by MS level."""
        if self.exp is None:
            return None

        best_idx = None
        best_rt_diff = float("inf")

        for i in range(self.exp.size()):
            spec = self.exp[i]
            if ms_level is not None and spec.getMSLevel() != ms_level:
                continue

            spec_rt = spec.getRT()
            rt_diff = abs(spec_rt - target_rt)
            if rt_diff < best_rt_diff:
                best_rt_diff = rt_diff
                best_idx = i

        return best_idx

    def show_spectrum_at_rt(self, rt: float, ms_level: Optional[int] = None):
        """Display the closest spectrum at the given retention time using the spectrum browser."""
        if self.exp is None:
            ui.notify("Load mzML file first", type="warning")
            return

        spec_idx = self.find_spectrum_idx_at_rt(rt, ms_level=ms_level)
        if spec_idx is None:
            level_str = f"MS{ms_level} " if ms_level else ""
            ui.notify(f"No {level_str}spectrum found near RT={rt:.1f}s", type="warning")
            return

        self.show_spectrum_in_browser(spec_idx)

    def show_ms1_spectrum(self, rt: float):
        """Display MS1 spectrum at the given retention time using the spectrum browser."""
        self.show_spectrum_at_rt(rt, ms_level=1)

    def zoom_to_feature(self, feature_idx: int, padding: float = 0.2):
        """Zoom to a specific feature."""
        if self.feature_map is None or feature_idx >= self.feature_map.size():
            return

        self.selected_feature_idx = feature_idx
        self.selected_id_idx = None
        feature = self.feature_map[feature_idx]

        rt = feature.getRT()
        mz = feature.getMZ()

        hulls = feature.getConvexHulls()
        if hulls:
            all_points = []
            for hull in hulls:
                points = hull.getHullPoints()
                all_points.extend([(p[0], p[1]) for p in points])

            if all_points:
                rt_coords = [p[0] for p in all_points]
                mz_coords = [p[1] for p in all_points]
                feat_rt_min, feat_rt_max = min(rt_coords), max(rt_coords)
                feat_mz_min, feat_mz_max = min(mz_coords), max(mz_coords)
            else:
                feat_rt_min, feat_rt_max = rt - 10, rt + 10
                feat_mz_min, feat_mz_max = mz - 2, mz + 2
        else:
            feat_rt_min, feat_rt_max = rt - 10, rt + 10
            feat_mz_min, feat_mz_max = mz - 2, mz + 2

        rt_range = max(feat_rt_max - feat_rt_min, 20)
        mz_range = max(feat_mz_max - feat_mz_min, 4)

        rt_pad = rt_range * padding
        mz_pad = mz_range * padding

        self.view_rt_min = max(self.rt_min, feat_rt_min - rt_pad)
        self.view_rt_max = min(self.rt_max, feat_rt_max + rt_pad)
        self.view_mz_min = max(self.mz_min, feat_mz_min - mz_pad)
        self.view_mz_max = min(self.mz_max, feat_mz_max + mz_pad)

        self.update_plot()
        ui.notify(f"Zoomed to feature {feature_idx + 1}", type="info")

        # NiceGUI 3.x: Emit selection change event
        self._emit_selection_changed("feature", feature_idx)

    def zoom_to_id(self, id_idx: int, padding: float = 0.3):
        """Zoom to a specific peptide identification and show annotated spectrum."""
        if not self.peptide_ids or id_idx >= len(self.peptide_ids):
            return

        self.selected_id_idx = id_idx
        self.selected_feature_idx = None
        pep_id = self.peptide_ids[id_idx]

        rt = pep_id.getRT()
        mz = pep_id.getMZ()

        rt_window = 30
        mz_window = 5

        self.view_rt_min = max(self.rt_min, rt - rt_window)
        self.view_rt_max = min(self.rt_max, rt + rt_window)
        self.view_mz_min = max(self.mz_min, mz - mz_window)
        self.view_mz_max = min(self.mz_max, mz + mz_window)

        self.update_plot()

        # Find matching spectrum and show it with annotations
        spec_idx = self.find_spectrum_for_id(id_idx)
        if spec_idx is not None:
            self.show_spectrum_in_browser(spec_idx)
        else:
            ui.notify("No matching MS2 spectrum found for this ID", type="warning")

        ui.notify(f"Zoomed to ID {id_idx + 1}", type="info")

        # NiceGUI 3.x: Emit selection change event
        self._emit_selection_changed("id", id_idx)

    def _data_to_plot_pixel(self, rt: float, mz: float) -> tuple[int, int]:
        """Convert RT/m/z to pixel coordinates (handles swapped axes)."""
        rt_range = self.view_rt_max - self.view_rt_min
        mz_range = self.view_mz_max - self.view_mz_min

        if rt_range == 0 or mz_range == 0:
            return (0, 0)

        if self.swap_axes:
            # m/z on x-axis, RT on y-axis
            x = int((mz - self.view_mz_min) / mz_range * self.plot_width)
            y = int((1 - (rt - self.view_rt_min) / rt_range) * self.plot_height)
        else:
            # RT on x-axis, m/z on y-axis (traditional)
            x = int((rt - self.view_rt_min) / rt_range * self.plot_width)
            y = int((1 - (mz - self.view_mz_min) / mz_range) * self.plot_height)

        return (x, y)

    def _is_in_view(self, rt: float, mz: float) -> bool:
        """Check if point is in current view."""
        return self.view_rt_min <= rt <= self.view_rt_max and self.view_mz_min <= mz <= self.view_mz_max

    def _feature_intersects_view(self, rt_min: float, rt_max: float, mz_min: float, mz_max: float) -> bool:
        """Check if feature intersects current view."""
        return not (
            rt_max < self.view_rt_min
            or rt_min > self.view_rt_max
            or mz_max < self.view_mz_min
            or mz_min > self.view_mz_max
        )

    def _draw_features_on_plot(self, img: Image.Image) -> Image.Image:
        """Draw feature overlays."""
        if self.feature_map is None or self.feature_map.size() == 0:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        features_drawn = 0
        max_features = 10000

        for idx, feature in enumerate(self.feature_map):
            if features_drawn >= max_features:
                break

            is_selected = idx == self.selected_feature_idx
            rt = feature.getRT()
            mz = feature.getMZ()

            hulls = feature.getConvexHulls()
            if hulls:
                all_points = []
                for hull in hulls:
                    points = hull.getHullPoints()
                    all_points.extend([(p[0], p[1]) for p in points])

                if all_points:
                    rt_coords = [p[0] for p in all_points]
                    mz_coords = [p[1] for p in all_points]
                    feat_rt_min, feat_rt_max = min(rt_coords), max(rt_coords)
                    feat_mz_min, feat_mz_max = min(mz_coords), max(mz_coords)
                else:
                    feat_rt_min, feat_rt_max = rt - 1, rt + 1
                    feat_mz_min, feat_mz_max = mz - 0.5, mz + 0.5
            else:
                feat_rt_min, feat_rt_max = rt - 1, rt + 1
                feat_mz_min, feat_mz_max = mz - 0.5, mz + 0.5

            if not self._feature_intersects_view(feat_rt_min, feat_rt_max, feat_mz_min, feat_mz_max):
                continue

            features_drawn += 1

            hull_color = self.selected_color if is_selected else self.hull_color
            bbox_color = self.selected_color if is_selected else self.bbox_color
            centroid_color = self.selected_color if is_selected else self.centroid_color
            line_width = 3 if is_selected else 1

            if self.show_convex_hulls and hulls:
                for hull in hulls:
                    points = hull.getHullPoints()
                    if len(points) >= 3:
                        pixel_points = [self._data_to_plot_pixel(p[0], p[1]) for p in points]
                        pixel_points.append(pixel_points[0])
                        fill_alpha = 100 if is_selected else 50
                        draw.polygon(pixel_points, outline=hull_color, fill=(*hull_color[:3], fill_alpha))

            if self.show_bounding_boxes:
                top_left = self._data_to_plot_pixel(feat_rt_min, feat_mz_max)
                bottom_right = self._data_to_plot_pixel(feat_rt_max, feat_mz_min)
                # Ensure x1 <= x2 and y1 <= y2 for PIL rectangle
                x1, y1 = top_left
                x2, y2 = bottom_right
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=line_width)

            if self.show_centroids:
                cx, cy = self._data_to_plot_pixel(rt, mz)
                r = 5 if is_selected else 3
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=centroid_color, outline=(255, 255, 255, 255))

        img = Image.alpha_composite(img, overlay)
        return img

    def _draw_spectrum_marker_on_plot(self, img: Image.Image) -> Image.Image:
        """Draw a crosshair at the selected spectrum's RT and precursor m/z (for MS2)."""
        if not self.show_spectrum_marker:
            return img
        if self.selected_spectrum_idx is None or self.exp is None:
            return img

        spec = self.exp[self.selected_spectrum_idx]
        rt = spec.getRT()
        ms_level = spec.getMSLevel()

        # Get precursor m/z for MS2 spectra
        precursor_mz = None
        if ms_level == 2:
            precursors = spec.getPrecursors()
            if precursors:
                precursor_mz = precursors[0].getMZ()

        # Check if RT is in view (still render if precursor_mz is in view)
        rt_in_view = self.view_rt_min <= rt <= self.view_rt_max
        mz_in_view = precursor_mz is not None and self.view_mz_min <= precursor_mz <= self.view_mz_max

        if not rt_in_view and not mz_in_view:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Colors: cyan for MS1, red for MS2
        line_color = (0, 212, 255, 200) if ms_level == 1 else (255, 107, 107, 200)
        crosshair_color = (255, 200, 50, 180)  # Yellow-orange for crosshair intersection

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except OSError:
            font = ImageFont.load_default()

        # Calculate pixel position based on axis orientation
        x, y = self._data_to_plot_pixel(rt, self.view_mz_min if not self.swap_axes else self.view_mz_max)

        # Draw marker at RT position - two lines with gap to not obscure peaks
        # When swap_axes=True: RT on y-axis, draw HORIZONTAL lines
        # When swap_axes=False: RT on x-axis, draw VERTICAL lines
        if rt_in_view:
            if self.swap_axes:
                # RT is on y-axis - draw horizontal lines at y position
                draw.line([(0, y - 1), (self.plot_width, y - 1)], fill=line_color, width=1)
                draw.line([(0, y + 1), (self.plot_width, y + 1)], fill=line_color, width=1)
                # Draw label at left
                label = f"MS{ms_level} #{self.selected_spectrum_idx}"
                draw.text((4, y + 4), label, fill=line_color, font=font)
            else:
                # RT is on x-axis - draw vertical lines at x position
                draw.line([(x - 1, 0), (x - 1, self.plot_height)], fill=line_color, width=1)
                draw.line([(x + 1, 0), (x + 1, self.plot_height)], fill=line_color, width=1)
                # Draw label at top
                label = f"MS{ms_level} #{self.selected_spectrum_idx}"
                draw.text((x + 4, 4), label, fill=line_color, font=font)

        # For MS2 spectra, draw line at precursor m/z to create crosshair
        # When swap_axes=True: m/z on x-axis, draw VERTICAL line
        # When swap_axes=False: m/z on y-axis, draw HORIZONTAL line
        if precursor_mz is not None and mz_in_view:
            prec_x, prec_y = self._data_to_plot_pixel(rt, precursor_mz)

            if self.swap_axes:
                # m/z is on x-axis - draw vertical line at prec_x
                draw.line([(prec_x, 0), (prec_x, self.plot_height)], fill=line_color, width=2)
                # Draw precursor m/z label at top
                mz_label = f"Prec: {precursor_mz:.4f}"
                draw.text((prec_x + 4, 4), mz_label, fill=line_color, font=font)
            else:
                # m/z is on y-axis - draw horizontal line at prec_y
                draw.line([(0, prec_y), (self.plot_width, prec_y)], fill=line_color, width=2)
                # Draw precursor m/z label on the right
                mz_label = f"Prec: {precursor_mz:.4f}"
                bbox = draw.textbbox((0, 0), mz_label, font=font)
                label_width = bbox[2] - bbox[0]
                draw.text((self.plot_width - label_width - 4, prec_y - 14), mz_label, fill=line_color, font=font)

            # Draw crosshair intersection marker (if both RT and m/z are in view)
            if rt_in_view:
                # Draw a small circle/crosshair at intersection
                r = 6
                draw.ellipse([(prec_x - r, prec_y - r), (prec_x + r, prec_y + r)], outline=crosshair_color, width=2)
                # Inner cross for visibility
                draw.line([(prec_x - r - 2, prec_y), (prec_x + r + 2, prec_y)], fill=crosshair_color, width=1)
                draw.line([(prec_x, prec_y - r - 2), (prec_x, prec_y + r + 2)], fill=crosshair_color, width=1)

        img = Image.alpha_composite(img, overlay)
        return img

    def _draw_ids_on_plot(self, img: Image.Image) -> Image.Image:
        """Draw peptide ID precursor positions and optionally sequence labels."""
        if not self.peptide_ids or not self.show_ids:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Load font for sequence labels
        font = None
        if self.show_id_sequences:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
            except Exception:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

        for idx, pep_id in enumerate(self.peptide_ids):
            rt = pep_id.getRT()
            mz = pep_id.getMZ()

            if not self._is_in_view(rt, mz):
                continue

            is_selected = idx == self.selected_id_idx
            color = self.id_selected_color if is_selected else self.id_color

            cx, cy = self._data_to_plot_pixel(rt, mz)

            r = 6 if is_selected else 4
            diamond = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            draw.polygon(diamond, fill=color, outline=(255, 255, 255, 255))

            if is_selected:
                draw.line([(cx - r - 3, cy), (cx + r + 3, cy)], fill=color, width=2)
                draw.line([(cx, cy - r - 3), (cx, cy + r + 3)], fill=color, width=2)

            # Draw sequence label if enabled
            if self.show_id_sequences and font is not None:
                hits = pep_id.getHits()
                if hits:
                    seq = hits[0].getSequence().toString()
                    # Truncate long sequences
                    if len(seq) > 12:
                        seq = seq[:10] + ".."
                    # Draw with background for readability
                    bbox = draw.textbbox((0, 0), seq, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_x = cx + r + 3
                    text_y = cy - text_height // 2
                    # Keep text within plot bounds
                    if text_x + text_width > self.plot_width:
                        text_x = cx - r - 3 - text_width
                    # Background rectangle
                    draw.rectangle(
                        [(text_x - 1, text_y - 1), (text_x + text_width + 1, text_y + text_height + 1)],
                        fill=(0, 0, 0, 180),
                    )
                    draw.text((text_x, text_y), seq, fill=(255, 255, 255, 255), font=font)

        img = Image.alpha_composite(img, overlay)
        return img

    def _draw_hover_overlay(self, img: Image.Image) -> Image.Image:
        """Draw hover highlights for features and IDs."""
        if self.hover_feature_idx is None and self.hover_id_idx is None:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        hover_color = (100, 255, 200, 180)  # Bright cyan-green for hover

        # Draw hover highlight for feature
        if self.hover_feature_idx is not None and self.feature_map is not None:
            try:
                feature = self.feature_map[self.hover_feature_idx]
                rt = feature.getRT()
                mz = feature.getMZ()

                # Get bounding box
                convex_hulls = feature.getConvexHulls()
                if convex_hulls:
                    hull = convex_hulls[0]
                    hull_points = hull.getHullPoints()
                    if hull_points:
                        rt_vals = [p.getX() for p in hull_points]
                        mz_vals = [p.getY() for p in hull_points]
                        rt_min_f, rt_max_f = min(rt_vals), max(rt_vals)
                        mz_min_f, mz_max_f = min(mz_vals), max(mz_vals)
                    else:
                        rt_min_f, rt_max_f = rt - 5, rt + 5
                        mz_min_f, mz_max_f = mz - 0.5, mz + 0.5
                else:
                    rt_min_f, rt_max_f = rt - 5, rt + 5
                    mz_min_f, mz_max_f = mz - 0.5, mz + 0.5

                # Draw bounding box preview
                x1, y1 = self._data_to_plot_pixel(rt_min_f, mz_max_f)
                x2, y2 = self._data_to_plot_pixel(rt_max_f, mz_min_f)

                # Clamp to plot area
                x1 = max(0, min(self.plot_width, x1))
                x2 = max(0, min(self.plot_width, x2))
                y1 = max(0, min(self.plot_height, y1))
                y2 = max(0, min(self.plot_height, y2))

                if x1 != x2 and y1 != y2:
                    # Draw dashed-style bounding box (corners + center cross)
                    draw.rectangle([(x1, y1), (x2, y2)], outline=hover_color, width=3)

                    # Draw centroid marker
                    cx, cy = self._data_to_plot_pixel(rt, mz)
                    r = 8
                    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=hover_color, width=3)

            except (IndexError, RuntimeError):
                pass

        # Draw hover highlight for ID
        if self.hover_id_idx is not None and self.peptide_ids:
            try:
                pep_id = self.peptide_ids[self.hover_id_idx]
                rt = pep_id.getRT()
                mz = pep_id.getMZ()

                if self._is_in_view(rt, mz):
                    cx, cy = self._data_to_plot_pixel(rt, mz)

                    # Draw pulsing ring effect
                    for r in [10, 14, 18]:
                        alpha = int(180 * (18 - r) / 8)  # Fade out
                        ring_color = (100, 255, 200, alpha)
                        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=ring_color, width=2)

            except (IndexError, RuntimeError):
                pass

        img = Image.alpha_composite(img, overlay)
        return img

    def set_hover_feature(self, idx: Optional[int]):
        """Set the feature being hovered."""
        if idx != self.hover_feature_idx:
            self.hover_feature_idx = idx
            self.hover_id_idx = None  # Clear other hover
            self.update_plot()

    def set_hover_id(self, idx: Optional[int]):
        """Set the ID being hovered."""
        if idx != self.hover_id_idx:
            self.hover_id_idx = idx
            self.hover_feature_idx = None  # Clear other hover
            self.update_plot()

    def clear_hover(self):
        """Clear all hover states."""
        if self.hover_feature_idx is not None or self.hover_id_idx is not None:
            self.hover_feature_idx = None
            self.hover_id_idx = None
            self.update_plot()

    def _draw_axes(self, canvas: Image.Image) -> Image.Image:
        """Draw axes on canvas."""
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 12)
                title_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
            except OSError:
                font = ImageFont.load_default()
                title_font = font

        plot_left = self.margin_left
        plot_right = self.margin_left + self.plot_width
        plot_top = self.margin_top
        plot_bottom = self.margin_top + self.plot_height

        draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=self.axis_color, width=1)

        # X-axis and Y-axis drawing depends on swap_axes setting
        if self.swap_axes:
            # X-axis: m/z
            x_ticks = calculate_nice_ticks(self.view_mz_min, self.view_mz_max, num_ticks=8)
            x_range = self.view_mz_max - self.view_mz_min
            x_min, x_max = self.view_mz_min, self.view_mz_max

            for tick_val in x_ticks:
                if x_min <= tick_val <= x_max:
                    x_frac = (tick_val - x_min) / x_range
                    x = plot_left + int(x_frac * self.plot_width)
                    draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=self.tick_color, width=1)
                    label = format_tick_label(tick_val, x_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    draw.text((x - label_width // 2, plot_bottom + 8), label, fill=self.label_color, font=font)

            x_title = "m/z"
            bbox = draw.textbbox((0, 0), x_title, font=title_font)
            title_width = bbox[2] - bbox[0]
            draw.text(
                (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
                x_title,
                fill=self.label_color,
                font=title_font,
            )

            # Y-axis: RT
            if self.rt_in_minutes:
                display_rt_min = self.view_rt_min / 60.0
                display_rt_max = self.view_rt_max / 60.0
                y_ticks_display = calculate_nice_ticks(display_rt_min, display_rt_max, num_ticks=8)
                y_ticks = [t * 60.0 for t in y_ticks_display]
            else:
                y_ticks = calculate_nice_ticks(self.view_rt_min, self.view_rt_max, num_ticks=8)
            y_range = self.view_rt_max - self.view_rt_min
            y_min, y_max = self.view_rt_min, self.view_rt_max

            for tick_val in y_ticks:
                if y_min <= tick_val <= y_max:
                    y_frac = 1 - (tick_val - y_min) / y_range
                    y = plot_top + int(y_frac * self.plot_height)
                    draw.line([(plot_left - 5, y), (plot_left, y)], fill=self.tick_color, width=1)
                    display_val = tick_val / 60.0 if self.rt_in_minutes else tick_val
                    display_range = y_range / 60.0 if self.rt_in_minutes else y_range
                    label = format_tick_label(display_val, display_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    label_height = bbox[3] - bbox[1]
                    draw.text(
                        (plot_left - label_width - 10, y - label_height // 2), label, fill=self.label_color, font=font
                    )

            y_title = "RT (min)" if self.rt_in_minutes else "RT (s)"
        else:
            # X-axis: RT (default)
            if self.rt_in_minutes:
                display_rt_min = self.view_rt_min / 60.0
                display_rt_max = self.view_rt_max / 60.0
                rt_ticks_display = calculate_nice_ticks(display_rt_min, display_rt_max, num_ticks=8)
                rt_ticks = [t * 60.0 for t in rt_ticks_display]
            else:
                rt_ticks = calculate_nice_ticks(self.view_rt_min, self.view_rt_max, num_ticks=8)
            rt_range = self.view_rt_max - self.view_rt_min

            for tick_val in rt_ticks:
                if self.view_rt_min <= tick_val <= self.view_rt_max:
                    x_frac = (tick_val - self.view_rt_min) / rt_range
                    x = plot_left + int(x_frac * self.plot_width)
                    draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=self.tick_color, width=1)
                    display_val = tick_val / 60.0 if self.rt_in_minutes else tick_val
                    display_range = rt_range / 60.0 if self.rt_in_minutes else rt_range
                    label = format_tick_label(display_val, display_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    draw.text((x - label_width // 2, plot_bottom + 8), label, fill=self.label_color, font=font)

            x_title = "RT (min)" if self.rt_in_minutes else "RT (s)"
            bbox = draw.textbbox((0, 0), x_title, font=title_font)
            title_width = bbox[2] - bbox[0]
            draw.text(
                (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
                x_title,
                fill=self.label_color,
                font=title_font,
            )

            # Y-axis: m/z (default)
            mz_ticks = calculate_nice_ticks(self.view_mz_min, self.view_mz_max, num_ticks=8)
            mz_range = self.view_mz_max - self.view_mz_min

            for tick_val in mz_ticks:
                if self.view_mz_min <= tick_val <= self.view_mz_max:
                    y_frac = 1 - (tick_val - self.view_mz_min) / mz_range
                    y = plot_top + int(y_frac * self.plot_height)
                    draw.line([(plot_left - 5, y), (plot_left, y)], fill=self.tick_color, width=1)
                    label = format_tick_label(tick_val, mz_range)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_width = bbox[2] - bbox[0]
                    label_height = bbox[3] - bbox[1]
                    draw.text(
                        (plot_left - label_width - 10, y - label_height // 2), label, fill=self.label_color, font=font
                    )

            y_title = "m/z"
        txt_img = Image.new("RGBA", (100, 30), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), y_title, fill=self.label_color, font=title_font)
        txt_img = txt_img.rotate(90, expand=True)

        y_title_x = 5
        y_title_y = plot_top + self.plot_height // 2 - txt_img.height // 2
        canvas.paste(txt_img, (y_title_x, y_title_y), txt_img)

        return canvas

    def render_image(self, fast: bool = False) -> str:
        """Render current view using datashader.

        Args:
            fast: If True, render at reduced resolution and skip overlays for faster panning.
        """
        if self.df is None or len(self.df) == 0:
            return ""

        mask = (
            (self.df["rt"] >= self.view_rt_min)
            & (self.df["rt"] <= self.view_rt_max)
            & (self.df["mz"] >= self.view_mz_min)
            & (self.df["mz"] <= self.view_mz_max)
        )
        view_df = self.df[mask]

        if len(view_df) == 0:
            return ""

        # Use reduced resolution for fast mode (panning)
        resolution_factor = 4 if fast else 1
        render_width = self.plot_width // resolution_factor
        render_height = self.plot_height // resolution_factor

        # Swap axes if enabled (m/z on x-axis, RT on y-axis)
        if self.swap_axes:
            ds_canvas = ds.Canvas(
                plot_width=render_width,
                plot_height=render_height,
                x_range=(self.view_mz_min, self.view_mz_max),
                y_range=(self.view_rt_min, self.view_rt_max),
            )
            agg = ds_canvas.points(view_df, "mz", "rt", ds.max("log_intensity"))
        else:
            ds_canvas = ds.Canvas(
                plot_width=render_width,
                plot_height=render_height,
                x_range=(self.view_rt_min, self.view_rt_max),
                y_range=(self.view_mz_min, self.view_mz_max),
            )
            agg = ds_canvas.points(view_df, "rt", "mz", ds.max("log_intensity"))
        img = tf.shade(agg, cmap=COLORMAPS[self.colormap], how="linear")
        # Skip dynspread in fast mode for speed
        if not fast:
            img = tf.dynspread(img, threshold=0.5, max_px=3)
        img = tf.set_background(img, get_colormap_background(self.colormap))

        plot_img = img.to_pil()

        # Upscale if using reduced resolution
        if fast and resolution_factor > 1:
            plot_img = plot_img.resize(
                (self.plot_width, self.plot_height), Image.Resampling.NEAREST
            )

        # Skip overlays in fast mode for speed
        if not fast:
            if self.feature_map is not None:
                plot_img = self._draw_features_on_plot(plot_img)

            if self.peptide_ids:
                plot_img = self._draw_ids_on_plot(plot_img)

            if self.show_spectrum_marker:
                plot_img = self._draw_spectrum_marker_on_plot(plot_img)

            # Draw hover highlights (last so they appear on top)
            plot_img = self._draw_hover_overlay(plot_img)

        canvas = Image.new("RGBA", (self.canvas_width, self.canvas_height), (0, 0, 0, 0))
        plot_img_rgba = plot_img.convert("RGBA")
        canvas.paste(plot_img_rgba, (self.margin_left, self.margin_top))

        # Skip axes in fast mode
        if not fast:
            canvas = self._draw_axes(canvas)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def render_faims_image(self, cv: float) -> str:
        """Render a single FAIMS CV peak map using datashader."""
        if cv not in self.faims_data or len(self.faims_data[cv]) == 0:
            return ""

        cv_df = self.faims_data[cv]

        mask = (
            (cv_df["rt"] >= self.view_rt_min)
            & (cv_df["rt"] <= self.view_rt_max)
            & (cv_df["mz"] >= self.view_mz_min)
            & (cv_df["mz"] <= self.view_mz_max)
        )
        view_df = cv_df[mask]

        if len(view_df) == 0:
            return ""

        # Smaller plot size for FAIMS panels
        faims_plot_width = self.plot_width // max(1, min(len(self.faims_cvs), 4))
        faims_plot_height = self.plot_height

        ds_canvas = ds.Canvas(
            plot_width=faims_plot_width,
            plot_height=faims_plot_height,
            x_range=(self.view_rt_min, self.view_rt_max),
            y_range=(self.view_mz_min, self.view_mz_max),
        )

        agg = ds_canvas.points(view_df, "rt", "mz", ds.max("log_intensity"))
        img = tf.shade(agg, cmap=COLORMAPS[self.colormap], how="linear")
        img = tf.dynspread(img, threshold=0.5, max_px=3)
        img = tf.set_background(img, get_colormap_background(self.colormap))

        plot_img = img.to_pil()

        # Add CV label at top
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(plot_img)
        label = f"CV: {cv:.1f}V"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        draw.rectangle([(5, 5), (label_width + 15, 25)], fill=(0, 0, 0, 180))
        draw.text((10, 7), label, fill=(255, 255, 255, 255), font=font)

        # Add border
        draw.rectangle([(0, 0), (faims_plot_width - 1, faims_plot_height - 1)], outline=(100, 100, 100, 255), width=1)

        buffer = io.BytesIO()
        plot_img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def render_im_image(self) -> str:
        """Render ion mobility peak map (m/z vs IM) using datashader with optional mobilogram.

        Returns base64-encoded PNG image string.
        """
        if self.im_df is None or len(self.im_df) == 0:
            return ""

        # Filter to current view bounds
        mask = (
            (self.im_df["mz"] >= self.view_mz_min)
            & (self.im_df["mz"] <= self.view_mz_max)
            & (self.im_df["im"] >= self.view_im_min)
            & (self.im_df["im"] <= self.view_im_max)
        )
        view_df = self.im_df[mask]

        if len(view_df) == 0:
            return ""

        # Use same dimensions as main peakmap
        im_plot_width = self.plot_width
        im_plot_height = self.plot_height

        # X-axis: m/z, Y-axis: IM
        ds_canvas = ds.Canvas(
            plot_width=im_plot_width,
            plot_height=im_plot_height,
            x_range=(self.view_mz_min, self.view_mz_max),
            y_range=(self.view_im_min, self.view_im_max),
        )
        agg = ds_canvas.points(view_df, "mz", "im", ds.max("log_intensity"))
        img = tf.shade(agg, cmap=COLORMAPS[self.colormap], how="linear")
        # Use higher max_px for sparse IM data to make individual points more visible
        img = tf.dynspread(img, threshold=0.3, max_px=5)
        img = tf.set_background(img, get_colormap_background(self.colormap))

        plot_img = img.to_pil()

        # Calculate canvas width - add space for mobilogram if enabled
        mobilogram_space = self.mobilogram_plot_width + 20 if self.show_mobilogram else 0
        total_canvas_width = self.canvas_width + mobilogram_space

        # Create canvas with margins for axes (wider if mobilogram is shown)
        canvas = Image.new("RGBA", (total_canvas_width, self.canvas_height), (0, 0, 0, 0))
        plot_img_rgba = plot_img.convert("RGBA")
        canvas.paste(plot_img_rgba, (self.margin_left, self.margin_top))

        # Draw axes
        canvas = self._draw_im_axes(canvas)

        # Draw mobilogram on the right side if enabled
        if self.show_mobilogram:
            canvas = self._draw_mobilogram(canvas)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _draw_mobilogram(self, canvas: Image.Image) -> Image.Image:
        """Draw mobilogram (summed intensity vs IM) on the right side of the IM peakmap."""
        draw = ImageDraw.Draw(canvas)

        # Get mobilogram data
        im_values, intensities = self.extract_mobilogram()
        if len(im_values) == 0 or len(intensities) == 0:
            return canvas

        # Sort by IM values for proper line drawing (prevents jumps)
        sort_idx = np.argsort(im_values)
        im_values = im_values[sort_idx]
        intensities = intensities[sort_idx]

        # Mobilogram plot area (to the right of the main plot)
        mob_left = self.margin_left + self.plot_width + 10
        mob_right = mob_left + self.mobilogram_plot_width
        mob_top = self.margin_top
        mob_bottom = self.margin_top + self.plot_height

        # Draw border for mobilogram area
        draw.rectangle([mob_left, mob_top, mob_right, mob_bottom], outline=self.axis_color, width=1)

        # Normalize intensities to plot width
        max_intensity = np.max(intensities) if len(intensities) > 0 else 1.0
        if max_intensity == 0:
            max_intensity = 1.0

        # Draw filled mobilogram as horizontal bars
        im_range = self.view_im_max - self.view_im_min
        if im_range == 0:
            im_range = 1.0

        # Draw as a filled area plot
        points = []
        for i, (im_val, intensity) in enumerate(zip(im_values, intensities)):
            # Y position (IM axis, inverted - low IM at bottom)
            y_frac = 1.0 - (im_val - self.view_im_min) / im_range
            y = mob_top + int(y_frac * self.plot_height)

            # X position (intensity, starting from left edge)
            x_frac = intensity / max_intensity
            x = mob_left + int(x_frac * self.mobilogram_plot_width)

            points.append((x, y))

        # Draw line connecting all points
        if len(points) > 1:
            # Draw filled polygon (intensity profile)
            # Points are sorted by IM ascending: first point = low IM (bottom), last point = high IM (top)
            # Build polygon path: bottom-left -> data points (bottom to top) -> top-left -> close
            fill_points = [(mob_left, mob_bottom)]  # Start at bottom-left
            fill_points.extend(points)  # Go through data points (bottom to top)
            fill_points.append((mob_left, mob_top))  # End at top-left

            # Fill with semi-transparent cyan
            draw.polygon(fill_points, fill=(0, 200, 255, 80))

            # Draw the line on top
            draw.line(points, fill=(0, 200, 255, 255), width=2)

        # Draw axis label at top
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 10)
            except OSError:
                font = ImageFont.load_default()

        label = "Mobilogram"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        draw.text(
            (mob_left + (self.mobilogram_plot_width - label_width) // 2, mob_top - 15),
            label,
            fill=self.label_color,
            font=font,
        )

        return canvas

    def _draw_im_axes(self, canvas: Image.Image) -> Image.Image:
        """Draw axes for IM peak map (m/z on X, IM on Y)."""
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 12)
                title_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
            except OSError:
                font = ImageFont.load_default()
                title_font = font

        plot_left = self.margin_left
        plot_right = self.margin_left + self.plot_width
        plot_top = self.margin_top
        plot_bottom = self.margin_top + self.plot_height

        # Draw border
        draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=self.axis_color, width=1)

        # X-axis: m/z
        x_ticks = calculate_nice_ticks(self.view_mz_min, self.view_mz_max, num_ticks=8)
        x_range = self.view_mz_max - self.view_mz_min
        if x_range == 0:
            x_range = 1.0  # Prevent division by zero

        for tick_val in x_ticks:
            if self.view_mz_min <= tick_val <= self.view_mz_max:
                x_frac = (tick_val - self.view_mz_min) / x_range
                x = plot_left + int(x_frac * self.plot_width)
                draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=self.tick_color, width=1)
                label = format_tick_label(tick_val, x_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                draw.text((x - label_width // 2, plot_bottom + 8), label, fill=self.label_color, font=font)

        x_title = "m/z"
        bbox = draw.textbbox((0, 0), x_title, font=title_font)
        title_width = bbox[2] - bbox[0]
        draw.text(
            (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
            x_title,
            fill=self.label_color,
            font=title_font,
        )

        # Y-axis: IM
        y_ticks = calculate_nice_ticks(self.view_im_min, self.view_im_max, num_ticks=8)
        y_range = self.view_im_max - self.view_im_min
        if y_range == 0:
            y_range = 1.0  # Prevent division by zero

        for tick_val in y_ticks:
            if self.view_im_min <= tick_val <= self.view_im_max:
                y_frac = 1 - (tick_val - self.view_im_min) / y_range
                y = plot_top + int(y_frac * self.plot_height)
                draw.line([(plot_left - 5, y), (plot_left, y)], fill=self.tick_color, width=1)
                label = format_tick_label(tick_val, y_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                draw.text(
                    (plot_left - label_width - 10, y - label_height // 2),
                    label,
                    fill=self.label_color,
                    font=font,
                )

        # Y-axis title (IM with unit)
        y_title = f"IM ({self.im_unit})" if self.im_unit else "Ion Mobility"

        # Draw rotated Y-axis title
        txt_img = Image.new("RGBA", (200, 30), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), y_title, fill=self.label_color, font=title_font)
        bbox = txt_draw.textbbox((0, 0), y_title, font=title_font)
        txt_width = bbox[2] - bbox[0]
        txt_img = txt_img.crop((0, 0, txt_width + 5, 25))
        txt_img = txt_img.rotate(90, expand=True)
        y_title_x = 5
        y_title_y = plot_top + self.plot_height // 2 - txt_img.height // 2
        canvas.paste(txt_img, (y_title_x, y_title_y), txt_img)

        return canvas

    def update_im_plot(self):
        """Update the IM peak map display."""
        if not self.has_ion_mobility or self.im_image_element is None:
            return
        img_data = self.render_im_image()
        if img_data:
            self.im_image_element.set_source(f"data:image/png;base64,{img_data}")

    def reset_im_view(self):
        """Reset IM view to full data range (both m/z and IM axes)."""
        if self.im_df is not None:
            # Reset both m/z (X-axis) and IM (Y-axis) to full range
            self.view_mz_min = self.mz_min
            self.view_mz_max = self.mz_max
            self.view_im_min = self.im_min
            self.view_im_max = self.im_max
            self.update_im_plot()
            if self.im_range_label:
                self.im_range_label.set_text(
                    f"m/z: {self.view_mz_min:.2f} - {self.view_mz_max:.2f} | "
                    f"IM: {self.view_im_min:.3f} - {self.view_im_max:.3f} {self.im_unit}"
                )

    def extract_mobilogram(self, mz_min: float = None, mz_max: float = None) -> tuple:
        """Extract mobilogram (summed intensity vs ion mobility) from IM data.

        Args:
            mz_min: Minimum m/z value for extraction (uses view_mz_min if None)
            mz_max: Maximum m/z value for extraction (uses view_mz_max if None)

        Returns:
            Tuple of (im_values, intensities) arrays for plotting
        """
        if self.im_df is None or len(self.im_df) == 0:
            return np.array([]), np.array([])

        # Use current view bounds if not specified
        if mz_min is None:
            mz_min = self.view_mz_min
        if mz_max is None:
            mz_max = self.view_mz_max

        # Filter to m/z range and current IM view
        mask = (
            (self.im_df["mz"] >= mz_min)
            & (self.im_df["mz"] <= mz_max)
            & (self.im_df["im"] >= self.view_im_min)
            & (self.im_df["im"] <= self.view_im_max)
        )
        filtered_df = self.im_df[mask]

        if len(filtered_df) == 0:
            return np.array([]), np.array([])

        # Bin IM values and sum intensities (similar to how TIC is calculated)
        im_range = self.view_im_max - self.view_im_min
        n_bins = min(200, max(50, int(len(filtered_df) / 100)))  # Adaptive bin count

        bin_edges = np.linspace(self.view_im_min, self.view_im_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Digitize IM values into bins
        bin_indices = np.digitize(filtered_df["im"].values, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Sum intensities per bin
        intensities = np.zeros(n_bins, dtype=np.float64)
        np.add.at(intensities, bin_indices, filtered_df["intensity"].values)

        return bin_centers, intensities

    def update_faims_plots(self):
        """Update all FAIMS CV peak map panels."""
        if not self.has_faims or not self.show_faims_view:
            return

        for cv in self.faims_cvs:
            if cv in self.faims_images and self.faims_images[cv] is not None:
                img_data = self.render_faims_image(cv)
                if img_data:
                    self.faims_images[cv].set_source(f"data:image/png;base64,{img_data}")

    def update_plot(self, lightweight: bool = False):
        """Update displayed plot.

        Args:
            lightweight: If True, only update the main peakmap image and range labels.
                        Skip minimap, TIC, 3D view, and breadcrumb updates (for panning).
                        Also uses fast rendering (lower resolution, no overlays).
        """
        if self.df is None:
            return

        if self.status_label and not lightweight:
            self.status_label.set_text("Rendering...")

        # Use fast rendering during panning (lower resolution, no overlays)
        img_data = self.render_image(fast=lightweight)
        if img_data and self.image_element:
            self.image_element.set_source(f"data:image/png;base64,{img_data}")

        # Update FAIMS plots if enabled (skip during lightweight update)
        if not lightweight and self.has_faims and self.show_faims_view:
            self.update_faims_plots()

        if self.rt_range_label:
            self.rt_range_label.set_text(f"RT: {self.view_rt_min:.2f} - {self.view_rt_max:.2f} s")
        if self.mz_range_label:
            self.mz_range_label.set_text(f"m/z: {self.view_mz_min:.2f} - {self.view_mz_max:.2f}")

        # Skip updates during lightweight mode (panning) to avoid flicker
        if not lightweight:
            # Update TIC plot (shows current view range)
            self.update_tic_plot()

            if self.status_label:
                self.status_label.set_text("Ready")

            # Update minimap
            self.update_minimap()

            # Update breadcrumb
            self.update_breadcrumb()

            # Update 3D view if enabled
            if self.show_3d_view:
                self.update_3d_view()

            # NiceGUI 3.x: Emit view change event
            self._emit_view_changed()

    def render_minimap(self) -> Optional[str]:
        """Render the minimap showing full data extent with view rectangle overlay."""
        if self.df is None or len(self.df) == 0:
            return None

        # Create minimap canvas - swap axes to match main view
        if self.swap_axes:
            # m/z on x-axis, RT on y-axis
            cvs = ds.Canvas(
                plot_width=self.minimap_width,
                plot_height=self.minimap_height,
                x_range=(self.mz_min, self.mz_max),
                y_range=(self.rt_min, self.rt_max),
            )
            agg = cvs.points(self.df, "mz", "rt", agg=ds.max("log_intensity"))
        else:
            # RT on x-axis, m/z on y-axis (traditional)
            cvs = ds.Canvas(
                plot_width=self.minimap_width,
                plot_height=self.minimap_height,
                x_range=(self.rt_min, self.rt_max),
                y_range=(self.mz_min, self.mz_max),
            )
            agg = cvs.points(self.df, "rt", "mz", agg=ds.max("log_intensity"))

        # Apply color map with linear scaling (matches main view)
        img = tf.shade(agg, cmap=COLORMAPS[self.colormap], how="linear")
        img = tf.dynspread(img, threshold=0.5, max_px=2)
        img = tf.set_background(img, get_colormap_background(self.colormap))

        # Convert to PIL
        plot_img = img.to_pil()

        # Draw view rectangle
        if (
            self.view_rt_min is not None
            and self.view_rt_max is not None
            and self.view_mz_min is not None
            and self.view_mz_max is not None
        ):
            draw = ImageDraw.Draw(plot_img)

            # Convert data coords to pixel coords
            rt_range = self.rt_max - self.rt_min
            mz_range = self.mz_max - self.mz_min

            if rt_range > 0 and mz_range > 0:
                if self.swap_axes:
                    # m/z on x-axis, RT on y-axis
                    x1 = int((self.view_mz_min - self.mz_min) / mz_range * self.minimap_width)
                    x2 = int((self.view_mz_max - self.mz_min) / mz_range * self.minimap_width)
                    y1 = int((self.rt_max - self.view_rt_max) / rt_range * self.minimap_height)
                    y2 = int((self.rt_max - self.view_rt_min) / rt_range * self.minimap_height)
                else:
                    # RT on x-axis, m/z on y-axis (traditional)
                    x1 = int((self.view_rt_min - self.rt_min) / rt_range * self.minimap_width)
                    x2 = int((self.view_rt_max - self.rt_min) / rt_range * self.minimap_width)
                    y1 = int((self.mz_max - self.view_mz_max) / mz_range * self.minimap_height)
                    y2 = int((self.mz_max - self.view_mz_min) / mz_range * self.minimap_height)

                # Clamp to minimap bounds and ensure x1 <= x2, y1 <= y2
                x1, x2 = max(0, x1), min(self.minimap_width - 1, x2)
                y1, y2 = max(0, y1), min(self.minimap_height - 1, y2)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Draw two concentric rectangles with complementary colors for visibility
                # Outer rectangle (blue)
                draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(0, 100, 255, 255), width=2)
                # Inner rectangle (yellow) - only draw if there's enough space
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=(255, 255, 0, 255), width=2)

        # Draw spectrum marker at selected spectrum RT
        # When swap_axes=True: RT on y-axis, draw HORIZONTAL lines
        # When swap_axes=False: RT on x-axis, draw VERTICAL lines
        if self.selected_spectrum_idx is not None and self.exp is not None:
            spec = self.exp[self.selected_spectrum_idx]
            rt = spec.getRT()
            ms_level = spec.getMSLevel()

            rt_range = self.rt_max - self.rt_min
            if rt_range > 0:
                draw = ImageDraw.Draw(plot_img)

                # Use different colors for MS1 vs MS2
                if ms_level == 1:
                    color = (0, 255, 0, 255)  # Green for MS1
                else:
                    color = (255, 0, 255, 255)  # Magenta for MS2

                if self.swap_axes:
                    # RT is on y-axis - draw horizontal lines
                    y = int((self.rt_max - rt) / rt_range * self.minimap_height)
                    y = max(0, min(self.minimap_height - 1, y))
                    draw.line([(0, y - 2), (self.minimap_width, y - 2)], fill=(0, 0, 0, 200), width=1)
                    draw.line([(0, y - 1), (self.minimap_width, y - 1)], fill=color, width=1)
                    draw.line([(0, y + 1), (self.minimap_width, y + 1)], fill=color, width=1)
                    draw.line([(0, y + 2), (self.minimap_width, y + 2)], fill=(0, 0, 0, 200), width=1)
                else:
                    # RT is on x-axis - draw vertical lines
                    x = int((rt - self.rt_min) / rt_range * self.minimap_width)
                    x = max(0, min(self.minimap_width - 1, x))
                    draw.line([(x - 2, 0), (x - 2, self.minimap_height)], fill=(0, 0, 0, 200), width=1)
                    draw.line([(x - 1, 0), (x - 1, self.minimap_height)], fill=color, width=1)
                    draw.line([(x + 1, 0), (x + 1, self.minimap_height)], fill=color, width=1)
                    draw.line([(x + 2, 0), (x + 2, self.minimap_height)], fill=(0, 0, 0, 200), width=1)

        # Convert to base64
        buffer = io.BytesIO()
        plot_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def update_minimap(self):
        """Update the minimap display."""
        if self.minimap_image is None:
            return

        img_data = self.render_minimap()
        if img_data:
            self.minimap_image.set_source(f"data:image/png;base64,{img_data}")

    def minimap_click_to_view(self, x_frac: float, y_frac: float):
        """Center the main view on the clicked position in minimap."""
        if self.df is None:
            return

        # Convert minimap fractions to data coordinates (depends on axis orientation)
        if self.swap_axes:
            # m/z on x-axis, RT on y-axis
            mz_click = self.mz_min + x_frac * (self.mz_max - self.mz_min)
            rt_click = self.rt_max - y_frac * (self.rt_max - self.rt_min)
        else:
            # RT on x-axis, m/z on y-axis (traditional)
            rt_click = self.rt_min + x_frac * (self.rt_max - self.rt_min)
            mz_click = self.mz_max - y_frac * (self.mz_max - self.mz_min)

        # Center current view on this point
        rt_half_range = (self.view_rt_max - self.view_rt_min) / 2
        mz_half_range = (self.view_mz_max - self.view_mz_min) / 2

        new_rt_min = rt_click - rt_half_range
        new_rt_max = rt_click + rt_half_range
        new_mz_min = mz_click - mz_half_range
        new_mz_max = mz_click + mz_half_range

        # Clamp to data bounds
        if new_rt_min < self.rt_min:
            new_rt_max += self.rt_min - new_rt_min
            new_rt_min = self.rt_min
        if new_rt_max > self.rt_max:
            new_rt_min -= new_rt_max - self.rt_max
            new_rt_max = self.rt_max

        if new_mz_min < self.mz_min:
            new_mz_max += self.mz_min - new_mz_min
            new_mz_min = self.mz_min
        if new_mz_max > self.mz_max:
            new_mz_min -= new_mz_max - self.mz_max
            new_mz_max = self.mz_max

        # Final clamp
        self.view_rt_min = max(self.rt_min, new_rt_min)
        self.view_rt_max = min(self.rt_max, new_rt_max)
        self.view_mz_min = max(self.mz_min, new_mz_min)
        self.view_mz_max = min(self.mz_max, new_mz_max)

        self.update_plot()

    def push_zoom_history(self):
        """Save current view state to zoom history."""
        if self.view_rt_min is None:
            return

        # Create label for this view state
        rt_range = self.view_rt_max - self.view_rt_min
        mz_range = self.view_mz_max - self.view_mz_min
        full_rt = self.rt_max - self.rt_min
        full_mz = self.mz_max - self.mz_min

        # Check if this is approximately full view
        if rt_range >= full_rt * 0.95 and mz_range >= full_mz * 0.95:
            label = "Full"
        else:
            label = f"RT {self.view_rt_min:.0f}-{self.view_rt_max:.0f}"

        state = (self.view_rt_min, self.view_rt_max, self.view_mz_min, self.view_mz_max, label)

        # Don't add if same as last entry
        if self.zoom_history and self.zoom_history[-1][:4] == state[:4]:
            return

        self.zoom_history.append(state)

        # Limit history size
        if len(self.zoom_history) > self.max_zoom_history:
            self.zoom_history = self.zoom_history[-self.max_zoom_history :]

    def go_to_zoom_history(self, index: int):
        """Jump to a specific point in zoom history."""
        if index < 0 or index >= len(self.zoom_history):
            return

        state = self.zoom_history[index]
        self.view_rt_min, self.view_rt_max, self.view_mz_min, self.view_mz_max, _ = state

        # Truncate history to this point (forward history is lost)
        self.zoom_history = self.zoom_history[: index + 1]

        self.update_plot()

    def update_breadcrumb(self):
        """Update the breadcrumb trail display."""
        if self.breadcrumb_label is None:
            return

        if not self.zoom_history:
            self.breadcrumb_label.set_text("Full view")
            return

        # Build breadcrumb string
        parts = []
        for _i, (_, _, _, _, label) in enumerate(self.zoom_history):
            parts.append(label)

        breadcrumb_text = " → ".join(parts)
        self.breadcrumb_label.set_text(breadcrumb_text)

    def pixel_to_data_coords(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        """Convert pixel coordinates to RT/m/z data coordinates."""
        # Account for margins
        plot_x = pixel_x - self.margin_left
        plot_y = pixel_y - self.margin_top

        # Clamp to plot area
        plot_x = max(0, min(self.plot_width, plot_x))
        plot_y = max(0, min(self.plot_height, plot_y))

        # Convert to data coordinates
        rt = self.view_rt_min + (plot_x / self.plot_width) * (self.view_rt_max - self.view_rt_min)
        mz = self.view_mz_max - (plot_y / self.plot_height) * (self.view_mz_max - self.view_mz_min)

        return rt, mz

    def update_coord_display(self, pixel_x: int, pixel_y: int):
        """Update the coordinate display label."""
        if self.coord_label is None or self.df is None:
            return

        rt, mz = self.pixel_to_data_coords(pixel_x, pixel_y)
        if self.rt_in_minutes:
            self.coord_label.set_text(f"RT: {rt / 60.0:.2f}min  m/z: {mz:.4f}")
        else:
            self.coord_label.set_text(f"RT: {rt:.2f}s  m/z: {mz:.4f}")

    # ==================== 3D Visualization Methods ====================

    def is_small_region(self) -> bool:
        """Check if current view is small enough for 3D visualization."""
        if self.view_rt_min is None or self.view_mz_min is None:
            return False
        rt_range = self.view_rt_max - self.view_rt_min
        mz_range = self.view_mz_max - self.view_mz_min
        return rt_range <= self.rt_threshold_3d and mz_range <= self.mz_threshold_3d

    def update_3d_view(self):
        """Update the 3D visualization with current view data using pyopenms-viz."""
        if not self.show_3d_view or self.plot_3d is None or self.df is None:
            return

        # Check if region is small enough
        if not self.is_small_region():
            # Show message that region is too large
            if self.view_3d_status:
                rt_range = self.view_rt_max - self.view_rt_min
                mz_range = self.view_mz_max - self.view_mz_min
                self.view_3d_status.set_text(
                    f"Zoom in more for 3D (current: RT={rt_range:.0f}s, m/z={mz_range:.0f} | need: RT≤{self.rt_threshold_3d:.0f}s, m/z≤{self.mz_threshold_3d:.0f})"
                )
            return

        # Get peaks in current view
        mask = (
            (self.df["rt"] >= self.view_rt_min)
            & (self.df["rt"] <= self.view_rt_max)
            & (self.df["mz"] >= self.view_mz_min)
            & (self.df["mz"] <= self.view_mz_max)
        )
        view_df = self.df[mask].copy()

        if len(view_df) == 0:
            if self.view_3d_status:
                self.view_3d_status.set_text("No peaks in view")
            return

        # Subsample if too many peaks
        num_peaks_total = len(view_df)
        if len(view_df) > self.max_3d_peaks:
            view_df = view_df.nlargest(self.max_3d_peaks, "intensity")
        num_peaks_shown = len(view_df)

        try:
            # Use pyopenms-viz for 3D plotting
            from pyopenms_viz._plotly.core import PLOTLYPeakMapPlot

            # Rename columns to match pyopenms-viz expectations
            plot_df = view_df.rename(columns={"rt": "RT", "mz": "mz", "intensity": "int"})

            # Create 3D plot (no title - header shows info)
            plot = PLOTLYPeakMapPlot(plot_df, x="RT", y="mz", z="int", plot_3d=True, title="")
            plot.plot()

            # Get the plotly figure
            fig = plot.fig

            # Update layout for light/dark mode compatibility - maximize space usage
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#888"},
                scene={
                    "xaxis": {"title": "RT (s)", "backgroundcolor": "rgba(128,128,128,0.1)", "gridcolor": "#888"},
                    "yaxis": {"title": "m/z", "backgroundcolor": "rgba(128,128,128,0.1)", "gridcolor": "#888"},
                    "zaxis": {"title": "Intensity", "backgroundcolor": "rgba(128,128,128,0.1)", "gridcolor": "#888"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "aspectmode": "manual",
                    "aspectratio": {"x": 1.5, "y": 1, "z": 0.8},
                },
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                width=self.canvas_width,
                height=500,
                autosize=False,
                showlegend=True,
                legend={"x": 0, "y": 1, "bgcolor": "rgba(128,128,128,0.3)"},
                modebar={"orientation": "v", "bgcolor": "rgba(0,0,0,0)"},
            )

            # Add feature markers if available
            if self.feature_map is not None and self.show_centroids:
                self._add_features_to_3d_plot(fig)

            # Update the plotly element
            self.plot_3d.update_figure(fig)

            # Update status
            if self.view_3d_status:
                if num_peaks_shown < num_peaks_total:
                    self.view_3d_status.set_text(
                        f"Showing {num_peaks_shown:,} of {num_peaks_total:,} peaks (top intensity)"
                    )
                else:
                    self.view_3d_status.set_text(f"Showing {num_peaks_shown:,} peaks")

        except Exception as e:
            if self.view_3d_status:
                self.view_3d_status.set_text(f"3D plot error: {str(e)[:50]}")

    def _add_features_to_3d_plot(self, fig):
        """Add feature bounding boxes to the 3D plotly figure."""
        import plotly.graph_objects as go

        if self.feature_map is None:
            return

        # Collect all box edges for a single trace (more efficient)
        box_x = []
        box_y = []
        box_z = []

        for _i, feature in enumerate(self.feature_map):
            rt = feature.getRT()
            mz = feature.getMZ()

            # Check if feature is in current view
            if not (self.view_rt_min <= rt <= self.view_rt_max and self.view_mz_min <= mz <= self.view_mz_max):
                continue

            # Get RT and m/z bounds from convex hull
            hulls = feature.getConvexHulls()
            if hulls and len(hulls) > 0:
                # Get bounds from convex hull points
                hull_points = hulls[0].getHullPoints()
                if len(hull_points) > 0:
                    rt_vals = [p[0] for p in hull_points]
                    mz_vals = [p[1] for p in hull_points]
                    rt_min, rt_max = min(rt_vals), max(rt_vals)
                    mz_min, mz_max = min(mz_vals), max(mz_vals)
                else:
                    # Fallback to small box around centroid
                    rt_min, rt_max = rt - 5, rt + 5
                    mz_min, mz_max = mz - 0.5, mz + 0.5
            else:
                # Fallback to small box around centroid
                rt_min, rt_max = rt - 5, rt + 5
                mz_min, mz_max = mz - 0.5, mz + 0.5

            # Draw box edges on the baseline (z=0)
            # Bottom rectangle (4 edges)
            z_base = 0
            # Edge 1: rt_min to rt_max at mz_min
            box_x.extend([rt_min, rt_max, None])
            box_y.extend([mz_min, mz_min, None])
            box_z.extend([z_base, z_base, None])
            # Edge 2: rt_max at mz_min to mz_max
            box_x.extend([rt_max, rt_max, None])
            box_y.extend([mz_min, mz_max, None])
            box_z.extend([z_base, z_base, None])
            # Edge 3: rt_max to rt_min at mz_max
            box_x.extend([rt_max, rt_min, None])
            box_y.extend([mz_max, mz_max, None])
            box_z.extend([z_base, z_base, None])
            # Edge 4: rt_min at mz_max to mz_min
            box_x.extend([rt_min, rt_min, None])
            box_y.extend([mz_max, mz_min, None])
            box_z.extend([z_base, z_base, None])

        if box_x:
            # Add all bounding boxes as a single trace
            fig.add_trace(
                go.Scatter3d(
                    x=box_x,
                    y=box_y,
                    z=box_z,
                    mode="lines",
                    line={"color": "#00ff66", "width": 3},
                    name="Features",
                    hoverinfo="skip",
                )
            )

    # ==================== Search and Filter Methods ====================

    def search_global(self, query: str) -> list[dict[str, Any]]:
        """
        Search across spectra, IDs, and features.
        Returns list of results with type, description, and action.
        """
        results = []
        query = query.strip()
        if not query:
            return results

        query_lower = query.lower()

        # Try to parse as spectrum number (e.g., "#123" or "123")
        spec_match = query.replace("#", "").strip()
        if spec_match.isdigit():
            spec_idx = int(spec_match)
            if self.exp and 0 <= spec_idx < self.exp.size():
                spec = self.exp[spec_idx]
                results.append(
                    {
                        "type": "spectrum",
                        "icon": "analytics",
                        "label": f"Spectrum #{spec_idx} (MS{spec.getMSLevel()}, RT={spec.getRT():.1f}s)",
                        "action": lambda idx=spec_idx: self.show_spectrum_in_browser(idx),
                    }
                )

        # Try to parse as m/z value (e.g., "500.25" or "mz:500.25")
        mz_query = query_lower.replace("mz:", "").replace("m/z:", "").strip()
        try:
            mz_val = float(mz_query)
            if 50 < mz_val < 10000:  # Reasonable m/z range
                # Find spectra with precursor near this m/z
                if self.exp:
                    for i in range(min(self.exp.size(), 1000)):  # Limit search
                        spec = self.exp[i]
                        if spec.getMSLevel() > 1:
                            precs = spec.getPrecursors()
                            if precs and abs(precs[0].getMZ() - mz_val) < 0.5:
                                results.append(
                                    {
                                        "type": "spectrum",
                                        "icon": "analytics",
                                        "label": f"MS2 #{i} precursor m/z={precs[0].getMZ():.4f}",
                                        "action": lambda idx=i: self.show_spectrum_in_browser(idx),
                                    }
                                )
                                if len(results) >= 10:
                                    break
                # Find IDs near this m/z
                for i, pid in enumerate(self.peptide_ids[:100]):
                    if abs(pid.getMZ() - mz_val) < 0.5:
                        hits = pid.getHits()
                        seq = hits[0].getSequence().toString() if hits else "?"
                        results.append(
                            {
                                "type": "id",
                                "icon": "biotech",
                                "label": f"ID: {seq} (m/z={pid.getMZ():.4f})",
                                "action": lambda idx=i: self.zoom_to_id(idx),
                            }
                        )
                        if len(results) >= 15:
                            break
        except ValueError:
            pass

        # Try to parse as RT value (e.g., "rt:100" or "100s")
        rt_query = query_lower.replace("rt:", "").replace("s", "").strip()
        try:
            rt_val = float(rt_query)
            if self.rt_min <= rt_val <= self.rt_max:
                results.append(
                    {
                        "type": "navigate",
                        "icon": "place",
                        "label": f"Go to RT={rt_val:.1f}s",
                        "action": lambda rt=rt_val: self.go_to_rt(rt),
                    }
                )
        except ValueError:
            pass

        # Search peptide sequences
        if len(query) >= 2:
            for i, pid in enumerate(self.peptide_ids[:500]):
                hits = pid.getHits()
                if hits:
                    seq = hits[0].getSequence().toString()
                    if query_lower in seq.lower():
                        results.append(
                            {
                                "type": "id",
                                "icon": "biotech",
                                "label": f"{seq} (RT={pid.getRT():.1f}s, m/z={pid.getMZ():.2f})",
                                "action": lambda idx=i: self.zoom_to_id(idx),
                            }
                        )
                        if len(results) >= 20:
                            break

        return results[:20]  # Limit total results

    def go_to_rt(self, rt: float):
        """Center view on given RT value."""
        if self.df is None:
            return
        self.push_zoom_history()
        rt_range = self.view_rt_max - self.view_rt_min
        self.view_rt_min = max(self.rt_min, rt - rt_range / 2)
        self.view_rt_max = min(self.rt_max, rt + rt_range / 2)
        self.push_zoom_history()
        self.update_plot()
        # Also show closest spectrum
        self.show_spectrum_at_rt(rt)

    def go_to_mz(self, mz: float):
        """Center view on given m/z value."""
        if self.df is None:
            return
        self.push_zoom_history()
        mz_range = self.view_mz_max - self.view_mz_min
        self.view_mz_min = max(self.mz_min, mz - mz_range / 2)
        self.view_mz_max = min(self.mz_max, mz + mz_range / 2)
        self.push_zoom_history()
        self.update_plot()

    def go_to_location(
        self, rt: Optional[float] = None, mz: Optional[float] = None, spectrum_idx: Optional[int] = None
    ):
        """Navigate to a specific location."""
        if spectrum_idx is not None and self.exp and 0 <= spectrum_idx < self.exp.size():
            self.show_spectrum_in_browser(spectrum_idx)
            spec = self.exp[spectrum_idx]
            rt = spec.getRT()

        if self.df is None:
            return

        self.push_zoom_history()

        if rt is not None:
            rt_range = self.view_rt_max - self.view_rt_min
            self.view_rt_min = max(self.rt_min, rt - rt_range / 2)
            self.view_rt_max = min(self.rt_max, rt + rt_range / 2)

        if mz is not None:
            mz_range = self.view_mz_max - self.view_mz_min
            self.view_mz_min = max(self.mz_min, mz - mz_range / 2)
            self.view_mz_max = min(self.mz_max, mz + mz_range / 2)

        self.push_zoom_history()
        self.update_plot()

    def filter_spectrum_data(
        self,
        ms_level: Optional[int] = None,
        rt_min: Optional[float] = None,
        rt_max: Optional[float] = None,
        min_peaks: Optional[int] = None,
        min_tic: Optional[float] = None,
    ) -> list[dict]:
        """Filter spectrum data based on criteria."""
        filtered = []
        for row in self.spectrum_data:
            if ms_level is not None and row["ms_level"] != ms_level:
                continue
            if rt_min is not None and row["rt"] < rt_min:
                continue
            if rt_max is not None and row["rt"] > rt_max:
                continue
            if min_peaks is not None and row["n_peaks"] < min_peaks:
                continue
            if min_tic is not None and row["tic"] < min_tic:
                continue
            filtered.append(row)
        return filtered

    def filter_id_data(
        self, sequence_pattern: Optional[str] = None, min_score: Optional[float] = None, charge: Optional[int] = None
    ) -> list[dict]:
        """Filter ID data based on criteria."""
        filtered = []
        for row in self.id_data:
            if sequence_pattern and sequence_pattern.lower() not in row["sequence"].lower():
                continue
            if min_score is not None and row["score"] < min_score:
                continue
            if charge is not None and row["charge"] != charge:
                continue
            filtered.append(row)
        return filtered

    def filter_feature_data(
        self, min_intensity: Optional[float] = None, min_quality: Optional[float] = None, charge: Optional[int] = None
    ) -> list[dict]:
        """Filter feature data based on criteria."""
        filtered = []
        for row in self.feature_data:
            if min_intensity is not None and row["intensity"] < min_intensity:
                continue
            if min_quality is not None and row["quality"] < min_quality:
                continue
            if charge is not None and row["charge"] != charge:
                continue
            filtered.append(row)
        return filtered

    def reset_view(self):
        """Reset to full view."""
        if self.df is None:
            return
        # Clear zoom history on reset
        self.zoom_history = []
        self.view_rt_min = self.rt_min
        self.view_rt_max = self.rt_max
        self.view_mz_min = self.mz_min
        self.view_mz_max = self.mz_max
        self.selected_feature_idx = None
        self.selected_id_idx = None
        self.push_zoom_history()  # Add full view as first entry
        self.update_plot()

    def zoom_in(self, factor=0.5):
        """Zoom in."""
        if self.df is None:
            return
        self.push_zoom_history()  # Save current state before zoom
        rt_center = (self.view_rt_min + self.view_rt_max) / 2
        mz_center = (self.view_mz_min + self.view_mz_max) / 2
        rt_range = (self.view_rt_max - self.view_rt_min) * factor / 2
        mz_range = (self.view_mz_max - self.view_mz_min) * factor / 2

        self.view_rt_min = rt_center - rt_range
        self.view_rt_max = rt_center + rt_range
        self.view_mz_min = mz_center - mz_range
        self.view_mz_max = mz_center + mz_range
        self.push_zoom_history()  # Save new state
        self.update_plot()

    def zoom_out(self, factor=2.0):
        """Zoom out."""
        if self.df is None:
            return
        self.push_zoom_history()  # Save current state before zoom
        rt_center = (self.view_rt_min + self.view_rt_max) / 2
        mz_center = (self.view_mz_min + self.view_mz_max) / 2
        rt_range = (self.view_rt_max - self.view_rt_min) * factor / 2
        mz_range = (self.view_mz_max - self.view_mz_min) * factor / 2

        self.view_rt_min = max(self.rt_min, rt_center - rt_range)
        self.view_rt_max = min(self.rt_max, rt_center + rt_range)
        self.view_mz_min = max(self.mz_min, mz_center - mz_range)
        self.view_mz_max = min(self.mz_max, mz_center + mz_range)
        self.push_zoom_history()  # Save new state
        self.update_plot()

    def pan(self, rt_frac=0, mz_frac=0):
        """Pan view."""
        if self.df is None:
            return
        rt_shift = (self.view_rt_max - self.view_rt_min) * rt_frac
        mz_shift = (self.view_mz_max - self.view_mz_min) * mz_frac

        if self.view_rt_min + rt_shift < self.rt_min:
            rt_shift = self.rt_min - self.view_rt_min
        if self.view_rt_max + rt_shift > self.rt_max:
            rt_shift = self.rt_max - self.view_rt_max
        if self.view_mz_min + mz_shift < self.mz_min:
            mz_shift = self.mz_min - self.view_mz_min
        if self.view_mz_max + mz_shift > self.mz_max:
            mz_shift = self.mz_max - self.view_mz_max

        self.view_rt_min += rt_shift
        self.view_rt_max += rt_shift
        self.view_mz_min += mz_shift
        self.view_mz_max += mz_shift
        self.update_plot()

    def apply_custom_range(self, rt_min, rt_max, mz_min, mz_max):
        """Apply custom range."""
        if self.df is None:
            return
        try:
            self.view_rt_min = max(self.rt_min, float(rt_min))
            self.view_rt_max = min(self.rt_max, float(rt_max))
            self.view_mz_min = max(self.mz_min, float(mz_min))
            self.view_mz_max = min(self.mz_max, float(mz_max))
            self.update_plot()
        except ValueError:
            ui.notify("Invalid range values", type="warning")

    def zoom_at_point(self, x_frac: float, y_frac: float, zoom_in: bool = True):
        """Zoom centered on a specific point (given as fraction of plot area).

        Args:
            x_frac: Horizontal position (0=left, 1=right) in plot area
            y_frac: Vertical position (0=top, 1=bottom) in plot area
            zoom_in: True to zoom in, False to zoom out
        """
        if self.df is None:
            return

        # Save current state to zoom history
        self.push_zoom_history()

        # Current ranges
        rt_range = self.view_rt_max - self.view_rt_min
        mz_range = self.view_mz_max - self.view_mz_min

        # Zoom factor
        factor = 0.7 if zoom_in else 1.4

        # New ranges
        new_rt_range = rt_range * factor
        new_mz_range = mz_range * factor

        if self.swap_axes:
            # swap_axes=True: m/z on x-axis, RT on y-axis
            mz_point = self.view_mz_min + x_frac * mz_range
            rt_point = self.view_rt_max - y_frac * rt_range  # Y is inverted

            # Keep the point under cursor at same position
            new_mz_min = mz_point - x_frac * new_mz_range
            new_mz_max = mz_point + (1 - x_frac) * new_mz_range
            new_rt_min = rt_point - (1 - y_frac) * new_rt_range
            new_rt_max = rt_point + y_frac * new_rt_range
        else:
            # swap_axes=False: RT on x-axis, m/z on y-axis
            rt_point = self.view_rt_min + x_frac * rt_range
            mz_point = self.view_mz_max - y_frac * mz_range  # Y is inverted

            # Keep the point under cursor at same position
            new_rt_min = rt_point - x_frac * new_rt_range
            new_rt_max = rt_point + (1 - x_frac) * new_rt_range
            new_mz_min = mz_point - (1 - y_frac) * new_mz_range
            new_mz_max = mz_point + y_frac * new_mz_range

        # Clamp to data bounds
        self.view_rt_min = max(self.rt_min, new_rt_min)
        self.view_rt_max = min(self.rt_max, new_rt_max)
        self.view_mz_min = max(self.mz_min, new_mz_min)
        self.view_mz_max = min(self.mz_max, new_mz_max)

        self.update_plot()

    def pan_by_pixels(self, dx: float, dy: float):
        """Pan the view by pixel amounts.

        Args:
            dx: Horizontal pixel delta (positive = pan right/increase RT)
            dy: Vertical pixel delta (positive = pan down/decrease mz)
        """
        if self.df is None:
            return

        # Convert pixels to data units
        rt_per_pixel = (self.view_rt_max - self.view_rt_min) / self.plot_width
        mz_per_pixel = (self.view_mz_max - self.view_mz_min) / self.plot_height

        rt_shift = -dx * rt_per_pixel  # Negative because dragging right should decrease RT view
        mz_shift = dy * mz_per_pixel  # Positive because dragging down should decrease mz view

        # Clamp shifts to stay within bounds
        if self.view_rt_min + rt_shift < self.rt_min:
            rt_shift = self.rt_min - self.view_rt_min
        if self.view_rt_max + rt_shift > self.rt_max:
            rt_shift = self.rt_max - self.view_rt_max
        if self.view_mz_min + mz_shift < self.mz_min:
            mz_shift = self.mz_min - self.view_mz_min
        if self.view_mz_max + mz_shift > self.mz_max:
            mz_shift = self.mz_max - self.view_mz_max

        self.view_rt_min += rt_shift
        self.view_rt_max += rt_shift
        self.view_mz_min += mz_shift
        self.view_mz_max += mz_shift

        self.update_plot()


def create_ui():
    """Create NiceGUI interface - root page function for NiceGUI 3.x."""
    viewer = Viewer()

    dark = ui.dark_mode()
    dark.enable()
    viewer.dark = dark  # Store dark mode reference for spectrum plot colors

    # Top-right corner buttons (dark mode toggle + fullscreen)
    with ui.element("div").classes("fixed top-2 right-2 z-50 flex gap-1"):

        def toggle_dark_mode():
            dark.toggle()
            dark_btn.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")
            dark_btn._props["icon"] = "light_mode" if dark.value else "dark_mode"
            dark_btn.update()

        dark_btn = (
            ui.button(icon="light_mode", on_click=toggle_dark_mode)
            .props("flat round dense color=grey")
            .tooltip("Toggle dark/light mode")
        )
        ui.button(
            icon="fullscreen",
            on_click=lambda: ui.run_javascript("""
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        """),
        ).props("flat round dense color=grey").tooltip("Toggle fullscreen (F11)")

    with ui.column().classes("w-full items-center p-2"):
        # Compact toolbar for file operations and info display
        with ui.row().classes("w-full max-w-[1700px] items-center gap-2 px-2 py-1 rounded").style(
            "background: rgba(128,128,128,0.1);"
        ):

            async def handle_upload(e):
                """Handle uploaded file - detect type and load appropriately."""
                # NiceGUI 3.x: UploadEventArguments has .file attribute (FileUpload object)
                # FileUpload has: .name, .content_type, async .read(), async .save(path)
                file = e.file
                filename = file.name.lower()
                original_name = file.name

                # Save to temp file using FileUpload.save()
                suffix = Path(filename).suffix
                tmp_path = tempfile.mktemp(suffix=suffix)
                await file.save(tmp_path)

                try:
                    if filename.endswith(".mzml"):
                        # Load mzML file (blocking pyOpenMS call runs in background thread)
                        success = await run.io_bound(viewer.load_mzml_sync, tmp_path)
                        if success:
                            viewer.update_plot()
                            viewer.update_tic_plot()
                            # Show first spectrum in 1D browser and expand panels
                            if viewer.exp and viewer.exp.size() > 0:
                                viewer.show_spectrum_in_browser(0)
                                if viewer.spectrum_expansion is not None:
                                    viewer.spectrum_expansion.set_value(True)
                                if viewer.tic_expansion is not None:
                                    viewer.tic_expansion.set_value(True)
                                if viewer.spectrum_table_expansion is not None:
                                    viewer.spectrum_table_expansion.set_value(True)
                            info_text = (
                                f"Loaded: {original_name} | Spectra: {viewer.exp.size():,} | Peaks: {len(viewer.df):,}"
                            )
                            if viewer.has_faims:
                                info_text += f" | FAIMS: {len(viewer.faims_cvs)} CVs"
                            if viewer.info_label:
                                viewer.info_label.set_text(info_text)
                            if viewer.spectrum_table is not None:
                                viewer.spectrum_table.rows = viewer.spectrum_data
                            if viewer.has_faims:
                                if viewer.faims_info_label:
                                    cv_str = ", ".join([f"{cv:.1f}V" for cv in viewer.faims_cvs])
                                    viewer.faims_info_label.set_text(f"FAIMS CVs: {cv_str}")
                                    viewer.faims_info_label.set_visibility(True)
                                if viewer.faims_toggle:
                                    viewer.faims_toggle.set_visibility(True)
                            # Update chromatogram panel
                            if viewer.has_chromatograms:
                                if viewer.chromatogram_info_label:
                                    viewer.chromatogram_info_label.set_text(
                                        f"Chromatograms: {len(viewer.chromatograms):,}"
                                    )
                                if viewer.chromatogram_table is not None:
                                    viewer.chromatogram_table.rows = viewer.chromatograms
                                if viewer.chromatogram_expansion is not None:
                                    viewer.chromatogram_expansion.set_value(True)
                            else:
                                if viewer.chromatogram_info_label:
                                    viewer.chromatogram_info_label.set_text("No chromatograms in file")
                            # Update ion mobility panel
                            if viewer.has_ion_mobility:
                                im_type_name = {
                                    "inverse_k0": "1/K₀",
                                    "drift_time": "Drift Time",
                                    "ion_mobility": "Ion Mobility"
                                }.get(viewer.im_type, "Ion Mobility")
                                if viewer.im_info_label:
                                    viewer.im_info_label.set_text(
                                        f"{im_type_name}: {len(viewer.im_df):,} peaks"
                                    )
                                if viewer.im_range_label:
                                    viewer.im_range_label.set_text(
                                        f"IM: {viewer.im_min:.3f} - {viewer.im_max:.3f} {viewer.im_unit}"
                                    )
                                viewer.update_im_plot()
                                if viewer.im_expansion is not None:
                                    viewer.im_expansion.set_value(True)
                            else:
                                if viewer.im_info_label:
                                    viewer.im_info_label.set_text("No ion mobility data")
                            # Update panel visibility based on loaded data
                            viewer.update_panel_visibility()
                            ui.notify(f"Loaded {len(viewer.df):,} peaks from {original_name}", type="positive")
                        else:
                            ui.notify(f"Failed to load {original_name}", type="negative")

                    elif filename.endswith(".featurexml") or (filename.endswith(".xml") and "feature" in filename):
                        success = await run.io_bound(viewer.load_featuremap_sync, tmp_path)
                        if success:
                            viewer.update_plot()
                            if viewer.feature_info_label:
                                viewer.feature_info_label.set_text(f"Features: {viewer.feature_map.size():,}")
                            if viewer.feature_table is not None:
                                viewer.feature_table.rows = viewer.feature_data
                            # Update panel visibility (features panel now has data)
                            viewer.update_panel_visibility()
                            ui.notify(
                                f"Loaded {viewer.feature_map.size():,} features from {original_name}", type="positive"
                            )

                    elif filename.endswith(".idxml") or (filename.endswith(".xml") and "id" in filename):
                        success = await run.io_bound(viewer.load_idxml_sync, tmp_path)
                        if success:
                            viewer.update_plot()
                            n_linked = sum(1 for s in viewer.spectrum_data if s.get("id_idx") is not None)
                            if viewer.id_info_label:
                                viewer.id_info_label.set_text(f"IDs: {len(viewer.peptide_ids):,} ({n_linked} linked)")
                            # Update unified spectrum table with ID info
                            if viewer.spectrum_table is not None:
                                viewer.spectrum_table.rows = viewer.spectrum_data
                            ui.notify(
                                f"Loaded {len(viewer.peptide_ids):,} IDs ({n_linked} linked) from {original_name}",
                                type="positive",
                            )

                    else:
                        ui.notify(
                            f"Unknown file type: {original_name}. Supported: .mzML, .featureXML, .idXML", type="warning"
                        )

                except Exception as ex:
                    ui.notify(f"Error loading {original_name}: {ex}", type="negative")
                    viewer.set_loading(False)

                finally:
                    # Clean up temp file
                    try:
                        Path(tmp_path).unlink()
                    except Exception:
                        pass

            async def open_native_file_dialog():
                """Open native file dialog to select files directly from filesystem.

                Only available in native mode (--native flag). Uses pywebview's
                create_file_dialog() to show a native OS file picker.
                """
                # Check if running in native mode
                if not app.native.main_window:
                    ui.notify("Native file dialog is only available in native mode (--native)", type="warning")
                    return

                try:
                    # Open file dialog with filters for supported file types
                    files = await app.native.main_window.create_file_dialog(
                        allow_multiple=True,
                        file_types=(
                            "Mass Spec Files (*.mzML;*.featureXML;*.idXML)",
                            "mzML Files (*.mzML)",
                            "Feature Files (*.featureXML)",
                            "ID Files (*.idXML)",
                            "All Files (*.*)",
                        ),
                    )

                    if not files:
                        return  # User cancelled

                    # Process each selected file
                    for filepath in files:
                        filename = Path(filepath).name
                        ext = Path(filepath).suffix.lower()

                        try:
                            if ext == ".mzml":
                                success = await run.io_bound(viewer.load_mzml_sync, filepath)
                                if success:
                                    viewer.update_plot()
                                    viewer.update_tic_plot()
                                    if viewer.exp and viewer.exp.size() > 0:
                                        viewer.show_spectrum_in_browser(0)
                                        if viewer.spectrum_expansion is not None:
                                            viewer.spectrum_expansion.set_value(True)
                                        if viewer.tic_expansion is not None:
                                            viewer.tic_expansion.set_value(True)
                                        if viewer.spectrum_table_expansion is not None:
                                            viewer.spectrum_table_expansion.set_value(True)
                                    info_text = (
                                        f"Loaded: {filename} | Spectra: {viewer.exp.size():,} | "
                                        f"Peaks: {len(viewer.df):,}"
                                    )
                                    if viewer.has_faims:
                                        info_text += f" | FAIMS: {len(viewer.faims_cvs)} CVs"
                                    if viewer.info_label:
                                        viewer.info_label.set_text(info_text)
                                    if viewer.spectrum_table is not None:
                                        viewer.spectrum_table.rows = viewer.spectrum_data
                                    if viewer.has_faims:
                                        if viewer.faims_info_label:
                                            cv_str = ", ".join([f"{cv:.1f}V" for cv in viewer.faims_cvs])
                                            viewer.faims_info_label.set_text(f"FAIMS CVs: {cv_str}")
                                            viewer.faims_info_label.set_visibility(True)
                                        if viewer.faims_toggle:
                                            viewer.faims_toggle.set_visibility(True)
                                    # Update chromatogram panel
                                    if viewer.has_chromatograms:
                                        if viewer.chromatogram_info_label:
                                            viewer.chromatogram_info_label.set_text(
                                                f"Chromatograms: {len(viewer.chromatograms):,}"
                                            )
                                        if viewer.chromatogram_table is not None:
                                            viewer.chromatogram_table.rows = viewer.chromatograms
                                        if viewer.chromatogram_expansion is not None:
                                            viewer.chromatogram_expansion.set_value(True)
                                    else:
                                        if viewer.chromatogram_info_label:
                                            viewer.chromatogram_info_label.set_text("No chromatograms in file")
                                    # Update ion mobility panel
                                    if viewer.has_ion_mobility:
                                        im_type_name = {
                                            "inverse_k0": "1/K₀",
                                            "drift_time": "Drift Time",
                                            "ion_mobility": "Ion Mobility"
                                        }.get(viewer.im_type, "Ion Mobility")
                                        if viewer.im_info_label:
                                            viewer.im_info_label.set_text(
                                                f"{im_type_name}: {len(viewer.im_df):,} peaks"
                                            )
                                        if viewer.im_range_label:
                                            viewer.im_range_label.set_text(
                                                f"IM: {viewer.im_min:.3f} - {viewer.im_max:.3f} {viewer.im_unit}"
                                            )
                                        viewer.update_im_plot()
                                        if viewer.im_expansion is not None:
                                            viewer.im_expansion.set_value(True)
                                    else:
                                        if viewer.im_info_label:
                                            viewer.im_info_label.set_text("No ion mobility data")
                                    # Update panel visibility based on loaded data
                                    viewer.update_panel_visibility()
                                    ui.notify(f"Loaded {len(viewer.df):,} peaks from {filename}", type="positive")
                                else:
                                    ui.notify(f"Failed to load {filename}", type="negative")

                            elif ext == ".featurexml":
                                success = await run.io_bound(viewer.load_featuremap_sync, filepath)
                                if success:
                                    viewer.update_plot()
                                    if viewer.feature_info_label:
                                        viewer.feature_info_label.set_text(f"Features: {viewer.feature_map.size():,}")
                                    if viewer.feature_table is not None:
                                        viewer.feature_table.rows = viewer.feature_data
                                    # Update panel visibility (features panel now has data)
                                    viewer.update_panel_visibility()
                                    ui.notify(
                                        f"Loaded {viewer.feature_map.size():,} features from {filename}",
                                        type="positive",
                                    )

                            elif ext == ".idxml":
                                success = await run.io_bound(viewer.load_idxml_sync, filepath)
                                if success:
                                    viewer.update_plot()
                                    n_linked = sum(1 for s in viewer.spectrum_data if s.get("id_idx") is not None)
                                    if viewer.id_info_label:
                                        viewer.id_info_label.set_text(f"IDs: {len(viewer.peptide_ids):,} ({n_linked} linked)")
                                    if viewer.spectrum_table is not None:
                                        viewer.spectrum_table.rows = viewer.spectrum_data
                                    ui.notify(
                                        f"Loaded {len(viewer.peptide_ids):,} IDs ({n_linked} linked) from {filename}",
                                        type="positive",
                                    )

                            else:
                                ui.notify(
                                    f"Unknown file type: {filename}. Supported: .mzML, .featureXML, .idXML",
                                    type="warning",
                                )

                        except Exception as ex:
                            ui.notify(f"Error loading {filename}: {ex}", type="negative")
                            viewer.set_loading(False)

                except Exception as ex:
                    ui.notify(f"File dialog error: {ex}", type="negative")

            # Open button
            ui.button(
                icon="folder_open",
                on_click=open_native_file_dialog,
            ).props("flat dense").tooltip("Open files (native mode)")

            # Compact drop zone
            ui.upload(
                label="Drop files here",
                on_upload=handle_upload,
                auto_upload=True,
                multiple=True,
            ).classes("w-40").props(
                'accept=".mzML,.mzml,.featureXML,.featurexml,.idXML,.idxml,.xml" flat dense bordered'
            ).style("min-height: 32px;")

            ui.separator().props("vertical").classes("h-6")

            # Clear menu (dropdown)
            def clear_features():
                viewer.clear_features()
                viewer.update_plot()
                if viewer.feature_info_label:
                    viewer.feature_info_label.set_text("")

            def clear_ids():
                viewer.clear_ids()
                viewer.update_plot()
                if viewer.id_info_label:
                    viewer.id_info_label.set_text("")

            def clear_all():
                clear_features()
                clear_ids()

            with ui.button(icon="delete_outline").props("flat dense size=sm").tooltip("Clear data"):
                with ui.menu().props("auto-close"):
                    ui.menu_item("Clear Features", on_click=clear_features).classes("text-cyan-400")
                    ui.menu_item("Clear IDs", on_click=clear_ids).classes("text-orange-400")
                    ui.separator()
                    ui.menu_item("Clear All", on_click=clear_all).classes("text-red-400")

            ui.separator().props("vertical").classes("h-6")

            # Settings button for panel order
            def show_panel_settings():
                with ui.dialog() as dialog, ui.card().classes("min-w-[400px]"):
                    ui.label("Panel Configuration").classes("text-lg font-bold mb-2")

                    # Panel visibility section
                    ui.label("Visibility").classes("text-sm font-semibold text-gray-400 mt-2")
                    ui.label("Toggle panels on/off. 'Auto' hides when no data.").classes(
                        "text-xs text-gray-500 mb-2"
                    )

                    visibility_container = ui.column().classes("w-full gap-1 mb-4")

                    # Panels that support "auto" visibility
                    auto_panels = {"chromatograms", "im_peakmap", "features_table"}

                    def refresh_visibility():
                        visibility_container.clear()
                        with visibility_container:
                            for panel_id in viewer.panel_order:
                                panel_def = viewer.panel_definitions.get(panel_id, {})
                                current_vis = viewer.panel_visibility.get(panel_id, True)

                                with ui.row().classes("w-full items-center gap-2"):
                                    ui.icon(panel_def.get("icon", "widgets")).classes("text-gray-400 text-sm")
                                    ui.label(panel_def.get("name", panel_id)).classes("flex-grow text-sm")

                                    if panel_id in auto_panels:
                                        # Three-state toggle for auto-capable panels
                                        options = ["Hide", "Auto", "Show"]
                                        if current_vis is False:
                                            current_val = "Hide"
                                        elif current_vis == "auto":
                                            current_val = "Auto"
                                        else:
                                            current_val = "Show"

                                        def make_toggle_handler(pid=panel_id):
                                            def handler(e):
                                                if e.value == "Hide":
                                                    viewer.panel_visibility[pid] = False
                                                elif e.value == "Auto":
                                                    viewer.panel_visibility[pid] = "auto"
                                                else:
                                                    viewer.panel_visibility[pid] = True

                                            return handler

                                        ui.toggle(options, value=current_val, on_change=make_toggle_handler()).props(
                                            "dense size=sm"
                                        ).classes("text-xs")
                                    else:
                                        # Simple checkbox for always-visible panels

                                        def make_checkbox_handler(pid=panel_id):
                                            def handler(e):
                                                viewer.panel_visibility[pid] = e.value

                                            return handler

                                        ui.checkbox(
                                            "", value=current_vis is True, on_change=make_checkbox_handler()
                                        ).props("dense")

                    refresh_visibility()

                    ui.separator().classes("my-2")

                    # Panel order section
                    ui.label("Order").classes("text-sm font-semibold text-gray-400")
                    ui.label("Reorder panels using arrows").classes("text-xs text-gray-500 mb-2")

                    panel_list = ui.column().classes("w-full gap-1")

                    def refresh_list():
                        panel_list.clear()
                        with panel_list:
                            for idx, panel_id in enumerate(viewer.panel_order):
                                panel_def = viewer.panel_definitions.get(panel_id, {})
                                with ui.row().classes("w-full items-center gap-2 p-1 rounded").style(
                                    "background: rgba(128,128,128,0.15);"
                                ):
                                    ui.icon(panel_def.get("icon", "widgets")).classes("text-gray-400 text-sm")
                                    ui.label(panel_def.get("name", panel_id)).classes("flex-grow text-sm")

                                    def move_up(i=idx):
                                        if i > 0:
                                            viewer.panel_order[i], viewer.panel_order[i - 1] = (
                                                viewer.panel_order[i - 1],
                                                viewer.panel_order[i],
                                            )
                                            refresh_list()
                                            refresh_visibility()

                                    def move_down(i=idx):
                                        if i < len(viewer.panel_order) - 1:
                                            viewer.panel_order[i], viewer.panel_order[i + 1] = (
                                                viewer.panel_order[i + 1],
                                                viewer.panel_order[i],
                                            )
                                            refresh_list()
                                            refresh_visibility()

                                    ui.button(icon="keyboard_arrow_up", on_click=move_up).props(
                                        "flat dense size=sm"
                                    ).set_enabled(idx > 0)
                                    ui.button(icon="keyboard_arrow_down", on_click=move_down).props(
                                        "flat dense size=sm"
                                    ).set_enabled(idx < len(viewer.panel_order) - 1)

                    refresh_list()

                    with ui.row().classes("w-full justify-end gap-2 mt-4"):

                        def apply_settings():
                            # Reorder panels using move()
                            if viewer.panels_container:
                                for idx, panel_id in enumerate(viewer.panel_order):
                                    if panel_id in viewer.panel_elements:
                                        viewer.panel_elements[panel_id].move(target_index=idx)
                            # Apply visibility
                            viewer.update_panel_visibility()
                            dialog.close()
                            ui.notify("Panel settings updated", type="positive")

                        ui.button("Cancel", on_click=dialog.close).props("flat")
                        ui.button("Apply", on_click=apply_settings).props("color=primary")

                dialog.open()

            ui.button(icon="tune", on_click=show_panel_settings).props("flat dense").tooltip("Panel Settings")

            # Spacer to push info to the right
            ui.space()

            # Inline info labels
            viewer.info_label = ui.label("No file loaded").classes("text-xs text-gray-400")
            viewer.feature_info_label = ui.label("").classes("text-xs text-cyan-400")
            viewer.id_info_label = ui.label("").classes("text-xs text-orange-400")
            viewer.faims_info_label = ui.label("").classes("text-xs text-purple-400")
            viewer.faims_info_label.set_visibility(False)
            viewer.status_label = ui.label("").classes("text-xs text-green-400")

            # Range labels
            viewer.rt_range_label = ui.label("").classes("text-xs text-blue-300")
            viewer.mz_range_label = ui.label("").classes("text-xs text-blue-300")

            # FAIMS toggle (hidden by default, shown when FAIMS data is detected)
            def toggle_faims_view():
                viewer.show_faims_view = faims_toggle.value
                if viewer.faims_container:
                    viewer.faims_container.set_visibility(viewer.show_faims_view)
                if viewer.df is not None and viewer.show_faims_view:
                    viewer.update_faims_plots()

            faims_toggle = ui.checkbox("FAIMS", value=False, on_change=toggle_faims_view).props("dense").classes(
                "text-xs text-purple-400"
            )
            faims_toggle.set_visibility(False)
            viewer.faims_toggle = faims_toggle

        # Panels container - holds all reorderable expansion panels
        viewer.panels_container = ui.column().classes("w-full items-center gap-2")

        # TIC Plot (clickable to show MS1 spectrum, zoomable to update peak map)
        viewer.tic_expansion = ui.expansion("TIC (Total Ion Chromatogram)", icon="show_chart", value=False).classes(
            "w-full max-w-[1700px]"
        )
        viewer.panel_elements["tic"] = viewer.tic_expansion
        viewer.tic_expansion.move(target_container=viewer.panels_container)
        with viewer.tic_expansion:
            ui.label("Click to view spectrum, drag to zoom RT range").classes("text-xs text-gray-500 mb-1")
            viewer.tic_plot = ui.plotly(viewer.create_tic_plot()).classes("w-full")

            # Handle click on TIC plot - show closest spectrum and center peak map
            def on_tic_click(e):
                try:
                    if e.args and "points" in e.args and e.args["points"]:
                        point = e.args["points"][0]
                        if "x" in point:
                            rt = point["x"]
                            # Show closest spectrum (any MS level) - this also highlights in table
                            viewer.show_spectrum_at_rt(rt)
                            # Also center the peak map on this RT
                            rt_range = viewer.view_rt_max - viewer.view_rt_min
                            viewer.view_rt_min = max(viewer.rt_min, rt - rt_range / 2)
                            viewer.view_rt_max = min(viewer.rt_max, rt + rt_range / 2)
                            viewer.update_plot()
                except Exception:
                    pass

            viewer.tic_plot.on("plotly_click", on_tic_click)

            # Handle zoom/pan on TIC plot - sync RT range to peak map
            def on_tic_relayout(e):
                try:
                    if e.args:
                        args = e.args
                        # Check for x-axis range changes (zoom or pan)
                        if "xaxis.range[0]" in args and "xaxis.range[1]" in args:
                            new_rt_min = float(args["xaxis.range[0]"])
                            new_rt_max = float(args["xaxis.range[1]"])
                            # Clamp to data bounds
                            viewer.view_rt_min = max(viewer.rt_min, new_rt_min)
                            viewer.view_rt_max = min(viewer.rt_max, new_rt_max)
                            # Set flag to prevent TIC reset during update
                            viewer._updating_from_tic = True
                            viewer.update_plot()
                            viewer._updating_from_tic = False
                        elif "xaxis.autorange" in args and args["xaxis.autorange"]:
                            # Reset to full range
                            viewer.view_rt_min = viewer.rt_min
                            viewer.view_rt_max = viewer.rt_max
                            viewer._updating_from_tic = True
                            viewer.update_plot()
                            viewer._updating_from_tic = False
                except Exception:
                    viewer._updating_from_tic = False

            viewer.tic_plot.on("plotly_relayout", on_tic_relayout)

        # Chromatograms Panel (for DIA/OpenSWATH/SRM data)
        viewer.chromatogram_expansion = ui.expansion(
            "Chromatograms", icon="timeline", value=False
        ).classes("w-full max-w-[1700px]")
        viewer.panel_elements["chromatograms"] = viewer.chromatogram_expansion
        viewer.chromatogram_expansion.move(target_container=viewer.panels_container)
        with viewer.chromatogram_expansion:
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                viewer.chromatogram_info_label = ui.label("No chromatograms loaded").classes("text-sm text-gray-400")
                ui.element("div").classes("flex-grow")

                def clear_chrom_selection():
                    viewer.clear_chromatogram_selection()
                    if viewer.chromatogram_table:
                        viewer.chromatogram_table.selected = []
                        viewer.chromatogram_table.update()

                ui.button("Clear Selection", on_click=clear_chrom_selection).props("dense outline size=sm color=grey")

            # Chromatogram plot
            viewer.chromatogram_plot = ui.plotly(viewer.create_chromatogram_plot()).classes("w-full")

            ui.label("Chromatogram Table (click to select, Ctrl+click to multi-select)").classes(
                "text-xs text-gray-500 mt-2"
            )

            # Chromatogram table
            chrom_columns = [
                {"name": "idx", "label": "#", "field": "idx", "sortable": True, "align": "left"},
                {"name": "type", "label": "Type", "field": "type", "sortable": True, "align": "left"},
                {"name": "native_id", "label": "Native ID", "field": "native_id", "sortable": True, "align": "left"},
                {"name": "precursor_mz", "label": "Q1 (m/z)", "field": "precursor_mz", "sortable": True, "align": "right"},
                {"name": "product_mz", "label": "Q3 (m/z)", "field": "product_mz", "sortable": True, "align": "right"},
                {"name": "rt_min", "label": "RT Start", "field": "rt_min", "sortable": True, "align": "right"},
                {"name": "rt_max", "label": "RT End", "field": "rt_max", "sortable": True, "align": "right"},
                {"name": "n_points", "label": "Points", "field": "n_points", "sortable": True, "align": "right"},
                {"name": "max_int", "label": "Max Int", "field": "max_int", "sortable": True, "align": "right"},
            ]

            def on_chrom_select(e):
                """Handle chromatogram selection from table."""
                if e.args and len(e.args) > 0:
                    selected_rows = e.args
                    # Get indices of selected chromatograms
                    viewer.selected_chromatogram_indices = [row["idx"] for row in selected_rows if "idx" in row]
                    viewer.update_chromatogram_plot()

            viewer.chromatogram_table = (
                ui.table(
                    columns=chrom_columns,
                    rows=viewer.chromatograms,
                    row_key="idx",
                    pagination={"rowsPerPage": 15, "sortBy": "idx"},
                    selection="multiple",
                    on_select=on_chrom_select,
                )
                .classes("w-full")
                .props('dense flat bordered virtual-scroll')
            )

        # Main visualization area - peak map with spectrum browser overlay (collapsible)
        peakmap_expansion = ui.expansion("2D Peak Map", icon="grid_on", value=False).classes("w-full max-w-[1700px]")
        viewer.panel_elements["peakmap"] = peakmap_expansion
        peakmap_expansion.move(target_container=viewer.panels_container)
        with peakmap_expansion:
            # Display options row
            with ui.row().classes("w-full items-center gap-4 mb-2 flex-wrap"):
                ui.label("Overlay:").classes("text-xs text-gray-400")

                def toggle_centroids():
                    viewer.show_centroids = centroid_cb.value
                    if viewer.df is not None:
                        viewer.update_plot()

                centroid_cb = (
                    ui.checkbox("Centroids", value=True, on_change=toggle_centroids)
                    .props("dense")
                    .classes("text-green-400")
                )

                def toggle_bboxes():
                    viewer.show_bounding_boxes = bbox_cb.value
                    if viewer.df is not None:
                        viewer.update_plot()

                bbox_cb = (
                    ui.checkbox("Bounding Boxes", value=False, on_change=toggle_bboxes)
                    .props("dense")
                    .classes("text-yellow-400")
                )

                def toggle_hulls():
                    viewer.show_convex_hulls = hull_cb.value
                    if viewer.df is not None:
                        viewer.update_plot()

                hull_cb = (
                    ui.checkbox("Convex Hulls", value=False, on_change=toggle_hulls)
                    .props("dense")
                    .classes("text-cyan-400")
                )

                def toggle_ids():
                    viewer.show_ids = ids_cb.value
                    if viewer.df is not None:
                        viewer.update_plot()

                ids_cb = (
                    ui.checkbox("Identifications", value=True, on_change=toggle_ids)
                    .props("dense")
                    .classes("text-orange-400")
                )

                def toggle_id_sequences():
                    viewer.show_id_sequences = id_seq_cb.value
                    if viewer.df is not None:
                        viewer.update_plot()

                id_seq_cb = (
                    ui.checkbox("Sequences", value=False, on_change=toggle_id_sequences)
                    .props("dense")
                    .classes("text-orange-300")
                )
                ui.tooltip("Show peptide sequences on 2D peakmap")

                ui.label("|").classes("text-gray-600 mx-2")
                ui.label("Colormap:").classes("text-xs text-gray-400")

                def change_colormap(e):
                    viewer.colormap = e.value
                    if viewer.df is not None:
                        viewer.update_plot()
                        viewer.update_minimap()

                colormap_options = list(COLORMAPS.keys())
                ui.select(colormap_options, value="jet", on_change=change_colormap).props("dense outlined").classes(
                    "w-28"
                )

                ui.label("|").classes("text-gray-600 mx-2")
                ui.label("RT:").classes("text-xs text-gray-400")

                def toggle_rt_unit(e):
                    viewer.rt_in_minutes = e.value == "min"
                    if viewer.df is not None:
                        viewer.update_plot()
                        viewer.update_minimap()
                        viewer.update_tic_plot()

                ui.toggle(["sec", "min"], value="sec", on_change=toggle_rt_unit).props("dense")

                ui.label("|").classes("text-gray-600 mx-2")

                def toggle_swap_axes():
                    viewer.swap_axes = swap_axes_cb.value
                    if viewer.df is not None:
                        viewer.update_plot()

                swap_axes_cb = (
                    ui.checkbox("Swap Axes", value=True, on_change=toggle_swap_axes)
                    .props("dense")
                    .classes("text-purple-400")
                )
                ui.tooltip(
                    "When checked: m/z on x-axis, RT on y-axis (default). When unchecked: RT on x-axis, m/z on y-axis."
                )

                def toggle_spectrum_marker():
                    viewer.show_spectrum_marker = spectrum_marker_cb.value
                    if viewer.df is not None:
                        viewer.update_plot()

                spectrum_marker_cb = (
                    ui.checkbox("Marker", value=True, on_change=toggle_spectrum_marker)
                    .props("dense")
                    .classes("text-cyan-400")
                )
                ui.tooltip("Show/hide the spectrum position marker (crosshair) on the 2D peakmap.")

            # Breadcrumb trail and coordinate display row
            with ui.row().classes("w-full items-center justify-between mb-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("navigation", size="xs").classes("text-gray-400")
                    viewer.breadcrumb_label = ui.label("Full view").classes("text-xs text-gray-400")
                viewer.coord_label = ui.label("RT: --  m/z: --").classes("text-xs text-cyan-400 font-mono")

            ui.label("Scroll to zoom, drag to select region, Shift+drag to measure, double-click to reset").classes(
                "text-xs text-gray-500 mb-1"
            )

            # Peak map with mouse interaction and minimap
            with ui.row().classes("w-full items-start gap-2"):
                # Peak map image with mouse handlers
                with ui.column().classes("flex-none"):
                    # Drag state for selection rectangle, measurement tool, and panning
                    drag_state = {
                        "dragging": False,
                        "measuring": False,
                        "panning": False,
                        "start_x": 0,
                        "start_y": 0,
                        # Initial view bounds for panning (stored on mousedown)
                        "pan_rt_min": 0,
                        "pan_rt_max": 0,
                        "pan_mz_min": 0,
                        "pan_mz_max": 0,
                        # Throttling for panning (timestamp of last render)
                        "last_pan_render": 0.0,
                    }

                    def pixel_to_data(px: float, py: float) -> tuple[float, float]:
                        """Convert pixel coordinates to (rt, mz) respecting swap_axes."""
                        plot_x = px - viewer.margin_left
                        plot_y = py - viewer.margin_top
                        plot_x = max(0, min(viewer.plot_width, plot_x))
                        plot_y = max(0, min(viewer.plot_height, plot_y))

                        x_frac = plot_x / viewer.plot_width
                        y_frac = plot_y / viewer.plot_height
                        rt_range = viewer.view_rt_max - viewer.view_rt_min
                        mz_range = viewer.view_mz_max - viewer.view_mz_min

                        if viewer.swap_axes:
                            # swap_axes=True: m/z on x-axis, RT on y-axis (inverted)
                            mz = viewer.view_mz_min + x_frac * mz_range
                            rt = viewer.view_rt_max - y_frac * rt_range
                        else:
                            # swap_axes=False: RT on x-axis, m/z on y-axis (inverted)
                            rt = viewer.view_rt_min + x_frac * rt_range
                            mz = viewer.view_mz_max - y_frac * mz_range
                        return rt, mz

                    def on_peakmap_mouse(e: MouseEventArguments):
                        """Handle mouse events on the peakmap for drag-to-zoom, measurement, and panning."""
                        if e.type == "mousedown":
                            # Only start drag if within plot area
                            plot_x = e.image_x - viewer.margin_left
                            plot_y = e.image_y - viewer.margin_top
                            if 0 <= plot_x <= viewer.plot_width and 0 <= plot_y <= viewer.plot_height:
                                drag_state["dragging"] = True
                                drag_state["measuring"] = e.shift  # Shift+drag = measurement mode
                                # Ctrl+drag = panning mode, but only if zoomed in (not at full view)
                                is_zoomed_in = (
                                    viewer.view_rt_min > viewer.rt_min + 0.01
                                    or viewer.view_rt_max < viewer.rt_max - 0.01
                                    or viewer.view_mz_min > viewer.mz_min + 0.01
                                    or viewer.view_mz_max < viewer.mz_max - 0.01
                                )
                                drag_state["panning"] = e.ctrl and is_zoomed_in
                                drag_state["start_x"] = e.image_x
                                drag_state["start_y"] = e.image_y
                                # Store initial view bounds for panning
                                if e.ctrl and is_zoomed_in:
                                    drag_state["pan_rt_min"] = viewer.view_rt_min
                                    drag_state["pan_rt_max"] = viewer.view_rt_max
                                    drag_state["pan_mz_min"] = viewer.view_mz_min
                                    drag_state["pan_mz_max"] = viewer.view_mz_max

                        elif e.type == "mousemove":
                            # Update coordinate display
                            try:
                                viewer.update_coord_display(e.image_x, e.image_y)
                            except Exception:
                                pass

                            # Draw overlay if dragging
                            if drag_state["dragging"]:
                                if drag_state["measuring"]:
                                    # Measurement mode: draw line with delta values
                                    x1, y1 = drag_state["start_x"], drag_state["start_y"]
                                    x2, y2 = e.image_x, e.image_y

                                    # Calculate delta RT and m/z
                                    rt1, mz1 = pixel_to_data(x1, y1)
                                    rt2, mz2 = pixel_to_data(x2, y2)
                                    delta_rt = abs(rt2 - rt1)
                                    delta_mz = abs(mz2 - mz1)

                                    # Format delta values
                                    if viewer.rt_in_minutes:
                                        rt_text = f"ΔRT: {delta_rt / 60.0:.3f} min"
                                    else:
                                        rt_text = f"ΔRT: {delta_rt:.2f} s"
                                    mz_text = f"Δm/z: {delta_mz:.4f}"

                                    # Position label near the midpoint of the line
                                    mid_x = (x1 + x2) / 2
                                    mid_y = (y1 + y2) / 2
                                    label_offset = 15

                                    viewer.image_element.content = f"""
                                        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                                              stroke="yellow" stroke-width="2"/>
                                        <circle cx="{x1}" cy="{y1}" r="4" fill="yellow"/>
                                        <circle cx="{x2}" cy="{y2}" r="4" fill="yellow"/>
                                        <rect x="{mid_x - 5}" y="{mid_y + label_offset - 2}"
                                              width="130" height="36" rx="3"
                                              fill="rgba(0, 0, 0, 0.8)" stroke="yellow" stroke-width="1"/>
                                        <text x="{mid_x}" y="{mid_y + label_offset + 12}"
                                              fill="yellow" font-size="12" font-family="monospace">{rt_text}</text>
                                        <text x="{mid_x}" y="{mid_y + label_offset + 28}"
                                              fill="yellow" font-size="12" font-family="monospace">{mz_text}</text>
                                    """
                                elif drag_state["panning"]:
                                    # Panning mode: calculate pixel delta and shift view
                                    delta_px = e.image_x - drag_state["start_x"]
                                    delta_py = e.image_y - drag_state["start_y"]

                                    # Convert pixel delta to data delta
                                    rt_range = drag_state["pan_rt_max"] - drag_state["pan_rt_min"]
                                    mz_range = drag_state["pan_mz_max"] - drag_state["pan_mz_min"]

                                    if viewer.swap_axes:
                                        # swap_axes=True: m/z on x-axis, RT on y-axis
                                        delta_mz = -(delta_px / viewer.plot_width) * mz_range
                                        delta_rt = (delta_py / viewer.plot_height) * rt_range  # Y inverted
                                    else:
                                        # swap_axes=False: RT on x-axis, m/z on y-axis
                                        delta_rt = -(delta_px / viewer.plot_width) * rt_range
                                        delta_mz = (delta_py / viewer.plot_height) * mz_range  # Y inverted

                                    # Calculate new bounds with clamping to data limits
                                    new_rt_min = drag_state["pan_rt_min"] + delta_rt
                                    new_rt_max = drag_state["pan_rt_max"] + delta_rt
                                    new_mz_min = drag_state["pan_mz_min"] + delta_mz
                                    new_mz_max = drag_state["pan_mz_max"] + delta_mz

                                    # Clamp to data limits (don't pan beyond data)
                                    if new_rt_min < viewer.rt_min:
                                        shift = viewer.rt_min - new_rt_min
                                        new_rt_min += shift
                                        new_rt_max += shift
                                    if new_rt_max > viewer.rt_max:
                                        shift = new_rt_max - viewer.rt_max
                                        new_rt_min -= shift
                                        new_rt_max -= shift
                                    if new_mz_min < viewer.mz_min:
                                        shift = viewer.mz_min - new_mz_min
                                        new_mz_min += shift
                                        new_mz_max += shift
                                    if new_mz_max > viewer.mz_max:
                                        shift = new_mz_max - viewer.mz_max
                                        new_mz_min -= shift
                                        new_mz_max -= shift

                                    # Update view bounds
                                    viewer.view_rt_min = new_rt_min
                                    viewer.view_rt_max = new_rt_max
                                    viewer.view_mz_min = new_mz_min
                                    viewer.view_mz_max = new_mz_max

                                    # Throttle rendering: only render if 50ms+ since last render
                                    current_time = time.time()
                                    if current_time - drag_state["last_pan_render"] >= 0.05:
                                        drag_state["last_pan_render"] = current_time
                                        # Update plot (lightweight/fast mode)
                                        viewer.update_plot(lightweight=True)

                                    # Show panning cursor indicator (always update)
                                    cx, cy = e.image_x, e.image_y
                                    viewer.image_element.content = f"""
                                        <circle cx="{cx}" cy="{cy}" r="8" fill="none"
                                                stroke="orange" stroke-width="2"/>
                                        <line x1="{cx - 12}" y1="{cy}" x2="{cx + 12}" y2="{cy}"
                                              stroke="orange" stroke-width="2"/>
                                        <line x1="{cx}" y1="{cy - 12}" x2="{cx}" y2="{cy + 12}"
                                              stroke="orange" stroke-width="2"/>
                                    """
                                else:
                                    # Zoom mode: draw selection rectangle
                                    x = min(drag_state["start_x"], e.image_x)
                                    y = min(drag_state["start_y"], e.image_y)
                                    w = abs(e.image_x - drag_state["start_x"])
                                    h = abs(e.image_y - drag_state["start_y"])
                                    viewer.image_element.content = f"""
                                        <rect x="{x}" y="{y}" width="{w}" height="{h}"
                                              fill="rgba(0, 200, 255, 0.15)"
                                              stroke="cyan" stroke-width="2" stroke-dasharray="5,5"/>
                                    """

                        elif e.type == "mouseup":
                            # Clear overlay
                            viewer.image_element.content = ""

                            if drag_state["dragging"]:
                                was_measuring = drag_state["measuring"]
                                was_panning = drag_state["panning"]
                                drag_state["dragging"] = False
                                drag_state["measuring"] = False
                                drag_state["panning"] = False

                                # Skip zoom if we were measuring or panning
                                if was_measuring or was_panning:
                                    # Do full UI update after panning finishes
                                    if was_panning:
                                        # Full resolution render (not lightweight)
                                        viewer.update_plot(lightweight=False)
                                    return

                                end_x = e.image_x
                                end_y = e.image_y

                                # Calculate selection in plot coordinates
                                start_plot_x = drag_state["start_x"] - viewer.margin_left
                                start_plot_y = drag_state["start_y"] - viewer.margin_top
                                end_plot_x = end_x - viewer.margin_left
                                end_plot_y = end_y - viewer.margin_top

                                # Ensure within bounds
                                start_plot_x = max(0, min(viewer.plot_width, start_plot_x))
                                start_plot_y = max(0, min(viewer.plot_height, start_plot_y))
                                end_plot_x = max(0, min(viewer.plot_width, end_plot_x))
                                end_plot_y = max(0, min(viewer.plot_height, end_plot_y))

                                # Only zoom if dragged a meaningful distance (>10 pixels)
                                dx = abs(end_plot_x - start_plot_x)
                                dy = abs(end_plot_y - start_plot_y)

                                if dx > 10 and dy > 10:
                                    # Save current state to zoom history before changing
                                    viewer.push_zoom_history()

                                    # Convert to data coordinates
                                    rt_range = viewer.view_rt_max - viewer.view_rt_min
                                    mz_range = viewer.view_mz_max - viewer.view_mz_min

                                    x1_frac = min(start_plot_x, end_plot_x) / viewer.plot_width
                                    x2_frac = max(start_plot_x, end_plot_x) / viewer.plot_width
                                    y1_frac = min(start_plot_y, end_plot_y) / viewer.plot_height
                                    y2_frac = max(start_plot_y, end_plot_y) / viewer.plot_height

                                    if viewer.swap_axes:
                                        # swap_axes=True: m/z on x-axis, RT on y-axis
                                        new_mz_min = viewer.view_mz_min + x1_frac * mz_range
                                        new_mz_max = viewer.view_mz_min + x2_frac * mz_range
                                        # Y is inverted (top = high RT)
                                        new_rt_max = viewer.view_rt_max - y1_frac * rt_range
                                        new_rt_min = viewer.view_rt_max - y2_frac * rt_range
                                    else:
                                        # swap_axes=False: RT on x-axis, m/z on y-axis
                                        new_rt_min = viewer.view_rt_min + x1_frac * rt_range
                                        new_rt_max = viewer.view_rt_min + x2_frac * rt_range
                                        # Y is inverted (top = high m/z)
                                        new_mz_max = viewer.view_mz_max - y1_frac * mz_range
                                        new_mz_min = viewer.view_mz_max - y2_frac * mz_range

                                    viewer.view_rt_min = new_rt_min
                                    viewer.view_rt_max = new_rt_max
                                    viewer.view_mz_min = new_mz_min
                                    viewer.view_mz_max = new_mz_max

                                    # Save new state to zoom history
                                    viewer.push_zoom_history()
                                    viewer.update_plot()

                    viewer.image_element = (
                        ui.interactive_image(
                            on_mouse=on_peakmap_mouse,
                            events=["mousedown", "mousemove", "mouseup"],
                            cross=False,
                        )
                        .classes("w-full")
                        .style(
                            f"width: {viewer.canvas_width}px; height: {viewer.canvas_height}px; "
                            f"background: transparent; cursor: crosshair;"
                        )
                    )

                    # Mouse wheel zoom handler (separate from interactive_image events)
                    def on_wheel(e):
                        try:
                            # Get mouse position relative to image
                            offset_x = e.args.get("offsetX", 0)
                            offset_y = e.args.get("offsetY", 0)
                            delta_y = e.args.get("deltaY", 0)

                            # Convert to plot area coordinates (account for margins)
                            plot_x = offset_x - viewer.margin_left
                            plot_y = offset_y - viewer.margin_top

                            # Check if within plot area
                            if 0 <= plot_x <= viewer.plot_width and 0 <= plot_y <= viewer.plot_height:
                                x_frac = plot_x / viewer.plot_width
                                y_frac = plot_y / viewer.plot_height
                                zoom_in = delta_y < 0  # Scroll up = zoom in
                                viewer.zoom_at_point(x_frac, y_frac, zoom_in)
                        except Exception:
                            pass

                    viewer.image_element.on("wheel.prevent", on_wheel)

                    def on_dblclick(e):
                        viewer.reset_view()

                    def on_mouseleave(e):
                        was_panning = drag_state["panning"]
                        drag_state["dragging"] = False
                        drag_state["measuring"] = False
                        drag_state["panning"] = False
                        viewer.image_element.content = ""  # Clear any overlay
                        if viewer.coord_label:
                            viewer.coord_label.set_text("RT: --  m/z: --")
                        # If panning was active, do full resolution render
                        if was_panning:
                            viewer.update_plot(lightweight=False)

                    viewer.image_element.on("dblclick", on_dblclick)
                    viewer.image_element.on("mouseleave", on_mouseleave)

                    # Handle ctrl key release during panning (triggers full render)
                    def on_keyup(e):
                        key = e.args.get("key", "")
                        if key == "Control" and drag_state["panning"]:
                            # Ctrl released during panning - end panning and do full render
                            drag_state["dragging"] = False
                            drag_state["measuring"] = False
                            drag_state["panning"] = False
                            viewer.image_element.content = ""
                            viewer.update_plot(lightweight=False)

                    viewer.image_element.on("keyup", on_keyup)

                    # Also handle document-level mouseup for when mouse is released outside image
                    async def on_document_mouseup():
                        if drag_state["panning"]:
                            drag_state["dragging"] = False
                            drag_state["measuring"] = False
                            drag_state["panning"] = False
                            viewer.image_element.content = ""
                            viewer.update_plot(lightweight=False)

                    ui.on("mouseup", on_document_mouseup)

                    # 3D View Container (below 2D peakmap, hidden by default)
                    viewer.scene_3d_container = ui.column().classes("w-full mt-1")
                    viewer.scene_3d_container.set_visibility(False)
                    with viewer.scene_3d_container:
                        viewer.view_3d_status = ui.label("").classes("text-xs text-yellow-400")
                        # Create empty plotly figure for 3D view
                        empty_fig = go.Figure()
                        empty_fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#888"},
                            width=viewer.canvas_width,
                            height=500,
                            autosize=False,
                            margin={"l": 0, "r": 0, "t": 0, "b": 0},
                        )
                        # Wrap in explicit div for sizing
                        with ui.element("div").style(f"width: {viewer.canvas_width}px; height: 500px;"):
                            viewer.plot_3d = ui.plotly(empty_fig).classes("w-full h-full")

                # Minimap panel (to the right of peak map)
                with ui.column().classes("flex-none"):
                    ui.label("Overview").classes("text-xs text-gray-400 mb-1")
                    viewer.minimap_image = ui.image().style(
                        f"width: {viewer.minimap_width}px; height: {viewer.minimap_height}px; "
                        f"background: transparent; cursor: pointer; border: 1px solid #888;"
                    )

                    # Minimap click handler
                    def on_minimap_click(e):
                        try:
                            offset_x = e.args.get("offsetX", 0)
                            offset_y = e.args.get("offsetY", 0)
                            x_frac = offset_x / viewer.minimap_width
                            y_frac = offset_y / viewer.minimap_height
                            viewer.minimap_click_to_view(x_frac, y_frac)
                        except Exception:
                            pass

                    viewer.minimap_image.on("click", on_minimap_click)

                    # Back and 3D View buttons in same row
                    def go_back():
                        if len(viewer.zoom_history) > 1:
                            viewer.go_to_zoom_history(len(viewer.zoom_history) - 2)

                    def toggle_3d_view():
                        viewer.show_3d_view = not viewer.show_3d_view
                        if viewer.scene_3d_container:
                            viewer.scene_3d_container.set_visibility(viewer.show_3d_view)
                        if viewer.show_3d_view and viewer.df is not None:
                            viewer.update_3d_view()
                        # Update button appearance
                        if viewer.show_3d_view:
                            view_3d_btn.props("color=purple")
                        else:
                            view_3d_btn.props("color=grey")

                    with ui.row().classes("mt-1 gap-1"):
                        ui.button("← Back", on_click=go_back).props("dense size=sm color=grey").tooltip(
                            "Go to previous view"
                        )
                        view_3d_btn = (
                            ui.button("3D", on_click=toggle_3d_view)
                            .props("dense size=sm color=grey")
                            .tooltip("Toggle 3D peak view")
                        )

        # Ion Mobility Peak Map Panel (for TIMS, drift tube data)
        viewer.im_expansion = ui.expansion(
            "Ion Mobility Map", icon="blur_on", value=False
        ).classes("w-full max-w-[1700px]")
        viewer.panel_elements["im_peakmap"] = viewer.im_expansion
        viewer.im_expansion.move(target_container=viewer.panels_container)
        with viewer.im_expansion:
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                viewer.im_info_label = ui.label("No ion mobility data").classes("text-sm text-gray-400")
                ui.element("div").classes("flex-grow")

                # Checkbox to link spectrum m/z zoom to IM peakmap
                def on_link_spectrum_change(e):
                    viewer.link_spectrum_mz_to_im = e.value
                    if e.value and viewer.spectrum_zoom_range:
                        # Apply current spectrum zoom to IM view
                        xmin, xmax = viewer.spectrum_zoom_range
                        viewer.view_mz_min = max(viewer.mz_min, xmin)
                        viewer.view_mz_max = min(viewer.mz_max, xmax)
                        viewer.update_im_plot()
                        if viewer.im_range_label:
                            viewer.im_range_label.set_text(
                                f"m/z: {viewer.view_mz_min:.2f} - {viewer.view_mz_max:.2f} | "
                                f"IM: {viewer.view_im_min:.3f} - {viewer.view_im_max:.3f} {viewer.im_unit}"
                            )

                viewer.link_spectrum_mz_checkbox = ui.checkbox(
                    "Link to Spectrum m/z", value=viewer.link_spectrum_mz_to_im, on_change=on_link_spectrum_change
                ).props("dense").tooltip("Sync m/z range with spectrum plot zoom")

                # Checkbox to show/hide mobilogram
                def on_mobilogram_change(e):
                    viewer.show_mobilogram = e.value
                    # Update image width when mobilogram toggle changes
                    mobilogram_space = viewer.mobilogram_plot_width + 20 if viewer.show_mobilogram else 0
                    new_width = viewer.canvas_width + mobilogram_space
                    if viewer.im_image_element:
                        viewer.im_image_element.style(
                            f"width: {new_width}px; height: {viewer.canvas_height}px; "
                            f"background: transparent; cursor: crosshair;"
                        )
                    viewer.update_im_plot()

                ui.checkbox(
                    "Show Mobilogram", value=viewer.show_mobilogram, on_change=on_mobilogram_change
                ).props("dense").tooltip("Show summed intensity profile vs ion mobility")

                def reset_im_view_click():
                    viewer.reset_im_view()

                ui.button("Reset View", icon="home", on_click=reset_im_view_click).props(
                    "dense outline size=sm"
                ).tooltip("Reset to full IM range")

            with ui.row().classes("w-full"):
                # IM range labels
                viewer.im_range_label = ui.label("IM: --").classes("text-xs text-gray-500")

            # IM peak map image (similar to main peakmap) - wider to accommodate mobilogram
            mobilogram_space = viewer.mobilogram_plot_width + 20 if viewer.show_mobilogram else 0
            im_canvas_width = viewer.canvas_width + mobilogram_space
            with ui.column().classes("w-full items-center"):
                viewer.im_image_element = (
                    ui.interactive_image()
                    .style(
                        f"width: {im_canvas_width}px; height: {viewer.canvas_height}px; "
                        f"background: transparent; cursor: crosshair;"
                    )
                    .classes("border border-gray-600")
                )

                # IM peakmap interaction handlers
                im_drag_state = {"dragging": False, "start_x": 0, "start_y": 0}

                def on_im_mousedown(e):
                    try:
                        offset_x = e.args.get("offsetX", 0)
                        offset_y = e.args.get("offsetY", 0)
                        im_drag_state["dragging"] = True
                        im_drag_state["start_x"] = offset_x
                        im_drag_state["start_y"] = offset_y
                    except Exception:
                        pass

                def on_im_mouseup(e):
                    try:
                        if not im_drag_state["dragging"]:
                            return
                        im_drag_state["dragging"] = False

                        offset_x = e.args.get("offsetX", 0)
                        offset_y = e.args.get("offsetY", 0)
                        start_x = im_drag_state["start_x"]
                        start_y = im_drag_state["start_y"]

                        # Check if it's a significant drag (zoom selection)
                        dx = abs(offset_x - start_x)
                        dy = abs(offset_y - start_y)
                        if dx < 5 and dy < 5:
                            viewer.im_image_element.content = ""
                            return

                        # Convert to data coordinates (X: m/z, Y: IM)
                        x1_frac = (min(start_x, offset_x) - viewer.margin_left) / viewer.plot_width
                        x2_frac = (max(start_x, offset_x) - viewer.margin_left) / viewer.plot_width
                        y1_frac = (min(start_y, offset_y) - viewer.margin_top) / viewer.plot_height
                        y2_frac = (max(start_y, offset_y) - viewer.margin_top) / viewer.plot_height

                        x1_frac = max(0, min(1, x1_frac))
                        x2_frac = max(0, min(1, x2_frac))
                        y1_frac = max(0, min(1, y1_frac))
                        y2_frac = max(0, min(1, y2_frac))

                        # Calculate new view bounds
                        mz_range = viewer.view_mz_max - viewer.view_mz_min
                        im_range = viewer.view_im_max - viewer.view_im_min

                        new_mz_min = viewer.view_mz_min + x1_frac * mz_range
                        new_mz_max = viewer.view_mz_min + x2_frac * mz_range
                        new_im_max = viewer.view_im_max - y1_frac * im_range  # Y is inverted
                        new_im_min = viewer.view_im_max - y2_frac * im_range

                        # Apply new bounds
                        viewer.view_mz_min = new_mz_min
                        viewer.view_mz_max = new_mz_max
                        viewer.view_im_min = new_im_min
                        viewer.view_im_max = new_im_max

                        viewer.im_image_element.content = ""
                        viewer.update_im_plot()
                        if viewer.im_range_label:
                            viewer.im_range_label.set_text(
                                f"IM: {viewer.view_im_min:.3f} - {viewer.view_im_max:.3f} {viewer.im_unit}"
                            )

                    except Exception:
                        pass

                def on_im_mousemove(e):
                    try:
                        offset_x = e.args.get("offsetX", 0)
                        offset_y = e.args.get("offsetY", 0)

                        # Draw selection rectangle while dragging
                        if im_drag_state["dragging"]:
                            start_x = im_drag_state["start_x"]
                            start_y = im_drag_state["start_y"]
                            rect_x = min(start_x, offset_x)
                            rect_y = min(start_y, offset_y)
                            rect_w = abs(offset_x - start_x)
                            rect_h = abs(offset_y - start_y)
                            viewer.im_image_element.content = (
                                f'<rect x="{rect_x}" y="{rect_y}" width="{rect_w}" height="{rect_h}" '
                                f'fill="rgba(255,255,0,0.15)" stroke="rgba(255,255,0,0.5)" stroke-width="1"/>'
                            )
                    except Exception:
                        pass

                def on_im_dblclick(e):
                    viewer.reset_im_view()

                def on_im_wheel(e):
                    try:
                        offset_x = e.args.get("offsetX", 0)
                        offset_y = e.args.get("offsetY", 0)
                        delta_y = e.args.get("deltaY", 0)

                        x_in_plot = viewer.margin_left <= offset_x <= viewer.margin_left + viewer.plot_width
                        y_in_plot = viewer.margin_top <= offset_y <= viewer.margin_top + viewer.plot_height

                        if x_in_plot and y_in_plot:
                            x_frac = (offset_x - viewer.margin_left) / viewer.plot_width
                            y_frac = (offset_y - viewer.margin_top) / viewer.plot_height
                            zoom_in = delta_y < 0

                            # Zoom at cursor position
                            factor = 0.8 if zoom_in else 1.25
                            mz_range = viewer.view_mz_max - viewer.view_mz_min
                            im_range = viewer.view_im_max - viewer.view_im_min

                            cursor_mz = viewer.view_mz_min + x_frac * mz_range
                            cursor_im = viewer.view_im_max - y_frac * im_range

                            new_mz_range = mz_range * factor
                            new_im_range = im_range * factor

                            viewer.view_mz_min = cursor_mz - x_frac * new_mz_range
                            viewer.view_mz_max = cursor_mz + (1 - x_frac) * new_mz_range
                            viewer.view_im_min = cursor_im - (1 - y_frac) * new_im_range
                            viewer.view_im_max = cursor_im + y_frac * new_im_range

                            # Clamp to data bounds
                            viewer.view_mz_min = max(viewer.mz_min, viewer.view_mz_min)
                            viewer.view_mz_max = min(viewer.mz_max, viewer.view_mz_max)
                            viewer.view_im_min = max(viewer.im_min, viewer.view_im_min)
                            viewer.view_im_max = min(viewer.im_max, viewer.view_im_max)

                            viewer.update_im_plot()
                            if viewer.im_range_label:
                                viewer.im_range_label.set_text(
                                    f"IM: {viewer.view_im_min:.3f} - {viewer.view_im_max:.3f} {viewer.im_unit}"
                                )
                    except Exception:
                        pass

                viewer.im_image_element.on("mousedown", on_im_mousedown)
                viewer.im_image_element.on("mouseup", on_im_mouseup)
                viewer.im_image_element.on("mousemove", on_im_mousemove)
                viewer.im_image_element.on("dblclick", on_im_dblclick)
                viewer.im_image_element.on("wheel.prevent", on_im_wheel)

        # 1D Spectrum Browser (collapsible panel, starts collapsed until file is loaded)
        viewer.spectrum_expansion = ui.expansion("1D Spectrum", icon="show_chart", value=False).classes(
            "w-full max-w-[1700px]"
        )
        viewer.panel_elements["spectrum"] = viewer.spectrum_expansion
        viewer.spectrum_expansion.move(target_container=viewer.panels_container)
        with viewer.spectrum_expansion:
            with ui.column().classes("w-full items-center"):
                # Navigation and info row
                with ui.row().classes("w-full items-center gap-2 mb-1").style(f"max-width: {viewer.canvas_width}px;"):
                    ui.button("|<", on_click=lambda: viewer.show_spectrum_in_browser(0)).props("dense size=sm").tooltip(
                        "First"
                    )
                    ui.button("< MS1", on_click=lambda: viewer.navigate_spectrum_by_ms_level(-1, 1)).props(
                        "dense size=sm color=cyan"
                    ).tooltip("Prev MS1")
                    ui.button("<", on_click=lambda: viewer.navigate_spectrum(-1)).props("dense size=sm").tooltip("Prev")

                    viewer.spectrum_nav_label = ui.label("No spectrum").classes("mx-2 text-gray-400 text-sm")

                    ui.button(">", on_click=lambda: viewer.navigate_spectrum(1)).props("dense size=sm").tooltip("Next")
                    ui.button("MS1 >", on_click=lambda: viewer.navigate_spectrum_by_ms_level(1, 1)).props(
                        "dense size=sm color=cyan"
                    ).tooltip("Next MS1")
                    ui.button(
                        ">|",
                        on_click=lambda: viewer.show_spectrum_in_browser(viewer.exp.size() - 1 if viewer.exp else 0),
                    ).props("dense size=sm").tooltip("Last")

                    ui.label("|").classes("mx-1 text-gray-600")
                    ui.button("< MS2", on_click=lambda: viewer.navigate_spectrum_by_ms_level(-1, 2)).props(
                        "dense size=sm color=orange"
                    ).tooltip("Prev MS2")
                    ui.button("MS2 >", on_click=lambda: viewer.navigate_spectrum_by_ms_level(1, 2)).props(
                        "dense size=sm color=orange"
                    ).tooltip("Next MS2")

                    ui.element("div").classes("flex-grow")  # Spacer

                    # Intensity display toggle
                    ui.label("Intensity:").classes("text-xs text-gray-400")
                    ui.toggle(
                        ["%", "abs"],
                        value="%",
                        on_change=lambda e: (
                            setattr(viewer, "spectrum_intensity_percent", e.value == "%"),
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)
                            if viewer.selected_spectrum_idx is not None
                            else None,
                        ),
                    ).props("dense size=sm color=grey").tooltip("Toggle between relative (%) and absolute intensity")

                    # Auto-scale checkbox
                    ui.checkbox(
                        "Auto Y",
                        value=False,
                        on_change=lambda e: (
                            setattr(viewer, "spectrum_auto_scale", e.value),
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)
                            if viewer.selected_spectrum_idx is not None
                            else None,
                        ),
                    ).props("dense size=sm color=grey").classes("text-xs").tooltip(
                        "Auto-scale Y-axis to fit visible peaks (highest peak at 95%)"
                    )

                    ui.label("|").classes("mx-1 text-gray-600")

                    # Measurement mode toggle (needs annotation_btn reference, defined below)
                    annotation_btn = None  # Forward declaration

                    def toggle_measure_mode():
                        viewer.spectrum_measure_mode = not viewer.spectrum_measure_mode
                        viewer.spectrum_measure_start = None  # Reset any pending measurement
                        viewer.spectrum_hover_peak = None  # Clear hover highlight
                        # Note: Don't clear zoom_range here - preserve zoom when toggling modes
                        # Disable annotation mode when measure mode is active
                        if viewer.spectrum_measure_mode and viewer.peak_annotation_mode:
                            viewer.peak_annotation_mode = False
                            if annotation_btn:
                                annotation_btn.props("color=grey")
                        measure_btn.props(f"color={'yellow' if viewer.spectrum_measure_mode else 'grey'}")
                        if viewer.spectrum_measure_mode:
                            ui.notify("Measure mode ON - click two peaks to measure Δm/z", type="info")
                        else:
                            ui.notify("Measure mode OFF", type="info")
                        # Refresh display to clear any preview
                        if viewer.selected_spectrum_idx is not None:
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                    measure_btn = ui.button("📏 Measure", on_click=toggle_measure_mode).props(
                        "dense size=sm color=grey"
                    ).tooltip("Toggle measurement mode - click two peaks to measure Δm/z")

                    ui.button(
                        "Clear Δ",
                        on_click=lambda: viewer.clear_spectrum_measurement(),
                    ).props("dense size=sm color=grey").tooltip("Clear measurements for this spectrum")

                    ui.label("|").classes("mx-1 text-gray-600")

                    # Annotation mode toggle
                    def toggle_annotation_mode():
                        viewer.peak_annotation_mode = not viewer.peak_annotation_mode
                        # Disable measure mode when annotation mode is active
                        if viewer.peak_annotation_mode and viewer.spectrum_measure_mode:
                            viewer.spectrum_measure_mode = False
                            viewer.spectrum_measure_start = None
                            measure_btn.props("color=grey")
                        annotation_btn.props(f"color={'green' if viewer.peak_annotation_mode else 'grey'}")
                        if viewer.peak_annotation_mode:
                            ui.notify("Label mode ON - click peaks to add labels", type="info")
                        else:
                            ui.notify("Label mode OFF", type="info")

                    annotation_btn = ui.button("🏷️ Label", on_click=toggle_annotation_mode).props(
                        "dense size=sm color=grey"
                    ).tooltip("Toggle label mode - click peaks to add/edit custom labels")

                    # Show m/z labels on all peaks toggle
                    def toggle_mz_labels(e):
                        viewer.show_mz_labels = e.value
                        if viewer.selected_spectrum_idx is not None:
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                    ui.checkbox("m/z", value=False, on_change=toggle_mz_labels).props(
                        "dense size=sm color=grey"
                    ).classes("text-xs").tooltip("Show m/z values on top peaks")

                    def clear_annotations():
                        if viewer.selected_spectrum_idx is not None:
                            viewer.clear_peak_annotations(viewer.selected_spectrum_idx)
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)
                            ui.notify("Annotations cleared", type="info")

                    ui.button("Clear 🏷️", on_click=clear_annotations).props(
                        "dense size=sm color=grey"
                    ).tooltip("Clear all labels for this spectrum")

                    ui.label("|").classes("mx-1 text-gray-600")
                    viewer.spectrum_browser_info = ui.label("Click TIC to select spectrum").classes(
                        "text-xs text-gray-500"
                    )

                # Spectrum plot
                viewer.spectrum_browser_plot = ui.plotly(go.Figure()).classes("w-full")

                # Spectrum measurement click handler
                def on_spectrum_click(e):
                    """Handle clicks on spectrum for peak measurement and selection."""
                    try:
                        if not e.args:
                            return

                        # Get clicked point
                        points = e.args.get("points", [])
                        if not points:
                            return

                        clicked_mz = points[0].get("x")
                        clicked_y = points[0].get("y")
                        if clicked_mz is None:
                            return

                        # Get current spectrum data
                        if viewer.selected_spectrum_idx is None or viewer.exp is None:
                            return

                        spec = viewer.exp[viewer.selected_spectrum_idx]
                        mz_array, int_array = spec.get_peaks()

                        if len(mz_array) == 0:
                            return

                        # Check if clicking on an existing measurement line (for selection)
                        if clicked_y is not None:
                            measurement_idx = viewer.find_measurement_at_position(clicked_mz, clicked_y)
                            if measurement_idx is not None:
                                # Toggle selection
                                if viewer.spectrum_selected_measurement_idx == measurement_idx:
                                    viewer.spectrum_selected_measurement_idx = None
                                    ui.notify("Measurement deselected", type="info")
                                else:
                                    viewer.spectrum_selected_measurement_idx = measurement_idx
                                    ui.notify("Measurement selected - press Delete to remove", type="info")
                                viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)
                                return

                        # Clear selection if clicking elsewhere (not on a measurement)
                        if viewer.spectrum_selected_measurement_idx is not None:
                            viewer.spectrum_selected_measurement_idx = None
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                        # Handle annotation mode - click to add/edit peak labels
                        if viewer.peak_annotation_mode:
                            # Snap to nearest peak
                            snapped = viewer.snap_to_peak(clicked_mz, mz_array, int_array, clicked_y)
                            if snapped is None:
                                ui.notify("No peak found near click position", type="warning")
                                return

                            snapped_mz, snapped_int = snapped
                            spectrum_idx = viewer.selected_spectrum_idx

                            # Check if annotation already exists at this m/z
                            existing_label = ""
                            if spectrum_idx in viewer.peak_annotations:
                                for ann in viewer.peak_annotations[spectrum_idx]:
                                    if abs(ann["mz"] - snapped_mz) < 0.01:
                                        existing_label = ann["label"]
                                        break

                            # Open dialog to edit annotation
                            with ui.dialog() as dialog, ui.card().classes("min-w-[300px]"):
                                ui.label("Peak Annotation").classes("text-lg font-bold")
                                ui.label(f"m/z: {snapped_mz:.4f}").classes("text-sm text-gray-400")

                                label_input = ui.input(
                                    "Label",
                                    value=existing_label,
                                    placeholder=f"{snapped_mz:.4f}",
                                ).classes("w-full")

                                with ui.row().classes("w-full justify-end gap-2 mt-4"):

                                    def delete_annotation(mz=snapped_mz, idx=spectrum_idx):
                                        viewer.remove_peak_annotation(idx, mz)
                                        dialog.close()
                                        viewer.show_spectrum_in_browser(idx)
                                        ui.notify("Annotation removed", type="info")

                                    def save_annotation(mz=snapped_mz, intensity=snapped_int, idx=spectrum_idx):
                                        label = label_input.value.strip()
                                        viewer.add_or_edit_peak_annotation(idx, mz, intensity, label if label else None)
                                        dialog.close()
                                        viewer.show_spectrum_in_browser(idx)
                                        ui.notify("Annotation saved", type="positive")

                                    if existing_label:
                                        ui.button("Delete", on_click=delete_annotation, color="red").props("flat")

                                    ui.button("Cancel", on_click=dialog.close).props("flat")
                                    ui.button("Save", on_click=save_annotation, color="primary")

                            dialog.open()
                            return

                        # Only handle peak measurement when measurement mode is active
                        if not viewer.spectrum_measure_mode:
                            return

                        # Snap to nearest peak
                        snapped = viewer.snap_to_peak(clicked_mz, mz_array, int_array)
                        if snapped is None:
                            ui.notify("No peak found near click position", type="warning")
                            return

                        snapped_mz, snapped_int = snapped

                        if viewer.spectrum_measure_start is None:
                            # First click - set start point (don't refresh to preserve zoom)
                            viewer.spectrum_measure_start = (snapped_mz, snapped_int)
                            ui.notify(f"Start: m/z {snapped_mz:.4f} - click second peak", type="info")
                        else:
                            # Second click - complete measurement
                            start_mz, start_int = viewer.spectrum_measure_start
                            viewer.spectrum_measure_start = None
                            viewer.spectrum_hover_peak = None  # Clear hover state

                            # Store the measurement
                            spectrum_idx = viewer.selected_spectrum_idx
                            if spectrum_idx not in viewer.spectrum_measurements:
                                viewer.spectrum_measurements[spectrum_idx] = []
                            viewer.spectrum_measurements[spectrum_idx].append(
                                (start_mz, start_int, snapped_mz, snapped_int)
                            )

                            delta_mz = abs(snapped_mz - start_mz)
                            ui.notify(f"Δm/z = {delta_mz:.4f}", type="positive")

                            # Refresh display to show the measurement
                            viewer.show_spectrum_in_browser(spectrum_idx)

                    except Exception:
                        pass

                viewer.spectrum_browser_plot.on("plotly_click", on_spectrum_click)

                # Hover handler to highlight nearest peak (works in all modes)
                def on_spectrum_hover(e):
                    """Highlight nearest (snap) peak on hover. Shows preview line in measure mode."""
                    try:
                        if not e.args:
                            return

                        points = e.args.get("points", [])
                        if not points:
                            return

                        hovered_mz = points[0].get("x")
                        hovered_int = points[0].get("y")
                        if hovered_mz is None:
                            return

                        if viewer.selected_spectrum_idx is None or viewer.exp is None:
                            return

                        spec = viewer.exp[viewer.selected_spectrum_idx]
                        mz_array, int_array = spec.get_peaks()

                        if len(mz_array) == 0:
                            return

                        # Snap to nearest peak using 2D distance (m/z and intensity)
                        snapped = viewer.snap_to_peak(hovered_mz, mz_array, int_array, hovered_int)
                        if snapped is None:
                            return

                        # Only refresh if the hovered peak actually changed (optimization)
                        if viewer.spectrum_hover_peak == snapped:
                            return

                        viewer.spectrum_hover_peak = snapped
                        # Refresh to show preview - zoom will be preserved by the fix
                        viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                    except Exception:
                        pass

                viewer.spectrum_browser_plot.on("plotly_hover", on_spectrum_hover)

                # Clear hover highlight when mouse leaves the plot
                def on_spectrum_unhover(e):
                    """Clear hover highlight when mouse leaves data points."""
                    if viewer.spectrum_hover_peak is not None:
                        viewer.spectrum_hover_peak = None
                        if viewer.selected_spectrum_idx is not None:
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                viewer.spectrum_browser_plot.on("plotly_unhover", on_spectrum_unhover)

                # Track zoom changes to preserve during measurement workflow and auto-scale
                def on_spectrum_relayout(e):
                    """Track zoom/pan changes on spectrum plot."""
                    try:
                        if not e.args:
                            return
                        # Check for zoom box selection or autorange
                        xmin = e.args.get("xaxis.range[0]")
                        xmax = e.args.get("xaxis.range[1]")
                        if xmin is not None and xmax is not None:
                            viewer.spectrum_zoom_range = (xmin, xmax)
                            # Re-render to apply auto-scale if enabled
                            if viewer.spectrum_auto_scale and viewer.selected_spectrum_idx is not None:
                                viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)
                            # Sync m/z range to IM peakmap if linking is enabled
                            if viewer.link_spectrum_mz_to_im and viewer.has_ion_mobility:
                                viewer.view_mz_min = max(viewer.mz_min, xmin)
                                viewer.view_mz_max = min(viewer.mz_max, xmax)
                                viewer.update_im_plot()
                                if viewer.im_range_label:
                                    viewer.im_range_label.set_text(
                                        f"m/z: {viewer.view_mz_min:.2f} - {viewer.view_mz_max:.2f} | "
                                        f"IM: {viewer.view_im_min:.3f} - {viewer.view_im_max:.3f} {viewer.im_unit}"
                                    )
                        # Check for autorange (reset)
                        elif e.args.get("xaxis.autorange"):
                            viewer.spectrum_zoom_range = None
                            # Re-render to reset y-axis if auto-scale enabled
                            if viewer.spectrum_auto_scale and viewer.selected_spectrum_idx is not None:
                                viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)
                            # Reset IM m/z range if linking is enabled
                            if viewer.link_spectrum_mz_to_im and viewer.has_ion_mobility:
                                viewer.view_mz_min = viewer.mz_min
                                viewer.view_mz_max = viewer.mz_max
                                viewer.update_im_plot()
                                if viewer.im_range_label:
                                    viewer.im_range_label.set_text(
                                        f"m/z: {viewer.view_mz_min:.2f} - {viewer.view_mz_max:.2f} | "
                                        f"IM: {viewer.view_im_min:.3f} - {viewer.view_im_max:.3f} {viewer.im_unit}"
                                    )
                    except Exception:
                        pass

                viewer.spectrum_browser_plot.on("plotly_relayout", on_spectrum_relayout)

        # FAIMS Multi-CV Peak Maps (hidden by default)
        faims_container = ui.card().classes("w-full max-w-6xl mt-2 p-2")
        faims_container.set_visibility(False)
        viewer.faims_container = faims_container

        with faims_container:
            ui.label("FAIMS Compensation Voltage Peak Maps").classes("text-lg font-semibold mb-2 text-purple-300")
            ui.label("Separate peak maps for each CV value - zoom/pan is synchronized").classes(
                "text-xs text-gray-500 mb-2"
            )

            # Container for dynamic FAIMS images
            faims_row = ui.row().classes("w-full gap-1 flex-wrap justify-center")

            # Note: Actual images will be created dynamically when FAIMS data is loaded
            # We need to create a method to dynamically populate this container

            def create_faims_images():
                """Create FAIMS image elements dynamically based on detected CVs."""
                faims_row.clear()
                viewer.faims_images = {}

                if not viewer.has_faims:
                    return

                n_cvs = len(viewer.faims_cvs)
                # Calculate width for each panel (max 4 per row)
                panel_width = viewer.plot_width // max(1, min(n_cvs, 4))
                panel_height = viewer.plot_height

                with faims_row:
                    for cv in viewer.faims_cvs:
                        with ui.column().classes("flex-none"):
                            img = ui.image().style(
                                f"width: {panel_width}px; height: {panel_height}px; background: transparent;"
                            )
                            viewer.faims_images[cv] = img

            # Store the function reference for later use
            viewer._create_faims_images = create_faims_images

        # Unified Spectra Table (combines spectrum metadata + ID info)
        viewer.spectrum_table_expansion = ui.expansion("Spectra", icon="list", value=False).classes(
            "w-full max-w-[1700px]"
        )
        viewer.panel_elements["spectra_table"] = viewer.spectrum_table_expansion
        viewer.spectrum_table_expansion.move(target_container=viewer.panels_container)
        with viewer.spectrum_table_expansion:
            ui.label("Click a row to view spectrum. Identified spectra show sequence and score.").classes(
                "text-sm text-gray-400 mb-2"
            )

            # View mode and column toggles
            with ui.row().classes("w-full items-center gap-4 mb-2"):
                # View filter: All, MS2 Only, Identified Only
                view_mode = (
                    ui.toggle(
                        ["All", "MS2", "Identified"],
                        value="All",
                    )
                    .props("dense size=sm")
                    .classes("text-xs")
                )

                # Advanced columns toggle
                show_advanced = ui.checkbox("Advanced", value=False).props("dense").classes("text-xs text-gray-400")
                ui.tooltip("Show additional columns: Peaks, TIC, BPI, m/z Range")

                # Meta Values columns toggle
                show_meta_values = (
                    ui.checkbox("Meta Values", value=False).props("dense").classes("text-xs text-gray-400")
                )
                ui.tooltip("Show PeptideIdentification (pid:) and PeptideHit (hit:) meta values")

                # All Hits toggle - shows all peptide hits, not just best hit
                show_all_hits = ui.checkbox("All Hits", value=False).props("dense").classes("text-xs text-gray-400")
                ui.tooltip("Show all peptide hits for each spectrum (default: best hit only)")

            # Additional filters row
            with ui.row().classes("w-full items-end gap-2 mb-2 flex-wrap"):
                ui.label("Filter:").classes("text-xs text-gray-400")
                spec_rt_min = ui.number(label="RT Min", format="%.0f").props("dense outlined").classes("w-20")
                spec_rt_max = ui.number(label="RT Max", format="%.0f").props("dense outlined").classes("w-20")
                spec_seq_pattern = (
                    ui.input(label="Sequence", placeholder="e.g. PEPTIDE").props("dense outlined").classes("w-28")
                )
                spec_min_score = ui.number(label="Min Score", format="%.2f").props("dense outlined").classes("w-24")

                # Annotation settings
                def toggle_annotate_peaks():
                    viewer.annotate_peaks = annotate_peaks_cb.value
                    if viewer.selected_spectrum_idx is not None:
                        viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                def update_tolerance():
                    if tolerance_input.value is not None and tolerance_input.value > 0:
                        viewer.annotation_tolerance_da = tolerance_input.value
                        if viewer.selected_spectrum_idx is not None and viewer.annotate_peaks:
                            viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                annotate_peaks_cb = (
                    ui.checkbox("Annotate", value=viewer.annotate_peaks, on_change=toggle_annotate_peaks)
                    .props("dense")
                    .classes("text-blue-400")
                )

                tolerance_input = (
                    ui.number(
                        label="Tol (Da)",
                        value=viewer.annotation_tolerance_da,
                        format="%.2f",
                        on_change=update_tolerance,
                    )
                    .props("dense outlined")
                    .classes("w-20")
                )
                ui.tooltip("Mass tolerance for matching peaks to theoretical ions (Da)")

                def toggle_mirror_view():
                    viewer.mirror_annotation_view = mirror_view_cb.value
                    if viewer.selected_spectrum_idx is not None:
                        viewer.show_spectrum_in_browser(viewer.selected_spectrum_idx)

                mirror_view_cb = (
                    ui.checkbox("Mirror", value=viewer.mirror_annotation_view, on_change=toggle_mirror_view)
                    .props("dense")
                    .classes("text-blue-400")
                )
                ui.tooltip("Mirror view: flip annotated peaks downward for comparison")

            # Define all columns - basic and advanced
            basic_columns = [
                {"name": "idx", "label": "#", "field": "idx", "sortable": True, "align": "left"},
                {"name": "rt", "label": "RT (s)", "field": "rt", "sortable": True, "align": "right"},
                {"name": "ms_level", "label": "MS", "field": "ms_level", "sortable": True, "align": "center"},
                {
                    "name": "precursor_mz",
                    "label": "Prec m/z",
                    "field": "precursor_mz",
                    "sortable": True,
                    "align": "right",
                },
                {"name": "precursor_z", "label": "Z", "field": "precursor_z", "sortable": True, "align": "center"},
                {"name": "sequence", "label": "Sequence", "field": "sequence", "sortable": True, "align": "left"},
                {"name": "score", "label": "Score", "field": "score", "sortable": True, "align": "right"},
            ]

            # Rank column (shown when All Hits is enabled)
            rank_column = {
                "name": "hit_rank",
                "label": "Rank",
                "field": "hit_rank",
                "sortable": True,
                "align": "center",
            }

            advanced_columns = [
                {"name": "n_peaks", "label": "Peaks", "field": "n_peaks", "sortable": True, "align": "right"},
                {"name": "tic", "label": "TIC", "field": "tic", "sortable": True, "align": "right"},
                {"name": "bpi", "label": "BPI", "field": "bpi", "sortable": True, "align": "right"},
                {"name": "mz_range", "label": "m/z Range", "field": "mz_range", "sortable": False, "align": "center"},
            ]

            def get_meta_columns():
                """Generate columns for meta values from PeptideIdentification and PeptideHit."""
                meta_cols = []
                for key in viewer.id_meta_keys:
                    # Create readable label from key (e.g., "pid:spectrum_reference" -> "Spectrum Ref (PID)")
                    prefix, name = key.split(":", 1) if ":" in key else ("", key)
                    # Shorten common prefixes
                    label_prefix = "PID" if prefix == "pid" else "Hit" if prefix == "hit" else prefix.upper()
                    # Convert underscores to spaces and title case
                    label_name = name.replace("_", " ").title()
                    # Truncate long names
                    if len(label_name) > 15:
                        label_name = label_name[:13] + ".."
                    label = f"{label_name} ({label_prefix})"
                    meta_cols.append(
                        {
                            "name": key,
                            "label": label,
                            "field": key,
                            "sortable": True,
                            "align": "left",
                        }
                    )
                return meta_cols

            def build_columns():
                """Build column list based on current toggle states."""
                cols = basic_columns[:3]  # idx, rt, ms_level
                if show_advanced.value:
                    cols = cols + advanced_columns
                cols = cols + basic_columns[3:5]  # precursor_mz, z
                # Add rank column if All Hits is enabled (before sequence)
                if show_all_hits.value:
                    cols = cols + [rank_column]
                cols = cols + basic_columns[5:]  # sequence, score
                if show_meta_values.value:
                    cols = cols + get_meta_columns()
                return cols

            def get_filtered_data():
                """Filter spectrum data based on view mode and filters."""
                data = viewer.spectrum_data

                # Apply view mode filter
                mode = view_mode.value
                if mode == "MS2":
                    data = [s for s in data if s["ms_level"] == 2]
                elif mode == "Identified":
                    data = [s for s in data if s.get("id_idx") is not None]

                # Apply RT filter
                if spec_rt_min.value is not None:
                    data = [s for s in data if s["rt"] >= spec_rt_min.value]
                if spec_rt_max.value is not None:
                    data = [s for s in data if s["rt"] <= spec_rt_max.value]

                # Expand to all hits if enabled
                if show_all_hits.value:
                    expanded_data = []
                    for s in data:
                        all_hits = s.get("all_hits", [])
                        if all_hits:
                            # Create a row for each hit
                            for hit_data in all_hits:
                                row = dict(s)  # Copy spectrum base data
                                row["sequence"] = hit_data["sequence"]
                                row["full_sequence"] = hit_data["full_sequence"]
                                row["score"] = hit_data["score"]
                                row["hit_rank"] = hit_data["hit_rank"]
                                row["hit_idx"] = hit_data["hit_idx"]
                                # Add meta values from this hit
                                for key, value in hit_data["meta_values"].items():
                                    row[key] = value
                                # Create unique row key for table
                                row["row_key"] = f"{s['idx']}_{hit_data['hit_rank']}"
                                expanded_data.append(row)
                        else:
                            # No hits - keep row as is
                            row = dict(s)
                            row["row_key"] = f"{s['idx']}_0"
                            expanded_data.append(row)
                    data = expanded_data

                # Apply sequence filter (after expansion so it filters all hits)
                if spec_seq_pattern.value:
                    pattern = spec_seq_pattern.value.upper()
                    data = [s for s in data if pattern in s.get("full_sequence", "").upper()]

                # Apply score filter
                if spec_min_score.value is not None:
                    data = [
                        s
                        for s in data
                        if s.get("score") != "-"
                        and isinstance(s.get("score"), (int, float))
                        and s["score"] >= spec_min_score.value
                    ]

                return data

            def update_table():
                """Update table with current filters and column visibility."""
                filtered = get_filtered_data()
                viewer.spectrum_table.rows = filtered
                viewer.spectrum_table.columns = build_columns()
                ui.notify(f"Showing {len(filtered)} spectra", type="info")

            def on_view_mode_change(e):
                update_table()

            def on_column_toggle_change(e):
                """Update columns when Advanced or Meta Values checkbox changes."""
                viewer.spectrum_table.columns = build_columns()

            def on_all_hits_change(e):
                """Update both columns and rows when All Hits checkbox changes."""
                update_table()

            view_mode.on("update:model-value", on_view_mode_change)
            show_advanced.on("update:model-value", on_column_toggle_change)
            show_meta_values.on("update:model-value", on_column_toggle_change)
            show_all_hits.on("update:model-value", on_all_hits_change)

            # Filter buttons
            with ui.row().classes("gap-2 mb-2"):
                ui.button("Apply", on_click=update_table).props("dense size=sm color=primary")

                def reset_filters():
                    view_mode.value = "All"
                    spec_rt_min.value = None
                    spec_rt_max.value = None
                    spec_seq_pattern.value = ""
                    spec_min_score.value = None
                    show_advanced.value = False
                    show_meta_values.value = False
                    show_all_hits.value = False
                    viewer.spectrum_table.rows = viewer.spectrum_data
                    viewer.spectrum_table.columns = basic_columns

                ui.button("Reset", on_click=reset_filters).props("dense size=sm color=grey")

            def on_spectrum_click(e):
                row = e.args[1]
                if row and "idx" in row:
                    # If this spectrum has an associated ID, zoom the peakmap to that location
                    if row.get("id_idx") is not None:
                        viewer.zoom_to_id(row["id_idx"])
                    else:
                        viewer.show_spectrum_in_browser(row["idx"])

            def on_spectrum_select(e):
                if e.selection:
                    row = e.selection[0]
                    if row.get("id_idx") is not None:
                        viewer.zoom_to_id(row["id_idx"])
                    else:
                        viewer.show_spectrum_in_browser(row["idx"])

            viewer.spectrum_table = (
                ui.table(
                    columns=basic_columns,
                    rows=viewer.spectrum_data,
                    row_key="idx",
                    pagination={"rowsPerPage": 10, "sortBy": "idx", "descending": False},
                    selection="single",
                    on_select=on_spectrum_select,
                )
                .classes("w-full")
                .on("rowClick", on_spectrum_click)
            )
            viewer.spectrum_table.props("flat bordered dense")

        # Feature Table
        viewer.feature_table_expansion = ui.expansion("Features", icon="scatter_plot").classes("w-full max-w-[1700px]")
        viewer.panel_elements["features_table"] = viewer.feature_table_expansion
        viewer.feature_table_expansion.move(target_container=viewer.panels_container)
        with viewer.feature_table_expansion:
            ui.label("Click a row to zoom to that feature").classes("text-sm text-gray-400 mb-2")

            # Feature filters row
            with ui.row().classes("w-full items-end gap-2 mb-2 flex-wrap"):
                ui.label("Filter:").classes("text-xs text-gray-400")
                feat_min_intensity = (
                    ui.number(label="Min Intensity", format="%.0f").props("dense outlined").classes("w-28")
                )
                feat_min_quality = ui.number(label="Min Quality", format="%.2f").props("dense outlined").classes("w-24")
                feat_charge = (
                    ui.select(["All", "1", "2", "3", "4", "5+"], value="All", label="Charge")
                    .props("dense outlined")
                    .classes("w-20")
                )

                def apply_feature_filter():
                    charge_val = None
                    if feat_charge.value and feat_charge.value != "All":
                        if feat_charge.value == "5+":
                            charge_val = 5  # Will match 5 or greater
                        else:
                            charge_val = int(feat_charge.value)

                    filtered = viewer.filter_feature_data(
                        min_intensity=feat_min_intensity.value if feat_min_intensity.value else None,
                        min_quality=feat_min_quality.value if feat_min_quality.value else None,
                        charge=charge_val,
                    )
                    viewer.feature_table.rows = filtered
                    ui.notify(f"Showing {len(filtered)} features", type="info")

                def reset_feature_filter():
                    feat_min_intensity.value = None
                    feat_min_quality.value = None
                    feat_charge.value = "All"
                    viewer.feature_table.rows = viewer.feature_data

                ui.button("Apply", on_click=apply_feature_filter).props("dense size=sm color=primary")
                ui.button("Reset", on_click=reset_feature_filter).props("dense size=sm color=grey")

            feature_columns = [
                {"name": "idx", "label": "#", "field": "idx", "sortable": True, "align": "left"},
                {"name": "rt", "label": "RT (s)", "field": "rt", "sortable": True, "align": "right"},
                {"name": "mz", "label": "m/z", "field": "mz", "sortable": True, "align": "right"},
                {"name": "intensity", "label": "Intensity", "field": "intensity", "sortable": True, "align": "right"},
                {"name": "charge", "label": "Z", "field": "charge", "sortable": True, "align": "center"},
                {"name": "quality", "label": "Quality", "field": "quality", "sortable": True, "align": "right"},
            ]

            def on_feature_click(e):
                row = e.args[1]
                if row and "idx" in row:
                    viewer.zoom_to_feature(row["idx"])

            def on_feature_select(e):
                """Handle feature row selection (checkbox)."""
                if e.selection:
                    row = e.selection[0]
                    if row and "idx" in row:
                        viewer.zoom_to_feature(row["idx"])

            def on_feature_hover(e):
                """Handle feature row hover for visual feedback."""
                try:
                    row = e.args[1] if len(e.args) > 1 else None
                    if row and "idx" in row:
                        viewer.set_hover_feature(row["idx"])
                except Exception:
                    pass

            def on_feature_leave(e):
                """Clear feature hover state."""
                viewer.clear_hover()

            viewer.feature_table = (
                ui.table(
                    columns=feature_columns,
                    rows=[],
                    row_key="idx",
                    pagination={"rowsPerPage": 8, "sortBy": "intensity", "descending": True},
                    selection="single",
                    on_select=on_feature_select,
                )
                .classes("w-full hover-highlight")
                .on("rowClick", on_feature_click)
            )
            viewer.feature_table.on("row-dblclick", on_feature_hover)  # Use dblclick as hover proxy
            viewer.feature_table.props("flat bordered dense")

        # Custom range
        viewer.custom_range_expansion = ui.expansion("Custom Range", icon="tune").classes("w-full max-w-[1700px]")
        viewer.panel_elements["custom_range"] = viewer.custom_range_expansion
        viewer.custom_range_expansion.move(target_container=viewer.panels_container)
        with viewer.custom_range_expansion:
            with ui.row().classes("w-full gap-4 items-end"):
                rt_min_input = ui.number(label="RT Min (s)", value=0, format="%.2f")
                rt_max_input = ui.number(label="RT Max (s)", value=1000, format="%.2f")
                mz_min_input = ui.number(label="m/z Min", value=0, format="%.2f")
                mz_max_input = ui.number(label="m/z Max", value=2000, format="%.2f")

                def apply_range():
                    viewer.apply_custom_range(
                        rt_min_input.value, rt_max_input.value, mz_min_input.value, mz_max_input.value
                    )

                ui.button("Apply Range", on_click=apply_range).props("color=primary")

        # Help panel
        viewer.legend_expansion = ui.expansion("Help", icon="help").classes("w-full max-w-[1700px]")
        viewer.panel_elements["legend"] = viewer.legend_expansion
        viewer.legend_expansion.move(target_container=viewer.panels_container)
        with viewer.legend_expansion:
            with ui.row().classes("gap-6 flex-wrap w-full"):
                # Keyboard Shortcuts
                with ui.card().classes("p-3").style("min-width: 280px;"):
                    ui.label("⌨️ Keyboard Shortcuts").classes("font-bold text-lg mb-2")
                    ui.markdown("""
| Key | Action |
|-----|--------|
| `+` or `=` | Zoom in |
| `-` | Zoom out |
| `←` `→` | Pan left/right (RT) |
| `↑` `↓` | Pan up/down (m/z) |
| `Home` | Reset to full view |
| `Delete` | Delete selected measurement |
| `F11` | Toggle fullscreen |
""").classes("text-sm")

                # Mouse Controls
                with ui.card().classes("p-3").style("min-width: 280px;"):
                    ui.label("🖱️ Mouse Controls").classes("font-bold text-lg mb-2")
                    ui.markdown("""
| Action | Effect |
|--------|--------|
| **Scroll wheel** | Zoom in/out at cursor |
| **Drag** | Select region to zoom |
| **Shift + Drag** | Measure distance (ΔRT, Δm/z) |
| **Ctrl + Drag** | Pan (grab & move) |
| **Double-click** | Reset to full view |
| **Click TIC** | Jump to spectrum at RT |
| **Click table row** | Select spectrum/feature |
""").classes("text-sm")

                # 1D Spectrum Controls
                with ui.card().classes("p-3").style("min-width: 280px;"):
                    ui.label("📊 1D Spectrum Tools").classes("font-bold text-lg mb-2")
                    ui.markdown("""
| Tool | Usage |
|------|-------|
| **📏 Measure** | Click two peaks to measure Δm/z |
| **🏷️ Label** | Click peak to add custom annotation |
| **m/z Labels** | Toggle to show all peak m/z values |
| **Auto Y** | Auto-scale Y-axis to visible peaks |
| **Navigation** | `< >` prev/next, `MS1`/`MS2` by level |
| **3D View** | Toggle 3D surface visualization |
""").classes("text-sm")

                # Overlay Colors
                with ui.card().classes("p-3").style("min-width: 220px;"):
                    ui.label("🎨 Overlay Colors").classes("font-bold text-lg mb-2")
                    with ui.column().classes("gap-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.html(
                                '<div style="width:14px;height:14px;background:#00ff64;border-radius:50%;'
                                'border:1px solid white;"></div>',
                                sanitize=False,
                            )
                            ui.label("Feature Centroid").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html(
                                '<div style="width:14px;height:14px;border:2px solid #ffff00;"></div>',
                                sanitize=False,
                            )
                            ui.label("Feature Bounding Box").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html(
                                '<div style="width:14px;height:14px;background:rgba(0,200,255,0.5);'
                                'border:1px solid #00c8ff;"></div>',
                                sanitize=False,
                            )
                            ui.label("Feature Convex Hull").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html(
                                '<div style="width:14px;height:14px;background:#ff9632;'
                                'transform:rotate(45deg);"></div>',
                                sanitize=False,
                            )
                            ui.label("ID Precursor (◆)").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html(
                                '<div style="width:14px;height:14px;background:#ff64ff;border-radius:50%;"></div>',
                                sanitize=False,
                            )
                            ui.label("Selected Item").classes("text-sm")

                # Ion Colors
                with ui.card().classes("p-3").style("min-width: 180px;"):
                    ui.label("🔬 Ion Annotations").classes("font-bold text-lg mb-2")
                    with ui.column().classes("gap-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:14px;height:14px;background:#1f77b4;"></div>', sanitize=False)
                            ui.label("b-ions").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:14px;height:14px;background:#d62728;"></div>', sanitize=False)
                            ui.label("y-ions").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:14px;height:14px;background:#2ca02c;"></div>', sanitize=False)
                            ui.label("a-ions").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:14px;height:14px;background:#ff7f0e;"></div>', sanitize=False)
                            ui.label("Precursor").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:14px;height:14px;background:#7f7f7f;"></div>', sanitize=False)
                            ui.label("Unmatched").classes("text-sm")

                # TIC & Markers
                with ui.card().classes("p-3").style("min-width: 200px;"):
                    ui.label("📈 TIC & Markers").classes("font-bold text-lg mb-2")
                    with ui.column().classes("gap-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:20px;height:3px;background:#00d4ff;"></div>', sanitize=False)
                            ui.label("TIC trace").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html(
                                '<div style="width:14px;height:14px;background:rgba(255,255,0,0.2);'
                                'border:1px solid rgba(255,255,0,0.6);"></div>',
                                sanitize=False,
                            )
                            ui.label("Current view").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:3px;height:14px;background:#00d4ff;"></div>', sanitize=False)
                            ui.label("MS1 marker").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:3px;height:14px;background:#ff6b6b;"></div>', sanitize=False)
                            ui.label("MS2 marker").classes("text-sm")
                        with ui.row().classes("items-center gap-2"):
                            ui.html('<div style="width:14px;height:14px;background:#00ff64;"></div>', sanitize=False)
                            ui.label("Spectrum peaks").classes("text-sm")

                # File Types
                with ui.card().classes("p-3").style("min-width: 200px;"):
                    ui.label("📁 Supported Files").classes("font-bold text-lg mb-2")
                    ui.markdown("""
| Extension | Content |
|-----------|---------|
| `.mzML` | MS peak data |
| `.featureXML` | Detected features |
| `.idXML` | Peptide IDs |
""").classes("text-sm")
                    ui.label("Tips:").classes("font-semibold mt-2 text-sm")
                    ui.markdown("""
- Drag & drop files to load
- Use `--native` for file dialog
- Load multiple files at once
""").classes("text-xs text-gray-400")

        # Apply initial panel visibility (hide panels with "auto" that have no data yet)
        viewer.update_panel_visibility()

        # Keyboard handlers
        def on_global_key(e):
            if not e.action.keydown:
                return
            if e.key in ["+", "="]:
                viewer.zoom_in()
            elif e.key == "-":
                viewer.zoom_out()
            elif e.key.arrow_left:
                viewer.pan(rt_frac=-0.1)
            elif e.key.arrow_right:
                viewer.pan(rt_frac=0.1)
            elif e.key.arrow_up:
                viewer.pan(mz_frac=0.1)
            elif e.key.arrow_down:
                viewer.pan(mz_frac=-0.1)
            elif e.key == "Home":
                viewer.reset_view()
            elif e.key in ["Delete", "Backspace"]:
                # Delete selected measurement in spectrum browser
                if viewer.spectrum_selected_measurement_idx is not None:
                    viewer.delete_selected_measurement()

        ui.keyboard(on_key=on_global_key)

    # Load CLI files after UI is ready
    if _cli_files["mzml"]:
        if viewer.load_mzml(_cli_files["mzml"]):
            viewer.update_plot()
            viewer.update_tic_plot()
            # Show first spectrum in 1D browser
            if viewer.exp and viewer.exp.size() > 0:
                viewer.show_spectrum_in_browser(0)
    if _cli_files["featurexml"]:
        if viewer.load_featuremap(_cli_files["featurexml"]):
            viewer.update_plot()
    if _cli_files["idxml"]:
        if viewer.load_idxml(_cli_files["idxml"]):
            viewer.update_plot()


@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--port", "-p", default=8080, help="Port to run the server on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to bind to")
@click.option("--open/--no-open", "-o/-n", default=True, help="Open browser automatically (default: open)")
@click.option(
    "--native", is_flag=True, default=False, help="Run as native desktop app (requires: pip install pywebview)"
)
def main(files, port, host, open, native):
    """
    pyopenms-viewer - Fast visualization of mass spectrometry data.

    Pass one or more files to load them automatically:

    \b
    Examples:
        pyopenms-viewer sample.mzML                  # Open with browser
        pyopenms-viewer sample.mzML --native         # Open as desktop app
        pyopenms-viewer sample.mzML --no-open        # Start server only
        pyopenms-viewer sample.mzML features.featureXML
        pyopenms-viewer sample.mzML ids.idXML
        pyopenms-viewer data.mzML features.featureXML ids.idXML

    Supported file types (detected by extension):
        .mzML       Mass spectrometry peak data
        .featureXML Detected features with convex hulls
        .idXML      Peptide identifications

    Native mode (--native):
        Runs as a standalone desktop application using pywebview.
        Install with: pip install pywebview
    """
    global _cli_files

    for filepath in files:
        path = Path(filepath)
        ext = path.suffix.lower()

        if ext == ".mzml":
            _cli_files["mzml"] = str(path)
            click.echo(f"Will load mzML: {path.name}")
        elif ext == ".featurexml":
            _cli_files["featurexml"] = str(path)
            click.echo(f"Will load featureXML: {path.name}")
        elif ext == ".idxml":
            _cli_files["idxml"] = str(path)
            click.echo(f"Will load idXML: {path.name}")
        elif ext == ".xml":
            name_lower = path.name.lower()
            if "feature" in name_lower:
                _cli_files["featurexml"] = str(path)
                click.echo(f"Will load as featureXML: {path.name}")
            elif "id" in name_lower:
                _cli_files["idxml"] = str(path)
                click.echo(f"Will load as idXML: {path.name}")
            else:
                click.echo(f"Unknown XML file type: {path.name} (skipping)")
        else:
            click.echo(f"Unknown file type: {path.name} (skipping)")

    if native:
        click.echo("\nStarting native desktop app...")
    else:
        click.echo(f"\nStarting server at http://{host}:{port}")
        if open:
            click.echo("Opening browser...")

    # NiceGUI 3.x: Use root parameter for cleaner single-page app structure
    ui.run(
        title="pyopenms-viewer",
        host=host,
        port=port,
        reload=False,
        show=open and not native,
        native=native,
        window_size=(1400, 900) if native else None,
        root=create_ui,
        reconnect_timeout=60.0,  # Allow longer reconnect time for large file loads
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
