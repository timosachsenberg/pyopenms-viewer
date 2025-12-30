"""Spectrum annotation using theoretical spectra matching."""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from pyopenms import (
    AASequence,
    MSSpectrum,
    SpectrumAlignment,
    SpectrumAnnotator,
    TheoreticalSpectrumGenerator,
)

from pyopenms_viewer.annotation.theoretical_spectrum import generate_theoretical_spectrum
from pyopenms_viewer.core.config import ION_COLORS


@dataclass
class MatchedIon:
    """A matched ion from experimental spectrum."""

    exp_mz: float  # Experimental m/z
    exp_intensity: float  # Experimental intensity (raw)
    exp_intensity_pct: float  # Experimental intensity (normalized %)
    exp_peak_idx: int  # Index in experimental spectrum
    theo_mz: float  # Theoretical m/z
    theo_intensity: float  # Theoretical intensity (predicted)
    ion_name: str  # Ion name (e.g., "b3", "y5+2")
    ion_type: str  # Ion type (e.g., "b", "y", "a")
    mz_error: float  # m/z error (exp - theo)


@dataclass
class UnmatchedIon:
    """An unmatched theoretical ion."""

    theo_mz: float  # Theoretical m/z
    theo_intensity: float  # Theoretical intensity (predicted)
    ion_name: str  # Ion name
    ion_type: str  # Ion type


@dataclass
class SpectrumAnnotationData:
    """Complete annotation data for a spectrum.

    Contains both matched and unmatched ions, computed once and cached.
    """

    sequence: str
    charge: int
    precursor_mz: float
    tolerance_da: float
    matched_ions: list[MatchedIon] = field(default_factory=list)
    unmatched_ions: list[UnmatchedIon] = field(default_factory=list)
    # Statistics
    n_theoretical: int = 0
    n_matched: int = 0
    coverage: float = 0.0  # n_matched / n_theoretical

    def get_matched_by_type(self, ion_type: str) -> list[MatchedIon]:
        """Get matched ions of a specific type."""
        return [ion for ion in self.matched_ions if ion.ion_type == ion_type]

    def get_unmatched_by_type(self, ion_type: str) -> list[UnmatchedIon]:
        """Get unmatched ions of a specific type."""
        return [ion for ion in self.unmatched_ions if ion.ion_type == ion_type]


def compute_spectrum_annotation(
    exp_mz: np.ndarray,
    exp_int: np.ndarray,
    sequence_str: str,
    charge: int,
    precursor_mz: float,
    tolerance_da: float = 0.05,
    external_annotations: Optional[list[tuple[int, str, str]]] = None,
) -> SpectrumAnnotationData:
    """Compute complete annotation data for a spectrum.

    This is the main function that computes both matched and unmatched ions.
    The result should be cached and reused for rendering.

    Args:
        exp_mz: Experimental m/z array
        exp_int: Experimental intensity array
        sequence_str: Peptide sequence string
        charge: Precursor charge
        precursor_mz: Precursor m/z
        tolerance_da: Mass tolerance in Da for matching
        external_annotations: Optional pre-computed annotations from SpectrumAnnotator
            as list of (peak_index, ion_name, ion_type)

    Returns:
        SpectrumAnnotationData with matched and unmatched ions
    """
    # Normalize intensities
    max_int = exp_int.max() if len(exp_int) > 0 else 1.0
    exp_int_pct = (exp_int / max_int) * 100 if max_int > 0 else exp_int

    # Generate theoretical spectrum
    try:
        seq = AASequence.fromString(sequence_str)
        theo_spec = generate_theoretical_spectrum(seq, charge)
    except Exception:
        # Return empty annotation data if sequence parsing fails
        return SpectrumAnnotationData(
            sequence=sequence_str,
            charge=charge,
            precursor_mz=precursor_mz,
            tolerance_da=tolerance_da,
        )

    matched_ions: list[MatchedIon] = []
    unmatched_ions: list[UnmatchedIon] = []
    matched_theo_mz: set[float] = set()  # Track which theoretical ions were matched

    if external_annotations:
        # Use external annotations (from SpectrumAnnotator or idXML)
        for peak_idx, ion_name, ion_type in external_annotations:
            if peak_idx < len(exp_mz):
                exp_mz_val = float(exp_mz[peak_idx])
                exp_int_val = float(exp_int[peak_idx])
                exp_int_pct_val = float(exp_int_pct[peak_idx])

                # Find matching theoretical ion
                theo_mz_val = exp_mz_val  # Default to exp m/z
                theo_int_val = 1.0
                mz_error = 0.0

                for theo_ion in theo_spec.ions:
                    if abs(theo_ion.mz - exp_mz_val) <= tolerance_da:
                        theo_mz_val = theo_ion.mz
                        theo_int_val = theo_ion.intensity
                        mz_error = exp_mz_val - theo_ion.mz
                        matched_theo_mz.add(theo_ion.mz)
                        break

                matched_ions.append(
                    MatchedIon(
                        exp_mz=exp_mz_val,
                        exp_intensity=exp_int_val,
                        exp_intensity_pct=exp_int_pct_val,
                        exp_peak_idx=peak_idx,
                        theo_mz=theo_mz_val,
                        theo_intensity=theo_int_val,
                        ion_name=ion_name,
                        ion_type=ion_type,
                        mz_error=mz_error,
                    )
                )
    else:
        # Match theoretical ions to experimental peaks
        for theo_ion in theo_spec.ions:
            if len(exp_mz) == 0:
                continue

            # Find closest experimental peak
            diffs = np.abs(exp_mz - theo_ion.mz)
            min_idx = int(np.argmin(diffs))

            if diffs[min_idx] <= tolerance_da:
                matched_theo_mz.add(theo_ion.mz)
                matched_ions.append(
                    MatchedIon(
                        exp_mz=float(exp_mz[min_idx]),
                        exp_intensity=float(exp_int[min_idx]),
                        exp_intensity_pct=float(exp_int_pct[min_idx]),
                        exp_peak_idx=min_idx,
                        theo_mz=theo_ion.mz,
                        theo_intensity=theo_ion.intensity,
                        ion_name=theo_ion.name,
                        ion_type=theo_ion.ion_type,
                        mz_error=float(exp_mz[min_idx]) - theo_ion.mz,
                    )
                )

    # Find unmatched theoretical ions
    for theo_ion in theo_spec.ions:
        is_matched = any(
            abs(theo_ion.mz - m) <= tolerance_da * 0.1  # Tighter tolerance for set membership
            for m in matched_theo_mz
        )
        if not is_matched:
            unmatched_ions.append(
                UnmatchedIon(
                    theo_mz=theo_ion.mz,
                    theo_intensity=theo_ion.intensity,
                    ion_name=theo_ion.name,
                    ion_type=theo_ion.ion_type,
                )
            )

    n_theoretical = len(theo_spec.ions)
    n_matched = len(matched_ions)
    coverage = n_matched / n_theoretical if n_theoretical > 0 else 0.0

    return SpectrumAnnotationData(
        sequence=sequence_str,
        charge=charge,
        precursor_mz=precursor_mz,
        tolerance_da=tolerance_da,
        matched_ions=matched_ions,
        unmatched_ions=unmatched_ions,
        n_theoretical=n_theoretical,
        n_matched=n_matched,
        coverage=coverage,
    )


def annotate_spectrum_with_id(
    spectrum: MSSpectrum,
    peptide_hit,
    tolerance_da: float = 0.05,
) -> list[tuple[int, str, str]]:
    """Annotate a spectrum using SpectrumAnnotator.

    Uses TheoreticalSpectrumGenerator and SpectrumAnnotator to generate
    annotations. Annotations are stored in the spectrum's string data array.

    Args:
        spectrum: The experimental MS2 spectrum
        peptide_hit: PeptideHit with sequence information
        tolerance_da: Mass tolerance in Da for matching

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
        string_arrays = spec_copy.getStringDataArrays()
        for arr in string_arrays:
            arr_name = arr.getName()
            # Handle both bytes and string for array name
            if arr_name == "IonNames" or arr_name == b"IonNames":
                for peak_idx, ion_name in enumerate(arr):
                    if ion_name:
                        # Handle bytes or string annotation
                        if isinstance(ion_name, bytes):
                            ion_name = ion_name.decode("utf-8", errors="ignore")

                        # Determine ion type from name
                        if ion_name.startswith("b"):
                            ion_type = "b"
                        elif ion_name.startswith("y"):
                            ion_type = "y"
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
        print(f"Error annotating spectrum: {e}")

    return annotations


def get_external_peak_annotations(spectrum: MSSpectrum) -> list[tuple[int, str, str]]:
    """Get peak annotations from external sources (idXML).

    Checks for annotations in:
    1. PeakAnnotations from getPeakAnnotations()
    2. UserParam "fragment_annotation" string

    Args:
        spectrum: MS2 spectrum that may have external annotations

    Returns:
        List of (peak_index, ion_name, ion_type) tuples
    """
    annotations = []

    try:
        # Try getPeakAnnotations() first (OpenMS >= 3.0)
        peak_annotations = spectrum.getPeakAnnotations()
        if peak_annotations:
            for pa in peak_annotations:
                peak_idx = pa.peak_index
                annotation = pa.annotation
                if annotation:
                    ion_name = str(annotation)
                    ion_type = _get_ion_type(ion_name)
                    annotations.append((peak_idx, ion_name, ion_type))
    except Exception:
        pass

    # Also check for UserParam annotations
    if spectrum.metaValueExists("fragment_annotation"):
        try:
            frag_annot = spectrum.getMetaValue("fragment_annotation")
            if isinstance(frag_annot, bytes):
                frag_annot = frag_annot.decode()
            annotations.extend(parse_fragment_annotation_string(frag_annot, spectrum))
        except Exception:
            pass

    return annotations


def parse_fragment_annotation_string(
    annotation_str: str,
    spectrum: MSSpectrum,
) -> list[tuple[int, str, str]]:
    """Parse fragment annotation string from UserParam.

    Handles various annotation formats like:
    - "y1@100.5" (ion@mz)
    - "b2+2@200.3" (ion+charge@mz)
    - Space or comma separated

    Args:
        annotation_str: Annotation string from UserParam
        spectrum: Spectrum to match peaks against

    Returns:
        List of (peak_index, ion_name, ion_type) tuples
    """
    annotations = []

    if not annotation_str:
        return annotations

    # Split by common separators
    parts = re.split(r"[,\s]+", annotation_str.strip())

    mz_array, _ = spectrum.get_peaks()

    for part in parts:
        if "@" in part:
            ion_part, mz_part = part.split("@", 1)
            try:
                target_mz = float(mz_part)
                # Find closest peak
                if len(mz_array) > 0:
                    diffs = abs(mz_array - target_mz)
                    min_idx = int(diffs.argmin())
                    if diffs[min_idx] < 0.5:  # Within 0.5 Da
                        ion_name = ion_part
                        ion_type = _get_ion_type(ion_name)
                        annotations.append((min_idx, ion_name, ion_type))
            except ValueError:
                pass

    return annotations


def format_ion_label_with_superscript(ion_name: str) -> str:
    """Format ion name with index as subscript and charge as superscript.

    For a,b,c,x,y,z ions, formats the numeric index as subscript and charge as superscript:
    - "y5" -> "y<sub>5</sub>"
    - "y15+" -> "y<sub>15</sub><sup>+</sup>"
    - "y5+2" -> "y<sub>5</sub><sup>2+</sup>"
    - "y5++" -> "y<sub>5</sub><sup>2+</sup>"
    - "b3-" -> "b<sub>3</sub><sup>-</sup>"
    - "y7+H2O++" -> "y<sub>7</sub>+H2O<sup>2+</sup>" (neutral loss preserved, only trailing charge)

    For other ion types, only formats charge as superscript.

    Args:
        ion_name: Ion name string (e.g., "b3", "y5+2", "y5++", "y7+H2O++")

    Returns:
        Ion name with index as subscript and charge as superscript HTML
    """
    if not ion_name:
        return ion_name

    # First, extract trailing charge from the end of the string
    # Charge patterns at end: +, ++, +++, -, --, ---, +2, +3, -2, -3
    trailing_charge = ""
    base_name = ion_name

    # Check for trailing charge patterns (must be at the very end)
    charge_match = re.search(r"(\++|\-+|\+\d+|\-\d+)$", ion_name)
    if charge_match:
        trailing_charge = charge_match.group(1)
        base_name = ion_name[: charge_match.start()]

    # Now parse the base name for a,b,c,x,y,z ion pattern: letter + digits + optional modifier
    match = re.match(r"^([abcxyzABCXYZ])(\d+)(.*)$", base_name)
    if match:
        ion_type = match.group(1)
        index = match.group(2)
        modifier = match.group(3)  # e.g., "", "+H2O", "-NH3", "-H2O"

        # Format index as subscript
        result = f"{ion_type}<sub>{index}</sub>"

        # Add modifier (neutral loss) as-is
        if modifier:
            result += modifier

        # Format charge as superscript if present
        if trailing_charge:
            charge_str = _parse_charge_string(trailing_charge)
            if charge_str:
                result += f"<sup>{charge_str}</sup>"

        return result

    # For non-standard ion names, just handle charge modifier at the end (fallback)
    return _format_charge_only(ion_name)


def _parse_charge_string(charge_part: str) -> str:
    """Parse charge string and return formatted charge (e.g., "+", "2+", "-", "2-").

    Args:
        charge_part: The charge portion of ion name (e.g., "+", "+2", "++", "-", "-2", "--")

    Returns:
        Formatted charge string for superscript display
    """
    if not charge_part:
        return ""

    # Pattern 1: Repeated + (e.g., "+", "++", "+++")
    match_repeated_plus = re.match(r"^\++$", charge_part)
    if match_repeated_plus:
        charge_count = len(charge_part)
        return "+" if charge_count == 1 else f"{charge_count}+"

    # Pattern 2: Repeated - (e.g., "-", "--", "---")
    match_repeated_minus = re.match(r"^-+$", charge_part)
    if match_repeated_minus:
        charge_count = len(charge_part)
        return "-" if charge_count == 1 else f"{charge_count}-"

    # Pattern 3: +N (e.g., "+2", "+3")
    match_plus_num = re.match(r"^\+(\d+)$", charge_part)
    if match_plus_num:
        charge_num = int(match_plus_num.group(1))
        return "+" if charge_num == 1 else f"{charge_num}+"

    # Pattern 4: -N (e.g., "-2", "-3")
    match_minus_num = re.match(r"^-(\d+)$", charge_part)
    if match_minus_num:
        charge_num = int(match_minus_num.group(1))
        return "-" if charge_num == 1 else f"{charge_num}-"

    # Unknown format, return as-is
    return charge_part


def _format_charge_only(ion_name: str) -> str:
    """Format only the charge portion of an ion name (fallback for non-standard ions).

    Args:
        ion_name: Ion name string

    Returns:
        Ion name with charge formatted as superscript HTML
    """
    # Pattern 1: Repeated + or - at the end (e.g., "precursor++")
    match_repeated_plus = re.search(r"\++$", ion_name)
    if match_repeated_plus:
        charge_count = len(match_repeated_plus.group())
        base_name = ion_name[: match_repeated_plus.start()]
        charge_str = "+" if charge_count == 1 else f"{charge_count}+"
        return f"{base_name}<sup>{charge_str}</sup>"

    match_repeated_minus = re.search(r"-+$", ion_name)
    if match_repeated_minus:
        charge_count = len(match_repeated_minus.group())
        base_name = ion_name[: match_repeated_minus.start()]
        charge_str = "-" if charge_count == 1 else f"{charge_count}-"
        return f"{base_name}<sup>{charge_str}</sup>"

    # Pattern 2: Number after + or - (e.g., "precursor+2")
    match_plus_num = re.search(r"\+(\d+)$", ion_name)
    if match_plus_num:
        charge_num = int(match_plus_num.group(1))
        base_name = ion_name[: match_plus_num.start()]
        charge_str = "+" if charge_num == 1 else f"{charge_num}+"
        return f"{base_name}<sup>{charge_str}</sup>"

    match_minus_num = re.search(r"-(\d+)$", ion_name)
    if match_minus_num:
        charge_num = int(match_minus_num.group(1))
        base_name = ion_name[: match_minus_num.start()]
        charge_str = "-" if charge_num == 1 else f"{charge_num}-"
        return f"{base_name}<sup>{charge_str}</sup>"

    # No charge modifier found, return as-is
    return ion_name


def _get_ion_type(ion_name: str) -> str:
    """Determine ion type from ion name.

    Args:
        ion_name: Ion name string (e.g., "b3", "y5+2")

    Returns:
        Ion type character ("b", "y", "a", "c", "x", "z", "precursor", "unknown")
    """
    name = ion_name.lower()
    if name.startswith("b"):
        return "b"
    elif name.startswith("y"):
        return "y"
    elif name.startswith("a"):
        return "a"
    elif name.startswith("c"):
        return "c"
    elif name.startswith("x"):
        return "x"
    elif name.startswith("z"):
        return "z"
    elif "precursor" in name or "prec" in name or "[m" in name:
        return "precursor"
    elif "mi:" in name or name.startswith("i"):
        return "unknown"  # Immonium ions
    else:
        return "unknown"


def get_external_peak_annotations_from_hit(
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
                ion_type = _get_ion_type(ion_name)
                annotations.append((int(min_idx), ion_name, ion_type))

    except Exception as e:
        print(f"Error getting external peak annotations: {e}")

    return annotations


def create_annotated_spectrum_plot(
    exp_mz: np.ndarray,
    exp_int: np.ndarray,
    sequence_str: str,
    charge: int,
    precursor_mz: float,
    annotate: bool = True,
    mirror_mode: bool = False,
    show_unmatched: bool = True,
    annotation_data: Optional[SpectrumAnnotationData] = None,
) -> go.Figure:
    """Create an annotated spectrum plot using Plotly.

    Args:
        exp_mz: Experimental m/z values
        exp_int: Experimental intensity values
        sequence_str: Peptide sequence string
        charge: Precursor charge
        precursor_mz: Precursor m/z value
        annotate: Whether to show annotations (if False, shows raw spectrum)
        mirror_mode: If True, flip annotated peaks downward for comparison view
        show_unmatched: If True and mirror_mode is True, show unmatched theoretical ions
        annotation_data: Pre-computed SpectrumAnnotationData (required when annotate=True)

    Returns:
        Plotly Figure object with annotated spectrum
    """
    # Normalize intensities to percentage
    max_int = exp_int.max() if len(exp_int) > 0 else 1
    exp_int_norm = (exp_int / max_int) * 100

    # Create figure
    fig = go.Figure()

    # Add experimental spectrum as vertical lines (stem plot)
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

    # Add invisible hover points for experimental peaks
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

    # Add annotations if enabled and annotation_data is provided
    if annotate and annotation_data is not None:
        _add_annotations_from_data(fig, annotation_data, mirror_mode, show_unmatched)

    # Add precursor marker
    fig.add_vline(
        x=precursor_mz, line_dash="dash", line_color="orange", annotation_text=f"Precursor ({precursor_mz:.2f})"
    )

    # Update layout
    fig.update_layout(
        title={"text": f"MS2 Spectrum: {sequence_str} (z={charge}+)", "font": {"size": 14, "color": "#888"}},
        xaxis_title="m/z",
        yaxis_title="Relative Intensity (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"size": 10}},
        modebar={"remove": ["lasso2d", "select2d"]},
        font={"color": "#888"},
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


def _draw_matched_ions(
    fig: go.Figure,
    ions: list[MatchedIon],
    intensities: list[float],
    color: str,
    legend_name: str,
    flip: bool = False,
    show_in_legend: bool = True,
    is_theoretical: bool = False,
    show_labels: bool = True,
) -> None:
    """Draw matched ion stems, markers, and labels.

    Args:
        fig: Plotly figure to add traces to
        ions: List of matched ions to draw
        intensities: Intensity values (0-100 scale) to use for each ion
        color: Color for stems, markers, and labels
        legend_name: Name for the legend entry
        flip: If True, draw downward (negative y). If False, draw upward.
        show_in_legend: Whether to show this trace in the legend
        is_theoretical: If True, show theoretical info in hover template
        show_labels: If True, show text labels for ion names
    """
    if not ions or not intensities:
        return

    sign = -1 if flip else 1
    text_offset = -5 if flip else 5  # Offset increased to accommodate subscript
    text_angle = 0

    # Build stems
    x_stems: list[float | None] = []
    y_stems: list[float | None] = []
    for ion, intensity in zip(ions, intensities):
        x_stems.extend([ion.exp_mz, ion.exp_mz, None])
        y_stems.extend([0, sign * intensity, None])

    fig.add_trace(
        go.Scatter(
            x=x_stems,
            y=y_stems,
            mode="lines",
            line={"color": color, "width": 2},
            name=legend_name,
            showlegend=show_in_legend,
            hoverinfo="skip",
        )
    )

    # Add markers and labels
    for ion, intensity in zip(ions, intensities):
        y_val = sign * intensity
        formatted_name = format_ion_label_with_superscript(ion.ion_name)
        if is_theoretical:
            hover = f"{formatted_name} (theoretical)<br>m/z: {ion.theo_mz:.4f}<br>Intensity: {intensity:.1f}%<extra></extra>"
        else:
            hover = f"{formatted_name}<br>m/z: {ion.exp_mz:.4f} (Δ{ion.mz_error:.4f})<br>Intensity: {ion.exp_intensity_pct:.1f}%<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=[ion.exp_mz],
                y=[y_val],
                mode="markers",
                marker={"color": color, "size": 4},
                showlegend=False,
                hovertemplate=hover,
            )
        )
        if show_labels:
            fig.add_annotation(
                x=ion.exp_mz,
                y=y_val + text_offset,
                text=format_ion_label_with_superscript(ion.ion_name),
                showarrow=False,
                font={"size": 9, "color": color},
                textangle=text_angle,
            )


def _add_annotations_from_data(
    fig: go.Figure,
    annotation_data: SpectrumAnnotationData,
    mirror_mode: bool,
    show_unmatched: bool,
) -> None:
    """Add annotations to figure from pre-computed SpectrumAnnotationData."""
    # Group matched ions by type
    matched_by_type: dict[str, list[MatchedIon]] = {}
    for ion in annotation_data.matched_ions:
        if ion.ion_type not in matched_by_type:
            matched_by_type[ion.ion_type] = []
        matched_by_type[ion.ion_type].append(ion)

    # Compute max theoretical intensity across ALL ions (matched + unmatched) for normalization
    all_theo_intensities = [ion.theo_intensity for ion in annotation_data.matched_ions]
    all_theo_intensities += [ion.theo_intensity for ion in annotation_data.unmatched_ions]
    max_theo_int = max(all_theo_intensities) if all_theo_intensities else 1.0
    if max_theo_int <= 0:
        max_theo_int = 1.0

    # Add unmatched theoretical ions in mirror mode (downward only)
    if mirror_mode and show_unmatched and annotation_data.unmatched_ions:
        x_unmatched: list[float | None] = []
        y_unmatched: list[float | None] = []
        for ion in annotation_data.unmatched_ions:
            # Normalize to 100% scale using global max
            display_int = (ion.theo_intensity / max_theo_int) * 100
            x_unmatched.extend([ion.theo_mz, ion.theo_mz, None])
            y_unmatched.extend([0, -display_int, None])

        fig.add_trace(
            go.Scatter(
                x=x_unmatched,
                y=y_unmatched,
                mode="lines",
                line={"color": "gray", "width": 1, "dash": "dash"},
                name="Unmatched theoretical",
                hoverinfo="skip",
                opacity=0.5,
            )
        )

        # Add hover points and annotations for unmatched ions
        for ion in annotation_data.unmatched_ions:
            display_int = (ion.theo_intensity / max_theo_int) * 100
            formatted_name = format_ion_label_with_superscript(ion.ion_name)
            fig.add_trace(
                go.Scatter(
                    x=[ion.theo_mz],
                    y=[-display_int],
                    mode="markers",
                    marker={"color": "gray", "size": 4, "opacity": 0.5},
                    showlegend=False,
                    hovertemplate=f"{formatted_name} (theoretical)<br>m/z: {ion.theo_mz:.4f}<br>Intensity: {display_int:.1f}%<extra></extra>",
                )
            )

            # Add text annotation
            fig.add_annotation(
                x=ion.theo_mz,
                y=-display_int - 5,
                text=format_ion_label_with_superscript(ion.ion_name),
                showarrow=False,
                font={"size": 8, "color": "gray"},
                textangle=0,
                opacity=0.6,
            )

    # Add matched peaks as colored lines grouped by ion type
    for ion_type, ions in matched_by_type.items():
        color = ION_COLORS.get(ion_type, ION_COLORS["unknown"])

        # Compute experimental intensities (always used for top half / non-mirror)
        exp_intensities = [ion.exp_intensity_pct for ion in ions]

        if mirror_mode:
            # Compute theoretical intensities normalized to 0-100% scale
            theo_intensities = [(ion.theo_intensity / max_theo_int) * 100 for ion in ions]

            # Mirror mode: draw matched peaks both upward (experimental) and downward (theoretical)
            # Top half: experimental intensities with labels (what we measured)
            _draw_matched_ions(
                fig,
                ions,
                exp_intensities,
                color,
                f"{ion_type}-ions",
                flip=False,
                show_in_legend=True,
                is_theoretical=False,
                show_labels=True,
            )
            # Bottom half: theoretical intensities without labels (what was predicted)
            _draw_matched_ions(
                fig,
                ions,
                theo_intensities,
                color,
                f"{ion_type}-ions",
                flip=True,
                show_in_legend=False,
                is_theoretical=True,
                show_labels=False,
            )
        else:
            # Non-mirror: draw upward with experimental intensities only
            _draw_matched_ions(
                fig,
                ions,
                exp_intensities,
                color,
                f"{ion_type}-ions",
                flip=False,
                show_in_legend=True,
                is_theoretical=False,
                show_labels=True,
            )
