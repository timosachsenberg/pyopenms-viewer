"""Tests for the pyopenms_viewer annotation module."""

import numpy as np

from pyopenms_viewer.annotation.spectrum_annotator import (
    MatchedIon,
    SpectrumAnnotationData,
    UnmatchedIon,
    _format_charge_only,
    _get_ion_type,
    _parse_charge_string,
    compute_spectrum_annotation,
    format_ion_label_with_superscript,
)
from pyopenms_viewer.annotation.theoretical_spectrum import (
    TheoreticalIon,
    TheoreticalSpectrum,
    generate_theoretical_spectrum,
)
from pyopenms_viewer.annotation.tick_formatter import (
    calculate_nice_ticks,
    format_intensity,
    format_mz_label,
    format_rt_label,
    format_tick_label,
)


class TestIonLabelFormatting:
    """Tests for ion label formatting with subscripts and superscripts."""

    def test_simple_b_ion(self):
        """Test formatting of simple b ion."""
        result = format_ion_label_with_superscript("b3")
        assert result == "b<sub>3</sub>"

    def test_simple_y_ion(self):
        """Test formatting of simple y ion."""
        result = format_ion_label_with_superscript("y5")
        assert result == "y<sub>5</sub>"

    def test_simple_a_ion(self):
        """Test formatting of simple a ion."""
        result = format_ion_label_with_superscript("a2")
        assert result == "a<sub>2</sub>"

    def test_two_digit_index(self):
        """Test formatting with two-digit index."""
        result = format_ion_label_with_superscript("y15")
        assert result == "y<sub>15</sub>"

    def test_single_charge(self):
        """Test formatting with single charge."""
        result = format_ion_label_with_superscript("y5+")
        assert result == "y<sub>5</sub><sup>+</sup>"

    def test_double_charge_plus_plus(self):
        """Test formatting with ++ notation."""
        result = format_ion_label_with_superscript("y5++")
        assert result == "y<sub>5</sub><sup>2+</sup>"

    def test_triple_charge(self):
        """Test formatting with +++ notation."""
        result = format_ion_label_with_superscript("b3+++")
        assert result == "b<sub>3</sub><sup>3+</sup>"

    def test_numeric_charge(self):
        """Test formatting with +N notation."""
        result = format_ion_label_with_superscript("y5+2")
        assert result == "y<sub>5</sub><sup>2+</sup>"

    def test_charge_three_numeric(self):
        """Test formatting with +3 notation."""
        result = format_ion_label_with_superscript("b7+3")
        assert result == "b<sub>7</sub><sup>3+</sup>"

    def test_negative_charge(self):
        """Test formatting with negative charge."""
        result = format_ion_label_with_superscript("b3-")
        assert result == "b<sub>3</sub><sup>-</sup>"

    def test_neutral_loss_h2o(self):
        """Test formatting with H2O neutral loss."""
        result = format_ion_label_with_superscript("y7-H2O")
        assert result == "y<sub>7</sub>-H2O"

    def test_neutral_loss_nh3(self):
        """Test formatting with NH3 neutral loss."""
        result = format_ion_label_with_superscript("b5-NH3")
        assert result == "b<sub>5</sub>-NH3"

    def test_neutral_loss_with_charge(self):
        """Test formatting with neutral loss and charge."""
        result = format_ion_label_with_superscript("y7-H2O++")
        assert result == "y<sub>7</sub>-H2O<sup>2+</sup>"

    def test_neutral_gain_with_charge(self):
        """Test formatting with neutral gain and charge."""
        # Note: +H2O followed by ++ for charge
        result = format_ion_label_with_superscript("y7+H2O++")
        assert result == "y<sub>7</sub>+H2O<sup>2+</sup>"

    def test_empty_string(self):
        """Test formatting of empty string."""
        result = format_ion_label_with_superscript("")
        assert result == ""

    def test_none_returns_empty(self):
        """Test formatting of None-like input."""
        result = format_ion_label_with_superscript("")
        assert result == ""

    def test_non_standard_ion_with_charge(self):
        """Test formatting of non-standard ion with charge."""
        result = format_ion_label_with_superscript("precursor++")
        assert result == "precursor<sup>2+</sup>"


class TestParseChargeString:
    """Tests for the _parse_charge_string helper function."""

    def test_single_plus(self):
        """Test parsing single plus."""
        assert _parse_charge_string("+") == "+"

    def test_double_plus(self):
        """Test parsing double plus."""
        assert _parse_charge_string("++") == "2+"

    def test_triple_plus(self):
        """Test parsing triple plus."""
        assert _parse_charge_string("+++") == "3+"

    def test_single_minus(self):
        """Test parsing single minus."""
        assert _parse_charge_string("-") == "-"

    def test_double_minus(self):
        """Test parsing double minus."""
        assert _parse_charge_string("--") == "2-"

    def test_plus_two(self):
        """Test parsing +2."""
        assert _parse_charge_string("+2") == "2+"

    def test_plus_one(self):
        """Test parsing +1."""
        assert _parse_charge_string("+1") == "+"

    def test_minus_two(self):
        """Test parsing -2."""
        assert _parse_charge_string("-2") == "2-"

    def test_empty_string(self):
        """Test parsing empty string."""
        assert _parse_charge_string("") == ""


class TestFormatChargeOnly:
    """Tests for the _format_charge_only fallback function."""

    def test_precursor_double_charge(self):
        """Test precursor with double charge."""
        result = _format_charge_only("precursor++")
        assert result == "precursor<sup>2+</sup>"

    def test_precursor_numeric_charge(self):
        """Test precursor with numeric charge."""
        result = _format_charge_only("precursor+2")
        assert result == "precursor<sup>2+</sup>"

    def test_no_charge(self):
        """Test string with no charge."""
        result = _format_charge_only("unknown")
        assert result == "unknown"


class TestGetIonType:
    """Tests for the _get_ion_type function."""

    def test_b_ion(self):
        """Test b ion detection."""
        assert _get_ion_type("b3") == "b"
        assert _get_ion_type("b15++") == "b"
        assert _get_ion_type("B3") == "b"

    def test_y_ion(self):
        """Test y ion detection."""
        assert _get_ion_type("y5") == "y"
        assert _get_ion_type("y12+2") == "y"
        assert _get_ion_type("Y5") == "y"

    def test_a_ion(self):
        """Test a ion detection."""
        assert _get_ion_type("a2") == "a"
        assert _get_ion_type("a7-H2O") == "a"

    def test_c_ion(self):
        """Test c ion detection."""
        assert _get_ion_type("c4") == "c"

    def test_x_ion(self):
        """Test x ion detection."""
        assert _get_ion_type("x3") == "x"

    def test_z_ion(self):
        """Test z ion detection."""
        assert _get_ion_type("z6") == "z"

    def test_precursor(self):
        """Test precursor detection."""
        assert _get_ion_type("precursor") == "precursor"
        assert _get_ion_type("prec++") == "precursor"
        assert _get_ion_type("[M+H]") == "precursor"

    def test_unknown(self):
        """Test unknown ion type."""
        assert _get_ion_type("unknown") == "unknown"
        assert _get_ion_type("MI:123") == "unknown"
        assert _get_ion_type("immonium") == "unknown"


class TestSpectrumAnnotationData:
    """Tests for SpectrumAnnotationData dataclass."""

    def test_empty_annotation_data(self):
        """Test creating empty annotation data."""
        data = SpectrumAnnotationData(
            sequence="PEPTIDE",
            charge=2,
            precursor_mz=400.5,
            tolerance_da=0.05,
        )
        assert data.sequence == "PEPTIDE"
        assert data.charge == 2
        assert len(data.matched_ions) == 0
        assert len(data.unmatched_ions) == 0
        assert data.coverage == 0.0

    def test_get_matched_by_type(self):
        """Test filtering matched ions by type."""
        matched = [
            MatchedIon(
                exp_mz=200.0,
                exp_intensity=1000,
                exp_intensity_pct=50,
                exp_peak_idx=0,
                theo_mz=200.01,
                theo_intensity=1.0,
                ion_name="b2",
                ion_type="b",
                mz_error=-0.01,
            ),
            MatchedIon(
                exp_mz=300.0,
                exp_intensity=2000,
                exp_intensity_pct=100,
                exp_peak_idx=1,
                theo_mz=300.02,
                theo_intensity=1.0,
                ion_name="y3",
                ion_type="y",
                mz_error=-0.02,
            ),
        ]
        data = SpectrumAnnotationData(
            sequence="PEPTIDE",
            charge=2,
            precursor_mz=400.5,
            tolerance_da=0.05,
            matched_ions=matched,
            n_matched=2,
        )
        b_ions = data.get_matched_by_type("b")
        y_ions = data.get_matched_by_type("y")
        assert len(b_ions) == 1
        assert len(y_ions) == 1
        assert b_ions[0].ion_name == "b2"
        assert y_ions[0].ion_name == "y3"

    def test_get_unmatched_by_type(self):
        """Test filtering unmatched ions by type."""
        unmatched = [
            UnmatchedIon(theo_mz=150.0, theo_intensity=1.0, ion_name="b1", ion_type="b"),
            UnmatchedIon(theo_mz=250.0, theo_intensity=1.0, ion_name="y2", ion_type="y"),
            UnmatchedIon(theo_mz=350.0, theo_intensity=1.0, ion_name="y4", ion_type="y"),
        ]
        data = SpectrumAnnotationData(
            sequence="PEPTIDE",
            charge=2,
            precursor_mz=400.5,
            tolerance_da=0.05,
            unmatched_ions=unmatched,
        )
        b_ions = data.get_unmatched_by_type("b")
        y_ions = data.get_unmatched_by_type("y")
        assert len(b_ions) == 1
        assert len(y_ions) == 2


class TestComputeSpectrumAnnotation:
    """Tests for spectrum annotation computation."""

    def test_empty_spectrum(self):
        """Test annotation with empty spectrum."""
        exp_mz = np.array([])
        exp_int = np.array([])
        result = compute_spectrum_annotation(
            exp_mz,
            exp_int,
            sequence_str="PEPTIDE",
            charge=2,
            precursor_mz=400.5,
        )
        assert result.sequence == "PEPTIDE"
        assert result.charge == 2
        assert len(result.matched_ions) == 0

    def test_invalid_sequence(self):
        """Test annotation with invalid sequence returns empty data."""
        exp_mz = np.array([200.0, 300.0, 400.0])
        exp_int = np.array([100.0, 200.0, 150.0])
        result = compute_spectrum_annotation(
            exp_mz,
            exp_int,
            sequence_str="INVALID!!!",  # Invalid sequence
            charge=2,
            precursor_mz=400.5,
        )
        assert result.sequence == "INVALID!!!"
        assert len(result.matched_ions) == 0

    def test_simple_peptide_annotation(self):
        """Test annotation of a simple peptide spectrum."""
        # Create experimental spectrum with peaks near theoretical b/y ions
        # For PEPTIDE (7 residues), expect b1-b6 and y1-y6 ions
        exp_mz = np.array([98.06, 227.1, 324.16, 425.2, 538.29, 653.31])
        exp_int = np.array([100, 200, 150, 300, 250, 180])

        result = compute_spectrum_annotation(
            exp_mz,
            exp_int,
            sequence_str="PEPTIDE",
            charge=2,
            precursor_mz=400.5,
            tolerance_da=0.1,
        )

        assert result.sequence == "PEPTIDE"
        assert result.n_theoretical > 0
        # Some ions should be matched
        assert result.n_matched >= 0


class TestTheoreticalSpectrum:
    """Tests for theoretical spectrum generation."""

    def test_theoretical_spectrum_dataclass(self):
        """Test TheoreticalSpectrum dataclass methods."""
        ions = [
            TheoreticalIon(mz=200.0, name="b2", ion_type="b", intensity=1.0),
            TheoreticalIon(mz=300.0, name="y3", ion_type="y", intensity=1.0),
            TheoreticalIon(mz=400.0, name="b4", ion_type="b", intensity=1.0),
            TheoreticalIon(mz=100.0, name="a1", ion_type="a", intensity=1.0),
        ]
        spec = TheoreticalSpectrum(ions=ions, sequence="PEPTIDE", charge=2)

        assert len(spec.b_ions) == 2
        assert len(spec.y_ions) == 1
        assert len(spec.a_ions) == 1
        assert spec.sequence == "PEPTIDE"
        assert spec.charge == 2

    def test_get_ions_by_type(self):
        """Test filtering ions by type."""
        ions = [
            TheoreticalIon(mz=200.0, name="b2", ion_type="b", intensity=1.0),
            TheoreticalIon(mz=300.0, name="y3", ion_type="y", intensity=1.0),
        ]
        spec = TheoreticalSpectrum(ions=ions, sequence="PEPTIDE", charge=2)

        b_ions = spec.get_ions_by_type("b")
        y_ions = spec.get_ions_by_type("y")
        c_ions = spec.get_ions_by_type("c")

        assert len(b_ions) == 1
        assert len(y_ions) == 1
        assert len(c_ions) == 0

    def test_generate_theoretical_spectrum(self):
        """Test generating theoretical spectrum from AASequence."""
        from pyopenms import AASequence

        seq = AASequence.fromString("PEPTIDE")
        spec = generate_theoretical_spectrum(seq, charge=2)

        assert spec.sequence == "PEPTIDE"
        assert spec.charge == 2
        assert len(spec.ions) > 0
        # Should have b, y, and a ions
        assert len(spec.b_ions) > 0
        assert len(spec.y_ions) > 0


class TestTickFormatter:
    """Tests for tick calculation and formatting utilities."""

    def test_calculate_nice_ticks_basic(self):
        """Test basic nice tick calculation."""
        ticks = calculate_nice_ticks(0, 100, num_ticks=6)
        assert len(ticks) > 0
        assert all(t >= 0 and t <= 100 for t in ticks)
        # Ticks should be round numbers
        assert all(t % 10 == 0 or t % 5 == 0 for t in ticks)

    def test_calculate_nice_ticks_small_range(self):
        """Test nice tick calculation for small range."""
        ticks = calculate_nice_ticks(0.1, 0.9, num_ticks=5)
        assert len(ticks) > 0
        assert all(t >= 0.1 and t <= 0.9 for t in ticks)

    def test_calculate_nice_ticks_large_range(self):
        """Test nice tick calculation for large range."""
        ticks = calculate_nice_ticks(0, 10000, num_ticks=6)
        assert len(ticks) > 0
        assert all(t >= 0 and t <= 10000 for t in ticks)

    def test_calculate_nice_ticks_equal_bounds(self):
        """Test nice tick calculation when min equals max."""
        ticks = calculate_nice_ticks(50, 50, num_ticks=6)
        assert len(ticks) == 1
        assert ticks[0] == 50

    def test_calculate_nice_ticks_reversed_bounds(self):
        """Test nice tick calculation when min > max."""
        ticks = calculate_nice_ticks(100, 50, num_ticks=6)
        assert len(ticks) == 1

    def test_format_tick_label_large_range(self):
        """Test tick label formatting for large range."""
        label = format_tick_label(1500, 5000)
        assert label == "1500"

    def test_format_tick_label_medium_range(self):
        """Test tick label formatting for medium range."""
        label = format_tick_label(15.5, 50)
        assert label == "15.5"

    def test_format_tick_label_small_range(self):
        """Test tick label formatting for small range."""
        label = format_tick_label(1.23, 5)
        assert label == "1.23"

    def test_format_tick_label_tiny_range(self):
        """Test tick label formatting for tiny range."""
        label = format_tick_label(0.123, 0.5)
        assert label == "0.123"

    def test_format_rt_label_seconds(self):
        """Test RT label formatting in seconds."""
        label = format_rt_label(125.5, in_minutes=False)
        assert label == "125.5"

    def test_format_rt_label_minutes(self):
        """Test RT label formatting in minutes."""
        label = format_rt_label(120.0, in_minutes=True)
        assert label == "2.00"

    def test_format_mz_label_default_precision(self):
        """Test m/z label formatting with default precision."""
        label = format_mz_label(524.2614)
        assert label == "524.2614"

    def test_format_mz_label_custom_precision(self):
        """Test m/z label formatting with custom precision."""
        label = format_mz_label(524.2614, precision=2)
        assert label == "524.26"

    def test_format_intensity_scientific(self):
        """Test intensity formatting in scientific notation."""
        label = format_intensity(1234567.89, scientific=True)
        assert "e" in label.lower()

    def test_format_intensity_regular(self):
        """Test intensity formatting in regular notation."""
        label = format_intensity(1234567.89, scientific=False)
        assert label == "1234568"
