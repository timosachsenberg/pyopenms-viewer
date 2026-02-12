"""Tests for the data export functionality."""

from pathlib import Path

import numpy as np
from pyopenms import MSExperiment, MSSpectrum, MzMLFile

from pyopenms_viewer.panels.export_panel import _export_filtered_mzml


def _make_test_experiment():
    """Create a small MSExperiment for testing.

    Returns an experiment with 5 spectra:
      - idx 0: MS1, RT=100, peaks at mz [100, 200, 300, 400, 500]
      - idx 1: MS2, RT=150, peaks at mz [110, 220, 330]
      - idx 2: MS1, RT=200, peaks at mz [150, 250, 350, 450]
      - idx 3: MS2, RT=250, peaks at mz [120, 240, 360]
      - idx 4: MS1, RT=300, peaks at mz [180, 280, 380, 480]
    """
    exp = MSExperiment()

    specs = [
        (1, 100.0, [100.0, 200.0, 300.0, 400.0, 500.0]),
        (2, 150.0, [110.0, 220.0, 330.0]),
        (1, 200.0, [150.0, 250.0, 350.0, 450.0]),
        (2, 250.0, [120.0, 240.0, 360.0]),
        (1, 300.0, [180.0, 280.0, 380.0, 480.0]),
    ]

    for ms_level, rt, mzs in specs:
        s = MSSpectrum()
        s.setMSLevel(ms_level)
        s.setRT(rt)
        mz_arr = np.array(mzs, dtype=np.float64)
        int_arr = np.ones(len(mzs), dtype=np.float64) * 1000.0
        s.set_peaks((mz_arr, int_arr))
        exp.addSpectrum(s)

    return exp


class TestExportFilteredMzml:
    """Tests for _export_filtered_mzml standalone function."""

    def test_export_all_spectra(self, tmp_path):
        """Export all spectra with no filtering."""
        exp = _make_test_experiment()
        output = str(tmp_path / "all.mzML")

        count = _export_filtered_mzml(exp, output, rt_min=0, rt_max=999, mz_min=0, mz_max=999, selected_levels={1, 2})

        assert count == 5
        assert Path(output).exists()

        # Verify output has correct number of spectra
        result = MSExperiment()
        MzMLFile().load(output, result)
        assert result.getNrSpectra() == 5

    def test_filter_by_rt_range(self, tmp_path):
        """Filter spectra to RT 100-200 (should get idx 0, 1, 2)."""
        exp = _make_test_experiment()
        output = str(tmp_path / "rt_filtered.mzML")

        count = _export_filtered_mzml(
            exp, output, rt_min=100, rt_max=200, mz_min=0, mz_max=999, selected_levels={1, 2}
        )

        assert count == 3

        result = MSExperiment()
        MzMLFile().load(output, result)
        assert result.getNrSpectra() == 3
        rts = [result.getSpectrum(i).getRT() for i in range(result.getNrSpectra())]
        assert all(100 <= rt <= 200 for rt in rts)

    def test_filter_by_ms_level(self, tmp_path):
        """Filter to only MS1 spectra (should get idx 0, 2, 4)."""
        exp = _make_test_experiment()
        output = str(tmp_path / "ms1_only.mzML")

        count = _export_filtered_mzml(exp, output, rt_min=0, rt_max=999, mz_min=0, mz_max=999, selected_levels={1})

        assert count == 3

        result = MSExperiment()
        MzMLFile().load(output, result)
        assert result.getNrSpectra() == 3
        for i in range(result.getNrSpectra()):
            assert result.getSpectrum(i).getMSLevel() == 1

    def test_filter_by_ms_level_2(self, tmp_path):
        """Filter to only MS2 spectra (should get idx 1, 3)."""
        exp = _make_test_experiment()
        output = str(tmp_path / "ms2_only.mzML")

        count = _export_filtered_mzml(exp, output, rt_min=0, rt_max=999, mz_min=0, mz_max=999, selected_levels={2})

        assert count == 2

        result = MSExperiment()
        MzMLFile().load(output, result)
        assert result.getNrSpectra() == 2
        for i in range(result.getNrSpectra()):
            assert result.getSpectrum(i).getMSLevel() == 2

    def test_mz_peak_trimming(self, tmp_path):
        """Trim peaks by m/z range within spectra."""
        exp = _make_test_experiment()
        output = str(tmp_path / "mz_trimmed.mzML")

        # Only keep peaks in m/z 150-350
        count = _export_filtered_mzml(
            exp, output, rt_min=0, rt_max=999, mz_min=150, mz_max=350, selected_levels={1, 2}
        )

        assert count == 5  # All spectra still present

        result = MSExperiment()
        MzMLFile().load(output, result)

        # Check spectrum 0 (MS1 RT=100): original mzs [100,200,300,400,500] -> [200,300] in range 150-350
        spec0 = result.getSpectrum(0)
        mzs0, _ = spec0.get_peaks()
        assert all(150 <= mz <= 350 for mz in mzs0)
        assert len(mzs0) == 2  # 200, 300

        # Check spectrum 1 (MS2 RT=150): original mzs [110,220,330] -> [220,330] in range 150-350
        spec1 = result.getSpectrum(1)
        mzs1, _ = spec1.get_peaks()
        assert all(150 <= mz <= 350 for mz in mzs1)
        assert len(mzs1) == 2  # 220, 330

    def test_empty_result(self, tmp_path):
        """No spectra match filters -> empty output."""
        exp = _make_test_experiment()
        output = str(tmp_path / "empty.mzML")

        # RT range that doesn't match any spectrum
        count = _export_filtered_mzml(
            exp, output, rt_min=500, rt_max=600, mz_min=0, mz_max=999, selected_levels={1, 2}
        )

        assert count == 0
        assert Path(output).exists()

        result = MSExperiment()
        MzMLFile().load(output, result)
        assert result.getNrSpectra() == 0

    def test_combined_rt_and_level_filter(self, tmp_path):
        """Combine RT and MS level filters."""
        exp = _make_test_experiment()
        output = str(tmp_path / "combined.mzML")

        # MS1 spectra in RT 150-300 -> idx 2 (MS1, RT=200) and idx 4 (MS1, RT=300)
        count = _export_filtered_mzml(
            exp, output, rt_min=150, rt_max=300, mz_min=0, mz_max=999, selected_levels={1}
        )

        assert count == 2

        result = MSExperiment()
        MzMLFile().load(output, result)
        assert result.getNrSpectra() == 2
        for i in range(result.getNrSpectra()):
            spec = result.getSpectrum(i)
            assert spec.getMSLevel() == 1
            assert 150 <= spec.getRT() <= 300

    def test_original_experiment_not_modified(self, tmp_path):
        """Verify the original experiment is not modified by export."""
        exp = _make_test_experiment()
        output = str(tmp_path / "trimmed.mzML")

        # Get original peak count for spectrum 0
        orig_mzs, _ = exp.getSpectrum(0).get_peaks()
        orig_count = len(orig_mzs)

        # Export with m/z trimming
        _export_filtered_mzml(exp, output, rt_min=0, rt_max=999, mz_min=200, mz_max=300, selected_levels={1, 2})

        # Original should be unchanged
        check_mzs, _ = exp.getSpectrum(0).get_peaks()
        assert len(check_mzs) == orig_count
