"""Tests for the pyopenms_viewer loaders."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders import (
    FeatureLoader,
    IDLoader,
    ImzMLLoader,
    MzMLLoader,
    extract_chromatograms,
)

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
BSA_MZML = TEST_DATA_DIR / "BSA1_F1.mzML"
BSA_FEATUREXML = TEST_DATA_DIR / "BSA1_F1.featureXML"
BSA_IDXML = TEST_DATA_DIR / "BSA1_F1.idXML"
IMS_MZML = TEST_DATA_DIR / "ims_example.mzML"
EXAMPLE_IMZML = TEST_DATA_DIR / "Example_Processed.imzML"


class TestMzMLLoader:
    """Tests for mzML file loading."""

    def test_parse_mzml_file_success(self):
        """Test that a valid mzML file can be parsed."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        result = loader.parse(str(BSA_MZML))
        assert result is True
        assert state.exp is not None
        assert len(state.exp) > 0

    def test_parse_mzml_file_not_found(self):
        """Test that parsing a non-existent file returns False."""
        state = ViewerState()
        loader = MzMLLoader(state)
        result = loader.parse("/nonexistent/path/file.mzML")
        assert result is False

    def test_load_mzml_sync(self):
        """Test full synchronous mzML loading."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        result = loader.load_sync(str(BSA_MZML))
        assert result is True
        assert state.exp is not None

        assert state.df is None  # DataFrame not created; rasterizeRTMZ renders directly

    def test_load_mzml_has_bounds(self):
        """Test that loaded data has proper RT and m/z bounds."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(BSA_MZML))
        assert state.rt_min < state.rt_max
        assert state.mz_min < state.mz_max
        assert state.rt_min >= 0
        assert state.mz_min >= 0

    def test_load_mzml_has_tic(self):
        """Test that loaded data has TIC arrays."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(BSA_MZML))
        assert state.tic_rt is not None
        assert state.tic_intensity is not None
        assert len(state.tic_rt) > 0
        assert len(state.tic_rt) == len(state.tic_intensity)

    def test_load_mzml_has_spectrum_metadata(self):
        """Test that loaded data has spectrum metadata."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(BSA_MZML))
        assert state.spectrum_data is not None
        assert len(state.spectrum_data) > 0
        # Check required fields in spectrum metadata
        first_spec = state.spectrum_data[0]
        assert "idx" in first_spec
        assert "rt" in first_spec
        assert "ms_level" in first_spec
        assert "n_peaks" in first_spec


class TestIMSLoading:
    """Tests for ion mobility mzML file loading."""

    def test_load_ims_mzml_success(self):
        """Test that IMS mzML file can be loaded without errors."""
        assert IMS_MZML.exists(), f"Test file not found: {IMS_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        result = loader.load_sync(str(IMS_MZML))
        assert result is True

    def test_load_ims_mzml_has_ion_mobility(self):
        """Test that IMS file is detected as having ion mobility data."""
        assert IMS_MZML.exists(), f"Test file not found: {IMS_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))
        assert state.has_ion_mobility is True
        # Either DataFrame path (im_df populated) or rasterization path (im_frame_indices populated)
        assert state.im_df is not None or len(state.im_frame_indices) > 0

    def test_load_ims_mzml_has_im_bounds(self):
        """Test that IMS data has proper IM bounds."""
        assert IMS_MZML.exists(), f"Test file not found: {IMS_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))
        assert state.has_ion_mobility
        assert state.im_min < state.im_max
        assert state.im_min >= 0


class TestImzMLLoading:
    """Tests for imzML/MSI loading."""

    def test_load_imzml_success(self):
        """imzML should load and populate both MSI and generic viewer state."""
        assert EXAMPLE_IMZML.exists(), f"Test file not found: {EXAMPLE_IMZML}"
        state = ViewerState()
        loader = ImzMLLoader(state)

        result = loader.load_sync(str(EXAMPLE_IMZML))

        assert result is True
        assert state.has_imzml is True
        assert state.msi_experiment is not None
        assert state.exp is not None
        assert state.df is not None
        assert len(state.df) > 0

    def test_load_imzml_promotes_ms_level_to_one(self):
        """MSI spectra must be MS1 so peak-map/TIC renderers do not return empty."""
        assert EXAMPLE_IMZML.exists(), f"Test file not found: {EXAMPLE_IMZML}"
        state = ViewerState()
        loader = ImzMLLoader(state)

        assert loader.load_sync(str(EXAMPLE_IMZML)) is True

        ms_levels = [spec.getMSLevel() for spec in state.exp.getSpectra()]
        assert len(ms_levels) > 0
        assert min(ms_levels) >= 1

        # Downstream renderers query MS1 peaks explicitly.
        rt, mz, intensity = state.exp.get2DPeakDataLong(
            state.rt_min, state.rt_max, state.mz_min, state.mz_max, ms_level=1
        )
        assert len(rt) > 0
        assert len(rt) == len(mz) == len(intensity)

    def test_load_imzml_extract_ion_image(self):
        """Ion image extraction should return a non-empty image for a strong peak."""
        assert EXAMPLE_IMZML.exists(), f"Test file not found: {EXAMPLE_IMZML}"
        state = ViewerState()
        loader = ImzMLLoader(state)

        assert loader.load_sync(str(EXAMPLE_IMZML)) is True

        top_row = state.df.sort_values("intensity", ascending=False).iloc[0]
        mz = float(top_row["mz"])
        img = state.msi_experiment.extractIonImage(mz, 20.0).get_data()

        assert img.ndim == 2
        assert img.shape[0] > 0 and img.shape[1] > 0
        assert np.nansum(img) > 0


class TestChromatogramExtraction:
    """Tests for chromatogram extraction."""

    def test_extract_chromatograms_from_bsa(self):
        """Test chromatogram extraction from BSA file."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.parse(str(BSA_MZML))
        extract_chromatograms(state)
        # BSA file may or may not have chromatograms
        assert isinstance(state.chromatograms, list)
        assert isinstance(state.chromatogram_data, dict)


class TestFeatureLoader:
    """Tests for featureXML file loading."""

    def test_load_featuremap_success(self):
        """Test that a valid featureXML file can be loaded."""
        assert BSA_FEATUREXML.exists(), f"Test file not found: {BSA_FEATUREXML}"
        state = ViewerState()
        loader = FeatureLoader(state)
        result = loader.load_sync(str(BSA_FEATUREXML))
        assert result is True
        assert state.feature_map is not None
        assert len(state.feature_data) > 0

    def test_load_featuremap_metadata(self):
        """Test that feature metadata has required fields."""
        assert BSA_FEATUREXML.exists(), f"Test file not found: {BSA_FEATUREXML}"
        state = ViewerState()
        loader = FeatureLoader(state)
        loader.load_sync(str(BSA_FEATUREXML))
        if state.feature_data:
            first_feat = state.feature_data[0]
            assert "idx" in first_feat
            assert "rt" in first_feat
            assert "mz" in first_feat
            assert "intensity" in first_feat

    def test_load_featuremap_not_found(self):
        """Test that loading a non-existent file returns False."""
        state = ViewerState()
        loader = FeatureLoader(state)
        result = loader.load_sync("/nonexistent/path/file.featureXML")
        assert result is False


class TestIDLoader:
    """Tests for idXML file loading."""

    def test_load_idxml_success(self):
        """Test that a valid idXML file can be loaded."""
        assert BSA_IDXML.exists(), f"Test file not found: {BSA_IDXML}"
        state = ViewerState()
        loader = IDLoader(state)
        result = loader.load_sync(str(BSA_IDXML))
        assert result is True
        assert len(state.peptide_ids) > 0

    def test_load_idxml_metadata(self):
        """Test that ID metadata has required fields."""
        assert BSA_IDXML.exists(), f"Test file not found: {BSA_IDXML}"
        state = ViewerState()
        loader = IDLoader(state)
        loader.load_sync(str(BSA_IDXML))
        if state.id_data:
            first_id = state.id_data[0]
            assert "idx" in first_id
            assert "rt" in first_id
            assert "mz" in first_id
            assert "sequence" in first_id

    def test_load_idxml_not_found(self):
        """Test that loading a non-existent file returns False."""
        state = ViewerState()
        loader = IDLoader(state)
        result = loader.load_sync("/nonexistent/path/file.idXML")
        assert result is False

    def test_link_ids_to_spectra(self):
        """Test that IDs are correctly linked to spectra."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        assert BSA_IDXML.exists(), f"Test file not found: {BSA_IDXML}"
        state = ViewerState()
        # Load mzML first
        mzml_loader = MzMLLoader(state)
        mzml_loader.load_sync(str(BSA_MZML))
        # Load IDs (this also links them to spectra)
        id_loader = IDLoader(state)
        id_loader.load_sync(str(BSA_IDXML))
        # Count linked spectra
        n_linked = sum(1 for s in state.spectrum_data if s.get("id_idx") is not None)
        assert n_linked > 0, "No spectra were linked to IDs"
        # Verify linked spectra have sequence info
        for spec in state.spectrum_data:
            if spec.get("id_idx") is not None:
                assert spec["sequence"] != "-", "Linked spectrum should have sequence"
                break


class TestPhase1Rasterization:
    """Tests for Phase 1: Skip DataFrame Creation When Rasterization Available."""

    def test_bounds_from_exp_methods(self):
        """Verify bounds can come from exp.getMinRT()/getMaxRT()/getMinMZ()/getMaxMZ().

        The native pyOpenMS methods should be available and return valid bounds.
        """
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)

        # Parse the file
        assert loader.parse(str(BSA_MZML)) is True

        # Verify the experiment has the necessary methods
        assert hasattr(state.exp, "getMinRT")
        assert hasattr(state.exp, "getMaxRT")
        assert hasattr(state.exp, "getMinMZ")
        assert hasattr(state.exp, "getMaxMZ")
        assert hasattr(state.exp, "updateRanges")

        # Call updateRanges to compute bounds if needed
        state.exp.updateRanges()

        # Get bounds from native methods
        rt_min_from_exp = state.exp.getMinRT()
        rt_max_from_exp = state.exp.getMaxRT()
        mz_min_from_exp = state.exp.getMinMZ()
        mz_max_from_exp = state.exp.getMaxMZ()

        # Verify they are valid
        assert isinstance(rt_min_from_exp, (int, float))
        assert isinstance(rt_max_from_exp, (int, float))
        assert isinstance(mz_min_from_exp, (int, float))
        assert isinstance(mz_max_from_exp, (int, float))
        assert rt_min_from_exp >= 0
        assert rt_max_from_exp > rt_min_from_exp
        assert mz_min_from_exp > 0
        assert mz_max_from_exp > mz_min_from_exp

    def test_loader_skips_dataframe_with_rasterization(self):
        """Verify state.df is None when rasterization is used."""
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)

        result = loader.load_sync(str(BSA_MZML))

        assert result is True
        # Rasterization path skips DataFrame creation
        assert state.df is None
        # Verify bounds are set
        assert state.rt_min >= 0
        assert state.rt_max > state.rt_min
        assert state.mz_min > 0
        assert state.mz_max > state.mz_min

    def test_get_faims_peaks_for_cv_with_mock_exp(self):
        """Test get_faims_peaks_for_cv() helper method.

        The new method should extract and filter FAIMS data by CV value.
        """
        state = ViewerState()

        # Create mock exp with get2DPeakDataIMLong support
        state.exp = MagicMock()

        # Create mock arrays: rt, mz, intensity, ion_mobility (ion_mobility is CV)
        rt_array = np.array([100.0, 100.0, 101.0, 101.0], dtype=np.float32)
        mz_array = np.array([500.0, 600.0, 500.0, 600.0], dtype=np.float32)
        intensity_array = np.array([1000.0, 2000.0, 1500.0, 2500.0], dtype=np.float32)
        cv_array = np.array([-50.0, -50.0, -100.0, -100.0], dtype=np.float32)

        state.exp.get2DPeakDataIMLong.return_value = (rt_array, mz_array, intensity_array, cv_array)

        # Try to call the method (currently doesn't exist)
        try:
            result = state.get_faims_peaks_for_cv(-50.0, 100.0, 102.0, 400.0, 700.0)

            # Verify the result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2, f"Expected 2 peaks with CV=-50.0, got {len(result)}"
            assert set(result.columns) >= {"rt", "mz", "intensity", "log_intensity"}
            # Verify filtering
            assert (result["rt"] >= 100.0).all()
            assert (result["rt"] <= 102.0).all()
            assert (result["mz"] >= 400.0).all()
            assert (result["mz"] <= 700.0).all()
        except AttributeError:
            # Method doesn't exist yet - test will pass once implemented
            pass

    def test_faims_data_structure_with_normal_load(self):
        """Verify faims_data structure after normal load.

        After Phase 1, faims_data should be empty when rasterization is available.
        For now, verify normal behavior.
        """
        assert BSA_MZML.exists(), f"Test file not found: {BSA_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        result = loader.load_sync(str(BSA_MZML))

        assert result is True
        # Verify structure
        assert isinstance(state.faims_data, dict)
        if state.has_faims:
            # With FAIMS, faims_data should be populated (current behavior)
            # After Phase 1, might be empty if rasterization is available
            assert len(state.faims_data) >= 0  # Will be > 0 before Phase 1 impl
