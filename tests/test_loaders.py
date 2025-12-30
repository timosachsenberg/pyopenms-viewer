"""Tests for the pyopenms_viewer loaders."""

from pathlib import Path

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders import (
    FeatureLoader,
    IDLoader,
    MzMLLoader,
    extract_chromatograms,
)

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
BSA_MZML = TEST_DATA_DIR / "BSA1_F1.mzML"
BSA_FEATUREXML = TEST_DATA_DIR / "BSA1_F1.featureXML"
BSA_IDXML = TEST_DATA_DIR / "BSA1_F1.idXML"
IMS_MZML = TEST_DATA_DIR / "ims_example.mzML"


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
        assert state.df is not None
        assert len(state.df) > 0

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
        assert state.im_df is not None
        assert len(state.im_df) > 0

    def test_load_ims_mzml_has_im_bounds(self):
        """Test that IMS data has proper IM bounds."""
        assert IMS_MZML.exists(), f"Test file not found: {IMS_MZML}"
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))
        assert state.has_ion_mobility
        assert state.im_min < state.im_max
        assert state.im_min >= 0


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
