"""Tests for ion mobility rasterization features.

Tests the frame selection state helpers, loader frame index population,
renderer rasterization path, and fallback behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders.mzml_loader import MzMLLoader
from pyopenms_viewer.rendering.peak_map_renderer import IMPeakMapRenderer

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
IMS_MZML = TEST_DATA_DIR / "ims_example.mzML"


class TestIMFrameSelectionState:
    """Tests for frame selection state and helper methods."""

    def test_im_frame_state_defaults(self):
        """Test that IM frame state initializes correctly."""
        state = ViewerState()
        assert state.selected_im_frame_idx is None
        assert state.im_frame_indices == []
        assert len(state.im_frame_rts) == 0

    def test_get_im_frame_spectrum_no_selection(self):
        """Test get_im_frame_spectrum returns None when no frame selected."""
        state = ViewerState()
        assert state.get_im_frame_spectrum() is None

    def test_get_im_frame_spectrum_no_exp(self):
        """Test get_im_frame_spectrum returns None when exp is None."""
        state = ViewerState()
        state.selected_im_frame_idx = 0
        assert state.get_im_frame_spectrum() is None

    def test_get_im_frame_spectrum_with_exp(self):
        """Test get_im_frame_spectrum returns spectrum when frame is selected."""
        state = ViewerState()
        mock_exp = MagicMock()
        mock_spec = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=10)
        mock_exp.__getitem__ = MagicMock(return_value=mock_spec)
        state.exp = mock_exp
        state.selected_im_frame_idx = 5
        result = state.get_im_frame_spectrum()
        assert result == mock_spec
        mock_exp.__getitem__.assert_called_with(5)

    def test_get_im_frame_spectrum_out_of_bounds(self):
        """Test get_im_frame_spectrum returns None for out-of-bounds index."""
        state = ViewerState()
        mock_exp = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=5)
        state.exp = mock_exp
        state.selected_im_frame_idx = 10
        assert state.get_im_frame_spectrum() is None

    def test_get_im_frame_spectrum_negative_index(self):
        """Test get_im_frame_spectrum returns None for negative index."""
        state = ViewerState()
        mock_exp = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=5)
        state.exp = mock_exp
        state.selected_im_frame_idx = -1
        assert state.get_im_frame_spectrum() is None

    def test_select_nearest_im_frame_exact(self):
        """Test select_nearest_im_frame with exact RT match."""
        state = ViewerState()
        state.im_frame_rts = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        state.im_frame_indices = [0, 2, 4, 6, 8]
        state.select_nearest_im_frame(300.0)
        assert state.selected_im_frame_idx == 4

    def test_select_nearest_im_frame_between(self):
        """Test select_nearest_im_frame with RT between frames."""
        state = ViewerState()
        state.im_frame_rts = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        state.im_frame_indices = [0, 2, 4, 6, 8]
        state.select_nearest_im_frame(250.0)
        # Should pick the closer one (200.0 at idx=2 or 300.0 at idx=4)
        assert state.selected_im_frame_idx in [2, 4]

    def test_select_nearest_im_frame_before_first(self):
        """Test select_nearest_im_frame with RT before all frames."""
        state = ViewerState()
        state.im_frame_rts = np.array([100.0, 200.0, 300.0])
        state.im_frame_indices = [0, 2, 4]
        state.select_nearest_im_frame(50.0)
        assert state.selected_im_frame_idx == 0

    def test_select_nearest_im_frame_after_last(self):
        """Test select_nearest_im_frame with RT after all frames."""
        state = ViewerState()
        state.im_frame_rts = np.array([100.0, 200.0, 300.0])
        state.im_frame_indices = [0, 2, 4]
        state.select_nearest_im_frame(500.0)
        assert state.selected_im_frame_idx == 4

    def test_select_nearest_im_frame_empty(self):
        """Test select_nearest_im_frame with no frames."""
        state = ViewerState()
        state.im_frame_rts = np.array([])
        state.im_frame_indices = []
        state.select_nearest_im_frame(100.0)
        assert state.selected_im_frame_idx is None

    def test_select_nearest_im_frame_none_rts(self):
        """Test select_nearest_im_frame when im_frame_rts is empty."""
        state = ViewerState()
        state.select_nearest_im_frame(100.0)
        assert state.selected_im_frame_idx is None

    def test_clear_mzml_data_clears_im_frame_state(self):
        """Test that clear_mzml_data clears all IM frame state."""
        state = ViewerState()
        state.selected_im_frame_idx = 5
        state.im_frame_indices = [0, 2, 4, 6, 8]
        state.im_frame_rts = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        state.has_ion_mobility = True

        state.clear_mzml_data()

        assert state.selected_im_frame_idx is None
        assert state.im_frame_indices == []
        assert len(state.im_frame_rts) == 0
        assert state.has_ion_mobility is False


class TestIMLoaderFrameIndices:
    """Tests for frame index population in the mzML loader."""

    def test_load_ims_mzml_populates_frame_indices(self):
        """Test that loading an IMS mzML file populates frame indices."""
        if not IMS_MZML.exists():
            return  # Skip if test data not available
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))

        if state.has_ion_mobility:
            # Frame indices should always be populated when IM data exists
            assert len(state.im_frame_indices) > 0
            assert len(state.im_frame_rts) > 0
            assert len(state.im_frame_rts) == len(state.im_frame_indices)

    def test_load_ims_mzml_frame_rts_sorted(self):
        """Test that frame RTs are sorted."""
        if not IMS_MZML.exists():
            return
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))

        if state.has_ion_mobility and len(state.im_frame_rts) > 0:
            # Check that RTs are in sorted order
            for i in range(len(state.im_frame_rts) - 1):
                assert state.im_frame_rts[i] <= state.im_frame_rts[i + 1]

    def test_load_ims_mzml_rasterization_path_skips_dataframe(self):
        """Test that rasterization path sets im_df to None."""
        if not IMS_MZML.exists():
            return
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))

        if state.has_ion_mobility:
            # Check if rasterization is available
            if state.exp is not None and len(state.exp) > 0:
                test_spec = state.exp[0]
                if hasattr(test_spec, "rasterizeIMFrame"):
                    # Rasterization path: no DataFrame, frame indices populated
                    assert state.im_df is None
                    assert len(state.im_frame_indices) > 0
                    assert state.selected_im_frame_idx is not None
                else:
                    # Fallback path: DataFrame created
                    assert state.im_df is not None

    def test_load_ims_mzml_first_frame_selected(self):
        """Test that first frame is selected by default on rasterization path."""
        if not IMS_MZML.exists():
            return
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))

        if state.has_ion_mobility and state.im_df is None and len(state.im_frame_indices) > 0:
            assert state.selected_im_frame_idx == state.im_frame_indices[0]

    def test_load_ims_mzml_has_valid_im_bounds(self):
        """Test that IM bounds are correctly computed from actual data."""
        if not IMS_MZML.exists():
            return
        state = ViewerState()
        loader = MzMLLoader(state)
        loader.load_sync(str(IMS_MZML))

        if state.has_ion_mobility:
            assert state.im_min < state.im_max
            assert state.im_min >= 0


class TestProcessIonMobilityDataBranching:
    """Tests for the _process_ion_mobility_data branching logic."""

    def test_rasterization_path_skips_dataframe(self):
        """Test rasterization path skips DataFrame creation."""
        state = ViewerState()
        # Create a mock experiment with rasterizeIMFrame
        mock_spec = MagicMock()
        mock_spec.rasterizeIMFrame = MagicMock()
        mock_spec.getRT.return_value = 100.0

        mock_exp = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=3)
        mock_exp.__getitem__ = MagicMock(return_value=mock_spec)
        state.exp = mock_exp

        loader = MzMLLoader(state)

        im_mz = [np.array([100.0, 200.0, 300.0], dtype=np.float32)]
        im_im = [np.array([0.5, 0.8, 1.1], dtype=np.float32)]
        im_int = [np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)]
        frame_indices = [0]

        loader._process_ion_mobility_data(im_mz, im_im, im_int, "ion mobility", "", frame_indices)

        assert state.has_ion_mobility is True
        assert state.im_df is None
        assert len(state.im_frame_indices) == 1
        assert state.selected_im_frame_idx == 0

    def test_empty_im_data_sets_no_ion_mobility(self):
        """Test that empty IM data sets has_ion_mobility to False."""
        state = ViewerState()
        loader = MzMLLoader(state)
        loader._process_ion_mobility_data([], [], [], None, "")
        assert state.has_ion_mobility is False
        assert state.im_df is None

    def test_im_bounds_computed_correctly(self):
        """Test that IM bounds are correctly computed."""
        state = ViewerState()
        mock_spec = MagicMock()
        mock_spec.rasterizeIMFrame = MagicMock()
        mock_spec.getRT.return_value = 100.0

        mock_exp = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=3)
        mock_exp.__getitem__ = MagicMock(return_value=mock_spec)
        state.exp = mock_exp

        loader = MzMLLoader(state)

        im_mz = [np.array([100.0, 200.0], dtype=np.float32)]
        im_im = [np.array([0.5, 1.5], dtype=np.float32)]
        im_int = [np.array([1000.0, 2000.0], dtype=np.float32)]
        frame_indices = [0]

        loader._process_ion_mobility_data(im_mz, im_im, im_int, "inverse reduced ion mobility", "", frame_indices)

        assert state.im_min == 0.5
        assert state.im_max == 1.5
        assert state.im_type == "inverse_k0"


class TestIMPeakMapRendererBranching:
    """Tests for IMPeakMapRenderer render method."""

    def test_render_no_frame_returns_empty(self):
        """Test render returns empty string when no frame selected."""
        renderer = IMPeakMapRenderer()
        state = ViewerState()
        assert renderer.render(state) == ""

    def test_render_im_rasterized_returns_image(self):
        """Test that _render_im_rasterized produces a base64 image."""
        renderer = IMPeakMapRenderer(plot_width=100, plot_height=50)
        state = ViewerState()
        state.has_ion_mobility = True
        state.im_min = 0.5
        state.im_max = 1.5
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.view_im_min = 0.5
        state.view_im_max = 1.5
        state.view_mz_min = 100.0
        state.view_mz_max = 2000.0
        state.show_mobilogram = False

        # Mock the spectrum with rasterizeIMFrame that fills the array
        mock_spec = MagicMock()

        def fill_array(arr, im_min, im_max, mz_min, mz_max, aggregation="sum"):
            arr[:] = np.random.rand(*arr.shape).astype(np.float32) * 1000

        mock_spec.rasterizeIMFrame = MagicMock(side_effect=fill_array)
        mock_spec.getRT.return_value = 100.0

        mock_exp = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=1)
        mock_exp.__getitem__ = MagicMock(return_value=mock_spec)
        state.exp = mock_exp
        state.selected_im_frame_idx = 0
        state.im_frame_indices = [0]
        state.im_frame_rts = np.array([100.0])

        result = renderer._render_im_rasterized(state)
        assert result != ""
        assert len(result) > 100  # Should be a real base64 PNG

    def test_render_im_rasterized_empty_array_returns_empty(self):
        """Test that _render_im_rasterized returns empty string when array is all zeros."""
        renderer = IMPeakMapRenderer(plot_width=100, plot_height=50)
        state = ViewerState()
        state.has_ion_mobility = True
        state.im_min = 0.5
        state.im_max = 1.5
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.view_im_min = 0.5
        state.view_im_max = 1.5
        state.view_mz_min = 100.0
        state.view_mz_max = 2000.0

        mock_spec = MagicMock()
        # rasterizeIMFrame that leaves array as zeros
        mock_spec.rasterizeIMFrame = MagicMock()
        mock_spec.getRT.return_value = 100.0

        mock_exp = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=1)
        mock_exp.__getitem__ = MagicMock(return_value=mock_spec)
        state.exp = mock_exp
        state.selected_im_frame_idx = 0

        result = renderer._render_im_rasterized(state)
        assert result == ""

    def test_render_im_rasterized_exception_returns_empty(self):
        """Test that _render_im_rasterized returns empty string on exception."""
        renderer = IMPeakMapRenderer(plot_width=100, plot_height=50)
        state = ViewerState()
        state.has_ion_mobility = True
        state.im_min = 0.5
        state.im_max = 1.5
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.view_im_min = 0.5
        state.view_im_max = 1.5
        state.view_mz_min = 100.0
        state.view_mz_max = 2000.0

        mock_spec = MagicMock()
        mock_spec.rasterizeIMFrame = MagicMock(side_effect=RuntimeError("test error"))
        mock_spec.getRT.return_value = 100.0

        mock_exp = MagicMock()
        mock_exp.__len__ = MagicMock(return_value=1)
        mock_exp.__getitem__ = MagicMock(return_value=mock_spec)
        state.exp = mock_exp
        state.selected_im_frame_idx = 0

        result = renderer._render_im_rasterized(state)
        assert result == ""

class TestMobilogramFromRaster:
    """Tests for _draw_mobilogram_from_raster method."""

    def test_mobilogram_from_raster_draws_on_canvas(self):
        """Test that mobilogram from raster draws correctly."""
        from PIL import Image

        renderer = IMPeakMapRenderer(plot_width=100, plot_height=50)
        state = ViewerState()
        state.mobilogram_plot_width = 80
        state.show_mobilogram = True

        canvas = Image.new("RGBA", (300, 150), (0, 0, 0, 0))

        # Create a raster array with some data
        raster = np.random.rand(50, 100).astype(np.float32) * 1000

        result = renderer._draw_mobilogram_from_raster(
            canvas, state, raster, 100.0, 2000.0, 0.5, 1.5
        )
        assert result is not None
        assert result.size == canvas.size

    def test_mobilogram_from_raster_zero_array(self):
        """Test that mobilogram from zero raster returns canvas unchanged."""
        from PIL import Image

        renderer = IMPeakMapRenderer(plot_width=100, plot_height=50)
        state = ViewerState()
        state.mobilogram_plot_width = 80
        state.show_mobilogram = True

        canvas = Image.new("RGBA", (300, 150), (0, 0, 0, 0))
        raster = np.zeros((50, 100), dtype=np.float32)

        result = renderer._draw_mobilogram_from_raster(
            canvas, state, raster, 100.0, 2000.0, 0.5, 1.5
        )
        assert result is not None
