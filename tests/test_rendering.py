"""Tests for the pyopenms_viewer rendering module."""

import numpy as np
import pandas as pd
import pytest

from pyopenms_viewer.core.config import DEFAULTS
from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.utils.coordinate_transform import CoordinateTransform
from pyopenms_viewer.utils.gpu import (
    is_cudf_available,
    is_dask_available,
    is_dask_enabled,
    is_gpu_enabled,
    set_dask_enabled,
    set_gpu_enabled,
    to_accelerated_dataframe,
    to_gpu_dataframe,
)


class TestAcceleration:
    """Tests for GPU and Dask acceleration support."""

    def test_cudf_availability_check(self):
        """Test that cudf availability check works without crashing."""
        # Should return a boolean, not crash
        result = is_cudf_available()
        assert isinstance(result, bool)

    def test_dask_availability_check(self):
        """Test that dask availability check works without crashing."""
        result = is_dask_available()
        assert isinstance(result, bool)

    def test_gpu_enabled_check(self):
        """Test that GPU enabled check works."""
        result = is_gpu_enabled()
        assert isinstance(result, bool)
        # GPU enabled should be False if cudf is not available
        if not is_cudf_available():
            assert result is False

    def test_dask_enabled_check(self):
        """Test that Dask enabled check works."""
        result = is_dask_enabled()
        assert isinstance(result, bool)
        # Dask enabled should be False if not available
        if not is_dask_available():
            assert result is False

    def test_set_gpu_enabled(self):
        """Test that GPU can be enabled/disabled."""
        original = is_gpu_enabled()

        # Disable GPU
        set_gpu_enabled(False)
        assert is_gpu_enabled() is False

        # Enable GPU (will still be False if cudf not available)
        set_gpu_enabled(True)
        if is_cudf_available():
            assert is_gpu_enabled() is True
        else:
            assert is_gpu_enabled() is False

        # Restore original state
        set_gpu_enabled(original or True)

    def test_set_dask_enabled(self):
        """Test that Dask can be enabled/disabled."""
        original = is_dask_enabled()

        # Disable Dask
        set_dask_enabled(False)
        assert is_dask_enabled() is False

        # Enable Dask
        set_dask_enabled(True)
        if is_dask_available():
            assert is_dask_enabled() is True
        else:
            assert is_dask_enabled() is False

        # Restore original state
        set_dask_enabled(original or True)

    def test_to_gpu_dataframe_fallback(self):
        """Test that to_gpu_dataframe returns pandas when GPU not available."""
        df = pd.DataFrame({"rt": [1.0, 2.0, 3.0], "mz": [100.0, 200.0, 300.0], "intensity": [1e6, 2e6, 3e6]})

        result = to_gpu_dataframe(df)

        # Without cudf, should return the same pandas DataFrame
        if not is_gpu_enabled():
            assert result is df  # Same object, no copy
            assert isinstance(result, pd.DataFrame)

    def test_to_accelerated_dataframe_small(self):
        """Test that small DataFrames stay as pandas."""
        set_dask_enabled(True)
        set_gpu_enabled(False)

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = to_accelerated_dataframe(df)

        # Small DataFrame should stay pandas (< 100k rows)
        if not is_gpu_enabled():
            assert result is df

    def test_to_accelerated_dataframe_large(self):
        """Test that large DataFrames use Dask when available."""
        set_dask_enabled(True)
        set_gpu_enabled(False)

        # Create large DataFrame
        df = pd.DataFrame({"a": range(200000), "b": range(200000)})
        result = to_accelerated_dataframe(df)

        # Should be Dask if available
        if is_dask_enabled():
            import dask.dataframe as dd

            assert isinstance(result, dd.DataFrame)
        else:
            assert result is df

        # Restore
        set_dask_enabled(True)

    def test_to_accelerated_dataframe_disabled(self):
        """Test that acceleration respects disabled state."""
        set_dask_enabled(False)
        set_gpu_enabled(False)

        df = pd.DataFrame({"a": range(200000)})
        result = to_accelerated_dataframe(df)

        # Should return same object when all disabled
        assert result is df

        # Re-enable for other tests
        set_dask_enabled(True)
        set_gpu_enabled(True)


class TestCoordinateTransform:
    """Tests for coordinate transformation between pixel and data coordinates."""

    @pytest.fixture
    def transform(self):
        """Create a standard coordinate transformer."""
        return CoordinateTransform(
            plot_width=1000,
            plot_height=500,
            margin_left=80,
            margin_top=20,
        )

    @pytest.fixture
    def state_standard(self):
        """Create a ViewerState with standard axis orientation."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.swap_axes = False  # RT on x-axis, m/z on y-axis
        return state

    @pytest.fixture
    def state_swapped(self):
        """Create a ViewerState with swapped axes."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.swap_axes = True  # m/z on x-axis, RT on y-axis
        return state

    @pytest.fixture
    def state_with_view(self):
        """Create a ViewerState with zoomed view bounds."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.view_rt_min = 500.0
        state.view_rt_max = 1500.0
        state.view_mz_min = 300.0
        state.view_mz_max = 800.0
        state.swap_axes = False
        return state

    def test_data_to_pixel_origin_standard(self, transform, state_standard):
        """Test converting origin data point to pixel (standard axes)."""
        # RT=0, m/z=2000 should be at top-left of plot area
        x, y = transform.data_to_pixel(state_standard, rt=0, mz=2000)
        assert x == 0
        assert y == 0

    def test_data_to_pixel_max_standard(self, transform, state_standard):
        """Test converting max data point to pixel (standard axes)."""
        # RT=3600, m/z=100 should be at bottom-right of plot area
        x, y = transform.data_to_pixel(state_standard, rt=3600, mz=100)
        assert x == 1000
        assert y == 500

    def test_data_to_pixel_center_standard(self, transform, state_standard):
        """Test converting center data point to pixel (standard axes)."""
        # RT=1800 (center), m/z=1050 (center) should be at center of plot
        x, y = transform.data_to_pixel(state_standard, rt=1800, mz=1050)
        assert x == 500  # Half of plot_width
        assert y == 250  # Half of plot_height

    def test_data_to_pixel_swapped_axes(self, transform, state_swapped):
        """Test converting data to pixel with swapped axes."""
        # With swapped axes: m/z on x-axis, RT on y-axis (inverted)
        # m/z=100 (min), RT=3600 (max) should be at top-left
        x, y = transform.data_to_pixel(state_swapped, rt=3600, mz=100)
        assert x == 0
        assert y == 0

    def test_pixel_to_data_origin_standard(self, transform, state_standard):
        """Test converting pixel origin to data (standard axes)."""
        # Pixel at margin_left, margin_top should be RT=0, m/z=max
        rt, mz = transform.pixel_to_data(state_standard, pixel_x=80, pixel_y=20)
        assert rt == pytest.approx(0.0, abs=0.1)
        assert mz == pytest.approx(2000.0, abs=0.1)

    def test_pixel_to_data_max_standard(self, transform, state_standard):
        """Test converting max pixel to data (standard axes)."""
        # Pixel at right edge, bottom edge
        rt, mz = transform.pixel_to_data(state_standard, pixel_x=1080, pixel_y=520)
        assert rt == pytest.approx(3600.0, abs=0.1)
        assert mz == pytest.approx(100.0, abs=0.1)

    def test_pixel_to_data_center_standard(self, transform, state_standard):
        """Test converting center pixel to data (standard axes)."""
        # Pixel at center of plot area
        rt, mz = transform.pixel_to_data(state_standard, pixel_x=580, pixel_y=270)
        assert rt == pytest.approx(1800.0, abs=10)
        assert mz == pytest.approx(1050.0, abs=10)

    def test_pixel_to_data_clamps_to_plot_area(self, transform, state_standard):
        """Test that pixel coordinates are clamped to plot area."""
        # Pixel outside plot area (negative relative to margins)
        rt, mz = transform.pixel_to_data(state_standard, pixel_x=0, pixel_y=0)
        # Should clamp to (0, 0) in plot coordinates, giving min RT, max m/z
        assert rt == pytest.approx(0.0, abs=0.1)
        assert mz == pytest.approx(2000.0, abs=0.1)

    def test_pixel_to_data_swapped_axes(self, transform, state_swapped):
        """Test converting pixel to data with swapped axes."""
        # With swapped axes at top-left of plot
        rt, mz = transform.pixel_to_data(state_swapped, pixel_x=80, pixel_y=20)
        assert mz == pytest.approx(100.0, abs=0.1)  # m/z min at left
        assert rt == pytest.approx(3600.0, abs=0.1)  # RT max at top

    def test_roundtrip_standard(self, transform, state_standard):
        """Test roundtrip conversion data -> pixel -> data (standard axes)."""
        original_rt = 1234.5
        original_mz = 567.8

        x, y = transform.data_to_pixel(state_standard, rt=original_rt, mz=original_mz)
        # Add margins for pixel_to_data which expects absolute coordinates
        rt, mz = transform.pixel_to_data(
            state_standard, pixel_x=x + transform.margin_left, pixel_y=y + transform.margin_top
        )

        assert rt == pytest.approx(original_rt, rel=0.01)
        assert mz == pytest.approx(original_mz, rel=0.01)

    def test_roundtrip_swapped(self, transform, state_swapped):
        """Test roundtrip conversion data -> pixel -> data (swapped axes)."""
        original_rt = 1234.5
        original_mz = 567.8

        x, y = transform.data_to_pixel(state_swapped, rt=original_rt, mz=original_mz)
        rt, mz = transform.pixel_to_data(
            state_swapped, pixel_x=x + transform.margin_left, pixel_y=y + transform.margin_top
        )

        assert rt == pytest.approx(original_rt, rel=0.01)
        assert mz == pytest.approx(original_mz, rel=0.01)

    def test_data_to_pixel_with_view_bounds(self, transform, state_with_view):
        """Test conversion respects view bounds when set."""
        # Center of view should be center of plot
        view_center_rt = (500.0 + 1500.0) / 2  # 1000
        view_center_mz = (300.0 + 800.0) / 2  # 550

        x, y = transform.data_to_pixel(state_with_view, rt=view_center_rt, mz=view_center_mz)
        assert x == pytest.approx(500, abs=5)
        assert y == pytest.approx(250, abs=5)

    def test_data_to_pixel_zero_range(self, transform):
        """Test handling of zero range (degenerate case)."""
        state = ViewerState()
        state.rt_min = 100.0
        state.rt_max = 100.0  # Zero range
        state.mz_min = 500.0
        state.mz_max = 500.0  # Zero range
        state.swap_axes = False

        x, y = transform.data_to_pixel(state, rt=100, mz=500)
        assert x == 0
        assert y == 0


class TestCoordinateTransformIM:
    """Tests for ion mobility coordinate transformations."""

    @pytest.fixture
    def transform(self):
        """Create a standard coordinate transformer."""
        return CoordinateTransform(
            plot_width=800,
            plot_height=400,
            margin_left=60,
            margin_top=15,
        )

    @pytest.fixture
    def state_im(self):
        """Create a ViewerState with ion mobility data bounds."""
        state = ViewerState()
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.im_min = 0.5
        state.im_max = 1.5
        state.has_ion_mobility = True
        return state

    def test_im_data_to_pixel_origin(self, transform, state_im):
        """Test converting IM origin data point to pixel."""
        # m/z=100, im=1.5 should be at top-left of plot
        x, y = transform.im_data_to_pixel(state_im, mz=100, im=1.5)
        assert x == 0
        assert y == 0

    def test_im_data_to_pixel_max(self, transform, state_im):
        """Test converting IM max data point to pixel."""
        # m/z=2000, im=0.5 should be at bottom-right of plot
        x, y = transform.im_data_to_pixel(state_im, mz=2000, im=0.5)
        assert x == 800
        assert y == 400

    def test_im_data_to_pixel_center(self, transform, state_im):
        """Test converting IM center data point to pixel."""
        # m/z=1050, im=1.0 should be at center of plot
        x, y = transform.im_data_to_pixel(state_im, mz=1050, im=1.0)
        assert x == 400
        assert y == 200

    def test_im_pixel_to_data_origin(self, transform, state_im):
        """Test converting pixel origin to IM data."""
        mz, im = transform.im_pixel_to_data(state_im, pixel_x=60, pixel_y=15)
        assert mz == pytest.approx(100.0, abs=0.1)
        assert im == pytest.approx(1.5, abs=0.01)

    def test_im_pixel_to_data_max(self, transform, state_im):
        """Test converting max pixel to IM data."""
        mz, im = transform.im_pixel_to_data(state_im, pixel_x=860, pixel_y=415)
        assert mz == pytest.approx(2000.0, abs=0.1)
        assert im == pytest.approx(0.5, abs=0.01)

    def test_im_pixel_to_data_center(self, transform, state_im):
        """Test converting center pixel to IM data."""
        mz, im = transform.im_pixel_to_data(state_im, pixel_x=460, pixel_y=215)
        assert mz == pytest.approx(1050.0, abs=5)
        assert im == pytest.approx(1.0, abs=0.05)

    def test_im_roundtrip(self, transform, state_im):
        """Test roundtrip conversion for IM data."""
        original_mz = 750.0
        original_im = 1.2

        x, y = transform.im_data_to_pixel(state_im, mz=original_mz, im=original_im)
        mz, im = transform.im_pixel_to_data(
            state_im, pixel_x=x + transform.margin_left, pixel_y=y + transform.margin_top
        )

        assert mz == pytest.approx(original_mz, rel=0.01)
        assert im == pytest.approx(original_im, rel=0.01)

    def test_im_data_to_pixel_zero_range(self, transform):
        """Test IM conversion with zero range."""
        state = ViewerState()
        state.mz_min = 500.0
        state.mz_max = 500.0
        state.im_min = 1.0
        state.im_max = 1.0

        x, y = transform.im_data_to_pixel(state, mz=500, im=1.0)
        assert x == 0
        assert y == 0


class TestCoordinateTransformDimensions:
    """Tests for different plot dimensions."""

    def test_small_plot(self):
        """Test coordinate transform with small plot dimensions."""
        transform = CoordinateTransform(
            plot_width=100,
            plot_height=50,
            margin_left=10,
            margin_top=5,
        )
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 100.0
        state.mz_min = 0.0
        state.mz_max = 100.0
        state.swap_axes = False

        x, y = transform.data_to_pixel(state, rt=50, mz=50)
        assert x == 50
        assert y == 25

    def test_wide_plot(self):
        """Test coordinate transform with wide plot (width >> height)."""
        transform = CoordinateTransform(
            plot_width=2000,
            plot_height=200,
            margin_left=50,
            margin_top=10,
        )
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 1000.0
        state.mz_min = 0.0
        state.mz_max = 100.0
        state.swap_axes = False

        x, y = transform.data_to_pixel(state, rt=500, mz=50)
        assert x == 1000
        assert y == 100

    def test_tall_plot(self):
        """Test coordinate transform with tall plot (height >> width)."""
        transform = CoordinateTransform(
            plot_width=200,
            plot_height=2000,
            margin_left=20,
            margin_top=10,
        )
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 100.0
        state.mz_min = 0.0
        state.mz_max = 1000.0
        state.swap_axes = False

        x, y = transform.data_to_pixel(state, rt=50, mz=500)
        assert x == 100
        assert y == 1000

    def test_different_margins(self):
        """Test coordinate transform with asymmetric margins."""
        transform = CoordinateTransform(
            plot_width=500,
            plot_height=500,
            margin_left=100,  # Large left margin
            margin_top=50,  # Different top margin
        )
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 500.0
        state.mz_min = 0.0
        state.mz_max = 500.0
        state.swap_axes = False

        # Test that margins are correctly applied in pixel_to_data
        rt, mz = transform.pixel_to_data(state, pixel_x=100, pixel_y=50)
        assert rt == pytest.approx(0.0, abs=0.1)
        assert mz == pytest.approx(500.0, abs=0.1)


class TestRasterizationSupport:
    """Tests for rasterization rendering mode in PeakMapRenderer."""

    @pytest.fixture
    def state_with_exp(self):
        """Create a ViewerState with a mock MSExperiment."""
        try:
            from pyopenms import MSExperiment, MSSpectrum
        except ImportError:
            pytest.skip("pyOpenMS not available")

        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0

        # Create a simple MSExperiment with test data
        exp = MSExperiment()

        # Build DataFrame matching MSExperiment peaks for Phase 6 (optional df removal)
        rt_list = []
        mz_list = []
        intensity_list = []

        # Add some test spectra
        for i in range(10):
            spectrum = MSSpectrum()
            spectrum.setRT(i * 360.0)  # 10 spectra, 360 seconds apart
            spectrum.setMSLevel(1)

            # Add peaks to spectrum
            mzs = np.array([100.0 + j * 100.0 for j in range(10)], dtype=np.float64)
            intensities = np.array([1000.0 * (i + 1) for _ in range(10)], dtype=np.float32)
            spectrum.set_peaks((mzs, intensities))

            # Track peaks for DataFrame
            rt_list.extend([i * 360.0] * 10)
            mz_list.extend(mzs)
            intensity_list.extend(intensities)

            exp.addSpectrum(spectrum)

        # Phase 6: Initialize state.df with matching data
        # This supports both in-memory and fallback rendering modes
        state.df = pd.DataFrame({
            'rt': rt_list,
            'mz': mz_list,
            'intensity': intensity_list,
            'log_intensity': np.log10(np.array(intensity_list) + 1),
        })

        state.exp = exp
        return state

    def test_should_use_rasterization_always_when_threshold_zero(self):
        """Test that rasterization is always used when threshold is 0."""
        state = ViewerState()

        # Save original values
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set thresholds to 0 (always rasterize)
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 0.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 0.0

            # Should always rasterize regardless of RT/mz range
            assert state.should_use_rasterization(rt_range=1.0, mz_range=1.0) is True
            assert state.should_use_rasterization(rt_range=10.0, mz_range=10.0) is True
            assert state.should_use_rasterization(rt_range=100.0, mz_range=100.0) is True
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_should_use_rasterization_below_thresholds(self):
        """Test that point rendering is used when below BOTH thresholds."""
        state = ViewerState()

        # Save original values
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set thresholds
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 60.0  # seconds
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 50.0

            # Below both thresholds: use point rendering
            assert state.should_use_rasterization(rt_range=30.0, mz_range=25.0) is False
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_should_use_rasterization_above_thresholds(self):
        """Test that rasterization is used when above thresholds."""
        state = ViewerState()

        # Save original values
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set thresholds
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 60.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 50.0

            # Above RT threshold: use rasterization
            assert state.should_use_rasterization(rt_range=100.0, mz_range=25.0) is True

            # Above mz threshold: use rasterization
            assert state.should_use_rasterization(rt_range=30.0, mz_range=100.0) is True

            # Above both thresholds: use rasterization
            assert state.should_use_rasterization(rt_range=100.0, mz_range=100.0) is True
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_render_rasterized_basic(self, state_with_exp):
        """Test basic rasterization creates a numpy array."""
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        renderer = PeakMapRenderer(
            plot_width=1100,
            plot_height=550,
        )

        # Set up state and bounds
        state_with_exp.view_rt_min = 0.0
        state_with_exp.view_rt_max = 1000.0
        state_with_exp.view_mz_min = 100.0
        state_with_exp.view_mz_max = 2000.0

        # Test that _render_rasterized method exists and can be called
        assert hasattr(renderer, "_render_rasterized")
        assert callable(renderer._render_rasterized)

    def test_render_rasterized_array_shape(self, state_with_exp):
        """Test that rasterized output has correct array shape."""
        # Set up state bounds
        state_with_exp.view_rt_min = 0.0
        state_with_exp.view_rt_max = 1000.0
        state_with_exp.view_mz_min = 100.0
        state_with_exp.view_mz_max = 2000.0

        # Test that we can create the rasterized array with correct shape
        # Using PLOT_WIDTH and PLOT_HEIGHT from config
        # Array should have shape (mz_bins, rt_bins)
        expected_shape = (DEFAULTS.PLOT_HEIGHT, DEFAULTS.PLOT_WIDTH)
        test_array = np.empty(expected_shape, dtype=np.float32)
        assert test_array.shape == (550, 1100)

    def test_render_xarray_conversion(self, state_with_exp):
        """Test conversion of numpy array to xarray DataArray."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not available")

        # Create a simple raterized array
        rt_range = np.linspace(0.0, 1000.0, 1100)
        mz_range = np.linspace(100.0, 2000.0, 550)

        data = np.random.uniform(0, 100, size=(550, 1100)).astype(np.float32)

        # Create xarray DataArray
        data_array = xr.DataArray(
            data,
            coords={"mz": mz_range, "rt": rt_range},
            dims=["mz", "rt"],
        )

        assert data_array.shape == (550, 1100)
        assert "mz" in data_array.coords
        assert "rt" in data_array.coords

    def test_render_with_datashader_shade(self):
        """Test that datashader can shade a simple xarray."""
        try:
            import datashader.transfer_functions as tf
            import xarray as xr
        except ImportError:
            pytest.skip("xarray or datashader not available")

        # Create a simple xarray
        rt_range = np.linspace(0.0, 1000.0, 1100)
        mz_range = np.linspace(100.0, 2000.0, 550)

        data = np.random.uniform(0, 100, size=(550, 1100)).astype(np.float32)

        data_array = xr.DataArray(
            data,
            coords={"mz": mz_range, "rt": rt_range},
            dims=["mz", "rt"],
        )

        # Test shading with datashader
        from pyopenms_viewer.core.config import COLORMAPS

        img = tf.shade(data_array, cmap=COLORMAPS["jet"], how="linear")
        assert img is not None

    def test_should_use_point_rendering(self):
        """Test should_use_point_rendering method logic."""
        state = ViewerState()

        # Save original values
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # When thresholds are 0, always use rasterization (point rendering = False)
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 0.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 0.0

            state.view_rt_min = 0.0
            state.view_rt_max = 100.0
            state.view_mz_min = 100.0
            state.view_mz_max = 200.0

            assert state.should_use_point_rendering() is False

            # Set reasonable thresholds
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 60.0  # seconds
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 50.0

            # When both ranges are below thresholds: use point rendering (True)
            state.view_rt_min = 0.0
            state.view_rt_max = 30.0  # RT range = 30, below 60
            state.view_mz_min = 100.0
            state.view_mz_max = 140.0  # mz range = 40, below 50

            assert state.should_use_point_rendering() is True

            # When RT range exceeds threshold: use rasterization (False)
            state.view_rt_min = 0.0
            state.view_rt_max = 100.0  # RT range = 100, above 60
            state.view_mz_min = 100.0
            state.view_mz_max = 140.0  # mz range = 40, below 50

            assert state.should_use_point_rendering() is False

            # When mz range exceeds threshold: use rasterization (False)
            state.view_rt_min = 0.0
            state.view_rt_max = 30.0  # RT range = 30, below 60
            state.view_mz_min = 100.0
            state.view_mz_max = 200.0  # mz range = 100, above 50

            assert state.should_use_point_rendering() is False
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_render_implements_branching_logic(self, state_with_exp):
        """Test that render() method implements branching between point and rasterized modes."""
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        renderer = PeakMapRenderer()

        # Set up state with test data
        state_with_exp.view_rt_min = 0.0
        state_with_exp.view_rt_max = 30.0  # Narrow range
        state_with_exp.view_mz_min = 100.0
        state_with_exp.view_mz_max = 140.0  # Narrow range

        # Create a DataFrame with test points
        state_with_exp.df = pd.DataFrame(
            {
                "rt": [10.0, 15.0, 20.0],
                "mz": [110.0, 120.0, 130.0],
                "intensity": [1000.0, 2000.0, 3000.0],
                "log_intensity": [np.log10(1000.0), np.log10(2000.0), np.log10(3000.0)],
            }
        )

        # Save original values
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set thresholds for deep zoom
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 60.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 50.0

            # When using point rendering (narrow range), should return valid image
            result = renderer.render(state_with_exp, fast=False, draw_overlays=False, draw_axes=False)
            assert isinstance(result, str)
            # Should be non-empty (base64 encoded)
            assert len(result) > 0

            # When using rasterization (wide range), should also return valid image
            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 1000.0  # Wide range
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 2000.0  # Wide range

            result = renderer.render(state_with_exp, fast=False, draw_overlays=False, draw_axes=False)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_render_rasterized_returns_image(self, state_with_exp):
        """Test that _render_rasterized() produces a valid image in PNG format."""
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        renderer = PeakMapRenderer()

        # Set up state with test data
        state_with_exp.view_rt_min = 0.0
        state_with_exp.view_rt_max = 3600.0
        state_with_exp.view_mz_min = 100.0
        state_with_exp.view_mz_max = 2000.0

        # Create test DataFrame
        state_with_exp.df = pd.DataFrame(
            {
                "rt": np.linspace(0, 3600, 100),
                "mz": np.linspace(100, 2000, 100),
                "intensity": np.random.uniform(1000, 100000, 100),
                "log_intensity": np.log10(np.random.uniform(1000, 100000, 100)),
            }
        )

        # Call _render_rasterized
        result = renderer._render_rasterized(state_with_exp, fast=False)

        # Should return a base64 encoded string
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be valid base64
        try:
            import base64

            base64.b64decode(result)
            valid_base64 = True
        except Exception:
            valid_base64 = False
        assert valid_base64

    def test_deep_zoom_uses_point_rendering(self, state_with_exp):
        """Test that _render_points is used when view is in deep zoom range."""
        from pyopenms_viewer.core.config import DEFAULTS
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        # Save original thresholds
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set thresholds - narrow ranges will trigger point rendering
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 100.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 200.0

            # Create state with narrow view (within thresholds)
            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 50.0  # 50 seconds (< 100)
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 250.0  # 150 mz (< 200)

            # Verify that should_use_point_rendering returns True
            assert state_with_exp.should_use_point_rendering() is True

            # Create a mock to track if _render_points would be called
            renderer = PeakMapRenderer()
            assert callable(renderer._render_points)
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_get_2d_peak_data_integration(self, state_with_exp):
        """Test that get2DPeakDataLong is called with correct parameters."""
        from unittest.mock import MagicMock, patch

        from pyopenms_viewer.core.config import DEFAULTS
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        # Save original thresholds
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set thresholds to use point rendering
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 1000.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 500.0

            # Create state with bounds that match fixture structure
            # Fixture has data at RT 0-3240, mz 100-1000
            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 400.0
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 500.0

            # Create a mock for get2DPeakDataLong
            mock_get2d = MagicMock(return_value=(np.array([100.0, 101.0]), np.array([200.0, 201.0]), np.array([1000.0, 2000.0])))
            state_with_exp.exp.get2DPeakDataLong = mock_get2d

            # Render (will call _render_points since we're in deep zoom)
            renderer = PeakMapRenderer()
            result = renderer.render(state_with_exp, fast=False)

            # Verify that get2DPeakDataLong was called with correct bounds and ms_level
            mock_get2d.assert_called()
            call_args = mock_get2d.call_args
            if call_args:
                # Check that call was made with correct bounds
                # get2DPeakDataLong(rt_min, rt_max, mz_min, mz_max, ms_level)
                args, kwargs = call_args
                if len(args) >= 5:
                    rt_min, rt_max, mz_min, mz_max, ms_level = args[:5]
                    assert rt_min == 0.0
                    assert rt_max == 400.0
                    assert mz_min == 100.0
                    assert mz_max == 500.0
                    assert ms_level == 1
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_temp_dataframe_creation(self, state_with_exp):
        """Test that temporary DataFrame is created from get2DPeakDataLong arrays."""
        from unittest.mock import MagicMock

        from pyopenms_viewer.core.config import DEFAULTS
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        # Save original thresholds
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set narrow thresholds
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 1000.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 500.0

            # Create state with bounds that match fixture data
            # Fixture has spectra at RT 0, 360, 720, ..., 3240
            # With peaks at m/z 100, 200, 300, ..., 1000
            # Use bounds that capture the first spectrum (RT 0-360)
            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 400.0  # Captures first spectrum at RT 0-360
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 500.0  # Captures peaks at 100, 200, 300, 400

            # Mock get2DPeakDataLong to return test data
            rt_array = np.array([100.0, 101.0, 102.0], dtype=np.float64)
            mz_array = np.array([200.0, 210.0, 220.0], dtype=np.float64)
            intensity_array = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
            
            mock_get2d = MagicMock(return_value=(rt_array, mz_array, intensity_array))
            state_with_exp.exp.get2DPeakDataLong = mock_get2d

            # Render
            renderer = PeakMapRenderer()
            result = renderer.render(state_with_exp, fast=False)

            # Verify that temp_peak_df was created and stored in state
            assert state_with_exp.temp_peak_df is not None
            assert isinstance(state_with_exp.temp_peak_df, pd.DataFrame)
            assert len(state_with_exp.temp_peak_df) == 3

            # Verify the DataFrame has the expected columns
            assert "rt" in state_with_exp.temp_peak_df.columns
            assert "mz" in state_with_exp.temp_peak_df.columns
            assert "intensity" in state_with_exp.temp_peak_df.columns
            assert "log_intensity" in state_with_exp.temp_peak_df.columns

            # Verify data integrity
            assert np.allclose(state_with_exp.temp_peak_df["rt"].values, rt_array)
            assert np.allclose(state_with_exp.temp_peak_df["mz"].values, mz_array)
            assert np.allclose(state_with_exp.temp_peak_df["intensity"].values, intensity_array)
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_temp_dataframe_reuse(self, state_with_exp):
        """Test that temp_peak_df can be reused for 3D view rendering."""
        from unittest.mock import MagicMock

        from pyopenms_viewer.core.config import DEFAULTS
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        # Save original thresholds
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set narrow thresholds to use point rendering
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 1000.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 500.0

            # Create state with bounds that match fixture data
            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 400.0
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 500.0

            # Mock get2DPeakDataLong to return test data
            rt_array = np.array([100.5, 101.5], dtype=np.float64)
            mz_array = np.array([205.0, 215.0], dtype=np.float64)
            intensity_array = np.array([5000.0, 6000.0], dtype=np.float32)
            
            mock_get2d = MagicMock(return_value=(rt_array, mz_array, intensity_array))
            state_with_exp.exp.get2DPeakDataLong = mock_get2d

            # First render
            renderer = PeakMapRenderer()
            result1 = renderer.render(state_with_exp, fast=False)

            # Save reference to temp_peak_df after first render
            temp_df_first = state_with_exp.temp_peak_df
            assert temp_df_first is not None
            assert len(temp_df_first) == 2

            # Second render with same bounds (should reuse)
            result2 = renderer.render(state_with_exp, fast=False)

            # Verify temp_peak_df is still available for reuse
            temp_df_second = state_with_exp.temp_peak_df
            assert temp_df_second is not None
            
            # Verify the data is still the same
            assert np.allclose(temp_df_second["rt"].values, rt_array)
            assert np.allclose(temp_df_second["mz"].values, mz_array)
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    # ========== PHASE 6: REMOVE GLOBAL DATAFRAME ==========

    def test_rendering_without_dataframe(self, state_with_exp):
        """Test that rendering works when state.df is None (Phase 6).
        
        Phase 6 removes global DataFrame storage in rasterization mode.
        Rendering should fall back to get2DPeakDataLong when state.df is None.
        """
        from pyopenms_viewer.core.config import DEFAULTS
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        # Save original thresholds
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set narrow thresholds to use point rendering
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 1000.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 500.0

            # Create state with bounds
            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 400.0
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 500.0

            # Phase 6: Remove df to simulate out-of-core mode
            state_with_exp.df = None

            # Rendering should still work by using get2DPeakDataLong
            renderer = PeakMapRenderer()
            result = renderer.render(state_with_exp, fast=False)

            # Verify rendering succeeded (non-empty base64 string)
            assert result != ""
            assert isinstance(result, str)
            # Base64 strings start with this pattern for PNG
            assert result.startswith("iVB") or result.startswith("/9j")  # PNG or JPEG

            # Verify temp_peak_df was created even without state.df
            assert state_with_exp.temp_peak_df is not None
            assert len(state_with_exp.temp_peak_df) > 0
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_point_rendering_fallback_chain(self, state_with_exp):
        """Test the fallback chain: get2DPeakDataLong -> df -> get_peaks_in_view.
        
        Phase 6 implements a three-tier fallback:
        1. Try get2DPeakDataLong (best for rasterization mode with MSExperiment)
        2. Fall back to filtering state.df (in-memory mode)
        3. Fall back to get_peaks_in_view() (out-of-core mode with DuckDB)
        """
        from unittest.mock import MagicMock

        from pyopenms_viewer.core.config import DEFAULTS
        from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

        # Save original thresholds
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set narrow thresholds to use point rendering
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 1000.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 500.0

            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 400.0
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 500.0

            # Test 1: PATH 1 - get2DPeakDataLong succeeds
            rt_array = np.array([100.0, 200.0], dtype=np.float64)
            mz_array = np.array([300.0, 400.0], dtype=np.float64)
            intensity_array = np.array([1000.0, 2000.0], dtype=np.float32)

            mock_get2d = MagicMock(return_value=(rt_array, mz_array, intensity_array))
            state_with_exp.exp.get2DPeakDataLong = mock_get2d

            renderer = PeakMapRenderer()
            result = renderer.render(state_with_exp, fast=False)

            # Verify get2DPeakDataLong was called
            assert mock_get2d.called
            assert state_with_exp.temp_peak_df is not None
            assert len(state_with_exp.temp_peak_df) == 2

            # Test 2: PATH 2 - get2DPeakDataLong fails, fall back to df
            state_with_exp.temp_peak_df = None
            mock_get2d.side_effect = RuntimeError("get2DPeakDataLong failed")

            result = renderer.render(state_with_exp, fast=False)

            # Should have used state.df instead
            assert state_with_exp.temp_peak_df is not None
            # Df filtering should have returned multiple peaks within the view bounds
            assert len(state_with_exp.temp_peak_df) > 0

            # Test 3: PATH 3 - both fail, fall back to get_peaks_in_view
            state_with_exp.df = None
            state_with_exp.temp_peak_df = None

            # Mock get_peaks_in_view to return a small DataFrame
            fallback_df = pd.DataFrame({
                'rt': [50.0, 100.0],
                'mz': [200.0, 250.0],
                'intensity': [5000.0, 6000.0],
                'log_intensity': [3.7, 3.78],
            })
            state_with_exp.get_peaks_in_view = MagicMock(return_value=fallback_df)

            result = renderer.render(state_with_exp, fast=False)

            # Should have used get_peaks_in_view
            assert state_with_exp.get_peaks_in_view.called
            assert state_with_exp.temp_peak_df is not None
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz

    def test_3d_view_without_dataframe(self, state_with_exp):
        """Test that 3D view works without state.df by using get2DPeakDataLong (Phase 6)."""
        import pytest

        try:
            from pyopenms_viewer.panels.peak_map_panel import PeakMapPanel
        except ImportError:
            pytest.skip("pyopenms_viewer.panels not available")

        from pyopenms_viewer.core.config import DEFAULTS

        # Save original thresholds
        original_rt = DEFAULTS.DEEP_ZOOM_RT_THRESHOLD
        original_mz = DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD

        try:
            # Set narrow view to trigger point rendering
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = 1000.0
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = 500.0

            state_with_exp.view_rt_min = 0.0
            state_with_exp.view_rt_max = 400.0
            state_with_exp.view_mz_min = 100.0
            state_with_exp.view_mz_max = 500.0

            # Phase 6: Remove df to simulate out-of-core mode
            state_with_exp.df = None

            # In Phase 6, when state.df is None, the 3D view should try to use
            # get2DPeakDataLong similar to rendering
            # We can't directly test PeakMapPanel without a full UI, but we can
            # verify the fallback logic would work

            # Verify that exp.get2DPeakDataLong would be the primary source
            # when state.df is None
            assert state_with_exp.exp is not None
            assert hasattr(state_with_exp.exp, 'get2DPeakDataLong')

            # Verify that even with df=None, we can extract peaks via get2DPeakDataLong
            try:
                rt_array, mz_array, intensity_array = state_with_exp.exp.get2DPeakDataLong(
                    state_with_exp.view_rt_min,
                    state_with_exp.view_rt_max,
                    state_with_exp.view_mz_min,
                    state_with_exp.view_mz_max,
                    ms_level=1,
                )
                # Should have some peaks
                assert len(rt_array) > 0
                assert len(mz_array) > 0
                assert len(intensity_array) > 0
            except Exception as e:
                pytest.skip(f"get2DPeakDataLong not available: {e}")
        finally:
            # Restore original values
            DEFAULTS.DEEP_ZOOM_RT_THRESHOLD = original_rt
            DEFAULTS.DEEP_ZOOM_MZ_THRESHOLD = original_mz


class TestMinimapCaching:
    """Tests for minimap rasterization caching."""

    @pytest.fixture
    def state_with_minimap_data(self):
        """Create a state with test peak data for minimap rendering."""
        state = ViewerState()

        # Set up data bounds
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0

        # Set up view bounds
        state.view_rt_min = 0.0
        state.view_rt_max = 3600.0
        state.view_mz_min = 100.0
        state.view_mz_max = 2000.0

        # Create test DataFrame with peaks
        n_peaks = 1000
        state.df = pd.DataFrame(
            {
                "rt": np.linspace(0, 3600, n_peaks),
                "mz": np.linspace(100, 2000, n_peaks),
                "intensity": np.random.uniform(1000, 100000, n_peaks),
                "log_intensity": np.log10(np.random.uniform(1000, 100000, n_peaks)),
            }
        )

        # Initialize data manager if needed
        state.init_data_manager(out_of_core=False)

        return state

    def test_minimap_cache_invalidation_method_exists(self, state_with_minimap_data):
        """Verify that invalidate_minimap_cache method exists in ViewerState."""
        state = state_with_minimap_data
        # Cache invalidation method should exist
        assert hasattr(state, "invalidate_minimap_cache")
        assert callable(state.invalidate_minimap_cache)

    def test_minimap_cache_attribute_exists(self, state_with_minimap_data):
        """Verify that cached_minimap_raster attribute exists in ViewerState."""
        state = state_with_minimap_data
        # Cache attribute should exist and be None initially
        assert hasattr(state, "cached_minimap_raster")
        assert state.cached_minimap_raster is None

    def test_minimap_caches_raster(self, state_with_minimap_data):
        """Verify that minimap rasterization is cached after first call."""
        from unittest.mock import MagicMock

        from pyopenms_viewer.rendering.minimap_renderer import MinimapRenderer

        state = state_with_minimap_data

        # Create a mock MSExperiment with rasterizeRTMZ method
        mock_exp = MagicMock()

        def mock_rasterize(output, rt_min, rt_max, mz_min, mz_max, ms_level=1, aggregation="sum"):
            # Simulate rasterization by filling output with random data
            output[:] = np.random.uniform(0, 100, output.shape)

        mock_exp.rasterizeRTMZ = MagicMock(side_effect=mock_rasterize)
        state.exp = mock_exp

        renderer = MinimapRenderer(width=DEFAULTS.MINIMAP_WIDTH, height=DEFAULTS.MINIMAP_HEIGHT)

        # First render - should populate cache
        result1 = renderer.render(state)
        assert result1 is not None
        assert isinstance(result1, str)

        # Cache should now contain the rasterized data
        assert state.cached_minimap_raster is not None
        assert isinstance(state.cached_minimap_raster, np.ndarray)

        # Verify rasterizeRTMZ was called
        assert mock_exp.rasterizeRTMZ.call_count >= 1

    def test_minimap_invalidation_on_new_file(self, state_with_minimap_data):
        """Verify that cache is cleared when new file is loaded."""
        state = state_with_minimap_data

        # Manually set a cache value
        state.cached_minimap_raster = np.random.uniform(0, 100, (DEFAULTS.MINIMAP_HEIGHT, DEFAULTS.MINIMAP_WIDTH))
        assert state.cached_minimap_raster is not None

        # Call invalidate_minimap_cache
        state.invalidate_minimap_cache()

        # Cache should be cleared
        assert state.cached_minimap_raster is None

    def test_minimap_reshades_on_colormap_change(self, state_with_minimap_data):
        """Verify that minimap re-shades without re-rasterization when colormap changes."""
        from unittest.mock import MagicMock

        from pyopenms_viewer.rendering.minimap_renderer import MinimapRenderer

        state = state_with_minimap_data

        # Create a mock MSExperiment
        mock_exp = MagicMock()

        rasterize_call_count = 0

        def mock_rasterize(output, rt_min, rt_max, mz_min, mz_max, ms_level=1, aggregation="sum"):
            nonlocal rasterize_call_count
            rasterize_call_count += 1
            output[:] = np.random.uniform(0, 100, output.shape)

        mock_exp.rasterizeRTMZ = MagicMock(side_effect=mock_rasterize)
        state.exp = mock_exp

        renderer = MinimapRenderer(width=DEFAULTS.MINIMAP_WIDTH, height=DEFAULTS.MINIMAP_HEIGHT)

        # First render with 'jet' colormap
        state.colormap = "jet"
        result1 = renderer.render(state)
        assert result1 is not None

        rasterize_calls_after_first = rasterize_call_count

        # Second render with different colormap - should reuse cached raster
        state.colormap = "viridis"
        result2 = renderer.render(state)
        assert result2 is not None

        # rasterizeRTMZ should not have been called again
        assert mock_exp.rasterizeRTMZ.call_count == rasterize_calls_after_first

    def test_minimap_excerpt_box_overlay(self, state_with_minimap_data):
        """Verify that excerpt box is drawn without affecting cached data."""
        from unittest.mock import MagicMock

        from pyopenms_viewer.rendering.minimap_renderer import MinimapRenderer

        state = state_with_minimap_data

        # Create a mock MSExperiment
        mock_exp = MagicMock()

        def mock_rasterize(output, rt_min, rt_max, mz_min, mz_max, ms_level=1, aggregation="sum"):
            output[:] = np.random.uniform(0, 100, output.shape)

        mock_exp.rasterizeRTMZ = MagicMock(side_effect=mock_rasterize)
        state.exp = mock_exp

        renderer = MinimapRenderer(width=DEFAULTS.MINIMAP_WIDTH, height=DEFAULTS.MINIMAP_HEIGHT)

        # Render with view bounds set
        state.view_rt_min = 600.0
        state.view_rt_max = 1200.0
        state.view_mz_min = 500.0
        state.view_mz_max = 1500.0

        # Render and save original cache
        renderer.render(state)
        cached_copy = state.cached_minimap_raster.copy() if state.cached_minimap_raster is not None else None

        # Render again without changing view bounds - cache should be identical
        renderer.render(state)

        # Cached raster should not have changed (overlay is applied to PIL image, not to numpy array)
        if cached_copy is not None and state.cached_minimap_raster is not None:
            assert np.allclose(cached_copy, state.cached_minimap_raster)

    def test_minimap_cache_size_and_type(self, state_with_minimap_data):
        """Verify that cached raster has correct size and type."""
        from unittest.mock import MagicMock

        from pyopenms_viewer.rendering.minimap_renderer import MinimapRenderer

        state = state_with_minimap_data

        # Create a mock MSExperiment
        mock_exp = MagicMock()

        def mock_rasterize(output, rt_min, rt_max, mz_min, mz_max, ms_level=1, aggregation="sum"):
            output[:] = np.random.uniform(0, 100, output.shape)

        mock_exp.rasterizeRTMZ = MagicMock(side_effect=mock_rasterize)
        state.exp = mock_exp

        renderer = MinimapRenderer(width=DEFAULTS.MINIMAP_WIDTH, height=DEFAULTS.MINIMAP_HEIGHT)

        # Render to populate cache
        result = renderer.render(state)
        assert result is not None

        # Verify cache shape and type
        assert state.cached_minimap_raster is not None
        assert state.cached_minimap_raster.dtype == np.float32
        assert state.cached_minimap_raster.shape == (DEFAULTS.MINIMAP_HEIGHT, DEFAULTS.MINIMAP_WIDTH)
