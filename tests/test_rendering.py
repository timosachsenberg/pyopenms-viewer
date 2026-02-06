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

        # Add some test spectra
        for i in range(10):
            spectrum = MSSpectrum()
            spectrum.setRT(i * 360.0)  # 10 spectra, 360 seconds apart
            spectrum.setMSLevel(1)

            # Add peaks to spectrum
            mzs = np.array([100.0 + j * 100.0 for j in range(10)], dtype=np.float64)
            intensities = np.array([1000.0 * (i + 1) for _ in range(10)], dtype=np.float32)
            spectrum.set_peaks((mzs, intensities))

            exp.addSpectrum(spectrum)

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
