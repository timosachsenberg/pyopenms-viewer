"""Tests for the pyopenms_viewer rendering module."""

import pytest

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.utils.coordinate_transform import CoordinateTransform


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
