"""Tests for the pyopenms_viewer core module (state, events, config)."""

from pathlib import Path

from pyopenms_viewer.core.config import (
    COLORMAPS,
    DEFAULT_PANEL_ORDER,
    DEFAULT_PANEL_VISIBILITY,
    DEFAULTS,
    ION_COLORS,
    PANEL_DEFINITIONS,
    get_colormap_background,
)
from pyopenms_viewer.core.events import EventBus
from pyopenms_viewer.core.state import ViewBounds, ViewerState

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"


class TestViewerState:
    """Tests for ViewerState class."""

    def test_init_defaults(self):
        """Test ViewerState initializes with correct defaults."""
        state = ViewerState()

        # Check data bounds have default values (rt_min/mz_min=0, max=1)
        assert state.rt_min == 0.0
        assert state.rt_max == 1.0  # Default is 1.0
        assert state.mz_min == 0.0
        assert state.mz_max == 1.0  # Default is 1.0

        # Check view bounds are None (use data bounds)
        assert state.view_rt_min is None
        assert state.view_rt_max is None
        assert state.view_mz_min is None
        assert state.view_mz_max is None

        # Check display options match DEFAULTS
        assert state.swap_axes == DEFAULTS.SWAP_AXES
        assert state.rt_in_minutes == DEFAULTS.RT_IN_MINUTES
        assert state.show_centroids == DEFAULTS.SHOW_CENTROIDS

        # Check experiment is None
        assert state.exp is None
        assert state.df is None

    def test_set_data_bounds(self):
        """Test setting data bounds."""
        state = ViewerState()
        state.rt_min = 100.0
        state.rt_max = 3600.0
        state.mz_min = 200.0
        state.mz_max = 2000.0

        assert state.rt_min == 100.0
        assert state.rt_max == 3600.0
        assert state.mz_min == 200.0
        assert state.mz_max == 2000.0

    def test_set_view_bounds(self):
        """Test setting view bounds for zooming."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0

        # Set zoomed view
        state.view_rt_min = 500.0
        state.view_rt_max = 1500.0
        state.view_mz_min = 300.0
        state.view_mz_max = 800.0

        assert state.view_rt_min == 500.0
        assert state.view_rt_max == 1500.0
        assert state.view_mz_min == 300.0
        assert state.view_mz_max == 800.0

    def test_get_view_bounds_no_zoom(self):
        """Test get_view_bounds returns data bounds when no zoom."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0

        bounds = state.get_view_bounds()

        assert bounds.rt_min == 0.0
        assert bounds.rt_max == 3600.0
        assert bounds.mz_min == 100.0
        assert bounds.mz_max == 2000.0

    def test_get_view_bounds_with_zoom(self):
        """Test get_view_bounds returns view bounds when zoomed."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.view_rt_min = 500.0
        state.view_rt_max = 1500.0
        state.view_mz_min = 300.0
        state.view_mz_max = 800.0

        bounds = state.get_view_bounds()

        assert bounds.rt_min == 500.0
        assert bounds.rt_max == 1500.0
        assert bounds.mz_min == 300.0
        assert bounds.mz_max == 800.0

    def test_reset_view(self):
        """Test resetting view to full data bounds."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0
        state.view_rt_min = 500.0
        state.view_rt_max = 1500.0
        state.view_mz_min = 300.0
        state.view_mz_max = 800.0

        state.reset_view(emit_event=False)

        # reset_view sets view bounds to data bounds (not None)
        assert state.view_rt_min == state.rt_min
        assert state.view_rt_max == state.rt_max
        assert state.view_mz_min == state.mz_min
        assert state.view_mz_max == state.mz_max

    def test_spectrum_data_list(self):
        """Test spectrum_data is initialized as empty list."""
        state = ViewerState()
        assert state.spectrum_data == []
        assert isinstance(state.spectrum_data, list)

    def test_feature_data_list(self):
        """Test feature_data is initialized as empty list."""
        state = ViewerState()
        assert state.feature_data == []

    def test_id_data_list(self):
        """Test id_data is initialized as empty list."""
        state = ViewerState()
        assert state.id_data == []

    def test_peptide_ids_list(self):
        """Test peptide_ids is initialized as empty list."""
        state = ViewerState()
        assert state.peptide_ids == []

    def test_tic_arrays(self):
        """Test TIC arrays are initialized as None."""
        state = ViewerState()
        assert state.tic_rt is None
        assert state.tic_intensity is None

    def test_ion_mobility_state(self):
        """Test ion mobility state initialization."""
        state = ViewerState()
        assert state.has_ion_mobility is False
        assert state.im_df is None
        assert state.im_min == 0.0
        assert state.im_max == 1.0  # Default is 1.0

    def test_selected_spectrum_index(self):
        """Test selected spectrum index."""
        state = ViewerState()
        assert state.selected_spectrum_idx is None

        state.selected_spectrum_idx = 5
        assert state.selected_spectrum_idx == 5

    def test_selected_feature_index(self):
        """Test selected feature index."""
        state = ViewerState()
        assert state.selected_feature_idx is None

        state.selected_feature_idx = 3
        assert state.selected_feature_idx == 3

    def test_axis_colors(self):
        """Test axis color settings."""
        state = ViewerState()
        assert state.axis_color is not None
        assert state.tick_color is not None
        assert state.label_color is not None

    def test_colormap_setting(self):
        """Test colormap setting."""
        state = ViewerState()
        assert state.colormap == DEFAULTS.COLORMAP

        state.colormap = "viridis"
        assert state.colormap == "viridis"

    def test_annotation_tolerance(self):
        """Test annotation tolerance setting."""
        state = ViewerState()
        assert state.annotation_tolerance_da == DEFAULTS.ANNOTATION_TOLERANCE_DA

        state.annotation_tolerance_da = 0.1
        assert state.annotation_tolerance_da == 0.1


class TestViewBounds:
    """Tests for ViewBounds dataclass."""

    def test_view_bounds_creation(self):
        """Test ViewBounds dataclass creation."""
        bounds = ViewBounds(
            rt_min=0.0,
            rt_max=3600.0,
            mz_min=100.0,
            mz_max=2000.0,
        )
        assert bounds.rt_min == 0.0
        assert bounds.rt_max == 3600.0
        assert bounds.mz_min == 100.0
        assert bounds.mz_max == 2000.0

    def test_view_bounds_with_im(self):
        """Test ViewBounds with ion mobility bounds."""
        bounds = ViewBounds(
            rt_min=0.0,
            rt_max=3600.0,
            mz_min=100.0,
            mz_max=2000.0,
            im_min=0.5,
            im_max=1.5,
        )
        assert bounds.im_min == 0.5
        assert bounds.im_max == 1.5


class TestEventBus:
    """Tests for EventBus class."""

    def test_subscribe_and_emit(self):
        """Test subscribing to and emitting events."""
        bus = EventBus()
        received_data = []

        def handler(**kwargs):
            received_data.append(kwargs)

        bus.subscribe("test_event", handler)
        bus.emit("test_event", key="value")

        assert len(received_data) == 1
        assert received_data[0] == {"key": "value"}

    def test_multiple_subscribers(self):
        """Test multiple subscribers to same event."""
        bus = EventBus()
        handler1_called = []
        handler2_called = []

        def handler1(**kwargs):
            handler1_called.append(kwargs)

        def handler2(**kwargs):
            handler2_called.append(kwargs)

        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)
        bus.emit("test_event", data="test_data")

        assert len(handler1_called) == 1
        assert len(handler2_called) == 1

    def test_emit_nonexistent_event(self):
        """Test emitting event with no subscribers doesn't raise."""
        bus = EventBus()
        # Should not raise
        bus.emit("nonexistent_event", data="test")

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        received_data = []

        def handler(**kwargs):
            received_data.append(kwargs)

        bus.subscribe("test_event", handler)
        bus.emit("test_event", data="first")
        assert len(received_data) == 1

        bus.unsubscribe("test_event", handler)
        bus.emit("test_event", data="second")
        assert len(received_data) == 1  # Should not receive second event

    def test_unsubscribe_nonexistent_handler(self):
        """Test unsubscribing non-existent handler doesn't raise."""
        bus = EventBus()

        def handler(**kwargs):
            pass

        # Should not raise
        bus.unsubscribe("test_event", handler)

    def test_clear_all_events(self):
        """Test clearing all event subscribers."""
        bus = EventBus()
        received = []

        def handler(**kwargs):
            received.append(kwargs)

        bus.subscribe("event1", handler)
        bus.subscribe("event2", handler)
        bus.clear()

        bus.emit("event1", data="test")
        bus.emit("event2", data="test")
        assert len(received) == 0

    def test_clear_specific_event(self):
        """Test clearing subscribers for a specific event."""
        bus = EventBus()
        received = []

        def handler(**kwargs):
            received.append(kwargs.get("source"))

        bus.subscribe("event1", handler)
        bus.subscribe("event2", handler)
        bus.clear("event1")

        bus.emit("event1", source="event1")
        bus.emit("event2", source="event2")
        assert len(received) == 1
        assert received[0] == "event2"

    def test_has_subscribers(self):
        """Test checking if event has subscribers."""
        bus = EventBus()

        def handler(**kwargs):
            pass

        assert bus.has_subscribers("test_event") is False
        bus.subscribe("test_event", handler)
        assert bus.has_subscribers("test_event") is True


class TestConfig:
    """Tests for configuration constants."""

    def test_ion_colors(self):
        """Test ION_COLORS contains expected ion types."""
        assert "b" in ION_COLORS
        assert "y" in ION_COLORS
        assert "a" in ION_COLORS
        assert "precursor" in ION_COLORS
        assert "unknown" in ION_COLORS

        # All values should be valid hex colors
        for color in ION_COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7

    def test_colormaps(self):
        """Test COLORMAPS contains expected colormaps."""
        assert "jet" in COLORMAPS
        assert "viridis" in COLORMAPS
        assert "hot" in COLORMAPS
        assert "fire" in COLORMAPS

    def test_get_colormap_background_jet(self):
        """Test getting background color from jet colormap."""
        bg = get_colormap_background("jet")
        assert bg.startswith("#")

    def test_get_colormap_background_fire(self):
        """Test getting background color from fire colormap."""
        bg = get_colormap_background("fire")
        # Fire from colorcet should return first color from list
        assert bg.startswith("#") or bg == "black"

    def test_get_colormap_background_invalid(self):
        """Test getting background color from invalid colormap."""
        bg = get_colormap_background("invalid_colormap")
        assert bg == "black"

    def test_defaults_class(self):
        """Test DEFAULTS class contains expected values."""
        assert DEFAULTS.PLOT_WIDTH > 0
        assert DEFAULTS.PLOT_HEIGHT > 0
        assert DEFAULTS.MARGIN_LEFT >= 0
        assert DEFAULTS.MARGIN_TOP >= 0
        assert DEFAULTS.MINIMAP_WIDTH > 0
        assert DEFAULTS.MINIMAP_HEIGHT > 0
        assert isinstance(DEFAULTS.SWAP_AXES, bool)
        assert isinstance(DEFAULTS.RT_IN_MINUTES, bool)
        assert DEFAULTS.ANNOTATION_TOLERANCE_DA > 0

    def test_panel_definitions(self):
        """Test PANEL_DEFINITIONS structure."""
        assert "tic" in PANEL_DEFINITIONS
        assert "peakmap" in PANEL_DEFINITIONS
        assert "spectrum" in PANEL_DEFINITIONS

        for _panel_id, panel_def in PANEL_DEFINITIONS.items():
            assert "name" in panel_def
            assert "icon" in panel_def

    def test_default_panel_order(self):
        """Test DEFAULT_PANEL_ORDER contains all panels."""
        for panel_id in PANEL_DEFINITIONS:
            assert panel_id in DEFAULT_PANEL_ORDER

    def test_default_panel_visibility(self):
        """Test DEFAULT_PANEL_VISIBILITY contains all panels."""
        for panel_id in PANEL_DEFINITIONS:
            assert panel_id in DEFAULT_PANEL_VISIBILITY


class TestStateZoomHistory:
    """Tests for zoom history functionality."""

    def test_zoom_history_init(self):
        """Test zoom history is initialized empty."""
        state = ViewerState()
        assert len(state.zoom_history) == 0

    def test_push_zoom_history(self):
        """Test pushing zoom state to history."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0

        # Set view bounds
        state.view_rt_min = 500.0
        state.view_rt_max = 1500.0
        state.view_mz_min = 300.0
        state.view_mz_max = 800.0

        state.push_zoom_history()

        assert len(state.zoom_history) == 1

    def test_go_to_zoom_history(self):
        """Test navigating to a point in zoom history."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0

        # First zoom level
        state.view_rt_min = 500.0
        state.view_rt_max = 1500.0
        state.view_mz_min = 300.0
        state.view_mz_max = 800.0
        state.push_zoom_history()

        # Second zoom level
        state.view_rt_min = 700.0
        state.view_rt_max = 900.0
        state.view_mz_min = 400.0
        state.view_mz_max = 600.0
        state.push_zoom_history()

        # Go back to first level (index 0)
        state.go_to_zoom_history(0, emit_event=False)

        assert state.view_rt_min == 500.0
        assert state.view_rt_max == 1500.0
        assert state.view_mz_min == 300.0
        assert state.view_mz_max == 800.0

    def test_go_to_invalid_history_index(self):
        """Test navigating to invalid history index doesn't raise."""
        state = ViewerState()
        # Should not raise
        state.go_to_zoom_history(999, emit_event=False)
        state.go_to_zoom_history(-1, emit_event=False)

    def test_zoom_history_max_size(self):
        """Test zoom history respects max size."""
        state = ViewerState()
        state.rt_min = 0.0
        state.rt_max = 3600.0
        state.mz_min = 100.0
        state.mz_max = 2000.0

        # Push more than max allowed (need unique views to avoid deduplication)
        for i in range(DEFAULTS.MAX_ZOOM_HISTORY + 5):
            state.view_rt_min = float(i * 100)
            state.view_rt_max = float(i * 100 + 500)
            state.view_mz_min = float(i * 50)
            state.view_mz_max = float(i * 50 + 200)
            state.push_zoom_history()

        assert len(state.zoom_history) <= DEFAULTS.MAX_ZOOM_HISTORY

    def test_max_zoom_history_attribute(self):
        """Test max_zoom_history attribute exists."""
        state = ViewerState()
        assert state.max_zoom_history == DEFAULTS.MAX_ZOOM_HISTORY


class TestStateChromatograms:
    """Tests for chromatogram state."""

    def test_chromatograms_init(self):
        """Test chromatograms are initialized as empty list."""
        state = ViewerState()
        assert state.chromatograms == []

    def test_chromatogram_data_init(self):
        """Test chromatogram_data is initialized as empty dict."""
        state = ViewerState()
        assert state.chromatogram_data == {}


class TestStateFAIMS:
    """Tests for FAIMS-related state."""

    def test_faims_cvs_init(self):
        """Test faims_cvs are initialized as empty list."""
        state = ViewerState()
        assert state.faims_cvs == []

    def test_selected_faims_cv_init(self):
        """Test selected_faims_cv is initialized as None."""
        state = ViewerState()
        assert state.selected_faims_cv is None

    def test_set_faims_cvs(self):
        """Test setting faims_cvs values."""
        state = ViewerState()
        state.faims_cvs = [-40, -50, -60]
        assert state.faims_cvs == [-40, -50, -60]

    def test_set_selected_faims_cv(self):
        """Test setting selected_faims_cv."""
        state = ViewerState()
        state.faims_cvs = [-40, -50, -60]
        state.selected_faims_cv = -50
        assert state.selected_faims_cv == -50

    def test_has_faims_flag(self):
        """Test has_faims flag."""
        state = ViewerState()
        assert state.has_faims is False
        state.has_faims = True
        assert state.has_faims is True

    def test_faims_data_dict(self):
        """Test faims_data is initialized as empty dict."""
        state = ViewerState()
        assert state.faims_data == {}
        assert isinstance(state.faims_data, dict)
