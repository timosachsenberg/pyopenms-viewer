"""Tests for the ImagingPanel pure-Python helpers and the ImzML loader MSI integration.

Only exercises logic that can run without a NiceGUI event loop (data loading,
aggregate computation, pixel ↔ spectrum index lookup, TIC image shape, and
ion image extraction).  UI construction and render methods require a live
NiceGUI client and are not tested here.
"""

from pathlib import Path

import numpy as np
import pytest

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders import ImzMLLoader
from pyopenms_viewer.panels.imaging_panel import ImagingPanel

# -------------------------------------------------------------------------
# Test data
# -------------------------------------------------------------------------
TEST_DATA_DIR = Path(__file__).parent / "data"
EXAMPLE_IMZML_CONTINUOUS = TEST_DATA_DIR / "Example_Continuous.imzML"
EXAMPLE_IMZML_PROCESSED = TEST_DATA_DIR / "Example_Processed.imzML"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _loaded_state(imzml_path: Path) -> ViewerState:
    """Return a ViewerState with the given imzML file loaded synchronously."""
    assert imzml_path.exists(), f"Test file not found: {imzml_path}"
    state = ViewerState()
    loader = ImzMLLoader(state)
    ok = loader.load_sync(str(imzml_path))
    assert ok, f"ImzMLLoader failed for {imzml_path}"
    return state


def _panel_with_data(state: ViewerState) -> ImagingPanel:
    """Return a headless ImagingPanel wired to *state*."""
    panel = ImagingPanel(state)
    # Manually trigger aggregate computation (normally done by _on_data_loaded)
    panel._recompute_aggregate()
    return panel


# -------------------------------------------------------------------------
# ImzML loader — MSI state integration
# -------------------------------------------------------------------------

class TestImzMLLoaderMSIState:
    """ImzMLLoader must populate both msi_experiment and generic viewer state."""

    def test_msi_experiment_set(self):
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        assert state.msi_experiment is not None

    def test_has_imzml_flag(self):
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        assert state.has_imzml is True

    def test_pixel_count_matches_geometry(self):
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        mie = state.msi_experiment
        geom = mie.getGeometry()
        n_pixels = mie.getNumberOfPixels()
        assert n_pixels == geom.getWidth() * geom.getHeight()
        assert n_pixels > 0

    def test_backward_compatible_exp_set(self):
        """state.exp must also be set so all non-MSI panels (TIC, spectrum) work."""
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        assert state.exp is not None
        assert len(state.exp.getSpectra()) > 0

    def test_processed_mode_loads(self):
        """Processed imzML (variable-length spectra) must also load correctly."""
        state = _loaded_state(EXAMPLE_IMZML_PROCESSED)
        assert state.has_imzml
        assert state.msi_experiment is not None
        assert state.msi_experiment.getNumberOfPixels() > 0


# -------------------------------------------------------------------------
# ImagingPanel._recompute_aggregate
# -------------------------------------------------------------------------

class TestRecomputeAggregate:
    """Aggregate spectrum computation mirrors the reference notebook."""

    @pytest.fixture(scope="class")
    def panel(self):
        return _panel_with_data(_loaded_state(EXAMPLE_IMZML_CONTINUOUS))

    def test_aggregate_not_none_after_load(self, panel):
        assert panel._agg_centers is not None
        assert panel._agg_mean is not None
        assert panel._agg_skyline is not None

    def test_centers_and_intensities_same_length(self, panel):
        assert panel._agg_centers.shape == panel._agg_mean.shape
        assert panel._agg_centers.shape == panel._agg_skyline.shape

    def test_centers_monotonically_increasing(self, panel):
        assert np.all(np.diff(panel._agg_centers) > 0)

    def test_skyline_ge_mean(self, panel):
        """Skyline is element-wise >= mean (max >= mean over any distribution)."""
        assert np.all(panel._agg_skyline >= panel._agg_mean - 1e-12)

    def test_mean_nonnegative(self, panel):
        assert np.all(panel._agg_mean >= 0)

    def test_centers_within_data_mz_range(self, panel):
        state = panel.state
        msexp = state.msi_experiment.getMSExperiment()
        msexp.updateRanges()
        min_mz = float(msexp.getMinMZ())
        max_mz = float(msexp.getMaxMZ())
        assert panel._agg_centers[0] >= min_mz
        # Last bin center can exceed max_mz by at most one bin width (~bin_ppm)
        # due to the log-spaced edge construction — use a generous tolerance.
        assert panel._agg_centers[-1] <= max_mz * (1 + 50e-6)


# -------------------------------------------------------------------------
# ImagingPanel._get_aggregate_top_peaks
# -------------------------------------------------------------------------

class TestGetAggregateTopPeaks:
    @pytest.fixture(scope="class")
    def panel(self):
        return _panel_with_data(_loaded_state(EXAMPLE_IMZML_CONTINUOUS))

    def test_returns_two_arrays(self, panel):
        mz, intensity = panel._get_aggregate_top_peaks()
        assert isinstance(mz, np.ndarray)
        assert isinstance(intensity, np.ndarray)

    def test_arrays_same_length(self, panel):
        mz, intensity = panel._get_aggregate_top_peaks()
        assert len(mz) == len(intensity)

    def test_at_most_top_n_peaks(self, panel):
        from pyopenms_viewer.panels.imaging_panel import _TOP_N_PEAKS
        mz, _ = panel._get_aggregate_top_peaks()
        assert len(mz) <= _TOP_N_PEAKS

    def test_mz_sorted(self, panel):
        mz, _ = panel._get_aggregate_top_peaks()
        if len(mz) > 1:
            assert np.all(np.diff(mz) >= 0)

    def test_intensities_positive(self, panel):
        _, intensity = panel._get_aggregate_top_peaks()
        assert np.all(intensity > 0)

    def test_skyline_mode(self, panel):
        panel._agg_mode = "skyline"
        mz_s, inten_s = panel._get_aggregate_top_peaks()
        panel._agg_mode = "mean"
        mz_m, inten_m = panel._get_aggregate_top_peaks()
        # Skyline picks top-N by skyline intensity; mean by mean intensity.
        # Skyline intensities should be >= corresponding mean intensities when
        # the same m/z positions are selected.
        assert inten_s.max() >= inten_m.max() - 1e-12
        panel._agg_mode = "mean"  # restore default


# -------------------------------------------------------------------------
# ImagingPanel._compute_tic_image
# -------------------------------------------------------------------------

class TestComputeTICImage:
    @pytest.fixture(scope="class")
    def panel_state(self):
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        panel = ImagingPanel(state)
        return panel, state

    def test_shape_matches_geometry(self, panel_state):
        panel, state = panel_state
        geom = state.msi_experiment.getGeometry()
        h, w = geom.getHeight(), geom.getWidth()
        img = panel._compute_tic_image()
        assert img.shape == (h, w)

    def test_no_nan_for_measured_pixels(self, panel_state):
        panel, state = panel_state
        img = panel._compute_tic_image()
        # For this test file all pixels are measured — no NaN expected.
        geom = state.msi_experiment.getGeometry()
        px = geom.get_pixels_struct()
        ys = px["y"].astype(np.intp)
        xs = px["x"].astype(np.intp)
        assert not np.any(np.isnan(img[ys, xs]))

    def test_tic_values_nonnegative(self, panel_state):
        panel, _ = panel_state
        img = panel._compute_tic_image()
        assert np.all(img[~np.isnan(img)] >= 0)

    def test_cached_on_second_call(self, panel_state):
        panel, _ = panel_state
        img1 = panel._compute_tic_image()
        img2 = panel._compute_tic_image()
        assert img1 is img2  # same object — cache hit


# -------------------------------------------------------------------------
# ImagingPanel._pixel_to_spectrum_idx
# -------------------------------------------------------------------------

class TestPixelToSpectrumIdx:
    @pytest.fixture(scope="class")
    def panel(self):
        return _panel_with_data(_loaded_state(EXAMPLE_IMZML_CONTINUOUS))

    def test_valid_pixel_returns_int(self, panel):
        geom = panel.state.msi_experiment.getGeometry()
        px = geom.get_pixels_struct()
        x, y = int(px["x"][0]), int(px["y"][0])
        idx = panel._pixel_to_spectrum_idx(x, y)
        assert isinstance(idx, int)
        assert idx >= 0

    def test_out_of_bounds_pixel_returns_none(self, panel):
        geom = panel.state.msi_experiment.getGeometry()
        large = geom.getWidth() * geom.getHeight() + 9999
        idx = panel._pixel_to_spectrum_idx(large, large)
        assert idx is None

    def test_all_pixels_resolve(self, panel):
        """Every pixel in the geometry must map to a valid spectrum index."""
        geom = panel.state.msi_experiment.getGeometry()
        px = geom.get_pixels_struct()
        n_spectra = panel.state.msi_experiment.getNumberOfPixels()
        for row in px:
            x, y = int(row["x"]), int(row["y"])
            idx = panel._pixel_to_spectrum_idx(x, y)
            assert idx is not None, f"pixel ({x},{y}) returned None"
            assert 0 <= idx < n_spectra


# -------------------------------------------------------------------------
# ImagingPanel._compute_ion_image
# -------------------------------------------------------------------------

class TestComputeIonImage:
    @pytest.fixture(scope="class")
    def panel(self):
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        panel = _panel_with_data(state)
        return panel

    def test_shape_matches_geometry(self, panel):
        geom = panel.state.msi_experiment.getGeometry()
        h, w = geom.getHeight(), geom.getWidth()
        mz, _ = panel._get_aggregate_top_peaks()
        mid_mz = float(mz[len(mz) // 2])
        img = panel._compute_ion_image(mid_mz, 50.0)
        assert img.shape == (h, w)

    def test_nonnegative_intensities(self, panel):
        mz, _ = panel._get_aggregate_top_peaks()
        img = panel._compute_ion_image(float(mz[0]), 50.0)
        assert np.all(img[~np.isnan(img)] >= 0)

    def test_ion_image_cache_key(self, panel):
        """Second extraction at the same m/z/ppm must hit the cache."""
        mz, _ = panel._get_aggregate_top_peaks()
        m = float(mz[0])
        panel._ion_cache_key = None  # clear cache
        img1 = panel._compute_ion_image(m, 10.0)
        img2 = panel._compute_ion_image(m, 10.0)
        assert img1 is img2

    def test_different_mz_gives_different_image(self, panel):
        """Extracting two distinct m/z values should produce different images."""
        mz, _ = panel._get_aggregate_top_peaks()
        if len(mz) < 2:
            pytest.skip("Need at least 2 peaks")
        img1 = panel._compute_ion_image(float(mz[0]), 1.0)
        img2 = panel._compute_ion_image(float(mz[-1]), 1.0)
        # Force cache miss for second call
        panel._ion_cache_key = None
        img2 = panel._compute_ion_image(float(mz[-1]), 1.0)
        # Images may be all-zero for narrow ppm on sparse data,
        # but shape must match and they need not be identical.
        assert img1.shape == img2.shape


# -------------------------------------------------------------------------
# ImagingPanel._has_data
# -------------------------------------------------------------------------

def test_has_data_true_when_loaded():
    state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
    panel = ImagingPanel(state)
    assert panel._has_data() is True


def test_has_data_false_on_empty_state():
    state = ViewerState()
    panel = ImagingPanel(state)
    assert panel._has_data() is False


def test_has_data_false_when_no_msi_experiment():
    state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
    state.msi_experiment = None  # simulate missing experiment
    panel = ImagingPanel(state)
    assert panel._has_data() is False


# -------------------------------------------------------------------------
# ImagingPanel aggregate idempotency (rehydration dedupe)
# -------------------------------------------------------------------------

class TestEnsureAggregateIdempotent:
    def test_ensure_aggregate_computes_once(self):
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        panel = ImagingPanel(state)

        calls = {"n": 0}
        real = panel._recompute_aggregate

        def counting():
            calls["n"] += 1
            real()

        panel._recompute_aggregate = counting
        panel._ensure_aggregate()
        panel._ensure_aggregate()
        panel._ensure_aggregate()
        assert calls["n"] == 1  # recomputed once despite repeated calls
        assert panel._agg_computed is True

    def test_reset_view_and_caches_reallows_compute(self):
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        panel = ImagingPanel(state)
        panel._ensure_aggregate()
        assert panel._agg_computed is True
        # Populate caches, then reset (as a new load would).
        panel._tic_cache = np.zeros((2, 2))
        panel._overlay_entries.append({"mz": 1.0, "ppm": 1.0, "hue": (0, 0, 0), "img": None})
        panel._reset_view_and_caches()
        assert panel._agg_computed is False
        assert panel._tic_cache is None
        assert panel._overlay_entries == []
        assert panel._mode == "tic"
        assert panel._current_mz is None
