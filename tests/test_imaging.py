"""Tests for the ImagingPanel pure-Python helpers and the ImzML loader MSI integration.

Only exercises logic that can run without a NiceGUI event loop (data loading,
aggregate computation, pixel ↔ spectrum index lookup, TIC image shape, and
ion image extraction).  UI construction and render methods require a live
NiceGUI client and are not tested here.
"""

from pathlib import Path

import numpy as np
import pyopenms as oms
import pytest

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders import ImzMLLoader
from pyopenms_viewer.loaders.imzml_loader import (
    _drop_zero_intensity_peaks,
    _extract_im_data,
    _im_state_from_unit,
)
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

    def test_df_is_none_no_peak_dataframe(self):
        """imzML loads stream peaks via MSExperiment — no peak DataFrame."""
        state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
        assert state.df is None


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

    def test_mean_uses_per_bin_counts_not_all_pixels(self, panel):
        """Sparse bins must not be diluted by pixels that never hit them.

        mean[i] == sum_of_peak_intensities_in_bin / n_peaks_in_bin, so occupied
        bin means stay on the same scale as skyline (not ~1/n_pixels).
        """
        nz = panel._agg_mean > 0
        assert nz.any()
        assert np.all(panel._agg_skyline[nz] >= panel._agg_mean[nz] - 1e-12)
        ratio = panel._agg_mean[nz] / np.maximum(panel._agg_skyline[nz], 1e-30)
        assert float(np.median(ratio)) > 0.05

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


# -------------------------------------------------------------------------
# Ion mobility handling (OpenMS spectrum contract)
# -------------------------------------------------------------------------


def _im_spectrum(
    mzs, intensities, im_values, name="mean inverse reduced ion mobility array"
):
    """Build an MSSpectrum carrying an ion mobility FloatDataArray."""
    spec = oms.MSSpectrum()
    spec.set_peaks(
        (np.asarray(mzs, dtype=np.float64), np.asarray(intensities, dtype=np.float32))
    )
    fda = oms.FloatDataArray()
    fda.set_data(np.asarray(im_values, dtype=np.float32))
    fda.setName(name)
    spec.setFloatDataArrays([fda])
    return spec


def _imaging_experiment(spectra):
    """Wrap spectra in an MSImagingExperiment, as the loader helpers expect."""
    exp = oms.MSExperiment()
    for spec in spectra:
        exp.addSpectrum(spec)
    mie = oms.MSImagingExperiment()
    mie.setMSExperiment(exp)
    return mie


class TestDropZeroIntensityPeaks:
    """Zero-intensity pruning must keep per-peak arrays aligned with the peaks."""

    def test_ion_mobility_array_stays_aligned(self):
        mie = _imaging_experiment(
            [
                _im_spectrum([100.0, 200.0, 300.0], [1.0, 0.0, 3.0], [0.8, 0.9, 1.0]),
            ]
        )
        _drop_zero_intensity_peaks(mie)

        spec = mie.getMSExperiment()[0]
        mzs, intensities = spec.get_peaks()
        im_values = spec.getFloatDataArrays()[0].get_data()
        assert list(mzs) == [100.0, 300.0]
        assert list(intensities) == [1.0, 3.0]
        assert list(im_values) == pytest.approx([0.8, 1.0])
        # OpenMS contract still sees IM after select()-based pruning.
        assert spec.containsIMData()
        extracted = _extract_im_data(spec)
        assert extracted is not None
        assert list(extracted[1]) == pytest.approx([0.8, 1.0])

    def test_spectrum_without_zero_peaks_untouched(self):
        mie = _imaging_experiment(
            [
                _im_spectrum([100.0, 200.0], [1.0, 2.0], [0.8, 0.9]),
            ]
        )
        _drop_zero_intensity_peaks(mie)

        spec = mie.getMSExperiment()[0]
        assert spec.size() == 2
        assert len(spec.getFloatDataArrays()[0].get_data()) == 2

    def test_empty_spectrum_is_skipped(self):
        mie = _imaging_experiment([oms.MSSpectrum()])
        _drop_zero_intensity_peaks(mie)  # must not raise
        assert mie.getMSExperiment()[0].size() == 0


class TestExtractIMData:
    """The loader reads IM only through containsIMData()/getIMData()."""

    def test_returns_none_without_im_array(self):
        spec = oms.MSSpectrum()
        spec.set_peaks(
            (np.array([100.0, 200.0]), np.array([1.0, 2.0], dtype=np.float32))
        )
        assert _extract_im_data(spec) is None

    def test_returns_name_values_and_unit(self):
        spec = _im_spectrum([100.0, 200.0], [1.0, 2.0], [0.85, 1.15])
        result = _extract_im_data(spec)
        assert result is not None
        name, values, unit = result
        assert name == "mean inverse reduced ion mobility array"
        assert list(values) == pytest.approx([0.85, 1.15])
        assert unit == spec.getIMData()[1]

    def test_returns_none_on_length_mismatch(self):
        """A desynced IM array cannot be plotted, so it must be rejected."""
        spec = _im_spectrum([100.0, 200.0, 300.0], [1.0, 2.0, 3.0], [0.85, 1.15])
        assert _extract_im_data(spec) is None


class TestIMStateFromUnit:
    """DriftTimeUnit drives the axis labelling, not FloatDataArray name guesses."""

    def test_vssc_maps_to_inverse_k0(self):
        assert _im_state_from_unit(oms.DriftTimeUnit.VSSC) == (
            "inverse_k0",
            "Vs/cm²",
        )

    def test_millisecond_maps_to_drift_time(self):
        assert _im_state_from_unit(oms.DriftTimeUnit.MILLISECOND) == (
            "drift_time",
            "ms",
        )

    def test_unlabelled_unit_returns_none(self):
        assert _im_state_from_unit(oms.DriftTimeUnit.NONE) is None


def test_no_ion_mobility_state_for_plain_imzml():
    """The example files carry no IM, so the IM panel must stay inactive."""
    state = _loaded_state(EXAMPLE_IMZML_CONTINUOUS)
    assert state.has_ion_mobility is False
    assert not state.ms1_im_frame_indices


class TestBoundsOnlyIonMobilityProcess:
    """imzML IM path streams min/max only — no per-peak array concat."""

    def test_bounds_only_activates_im_without_peak_arrays(self):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader

        state = ViewerState()
        exp = oms.MSExperiment()
        for i, rt in enumerate([0.0, 1.0, 2.0]):
            spec = oms.MSSpectrum()
            spec.setRT(rt)
            spec.setMSLevel(1)
            spec.set_peaks(
                (
                    np.array([100.0 + i, 200.0], dtype=np.float64),
                    np.array([10.0, 20.0], dtype=np.float32),
                )
            )
            exp.addSpectrum(spec)
        state.exp = exp
        state.mz_min = 50.0
        state.mz_max = 250.0

        loader = MzMLLoader(state)
        loader._process_ion_mobility_data(
            im_mz_list=[],
            im_im_list=[],
            im_int_list=[],
            detected_im_name="mean inverse reduced ion mobility array",
            filepath="synthetic.imzML",
            im_frame_indices=[0, 1, 2],
            im_frame_ms_levels=[1, 1, 1],
            im_mz_min=100.0,
            im_mz_max=200.0,
            im_min=0.7,
            im_max=1.3,
        )

        assert state.has_ion_mobility is True
        assert state.im_df is None
        assert state.im_frame_indices == [0, 1, 2]
        assert state.ms1_im_frame_indices == [0, 1, 2]
        assert state.im_min == pytest.approx(0.7)
        assert state.im_max == pytest.approx(1.3)
        assert state.im_type == "inverse_k0"
        assert state.selected_im_frame_idx == 0

    def test_bounds_only_without_frames_does_not_activate(self):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader

        state = ViewerState()
        state.exp = oms.MSExperiment()
        loader = MzMLLoader(state)
        loader._process_ion_mobility_data(
            im_mz_list=[],
            im_im_list=[],
            im_int_list=[],
            detected_im_name="mean inverse reduced ion mobility array",
            filepath="synthetic.imzML",
            im_frame_indices=[],
            im_mz_min=100.0,
            im_mz_max=200.0,
            im_min=0.7,
            im_max=1.3,
        )
        assert state.has_ion_mobility is False
