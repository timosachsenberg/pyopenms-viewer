"""Tests for MS2 ion mobility frame support."""

import numpy as np
from pyopenms import MSExperiment, MzMLFile

from pyopenms_viewer.core.state import ViewerState


def test_fixture_roundtrips(ms2_im_mzml_path):
    exp = MSExperiment()
    MzMLFile().load(str(ms2_im_mzml_path), exp)
    assert len(exp) == 8
    levels = [exp[i].getMSLevel() for i in range(len(exp))]
    assert levels == [1, 2, 2, 2, 1, 2, 2, 1]
    rts = [round(exp[i].getRT(), 2) for i in range(len(exp))]
    assert rts == [1.00, 1.10, 1.20, 1.30, 2.00, 2.10, 2.20, 3.00]

    has_im = []
    for i in range(len(exp)):
        names = [fda.getName() for fda in exp[i].getFloatDataArrays()]
        has_im.append(any("mobility" in (n or "").lower() for n in names))
    assert has_im == [True, True, True, False, True, True, True, False]


class TestNewIMStateAttributes:
    def test_defaults(self):
        state = ViewerState()
        assert len(state.im_frame_ms_levels) == 0
        assert len(state.im_frame_precursor_mz) == 0
        assert len(state.im_frame_precursor_lower) == 0
        assert len(state.im_frame_precursor_upper) == 0
        assert state.im_frame_position_by_index == {}
        assert state.ms1_im_frame_indices == []
        assert len(state.ms1_im_frame_rts) == 0

    def test_clear_mzml_data_resets_new_attrs(self):
        state = ViewerState()
        state.im_frame_ms_levels = np.array([1, 2, 2], dtype=np.int32)
        state.im_frame_precursor_mz = np.array([np.nan, 500.0, 510.0])
        state.im_frame_precursor_lower = np.array([np.nan, 495.0, 505.0])
        state.im_frame_precursor_upper = np.array([np.nan, 505.0, 515.0])
        state.im_frame_position_by_index = {0: 0, 1: 1, 2: 2}
        state.ms1_im_frame_indices = [0]
        state.ms1_im_frame_rts = np.array([1.0])
        state.has_ion_mobility = True

        state.clear_mzml_data()

        assert len(state.im_frame_ms_levels) == 0
        assert len(state.im_frame_precursor_mz) == 0
        assert len(state.im_frame_precursor_lower) == 0
        assert len(state.im_frame_precursor_upper) == 0
        assert state.im_frame_position_by_index == {}
        assert state.ms1_im_frame_indices == []
        assert len(state.ms1_im_frame_rts) == 0
        assert state.has_ion_mobility is False


class TestIMStateHelpers:
    def _state_with_synth_data(self):
        """Build state matching the synthetic fixture's IM frames."""
        state = ViewerState()
        state.has_ion_mobility = True
        state.im_frame_indices = [0, 1, 2, 4, 5, 6]
        state.im_frame_rts = np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2], dtype=np.float64)
        state.im_frame_ms_levels = np.array([1, 2, 2, 1, 2, 2], dtype=np.int32)
        state.im_frame_precursor_mz = np.array([np.nan, 500.0, 510.0, np.nan, 500.0, 510.0])
        state.im_frame_precursor_lower = np.array([np.nan, 495.0, 505.0, np.nan, 495.0, 505.0])
        state.im_frame_precursor_upper = np.array([np.nan, 505.0, 515.0, np.nan, 505.0, 515.0])
        state.im_frame_position_by_index = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
        state.ms1_im_frame_indices = [0, 4]
        state.ms1_im_frame_rts = np.array([1.0, 2.0], dtype=np.float64)
        return state

    def test_find_nearest_ms1_im_frame_idx_picks_ms1(self):
        state = self._state_with_synth_data()
        # RT 1.15 is closest to MS2 idx 1 (RT 1.1) and MS2 idx 2 (RT 1.2)
        # but find_nearest_ms1_im_frame_idx must return MS1 idx 0 (RT 1.0).
        assert state.find_nearest_ms1_im_frame_idx(1.15) == 0
        assert state.find_nearest_ms1_im_frame_idx(1.6) == 4  # closer to RT 2.0 MS1
        assert state.find_nearest_ms1_im_frame_idx(0.0) == 0
        assert state.find_nearest_ms1_im_frame_idx(99.0) == 4

    def test_find_nearest_ms1_im_frame_idx_empty(self):
        state = ViewerState()
        assert state.find_nearest_ms1_im_frame_idx(1.0) is None

    def test_get_im_frame_ms_level(self):
        state = self._state_with_synth_data()
        assert state.get_im_frame_ms_level(0) == 1
        assert state.get_im_frame_ms_level(1) == 2
        assert state.get_im_frame_ms_level(99) is None
        assert state.get_im_frame_ms_level(3) is None  # idx 3 not in im_frame_indices

    def test_get_im_frame_precursor_bounds(self):
        state = self._state_with_synth_data()
        assert state.get_im_frame_precursor_lower(0) is None  # MS1 (nan)
        assert state.get_im_frame_precursor_upper(0) is None
        assert state.get_im_frame_precursor_lower(1) == 495.0
        assert state.get_im_frame_precursor_upper(1) == 505.0
        assert state.get_im_frame_precursor_lower(99) is None


class TestLoaderMS2IM:
    def _load(self, ms2_im_mzml_path):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader

        state = ViewerState()
        loader = MzMLLoader(state)
        assert loader.load_sync(str(ms2_im_mzml_path)) is True
        return state

    def test_im_frame_indices_includes_all_im_levels(self, ms2_im_mzml_path):
        state = self._load(ms2_im_mzml_path)
        # Synthetic file has 2 MS1+IM, 4 MS2+IM, plus 1 MS2 no-IM and 1 MS1 no-IM.
        # Frame indices are the spectrum indices of IM-bearing frames.
        assert state.has_ion_mobility is True
        assert sorted(state.im_frame_indices) == [0, 1, 2, 4, 5, 6]

    def test_im_frame_arrays_parallel_and_sorted(self, ms2_im_mzml_path):
        state = self._load(ms2_im_mzml_path)
        # Sorted by RT (and spec_idx for tie-breaking). RTs are 1.0..2.2 for these 6.
        rts = state.im_frame_rts.tolist()
        assert rts == sorted(rts)
        assert len(state.im_frame_indices) == len(state.im_frame_rts)
        assert len(state.im_frame_indices) == len(state.im_frame_ms_levels)
        assert len(state.im_frame_indices) == len(state.im_frame_precursor_mz)
        assert len(state.im_frame_indices) == len(state.im_frame_precursor_lower)
        assert len(state.im_frame_indices) == len(state.im_frame_precursor_upper)

    def test_im_frame_ms_levels(self, ms2_im_mzml_path):
        state = self._load(ms2_im_mzml_path)
        # idx -> ms_level pairs derived from fixture layout.
        expected = {0: 1, 1: 2, 2: 2, 4: 1, 5: 2, 6: 2}
        for spec_idx, expected_level in expected.items():
            assert state.get_im_frame_ms_level(spec_idx) == expected_level

    def test_im_frame_precursor_bounds(self, ms2_im_mzml_path):
        state = self._load(ms2_im_mzml_path)
        # MS1 frames: nan -> None via helper.
        assert state.get_im_frame_precursor_lower(0) is None
        assert state.get_im_frame_precursor_upper(0) is None
        # MS2 idx 1: precursor 500.0 with offsets 5.0 -> [495.0, 505.0]
        assert state.get_im_frame_precursor_lower(1) == 495.0
        assert state.get_im_frame_precursor_upper(1) == 505.0
        # MS2 idx 2: precursor 510.0 -> [505.0, 515.0]
        assert state.get_im_frame_precursor_lower(2) == 505.0
        assert state.get_im_frame_precursor_upper(2) == 515.0

    def test_position_by_index_mapping(self, ms2_im_mzml_path):
        state = self._load(ms2_im_mzml_path)
        for pos, spec_idx in enumerate(state.im_frame_indices):
            assert state.im_frame_position_by_index[spec_idx] == pos

    def test_ms1_im_frame_subset(self, ms2_im_mzml_path):
        state = self._load(ms2_im_mzml_path)
        assert state.ms1_im_frame_indices == [0, 4]
        assert state.ms1_im_frame_rts.tolist() == [1.0, 2.0]

    def test_no_im_spectrum_excluded(self, ms2_im_mzml_path):
        state = self._load(ms2_im_mzml_path)
        # idx 3 (MS2 no-IM) and idx 7 (MS1 no-IM) must not appear anywhere.
        assert 3 not in state.im_frame_indices
        assert 7 not in state.im_frame_indices
        assert 3 not in state.im_frame_position_by_index
        assert 7 not in state.im_frame_position_by_index


class TestApplySpectrumSelectionToIM:
    """Pure-function tests for the selection -> IM frame mapping."""

    def _state(self):
        state = ViewerState()
        state.has_ion_mobility = True
        state.im_frame_indices = [0, 1, 2, 4, 5, 6]
        state.im_frame_position_by_index = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
        return state

    def test_sets_for_ms2_im_frame(self):
        from pyopenms_viewer.panels.im_peak_map_panel import (
            apply_spectrum_selection_to_im,
        )

        state = self._state()
        apply_spectrum_selection_to_im(state, "spectrum", 1)
        assert state.selected_im_frame_idx == 1

    def test_clears_for_spectrum_without_im(self):
        from pyopenms_viewer.panels.im_peak_map_panel import (
            apply_spectrum_selection_to_im,
        )

        state = self._state()
        state.selected_im_frame_idx = 0
        apply_spectrum_selection_to_im(state, "spectrum", 3)
        assert state.selected_im_frame_idx is None

    def test_sets_for_ms1_im_frame(self):
        from pyopenms_viewer.panels.im_peak_map_panel import (
            apply_spectrum_selection_to_im,
        )

        state = self._state()
        apply_spectrum_selection_to_im(state, "spectrum", 4)
        assert state.selected_im_frame_idx == 4

    def test_clears_for_none_index(self):
        from pyopenms_viewer.panels.im_peak_map_panel import (
            apply_spectrum_selection_to_im,
        )

        state = self._state()
        state.selected_im_frame_idx = 0
        apply_spectrum_selection_to_im(state, "spectrum", None)
        assert state.selected_im_frame_idx is None

    def test_noop_when_no_ion_mobility(self):
        from pyopenms_viewer.panels.im_peak_map_panel import (
            apply_spectrum_selection_to_im,
        )

        state = self._state()
        state.has_ion_mobility = False
        state.selected_im_frame_idx = 5
        apply_spectrum_selection_to_im(state, "spectrum", 1)
        assert state.selected_im_frame_idx == 5

    def test_ignores_non_spectrum_selection(self):
        from pyopenms_viewer.panels.im_peak_map_panel import (
            apply_spectrum_selection_to_im,
        )

        state = self._state()
        state.selected_im_frame_idx = 0
        apply_spectrum_selection_to_im(state, "feature", 1)
        assert state.selected_im_frame_idx == 0


class TestSelectSpectrumDrivesIMFrame:
    """Verify state.select_spectrum -> apply_spectrum_selection_to_im end-to-end."""

    def test_select_spectrum_updates_im_via_subscribed_handler(self, ms2_im_mzml_path):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader
        from pyopenms_viewer.panels.im_peak_map_panel import (
            apply_spectrum_selection_to_im,
        )

        state = ViewerState()
        MzMLLoader(state).load_sync(str(ms2_im_mzml_path))
        state.on_selection_changed(
            lambda selection_type, index: apply_spectrum_selection_to_im(state, selection_type, index)
        )

        state.select_spectrum(1)
        assert state.selected_im_frame_idx == 1

        state.select_spectrum(3)
        assert state.selected_im_frame_idx is None

        state.select_spectrum(0)
        assert state.selected_im_frame_idx == 0
