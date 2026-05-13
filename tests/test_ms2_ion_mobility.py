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
        assert state.im_frame_ms_levels is not None
        assert len(state.im_frame_ms_levels) == 0
        assert state.im_frame_precursor_mz is not None
        assert len(state.im_frame_precursor_mz) == 0
        assert state.im_frame_precursor_lower is not None
        assert len(state.im_frame_precursor_lower) == 0
        assert state.im_frame_precursor_upper is not None
        assert len(state.im_frame_precursor_upper) == 0
        assert state.im_frame_position_by_index == {}
        assert state.ms1_im_frame_indices == []
        assert state.ms1_im_frame_rts is not None
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
