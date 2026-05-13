"""Shared pytest fixtures for the test suite."""

import numpy as np
import pytest
from pyopenms import FloatDataArray, MSExperiment, MSSpectrum, MzMLFile, Precursor


@pytest.fixture(scope="session")
def ms2_im_mzml_path(tmp_path_factory):
    """Synthetic mzML covering MS1 IM, MS2 IM, and 'no IM' edge cases.

    Layout (sorted by RT):
        idx 0: MS1 RT=1.0, IM array present
        idx 1: MS2 RT=1.1, IM array present, precursor mz=500.0 lo=5.0 hi=5.0
        idx 2: MS2 RT=1.2, IM array present, precursor mz=510.0 lo=5.0 hi=5.0
        idx 3: MS2 RT=1.3, NO IM array, precursor mz=520.0 (placeholder path)
        idx 4: MS1 RT=2.0, IM array present
        idx 5: MS2 RT=2.1, IM array present, precursor mz=500.0 lo=5.0 hi=5.0
        idx 6: MS2 RT=2.2, IM array present, precursor mz=510.0 lo=5.0 hi=5.0
        idx 7: MS1 RT=3.0, NO IM array (MS1 placeholder edge)
    """

    def _make_spec(ms_level, rt, mzs, intens, im_values=None, precursor=None):
        spec = MSSpectrum()
        spec.setMSLevel(ms_level)
        spec.setRT(rt)
        spec.set_peaks((np.asarray(mzs, dtype=np.float64),
                        np.asarray(intens, dtype=np.float32)))
        if im_values is not None:
            fda = FloatDataArray()
            fda.setName("Ion Mobility Drift Time")
            fda.set_data(np.asarray(im_values, dtype=np.float32))
            spec.setFloatDataArrays([fda])
        if precursor is not None:
            mz, lo_off, hi_off = precursor
            p = Precursor()
            p.setMZ(mz)
            p.setIsolationWindowLowerOffset(lo_off)
            p.setIsolationWindowUpperOffset(hi_off)
            spec.setPrecursors([p])
        return spec

    exp = MSExperiment()
    exp.addSpectrum(_make_spec(1, 1.0, [100.0, 200.0], [10.0, 20.0], im_values=[0.80, 0.90]))
    exp.addSpectrum(_make_spec(2, 1.1, [110.0, 220.0], [11.0, 22.0], im_values=[1.00, 1.10],
                               precursor=(500.0, 5.0, 5.0)))
    exp.addSpectrum(_make_spec(2, 1.2, [120.0, 230.0], [12.0, 23.0], im_values=[1.05, 1.15],
                               precursor=(510.0, 5.0, 5.0)))
    exp.addSpectrum(_make_spec(2, 1.3, [130.0, 240.0], [13.0, 24.0], im_values=None,
                               precursor=(520.0, 5.0, 5.0)))
    exp.addSpectrum(_make_spec(1, 2.0, [105.0, 205.0], [15.0, 25.0], im_values=[0.82, 0.92]))
    exp.addSpectrum(_make_spec(2, 2.1, [115.0, 225.0], [16.0, 26.0], im_values=[1.02, 1.12],
                               precursor=(500.0, 5.0, 5.0)))
    exp.addSpectrum(_make_spec(2, 2.2, [125.0, 235.0], [17.0, 27.0], im_values=[1.07, 1.17],
                               precursor=(510.0, 5.0, 5.0)))
    exp.addSpectrum(_make_spec(1, 3.0, [108.0, 208.0], [18.0, 28.0], im_values=None))

    path = tmp_path_factory.mktemp("ms2_im") / "synthetic_ms2_im.mzML"
    MzMLFile().store(str(path), exp)
    return path
