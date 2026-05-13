"""Tests for MS2 ion mobility frame support."""

import numpy as np
from pyopenms import MSExperiment, MzMLFile


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
