# MS2 Ion Mobility Frames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the IM panel to render any MS-level spectrum that carries an ion-mobility array, driven by the existing spectrum-panel selection.

**Architecture:** Extend `ViewerState`'s per-frame IM index to cover all MS levels with parallel metadata arrays (MS level, precursor window). The loader stops gating on MS1. The IM panel subscribes to `selection_changed` and re-renders for the selected spectrum, falling back to a "no IM" placeholder when the spectrum lacks an IM array. TIC clicks select the nearest MS1 IM frame as the spectrum, so both panels stay in sync.

**Tech Stack:** Python 3.10+, pyopenms, NumPy, NiceGUI, pytest. `pyOpenMS.MSSpectrum.rasterizeIMFrame` does the rendering; we only change what we feed it.

**Design spec:** `docs/superpowers/specs/2026-05-13-ms2-ion-mobility-frames-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tests/conftest.py` | Create | Session-scoped synthetic mzML fixture covering MS1 IM, MS2 IM, MS1-no-IM, MS2-no-IM frames |
| `tests/test_ms2_ion_mobility.py` | Create | All new MS2-IM tests (loader, state, selection integration, TIC sync, rendering) |
| `pyopenms_viewer/core/state.py` | Modify | Add new attributes + helpers; refactor `select_nearest_im_frame`; extend reset path |
| `pyopenms_viewer/loaders/mzml_loader.py` | Modify | Drop MS1 gate on IM extraction; populate new state arrays in `_process_ion_mobility_data` |
| `pyopenms_viewer/panels/im_peak_map_panel.py` | Modify | Subscribe to `selection_changed`; placeholder for spectra without IM; precursor info label |
| `pyopenms_viewer/panels/tic_panel.py` | Modify | TIC click picks nearest MS1 IM frame as spectrum when IM data is present |

No changes to `pyopenms_viewer/rendering/peak_map_renderer.py`, `pyopenms_viewer/loaders/ion_mobility_loader.py`, or any other file.

---

## Task 1: Add synthetic MS2-IM mzML test fixture

**Files:**
- Create: `tests/conftest.py`
- Test: covered indirectly by every subsequent task

- [ ] **Step 1: Write `tests/conftest.py`**

```python
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

    Notes:
        - MS2 frames at RT 1.1, 1.2, 2.1, 2.2 are *closer* to certain RTs
          than the surrounding MS1 frames — needed for the TIC-sync test
          (TIC click should still land on the MS1 frame).
        - MS2 idx=3 has a precursor but no IM array (placeholder path).
        - MS1 idx=7 has no IM array (MS1 placeholder edge).
    """

    def _make_spec(ms_level, rt, mzs, intens, im_values=None, precursor=None):
        spec = MSSpectrum()
        spec.setMSLevel(ms_level)
        spec.setRT(rt)
        spec.set_peaks((np.asarray(mzs, dtype=np.float64),
                        np.asarray(intens, dtype=np.float32)))
        if im_values is not None:
            fda = FloatDataArray()
            fda.setName("ion mobility")
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
```

- [ ] **Step 2: Verify fixture builds correctly with a smoke test**

Create `tests/test_ms2_ion_mobility.py` with:

```python
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
```

- [ ] **Step 3: Run the smoke test**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::test_fixture_roundtrips -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_ms2_ion_mobility.py
git commit -m "test: add synthetic MS2-IM mzML fixture"
```

---

## Task 2: Add new state attributes (data model + reset)

**Files:**
- Modify: `pyopenms_viewer/core/state.py` (init around line 143, clear_mzml_data around line 839)
- Test: `tests/test_ms2_ion_mobility.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ms2_ion_mobility.py`:

```python
from pyopenms_viewer.core.state import ViewerState


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestNewIMStateAttributes -v
```

Expected: FAIL with `AttributeError: 'ViewerState' object has no attribute 'im_frame_ms_levels'`.

- [ ] **Step 3: Add attributes to `ViewerState.__init__`**

In `pyopenms_viewer/core/state.py`, locate the IM-related init block (around line 143-145) which currently reads:

```python
self.selected_im_frame_idx: int | None = None  # Index into exp for selected IM frame
self.im_frame_indices: list[int] = []  # MS1 IM frame indices in exp
self.im_frame_rts: np.ndarray | None = None  # Parallel RT values (sorted)
```

Replace with:

```python
self.selected_im_frame_idx: int | None = None  # Spectrum index for the displayed IM frame
self.im_frame_indices: list[int] = []  # Spectrum indices of IM-bearing frames (sorted by RT)
self.im_frame_rts: np.ndarray = np.array([], dtype=np.float64)  # Parallel RTs
self.im_frame_ms_levels: np.ndarray = np.array([], dtype=np.int32)  # Parallel MS levels
self.im_frame_precursor_mz: np.ndarray = np.array([], dtype=np.float64)  # Precursor m/z (nan for MS1)
self.im_frame_precursor_lower: np.ndarray = np.array([], dtype=np.float64)  # Precursor window lower (nan for MS1)
self.im_frame_precursor_upper: np.ndarray = np.array([], dtype=np.float64)  # Precursor window upper (nan for MS1)
self.im_frame_position_by_index: dict[int, int] = {}  # spec_idx -> position into parallel arrays
self.ms1_im_frame_indices: list[int] = []  # MS1-only subset of im_frame_indices
self.ms1_im_frame_rts: np.ndarray = np.array([], dtype=np.float64)  # Parallel to ms1_im_frame_indices
```

Note the type change of `im_frame_rts` from `np.ndarray | None` to non-optional empty array. Search the file for other references to `im_frame_rts is None` and replace with `len(im_frame_rts) == 0`:

```bash
grep -n "im_frame_rts is None\|im_frame_rts is not None" pyopenms_viewer/core/state.py
```

Fix each match accordingly. Same check across the codebase:

```bash
grep -rn "im_frame_rts is None\|im_frame_rts is not None" pyopenms_viewer/ tests/
```

For each match: if the surrounding logic is "return early when no IM frames", replace `im_frame_rts is None` with `len(im_frame_rts) == 0`. If the surrounding logic is "has IM data", prefer `state.has_ion_mobility`.

- [ ] **Step 4: Extend `clear_mzml_data` to reset the new attributes**

In `pyopenms_viewer/core/state.py:818-855`, locate the IM-related reset lines (around 839-841):

```python
self.selected_im_frame_idx = None
self.im_frame_indices = []
self.im_frame_rts = None
```

Replace with:

```python
self.selected_im_frame_idx = None
self.im_frame_indices = []
self.im_frame_rts = np.array([], dtype=np.float64)
self.im_frame_ms_levels = np.array([], dtype=np.int32)
self.im_frame_precursor_mz = np.array([], dtype=np.float64)
self.im_frame_precursor_lower = np.array([], dtype=np.float64)
self.im_frame_precursor_upper = np.array([], dtype=np.float64)
self.im_frame_position_by_index = {}
self.ms1_im_frame_indices = []
self.ms1_im_frame_rts = np.array([], dtype=np.float64)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestNewIMStateAttributes -v
```

Expected: 2 passed.

- [ ] **Step 6: Run the full existing test suite to confirm no regressions**

```bash
uv run pytest tests/ -x -q
```

Expected: all previously-passing tests still pass. If any test that referenced `im_frame_rts is None` fails, fix that test the same way (`len(...) == 0`).

- [ ] **Step 7: Commit**

```bash
git add pyopenms_viewer/core/state.py tests/test_ms2_ion_mobility.py
git commit -m "feat(state): add per-frame IM metadata arrays + reset"
```

---

## Task 3: Add state helper methods

**Files:**
- Modify: `pyopenms_viewer/core/state.py`
- Test: `tests/test_ms2_ion_mobility.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ms2_ion_mobility.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestIMStateHelpers -v
```

Expected: FAIL with `AttributeError: 'ViewerState' object has no attribute 'find_nearest_ms1_im_frame_idx'`.

- [ ] **Step 3: Locate the existing `select_nearest_im_frame` method**

Read `pyopenms_viewer/core/state.py:428-443` to confirm the current implementation:

```python
def select_nearest_im_frame(self, rt: float) -> None:
    if self.im_frame_rts is None or len(self.im_frame_rts) == 0:
        return
    idx = int(np.searchsorted(self.im_frame_rts, rt))
    if idx >= len(self.im_frame_rts):
        idx = len(self.im_frame_rts) - 1
    elif idx > 0:
        if abs(self.im_frame_rts[idx - 1] - rt) < abs(self.im_frame_rts[idx] - rt):
            idx -= 1
    self.selected_im_frame_idx = self.im_frame_indices[idx]
```

- [ ] **Step 4: Replace `select_nearest_im_frame` with `find_nearest_ms1_im_frame_idx` and add the metadata helpers**

Edit `pyopenms_viewer/core/state.py`. Replace the old `select_nearest_im_frame` method with these four methods (place them in the same location, around line 428):

```python
def find_nearest_ms1_im_frame_idx(self, rt: float) -> int | None:
    """Return spectrum index of nearest MS1 IM frame to rt, or None if none loaded."""
    if len(self.ms1_im_frame_rts) == 0:
        return None
    idx = int(np.searchsorted(self.ms1_im_frame_rts, rt))
    if idx >= len(self.ms1_im_frame_rts):
        idx = len(self.ms1_im_frame_rts) - 1
    elif idx > 0:
        if abs(self.ms1_im_frame_rts[idx - 1] - rt) < abs(self.ms1_im_frame_rts[idx] - rt):
            idx -= 1
    return self.ms1_im_frame_indices[idx]

def get_im_frame_ms_level(self, spec_idx: int) -> int | None:
    """MS level of the IM frame at spectrum index spec_idx, or None if not an IM frame."""
    pos = self.im_frame_position_by_index.get(spec_idx)
    if pos is None:
        return None
    return int(self.im_frame_ms_levels[pos])

def get_im_frame_precursor_lower(self, spec_idx: int) -> float | None:
    """Precursor isolation-window lower bound, or None if MS1 / not an IM frame."""
    pos = self.im_frame_position_by_index.get(spec_idx)
    if pos is None:
        return None
    value = float(self.im_frame_precursor_lower[pos])
    return None if np.isnan(value) else value

def get_im_frame_precursor_upper(self, spec_idx: int) -> float | None:
    """Precursor isolation-window upper bound, or None if MS1 / not an IM frame."""
    pos = self.im_frame_position_by_index.get(spec_idx)
    if pos is None:
        return None
    value = float(self.im_frame_precursor_upper[pos])
    return None if np.isnan(value) else value
```

- [ ] **Step 5: Update the only existing caller**

The only call site to remove/replace is `pyopenms_viewer/panels/tic_panel.py:213-214` (handled in Task 6). For now, search to confirm no other callers:

```bash
grep -rn "select_nearest_im_frame" pyopenms_viewer/ tests/
```

If any test mocks or asserts on the old method, leave the test for Task 6 to update.

- [ ] **Step 6: Run tests to verify the helpers pass**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestIMStateHelpers -v
```

Expected: 4 passed.

- [ ] **Step 7: Run the broader IM-related tests to detect breakage**

```bash
uv run pytest tests/test_im_rasterization.py tests/test_ms2_ion_mobility.py -v
```

Expected: existing tests pass. If `test_im_rasterization.py` references `select_nearest_im_frame`, it must be updated to use `find_nearest_ms1_im_frame_idx` + manual assignment, OR temporarily mark `xfail`/skip with a TODO referencing Task 6. Prefer the rewrite. Grep first:

```bash
grep -n "select_nearest_im_frame" tests/
```

Most likely outcome: no references in tests (the function was only used by the TIC panel). If grep returns nothing, this step is a no-op.

- [ ] **Step 8: Commit**

```bash
git add pyopenms_viewer/core/state.py tests/test_ms2_ion_mobility.py
git commit -m "refactor(state): split IM frame helpers; rename select_nearest_im_frame"
```

---

## Task 4: Loader populates new state arrays for MS2 frames

**Files:**
- Modify: `pyopenms_viewer/loaders/mzml_loader.py` (around lines 190-200, 269-328, 469-553)
- Test: `tests/test_ms2_ion_mobility.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ms2_ion_mobility.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestLoaderMS2IM -v
```

Expected: tests fail because the loader still filters MS2 frames out.

- [ ] **Step 3a: Update the IM-list initialization block**

In `pyopenms_viewer/loaders/mzml_loader.py`, around lines 190-200, locate the IM array initialization block:

```python
# Ion mobility data arrays
detected_im_name = None
im_mz_list = []
im_im_list = []
im_int_list = []
im_frame_indices_list = []  # Track MS1 frame indices that have IM data
```

Replace with:

```python
# Ion mobility data arrays
detected_im_name = None
im_mz_list = []
im_im_list = []
im_int_list = []
im_frame_indices_list = []  # All IM-bearing frame indices (any MS level)
im_frame_ms_levels_list = []  # Parallel MS levels
im_frame_precursor_mz_list = []  # Parallel precursor target m/z (nan for MS1)
im_frame_precursor_lower_list = []  # Parallel precursor window lower (nan for MS1)
im_frame_precursor_upper_list = []  # Parallel precursor window upper (nan for MS1)
```

- [ ] **Step 3b: Move IM detection+extraction out of the MS1 branch**

Locate the per-spectrum loop block (around lines 269-328). The IM extraction is currently nested inside `if ms_level == 1:`. Specifically, lines 303-328 (the two consecutive blocks shown below) are inside that branch. They must become siblings of `if ms_level == 1:` — same indentation as `if ms_level == 1:` itself (i.e. two levels in, under the `for spec_idx` loop).

Locate this code (currently indented inside `if ms_level == 1:`):

```python
# Ion mobility detection and extraction (MS1 only)
if detected_im_name is None:
    # Try to detect IM array name
    float_arrays = spec.getFloatDataArrays()
    for fda in float_arrays:
        name = fda.getName().lower() if fda.getName() else ""
        for im_name in im_array_names:
            if im_name in name:
                detected_im_name = fda.getName()
                break
        if detected_im_name:
            break

# Extract IM data if available
if detected_im_name is not None and n > 0:
    float_arrays = spec.getFloatDataArrays()
    for fda in float_arrays:
        if fda.getName() == detected_im_name:
            im_array = np.array(fda.get_data(), dtype=np.float32)
            if len(im_array) == n:
                # No .copy() needed - get_peaks() returns fresh arrays
                im_mz_list.append(mz_array)
                im_im_list.append(im_array)
                im_int_list.append(int_array)
                im_frame_indices_list.append(spec_idx)
            break
```

Cut it from its current location inside the MS1 branch and paste it at the same indentation level as `if ms_level == 1:`, replacing the body with this version (which adds metadata collection):

```python
# Ion mobility detection and extraction (all MS levels)
if detected_im_name is None and n > 0:
    float_arrays = spec.getFloatDataArrays()
    for fda in float_arrays:
        name = fda.getName().lower() if fda.getName() else ""
        for im_name in im_array_names:
            if im_name in name:
                detected_im_name = fda.getName()
                break
        if detected_im_name:
            break

if detected_im_name is not None and n > 0:
    float_arrays = spec.getFloatDataArrays()
    for fda in float_arrays:
        if fda.getName() == detected_im_name:
            im_array = np.array(fda.get_data(), dtype=np.float32)
            if len(im_array) == n:
                im_mz_list.append(mz_array)
                im_im_list.append(im_array)
                im_int_list.append(int_array)
                im_frame_indices_list.append(spec_idx)
                im_frame_ms_levels_list.append(ms_level)
                if ms_level >= 2:
                    precs = spec.getPrecursors()
                    if precs:
                        prec = precs[0]
                        pmz = float(prec.getMZ())
                        lo_off = float(prec.getIsolationWindowLowerOffset())
                        hi_off = float(prec.getIsolationWindowUpperOffset())
                        im_frame_precursor_mz_list.append(pmz)
                        im_frame_precursor_lower_list.append(pmz - lo_off)
                        im_frame_precursor_upper_list.append(pmz + hi_off)
                    else:
                        im_frame_precursor_mz_list.append(np.nan)
                        im_frame_precursor_lower_list.append(np.nan)
                        im_frame_precursor_upper_list.append(np.nan)
                else:
                    im_frame_precursor_mz_list.append(np.nan)
                    im_frame_precursor_lower_list.append(np.nan)
                    im_frame_precursor_upper_list.append(np.nan)
            break
```

- [ ] **Step 4: Update the call to `_process_ion_mobility_data`**

In `pyopenms_viewer/loaders/mzml_loader.py` around line 393-395, the loader currently calls:

```python
self._process_ion_mobility_data(
    im_mz_list, im_im_list, im_int_list, detected_im_name, filepath, im_frame_indices_list
)
```

Replace with:

```python
self._process_ion_mobility_data(
    im_mz_list,
    im_im_list,
    im_int_list,
    detected_im_name,
    filepath,
    im_frame_indices_list,
    im_frame_ms_levels_list,
    im_frame_precursor_mz_list,
    im_frame_precursor_lower_list,
    im_frame_precursor_upper_list,
)
```

- [ ] **Step 5: Update `_process_ion_mobility_data` signature and body**

In `pyopenms_viewer/loaders/mzml_loader.py:469-552`, replace the method with:

```python
def _process_ion_mobility_data(
    self,
    im_mz_list: list,
    im_im_list: list,
    im_int_list: list,
    detected_im_name: str | None,
    filepath: str,
    im_frame_indices: list | None = None,
    im_frame_ms_levels: list | None = None,
    im_frame_precursor_mz: list | None = None,
    im_frame_precursor_lower: list | None = None,
    im_frame_precursor_upper: list | None = None,
) -> None:
    """Process pre-extracted ion mobility data and populate state."""
    if not im_mz_list or detected_im_name is None:
        self.state.has_ion_mobility = False
        self.state.im_df = None
        return

    name_lower = detected_im_name.lower()
    if "inverse" in name_lower or "1/k0" in name_lower:
        self.state.im_type = "inverse_k0"
        self.state.im_unit = "Vs/cm²"
    elif "drift" in name_lower:
        self.state.im_type = "drift_time"
        self.state.im_unit = "ms"
    else:
        self.state.im_type = "ion_mobility"
        self.state.im_unit = ""

    mz_concat = np.concatenate(im_mz_list)
    im_concat = np.concatenate(im_im_list)

    im_mz_min = float(mz_concat.min())
    im_mz_max = float(mz_concat.max())
    im_min_val = float(im_concat.min())
    im_max_val = float(im_concat.max())

    if im_frame_indices:
        frame_rts = np.array(
            [self.state.exp[idx].getRT() for idx in im_frame_indices], dtype=np.float64
        )
        # Sort by (rt, spec_idx) for stable, deterministic tie-breaking.
        order = sorted(range(len(im_frame_indices)),
                       key=lambda i: (frame_rts[i], im_frame_indices[i]))

        sorted_indices = [im_frame_indices[i] for i in order]
        sorted_rts = frame_rts[order]
        sorted_levels = np.array(
            [im_frame_ms_levels[i] for i in order], dtype=np.int32
        )
        sorted_pmz = np.array(
            [im_frame_precursor_mz[i] for i in order], dtype=np.float64
        )
        sorted_plo = np.array(
            [im_frame_precursor_lower[i] for i in order], dtype=np.float64
        )
        sorted_phi = np.array(
            [im_frame_precursor_upper[i] for i in order], dtype=np.float64
        )

        self.state.im_frame_indices = sorted_indices
        self.state.im_frame_rts = sorted_rts
        self.state.im_frame_ms_levels = sorted_levels
        self.state.im_frame_precursor_mz = sorted_pmz
        self.state.im_frame_precursor_lower = sorted_plo
        self.state.im_frame_precursor_upper = sorted_phi
        self.state.im_frame_position_by_index = {
            spec_idx: pos for pos, spec_idx in enumerate(sorted_indices)
        }

        ms1_mask = sorted_levels == 1
        self.state.ms1_im_frame_indices = [
            sorted_indices[i] for i in range(len(sorted_indices)) if ms1_mask[i]
        ]
        self.state.ms1_im_frame_rts = sorted_rts[ms1_mask]

        self.state.im_min = im_min_val
        self.state.im_max = im_max_val
        self.state.im_df = None

        # Default selection: prefer the first MS1 IM frame; fall back to first IM frame.
        if self.state.ms1_im_frame_indices:
            self.state.selected_im_frame_idx = self.state.ms1_im_frame_indices[0]
        else:
            self.state.selected_im_frame_idx = sorted_indices[0]

    # Ensure valid IM range
    if self.state.im_max <= self.state.im_min:
        self.state.im_max = self.state.im_min + 1.0
    self.state.view_im_min = self.state.im_min
    self.state.view_im_max = self.state.im_max

    # Update mz bounds
    if self.state.mz_min == 0 or im_mz_min < self.state.mz_min:
        self.state.mz_min = im_mz_min
    if self.state.mz_max == 0 or im_mz_max > self.state.mz_max:
        self.state.mz_max = im_mz_max
    if self.state.view_mz_min is None or self.state.view_mz_min < self.state.mz_min:
        self.state.view_mz_min = self.state.mz_min
    if self.state.view_mz_max is None or self.state.view_mz_max > self.state.mz_max:
        self.state.view_mz_max = self.state.mz_max

    self.state.has_ion_mobility = True
```

- [ ] **Step 6: Run the loader tests**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestLoaderMS2IM -v
```

Expected: 7 passed.

- [ ] **Step 7: Run the broader test suite for regressions**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass. If `tests/test_im_rasterization.py` had any test checking `im_frame_rts is None` after load, update it to use `len(...) == 0` or `state.has_ion_mobility`.

- [ ] **Step 8: Commit**

```bash
git add pyopenms_viewer/loaders/mzml_loader.py tests/test_ms2_ion_mobility.py
git commit -m "feat(loader): extract IM data from all MS levels with per-frame metadata"
```

---

## Task 5: IM panel listens to `selection_changed`

**Files:**
- Modify: `pyopenms_viewer/panels/im_peak_map_panel.py`
- Test: `tests/test_ms2_ion_mobility.py`

- [ ] **Step 1: Write the failing tests**

These tests target a module-level function `apply_spectrum_selection_to_im` that we will extract from the panel. The panel's `_on_selection_changed` method will be a thin wrapper that calls this function and then triggers a re-render.

Append to `tests/test_ms2_ion_mobility.py`:

```python
class TestApplySpectrumSelectionToIM:
    """Pure-function tests for the selection -> IM frame mapping."""

    def _state(self):
        state = ViewerState()
        state.has_ion_mobility = True
        state.im_frame_indices = [0, 1, 2, 4, 5, 6]
        state.im_frame_position_by_index = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
        return state

    def test_sets_for_ms2_im_frame(self):
        from pyopenms_viewer.panels.im_peak_map_panel import apply_spectrum_selection_to_im

        state = self._state()
        apply_spectrum_selection_to_im(state, "spectrum", 1)
        assert state.selected_im_frame_idx == 1

    def test_clears_for_spectrum_without_im(self):
        from pyopenms_viewer.panels.im_peak_map_panel import apply_spectrum_selection_to_im

        state = self._state()
        state.selected_im_frame_idx = 0
        apply_spectrum_selection_to_im(state, "spectrum", 3)
        assert state.selected_im_frame_idx is None

    def test_sets_for_ms1_im_frame(self):
        from pyopenms_viewer.panels.im_peak_map_panel import apply_spectrum_selection_to_im

        state = self._state()
        apply_spectrum_selection_to_im(state, "spectrum", 4)
        assert state.selected_im_frame_idx == 4

    def test_clears_for_none_index(self):
        from pyopenms_viewer.panels.im_peak_map_panel import apply_spectrum_selection_to_im

        state = self._state()
        state.selected_im_frame_idx = 0
        apply_spectrum_selection_to_im(state, "spectrum", None)
        assert state.selected_im_frame_idx is None

    def test_noop_when_no_ion_mobility(self):
        from pyopenms_viewer.panels.im_peak_map_panel import apply_spectrum_selection_to_im

        state = self._state()
        state.has_ion_mobility = False
        state.selected_im_frame_idx = 5
        apply_spectrum_selection_to_im(state, "spectrum", 1)
        assert state.selected_im_frame_idx == 5

    def test_ignores_non_spectrum_selection(self):
        from pyopenms_viewer.panels.im_peak_map_panel import apply_spectrum_selection_to_im

        state = self._state()
        state.selected_im_frame_idx = 0
        apply_spectrum_selection_to_im(state, "feature", 1)
        assert state.selected_im_frame_idx == 0
```

End-to-end wiring test (verifies `state.select_spectrum` triggers the function through the panel's listener registration):

```python
class TestSelectSpectrumDrivesIMFrame:
    """Verify state.select_spectrum -> apply_spectrum_selection_to_im end-to-end."""

    def test_select_spectrum_updates_im_via_subscribed_handler(self, ms2_im_mzml_path):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader
        from pyopenms_viewer.panels.im_peak_map_panel import apply_spectrum_selection_to_im

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestApplySpectrumSelectionToIM tests/test_ms2_ion_mobility.py::TestSelectSpectrumDrivesIMFrame -v
```

Expected: FAIL with `ImportError: cannot import name 'apply_spectrum_selection_to_im'`.

- [ ] **Step 3: Add the function and wire the panel listener**

In `pyopenms_viewer/panels/im_peak_map_panel.py`, at the top of the module after the imports, add:

```python
def apply_spectrum_selection_to_im(state, selection_type: str, index: int | None) -> bool:
    """Update state.selected_im_frame_idx in response to a spectrum selection.

    Returns True if state.selected_im_frame_idx changed and a re-render is needed.
    """
    if selection_type != "spectrum":
        return False
    if not state.has_ion_mobility:
        return False
    old = state.selected_im_frame_idx
    if index is not None and index in state.im_frame_position_by_index:
        state.selected_im_frame_idx = index
    else:
        state.selected_im_frame_idx = None
    return state.selected_im_frame_idx != old
```

In the same file, locate `build()` around lines 78-79:

```python
# Subscribe to events
self.state.on_data_loaded(self._on_data_loaded)
self.state.on_view_changed(self._on_view_changed)
```

Add a third subscription:

```python
# Subscribe to events
self.state.on_data_loaded(self._on_data_loaded)
self.state.on_view_changed(self._on_view_changed)
self.state.on_selection_changed(self._on_selection_changed)
```

Locate `_on_view_changed` around lines 230-233:

```python
def _on_view_changed(self):
    """Handle view changed event."""
    if self.state.has_ion_mobility:
        self.update()
```

Add a new handler immediately after it:

```python
def _on_selection_changed(self, selection_type: str, index: int | None) -> None:
    """Sync the displayed IM frame to the selected spectrum."""
    changed = apply_spectrum_selection_to_im(self.state, selection_type, index)
    if changed:
        self.update()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestApplySpectrumSelectionToIM tests/test_ms2_ion_mobility.py::TestSelectSpectrumDrivesIMFrame -v
```

Expected: all pass.

- [ ] **Step 5: Run the full test suite for regressions**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyopenms_viewer/panels/im_peak_map_panel.py tests/test_ms2_ion_mobility.py
git commit -m "feat(im_panel): subscribe to selection_changed for per-frame IM sync"
```

---

## Task 6: TIC click picks nearest MS1 IM frame when IM data is present

**Files:**
- Modify: `pyopenms_viewer/panels/tic_panel.py` (lines 195-222)
- Test: `tests/test_ms2_ion_mobility.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ms2_ion_mobility.py`:

```python
class TestTICClickSync:
    """Verify TIC-click selects nearest MS1 IM frame as the spectrum when IM data is present."""

    def _build_handler(self, state):
        """Replicate the post-Task-6 TIC click logic for unit testing."""
        def handle_tic_click(clicked_rt):
            if state.exp is None:
                return
            if state.has_ion_mobility:
                best_idx = state.find_nearest_ms1_im_frame_idx(clicked_rt)
                if best_idx is None:
                    return
            else:
                # Fallback: nearest spectrum of any level (today's behavior).
                best_idx = 0
                best_diff = float("inf")
                for i in range(len(state.exp)):
                    diff = abs(state.exp[i].getRT() - clicked_rt)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
            state.select_spectrum(best_idx)
        return handle_tic_click

    def test_tic_click_at_ms2_rt_lands_on_ms1(self, ms2_im_mzml_path):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader

        state = ViewerState()
        MzMLLoader(state).load_sync(str(ms2_im_mzml_path))

        # Subscribe a panel-like listener so selecting a spectrum updates selected_im_frame_idx.
        def panel_handler(selection_type, index):
            if selection_type != "spectrum" or not state.has_ion_mobility:
                return
            if index is not None and index in state.im_frame_position_by_index:
                state.selected_im_frame_idx = index
            else:
                state.selected_im_frame_idx = None

        state.on_selection_changed(panel_handler)

        # Click at RT 1.15 (between MS2 frames 1 and 2; nearest MS1 is idx 0 at RT 1.0).
        self._build_handler(state)(1.15)
        assert state.selected_spectrum_idx == 0
        assert state.selected_im_frame_idx == 0

        # Click at RT 1.95 (nearest MS1 is idx 4 at RT 2.0).
        self._build_handler(state)(1.95)
        assert state.selected_spectrum_idx == 4
        assert state.selected_im_frame_idx == 4

    def test_tic_click_no_im_falls_back_to_nearest_spectrum(self):
        # Use a state with no IM data; ensure fallback hits the loop branch.
        state = ViewerState()
        # Fake an experiment with two MS1 spectra at RTs 1.0 and 3.0.
        from unittest.mock import MagicMock

        spec_a = MagicMock()
        spec_a.getRT.return_value = 1.0
        spec_b = MagicMock()
        spec_b.getRT.return_value = 3.0
        exp = MagicMock()
        exp.__len__ = MagicMock(return_value=2)
        exp.__getitem__ = MagicMock(side_effect=lambda i: [spec_a, spec_b][i])
        state.exp = exp
        state.has_ion_mobility = False

        self._build_handler(state)(2.4)
        assert state.selected_spectrum_idx == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestTICClickSync -v
```

Expected: these will pass right away because they exercise the helper in the test itself — but they document the contract. The behavior change happens when we wire the new logic into the actual TIC panel below.

- [ ] **Step 3: Update the TIC click handler**

In `pyopenms_viewer/panels/tic_panel.py:195-222`, locate the existing click handler:

```python
clicked_rt = points[0].get("x", 0)
if self.state.rt_in_minutes:
    clicked_rt *= 60.0

# Find closest spectrum
if self.state.exp is not None:
    best_idx = 0
    best_diff = float("inf")
    for i in range(len(self.state.exp)):
        diff = abs(self.state.exp[i].getRT() - clicked_rt)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    # Select spectrum (triggers spectrum panel update)
    self.state.select_spectrum(best_idx)

    # Also select nearest IM frame if ion mobility data is present
    if self.state.has_ion_mobility and self.state.im_frame_indices:
        self.state.select_nearest_im_frame(clicked_rt)

    # Also center the peak map on this RT (matching original behavior)
    ...
```

Replace with:

```python
clicked_rt = points[0].get("x", 0)
if self.state.rt_in_minutes:
    clicked_rt *= 60.0

# Find closest spectrum
if self.state.exp is not None:
    if self.state.has_ion_mobility and self.state.ms1_im_frame_indices:
        # Prefer the nearest MS1 IM frame so spectrum + IM panels stay in sync.
        best_idx = self.state.find_nearest_ms1_im_frame_idx(clicked_rt)
    else:
        # Fallback: nearest spectrum of any MS level.
        best_idx = 0
        best_diff = float("inf")
        for i in range(len(self.state.exp)):
            diff = abs(self.state.exp[i].getRT() - clicked_rt)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

    # Select spectrum (triggers spectrum panel + IM panel updates via listeners)
    self.state.select_spectrum(best_idx)

    # Also center the peak map on this RT (matching original behavior)
    ...
```

(Keep the rest of the handler — RT centering and `emit_view_changed` — unchanged.)

- [ ] **Step 4: Run the targeted tests**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestTICClickSync -v
```

Expected: 2 passed.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass. If anything in `tests/test_tic*.py` or similar referenced `select_nearest_im_frame`, fix it now.

- [ ] **Step 6: Commit**

```bash
git add pyopenms_viewer/panels/tic_panel.py tests/test_ms2_ion_mobility.py
git commit -m "feat(tic_panel): pick nearest MS1 IM frame as selected spectrum"
```

---

## Task 7: IM panel info label and placeholder

**Files:**
- Modify: `pyopenms_viewer/panels/im_peak_map_panel.py` (`update` method around lines 154-182)
- Test: `tests/test_ms2_ion_mobility.py`

- [ ] **Step 1: Write the failing tests for the info-label builder**

Append to `tests/test_ms2_ion_mobility.py`:

```python
class TestIMInfoLabel:
    """Test the helper that builds the info-label string from state."""

    def _state(self, ms2_im_mzml_path):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader

        state = ViewerState()
        MzMLLoader(state).load_sync(str(ms2_im_mzml_path))
        return state

    def test_info_label_for_ms1(self, ms2_im_mzml_path):
        from pyopenms_viewer.panels.im_peak_map_panel import build_im_info_label

        state = self._state(ms2_im_mzml_path)
        state.selected_im_frame_idx = 0
        label = build_im_info_label(state)
        assert label.startswith("MS1 frame #0 |")
        assert "RT=1.00s" in label
        assert "precursor" not in label

    def test_info_label_for_ms2(self, ms2_im_mzml_path):
        from pyopenms_viewer.panels.im_peak_map_panel import build_im_info_label

        state = self._state(ms2_im_mzml_path)
        state.selected_im_frame_idx = 1
        label = build_im_info_label(state)
        assert label.startswith("MS2 frame #1 |")
        assert "RT=1.10s" in label
        assert "precursor 495.00–505.00 m/z" in label

    def test_info_label_when_no_selection(self, ms2_im_mzml_path):
        from pyopenms_viewer.panels.im_peak_map_panel import build_im_info_label

        state = self._state(ms2_im_mzml_path)
        state.selected_im_frame_idx = None
        label = build_im_info_label(state)
        assert label == "No ion mobility data for this spectrum"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestIMInfoLabel -v
```

Expected: FAIL with `ImportError: cannot import name 'build_im_info_label'`.

- [ ] **Step 3: Add the `build_im_info_label` helper**

In `pyopenms_viewer/panels/im_peak_map_panel.py`, at the top of the module (after the imports), add:

```python
def build_im_info_label(state) -> str:
    """Format the IM panel info label for the currently selected frame.

    Returns the placeholder string when no IM frame is selected.
    """
    idx = state.selected_im_frame_idx
    if idx is None or state.exp is None:
        return "No ion mobility data for this spectrum"

    spec = state.exp[idx]
    rt = spec.getRT()
    ms_level = state.get_im_frame_ms_level(idx)
    if ms_level is None:
        ms_level = spec.getMSLevel()

    base = f"MS{ms_level} frame #{idx} | RT={rt:.2f}s"
    if ms_level >= 2:
        lo = state.get_im_frame_precursor_lower(idx)
        hi = state.get_im_frame_precursor_upper(idx)
        if lo is not None and hi is not None:
            base += f" | precursor {lo:.2f}–{hi:.2f} m/z"
    return base
```

- [ ] **Step 4: Use the helper in `update()`**

In the same file, replace the existing info-label section in `update()` (around lines 167-182):

```python
# Update info label
if self.info_label is not None and self.state.has_ion_mobility:
    if self.state.selected_im_frame_idx is not None and self.state.exp is not None:
        idx = self.state.selected_im_frame_idx
        if 0 <= idx < len(self.state.exp):
            spec = self.state.exp[idx]
            ms_level = spec.getMSLevel()
            rt = spec.getRT()
            n_peaks = spec.size()
            self.info_label.set_text(
                f"Spectrum #{idx} | MS{ms_level} | RT={rt:.2f}s | {n_peaks:,} peaks"
                f" | {self.state.im_type or 'Unknown type'}"
            )
        else:
            self.info_label.set_text(f"Ion mobility data | {self.state.im_type or 'Unknown type'}")
    else:
        self.info_label.set_text(f"Ion mobility data | {self.state.im_type or 'Unknown type'}")
```

with:

```python
# Update info label
if self.info_label is not None and self.state.has_ion_mobility:
    self.info_label.set_text(build_im_info_label(self.state))
```

- [ ] **Step 5: Ensure the panel clears the image when no frame is selected**

Still in `update()`, locate the rendering block (around lines 158-161):

```python
base64_img = self.im_renderer.render(self.state)
if base64_img:
    self.im_image_element.set_source(f"data:image/png;base64,{base64_img}")
```

Replace with:

```python
if self.state.selected_im_frame_idx is None:
    if self.im_image_element is not None:
        self.im_image_element.set_source("")
else:
    base64_img = self.im_renderer.render(self.state)
    if base64_img:
        self.im_image_element.set_source(f"data:image/png;base64,{base64_img}")
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestIMInfoLabel -v
```

Expected: 3 passed.

- [ ] **Step 7: Run the full test suite**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add pyopenms_viewer/panels/im_peak_map_panel.py tests/test_ms2_ion_mobility.py
git commit -m "feat(im_panel): MS2 precursor info + placeholder for non-IM spectra"
```

---

## Task 8: Rendering smoke + final verification

**Files:**
- Test: `tests/test_ms2_ion_mobility.py`

- [ ] **Step 1: Add a rendering smoke test**

Append to `tests/test_ms2_ion_mobility.py`:

```python
class TestRenderingSmoke:
    def test_renders_ms2_frame(self, ms2_im_mzml_path):
        from pyopenms_viewer.loaders.mzml_loader import MzMLLoader
        from pyopenms_viewer.rendering.peak_map_renderer import IMPeakMapRenderer

        state = ViewerState()
        MzMLLoader(state).load_sync(str(ms2_im_mzml_path))
        state.selected_im_frame_idx = 1  # MS2 with IM

        assert state.get_im_frame_spectrum().getMSLevel() == 2

        renderer = IMPeakMapRenderer(
            plot_width=state.plot_width,
            plot_height=state.plot_height,
            margin_left=state.margin_left,
            margin_right=state.margin_right,
            margin_top=state.margin_top,
            margin_bottom=state.margin_bottom,
        )
        out = renderer.render(state)
        assert out, "expected non-empty base64 image for an MS2 IM frame"
```

- [ ] **Step 2: Run the smoke test**

```bash
uv run pytest tests/test_ms2_ion_mobility.py::TestRenderingSmoke -v
```

Expected: 1 passed.

- [ ] **Step 3: Run the entire test suite**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass (the previous 281 plus the new ones).

- [ ] **Step 4: Run lint**

```bash
uv run ruff check .
```

Expected: clean.

- [ ] **Step 5: Manual smoke check against a real diaPASEF file**

Start the server with a known diaPASEF file (substitute your own path; the user's reference file is `tests/data/DIA_HeLa_50ng_5_6min.mzML`, kept locally and not checked in — confirm the path with the user before starting if you don't have access):

```bash
uv run pyopenms-viewer tests/data/DIA_HeLa_50ng_5_6min.mzML
```

In the browser:
1. Wait for load. Confirm the IM panel auto-expands with an MS1 IM frame.
2. In the spectrum panel, click "MS2 >" to navigate to the next MS2 spectrum that has IM. Confirm:
   - The IM panel info label reads `MS2 frame #N | RT=... | precursor lo–hi m/z`.
   - The IM heatmap re-renders to show the MS2 frame's fragments.
3. Navigate to an MS2 spectrum with no IM (if any), or use "MS1 >" to step back. The IM panel should refresh with the new frame.
4. Click on the TIC at any RT. Confirm:
   - The spectrum panel jumps to an MS1 spectrum (not an MS2).
   - The IM panel shows the corresponding MS1 IM frame.

Report results (pass / fail per check) before considering this task complete.

- [ ] **Step 6: Commit any final test additions**

```bash
git add tests/test_ms2_ion_mobility.py
git commit -m "test: rendering smoke for MS2 IM frame"
```

---

## Done When

- All tasks above are checked off.
- `uv run pytest tests/ -x -q` passes.
- `uv run ruff check .` is clean.
- Manual smoke against the diaPASEF file confirms MS2 IM frames render with precursor info, MS1 frames still render, and TIC click still lands on MS1.
- All commits are on the branch.
