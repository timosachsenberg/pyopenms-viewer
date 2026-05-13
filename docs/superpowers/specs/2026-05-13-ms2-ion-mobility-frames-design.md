# MS2 Ion Mobility Frames — Design

**Date:** 2026-05-13
**Status:** Approved, ready for implementation planning

## Problem

The IM panel currently displays one frame at a time via `MSSpectrum.rasterizeIMFrame`, driven by `state.selected_im_frame_idx`. However, the mzML loader only collects MS1 spectra into `state.im_frame_indices` — MS2 IM frames (e.g., diaPASEF fragment frames) are filtered out at extraction time. As a result, users cannot view the m/z×IM cloud of any MS2 spectrum, even though `rasterizeIMFrame` would work on it.

Target data: diaPASEF (wide, repeating precursor isolation windows). The user's primary inspection workflow is per-frame — view the m/z×IM fragments of a single MS2 spectrum, selected from the existing spectrum panel (e.g., via MS2 navigation buttons, idXML, or precursor click). No aggregation across cycles.

## Goals

1. Load IM data from all MS levels, not just MS1.
2. Allow the IM panel to render any selected spectrum that has an IM array, regardless of MS level.
3. Drive IM-panel selection from the existing spectrum panel: when an MS2 spectrum with IM is selected, the IM panel updates to show that frame.
4. When the selected spectrum has no IM array, the IM panel clears and shows a placeholder.
5. Preserve the visible MS1 experience: the IM panel still renders an MS1 frame by default, and TIC clicks still land on MS1 IM frames (now via spectrum selection rather than a parallel IM-only path).

## Non-Goals

- No aggregated "per precursor window" view across cycles.
- No new MS2 frame navigation UI inside the IM panel (no MS-level toggle, no window dropdown, no RT slider). Selection is entirely spectrum-panel-driven.
- No changes to the dead-code `extract_ion_mobility_data` fallback path in `pyopenms_viewer/loaders/ion_mobility_loader.py` (not on the live load path).
- No performance work on real diaPASEF files in this iteration (validated manually after merge).

## Approach Chosen

**Approach A — Extend the existing IM-frame model to all MS levels.** Treat MS1 and MS2 IM frames uniformly: one `selected_im_frame_idx`, one frame-index table, single rendering path. Add per-frame metadata (MS level, precursor isolation window) as parallel numpy arrays so consumers can filter by MS level cheaply.

Alternatives considered:
- **B (parallel MS2 tracking):** separate `ms2_im_frame_indices` + `selected_ms2_im_frame_idx`. Rejected — duplicates state and forces the panel to pick between two selections.
- **C (no state change, read peaks directly from the selected spectrum):** rejected — loses the frame-index table needed for the placeholder logic and future MS2 frame navigation, and breaks symmetry with the MS1 path.

## Data Model

New and modified attributes on `ViewerState`:

- `im_frame_indices: list[int]` *(existing)* — now includes every spectrum index whose spectrum carries an IM float-data-array, regardless of MS level. Sorted by RT (existing invariant preserved).
- `im_frame_rts: np.ndarray` *(existing)* — parallel to `im_frame_indices`.
- `im_frame_ms_levels: np.ndarray` **(new)**, dtype `np.int32` — parallel; MS level per frame.
- `im_frame_precursor_mz: np.ndarray` **(new)**, dtype `np.float64` — parallel; precursor target m/z for MS2, `np.nan` for MS1.
- `im_frame_precursor_lower: np.ndarray` **(new)**, dtype `np.float64` — parallel; isolation-window lower bound (`mz - getIsolationWindowLowerOffset()`), `np.nan` for MS1.
- `im_frame_precursor_upper: np.ndarray` **(new)**, dtype `np.float64` — parallel; isolation-window upper bound (`mz + getIsolationWindowUpperOffset()`), `np.nan` for MS1.
- `im_frame_position_by_index: dict[int, int]` **(new)** — maps spectrum index → position into the parallel arrays. Built once at load time. Used for O(1) membership testing (via `in`) from the spectrum panel and for O(1) lookup of per-frame metadata from helper methods.
- `ms1_im_frame_indices: list[int]` **(new, derived)** — `im_frame_indices` filtered to MS1, used exclusively by `find_nearest_ms1_im_frame_idx` so TIC click never lands on MS2.
- `ms1_im_frame_rts: np.ndarray` **(new, derived)** — parallel to `ms1_im_frame_indices`.

Rationale for parallel numpy arrays: a few thousand floats fit easily in memory, allow vectorized filtering by MS level, and avoid per-frame dict lookups in hot paths. Storing precursor center plus lower/upper covers DDA-PASEF (narrow windows) and diaPASEF (wider windows) uniformly. NaN sentinels for MS1 let the info-label code decide whether to display precursor-window text without branching on MS level explicitly.

**State teardown:** all new attributes must be cleared alongside `selected_im_frame_idx` in `state.py:839` (the existing reset path). Reset arrays to empty `np.ndarray` / empty dict / empty list as appropriate so the panel can rely on truthiness checks.

## Loader Changes

`pyopenms_viewer/loaders/mzml_loader.py`:

- In the per-spectrum loop (~lines 221–328), move the IM detection block (currently lines 304–314) and the IM extraction block (currently lines 316–328) **out of** the `if ms_level == 1:` branch (line 270). They become unconditional within the loop, gated only by `n > 0`.
- For every spectrum with an IM array present, append:
  - peak data to `im_mz_list`, `im_im_list`, `im_int_list`,
  - the frame index to `im_frame_indices_list`,
  - `ms_level` to a new `im_frame_ms_levels_list`,
  - precursor metadata to new `im_frame_precursor_mz_list`, `im_frame_precursor_lower_list`, `im_frame_precursor_upper_list`. For MS1, append `np.nan`. For MS2+, read from `spec.getPrecursors()[0]` if present; if absent (malformed mzML), append `np.nan` defensively.
- In `_process_ion_mobility_data` (line 469), accept the new lists, sort them with the existing `sort_order` argsort applied to RTs (stable sort; ties on identical RT are broken by spectrum index — set the sort key to `(rt, spec_idx)` to make tie-breaking explicit), and assign to the new state arrays. Also build `im_frame_position_by_index`, `ms1_im_frame_indices`, and `ms1_im_frame_rts`.
- During implementation, sanity-check the isolation-window math against a known input (e.g., precursor 500.0 m/z with `getIsolationWindowLowerOffset() == 5.0` should produce `lower = 495.0`). The OpenMS offset convention is positive offsets subtracted/added from the target m/z, but verify before relying on it.

`pyopenms_viewer/loaders/ion_mobility_loader.py`: **no changes**. This standalone loader is exported but not called on the live load path (confirmed by grep). Leaving it untouched keeps scope tight; a separate cleanup task can remove it later.

Memory impact: rasterization path stores no per-frame peaks (only the index arrays grow). Each new array adds ~8 bytes × n_IM_frames. For a 60-minute PASEF run with ~50k MS2 frames, this is ~400 KB per array — negligible.

## Selection Integration

**Event flow today:** `state.select_spectrum(idx)` sets `selected_spectrum_idx` and emits `selection_changed`. The spectrum panel subscribes to `on_selection_changed` and calls `show_spectrum` to redraw. The IM panel does *not* subscribe to selection events — it only listens to `on_view_changed`.

**New integration:** the IM panel subscribes to `state.on_selection_changed` directly. No changes to `spectrum_panel.show_spectrum`. This avoids cross-panel coupling and avoids touching the ~25 call sites of `show_spectrum`.

In `pyopenms_viewer/panels/im_peak_map_panel.py.build()`, after the existing `on_view_changed` subscription:

```python
self.state.on_selection_changed(self._on_selection_changed)

def _on_selection_changed(self, selection_type: str, index: int | None) -> None:
    if selection_type != "spectrum":
        return
    if not self.state.has_ion_mobility:
        return
    if index is not None and index in self.state.im_frame_position_by_index:
        self.state.selected_im_frame_idx = index
    else:
        self.state.selected_im_frame_idx = None
    self._refresh_image()  # whichever existing method re-renders the canvas
```

The IM panel becomes the single writer of `selected_im_frame_idx` for selection-driven updates (loader still writes the default at load time; reset still clears it).

**TIC-click path (`pyopenms_viewer/panels/tic_panel.py:200-214`):**

Today TIC click picks the nearest spectrum of any MS level via a linear scan and also calls `state.select_nearest_im_frame(rt)`. Replace both:

1. If `state.has_ion_mobility` is True: compute `best_idx = state.find_nearest_ms1_im_frame_idx(rt)` (see below) and call `state.select_spectrum(best_idx)`. The IM panel's new `on_selection_changed` handler then updates the IM frame automatically — no separate `select_nearest_im_frame` call.
2. If `state.has_ion_mobility` is False (no IM data in the file): keep today's "nearest spectrum of any level" behavior unchanged.

**Helper refactor:** `state.select_nearest_im_frame(rt)` (state.py:428) is the only existing function whose signature changes. Today it mutates `selected_im_frame_idx`. Refactor to a non-mutating helper `state.find_nearest_ms1_im_frame_idx(rt) -> int | None` that returns the spectrum index of the nearest MS1 IM frame using `ms1_im_frame_indices` / `ms1_im_frame_rts` and binary search (matching today's logic). Returns `None` if no MS1 IM frames are loaded. The TIC handler uses the returned index. Grep before deleting the old name to confirm it has only the one caller in `tic_panel.py:214`; remove it.

**IM panel helpers (`ViewerState`):**

- `state.get_im_frame_ms_level(spec_idx: int) -> int | None` — returns `int(im_frame_ms_levels[im_frame_position_by_index[spec_idx]])` if `spec_idx` is an IM frame, else `None`.
- `state.get_im_frame_precursor_lower(spec_idx: int) -> float | None` — same pattern, returns `None` if NaN or not present.
- `state.get_im_frame_precursor_upper(spec_idx: int) -> float | None` — same.

All three take a *spectrum index* (the value of `selected_im_frame_idx`), not an array position; internal lookup goes through `im_frame_position_by_index`.

## IM Panel Display

`pyopenms_viewer/panels/im_peak_map_panel.py`:

- When `state.selected_im_frame_idx is None`, render a placeholder: clear the image element to a blank canvas, set `info_label` to "No ion mobility data for this spectrum".
- When `state.selected_im_frame_idx is not None`, render via the existing `rasterizeIMFrame` path. Use the `ViewerState` helpers defined in the Selection Integration section to populate `info_label`:
  - MS1: `"MS1 frame #{idx} | RT={rt:.2f}s"`
  - MS2+: `"MS{level} frame #{idx} | RT={rt:.2f}s | precursor {lo:.2f}–{hi:.2f} m/z"` when both precursor bounds are non-`None`; fall back to the MS1-shaped label when precursor metadata is missing.
- Panel title ("Ion Mobility Frame") stays as-is. The MS level lives in the info label.

`pyopenms_viewer/rendering/peak_map_renderer.py`: the existing early-return in `IMPeakMapRenderer.render` when `selected_im_frame_idx is None` (line 783) is reviewed to ensure it produces a clean empty canvas rather than stale pixels. If it currently returns an empty string / no-op (which would leave a stale image in the NiceGUI `interactive_image`), update the panel's `_refresh_image` to explicitly set the element source to a 1×1 transparent PNG (or equivalent) when the renderer signals empty.

## Error Handling

- **Selected spectrum has no IM array:** IM panel clears, placeholder shown. See above.
- **MS2 spectrum has no precursor metadata:** isolation-window fields are `np.nan`. Info-label rendering falls back to `"MS{level} frame #{idx} | RT={rt:.2f}s"` (no precursor segment). No crash.
- **`rasterizeIMFrame` raises on a malformed MS2 spectrum:** propagates the same way it does today for malformed MS1 frames. Out of scope to add new try/except.

## Testing

**Fixture status:** `tests/data/ims_example.mzML` is MS1-only IM (4 spectra) — useful for the existing MS1 IM tests but does not exercise MS2. `tests/data/DIA_HeLa_50ng_5_6min.mzML` is 2.6 GB diaPASEF — too large for the test suite. We add a synthetic mzML fixture.

**Fixture:** built in a `conftest.py` session-scoped fixture using `pyopenms.MSExperiment` + `MzMLFile().store()`, written to a `tmp_path_factory` directory and reused across tests. API verified end-to-end during design (synthetic file roundtrips MS level, RT, `Precursor.getMZ`/`getIsolationWindowLowerOffset`/`getIsolationWindowUpperOffset`, and FloatDataArray named `"ion mobility"`).

Contents:
- 2 MS1 frames with IM arrays at distinct RTs.
- 4 MS2 frames with IM arrays and `Precursor` set up with non-zero isolation-window offsets, interleaved by RT so that some MS2 frames are temporally closer to certain RTs than the MS1 frames (needed for the TIC-sync test).
- 1 MS2 frame *without* an IM array (for the placeholder path).
- 1 MS1 frame *without* an IM array (covers MS1 placeholder edge case; cheap to include).

Total: 8 spectra, ~10 KB on disk.

**Loader tests** (extend `tests/test_im_rasterization.py` or create `tests/test_mzml_loader_ms2_im.py`):
1. After load, `state.im_frame_indices` has length 6 (2 MS1 with IM + 4 MS2 with IM; MS1-without-IM and MS2-without-IM are excluded).
2. `state.im_frame_ms_levels` matches expected per-frame MS levels.
3. `state.im_frame_precursor_lower` / `_upper` are `nan` for MS1 frames, match synthesized offsets for MS2 frames (e.g., precursor 500.0 with `lo_offset=5.0` → `lower == 495.0`).
4. `im_frame_indices` and `im_frame_rts` are sorted by RT and parallel to the new arrays.
5. `set(state.im_frame_position_by_index.keys()) == set(state.im_frame_indices)` and each `position_by_index[spec_idx]` correctly indexes back to `spec_idx` in `im_frame_indices`.

**State tests:**
6. `find_nearest_ms1_im_frame_idx(rt)` returns only MS1 frame indices even when MS2 frames are closer in RT. Returns `None` when no MS1 IM frames are loaded.
7. `get_im_frame_ms_level`, `get_im_frame_precursor_lower`, `get_im_frame_precursor_upper` return correct values for both MS1 and MS2 frames; return `None` for spectrum indices that aren't IM frames.
7a. After a fresh `state.reset()`, all new attributes are empty/cleared and `state.has_ion_mobility is False`.

**Selection integration tests** (the IM panel reacts to `selection_changed` events):
8. Calling `state.select_spectrum(ms2_with_im_idx)` causes the IM panel's listener to set `state.selected_im_frame_idx == ms2_with_im_idx`.
9. Calling `state.select_spectrum(ms2_no_im_idx)` causes the listener to set `state.selected_im_frame_idx is None`.
10. Calling `state.select_spectrum(ms1_im_idx)` causes the listener to set `state.selected_im_frame_idx == ms1_im_idx`.
10a. When `state.has_ion_mobility is False`, the listener is a no-op (does not touch `selected_im_frame_idx`).

**TIC-click sync test:**
11. Simulating a TIC click handler at an RT where an MS2 frame is *closer* than any MS1 IM frame ends with both `state.selected_spectrum_idx` and `state.selected_im_frame_idx` pointing at the nearest MS1 IM frame.
11a. With no IM data loaded, the TIC handler falls back to today's nearest-spectrum-of-any-level behavior (sanity coverage).

**Rendering smoke tests:**
12. After selecting an MS2 frame, `state.get_im_frame_spectrum().getMSLevel() == 2` and an `IMPeakMapRenderer.render(state)` call returns a non-empty image source string. Asserts inputs/outputs, no mocking of `rasterizeIMFrame`.
13. With `selected_im_frame_idx = None`, `IMPeakMapRenderer.render(state)` returns the placeholder source (or empty marker) and the panel's info label contains "No ion mobility data".

**Regression coverage:**
14. Existing IM tests in `tests/test_im_rasterization.py` continue to pass unchanged.

## Out of Scope

- Aggregated "per precursor window" view across cycles.
- MS2 frame navigation UI inside the IM panel.
- Real-file performance testing of diaPASEF datasets (manual after merge).
- Cleanup of the dead `extract_ion_mobility_data` standalone loader.
- Mobilogram behavior changes for MS2 (the mobilogram code currently reads `im_df`, which is `None` on the live path; this is a pre-existing limitation that affects MS1 too and is out of scope here).
