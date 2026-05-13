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
5. Preserve all existing MS1 behavior, including TIC-click → nearest MS1 IM frame.

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
- `im_frame_ms_levels: np.ndarray[int]` **(new)** — parallel; MS level per frame.
- `im_frame_precursor_mz: np.ndarray[float]` **(new)** — parallel; precursor target m/z for MS2, `np.nan` for MS1.
- `im_frame_precursor_lower: np.ndarray[float]` **(new)** — parallel; isolation-window lower bound (`mz - getIsolationWindowLowerOffset()`), `np.nan` for MS1.
- `im_frame_precursor_upper: np.ndarray[float]` **(new)** — parallel; isolation-window upper bound (`mz + getIsolationWindowUpperOffset()`), `np.nan` for MS1.
- `im_frame_position_by_index: dict[int, int]` **(new)** — maps spectrum index → position into the parallel arrays. Built once at load time. Used for O(1) membership testing (via `in`) from the spectrum panel and for O(1) lookup of per-frame metadata from helper methods.
- `ms1_im_frame_indices: list[int]` **(new, derived)** — `im_frame_indices` filtered to MS1, used exclusively by `select_nearest_im_frame` so TIC click never lands on MS2.
- `ms1_im_frame_rts: np.ndarray` **(new, derived)** — parallel to `ms1_im_frame_indices`.

Rationale for parallel numpy arrays: a few thousand floats fit easily in memory, allow vectorized filtering by MS level, and avoid per-frame dict lookups in hot paths. Storing precursor center plus lower/upper covers DDA-PASEF (narrow windows) and diaPASEF (wider windows) uniformly. NaN sentinels for MS1 let the info-label code decide whether to display precursor-window text without branching on MS level explicitly.

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

## Spectrum-Panel Integration

In `pyopenms_viewer/panels/spectrum_panel.py:show_spectrum` (~line 280), after `state.selected_spectrum_idx = spectrum_idx`:

```python
if state.has_ion_mobility:
    if spectrum_idx in state.im_frame_position_by_index:
        state.selected_im_frame_idx = spectrum_idx
    else:
        state.selected_im_frame_idx = None
    state.emit_view_changed()
```

This makes the IM panel always reflect the spectrum-panel selection.

**TIC-click path:** Currently `tic_panel.py:200-214` selects the nearest spectrum of *any* MS level and *separately* selects the nearest MS1 IM frame — under the new design this would diverge the two panels. Change TIC click so the selected spectrum is the nearest MS1 IM frame (use `ms1_im_frame_indices` / `ms1_im_frame_rts` to find it), then call `state.select_spectrum(best_idx)`. The downstream `show_spectrum` hook sets `selected_im_frame_idx` automatically, so the separate `select_nearest_im_frame` call becomes redundant and is removed. Both panels end up synced on the same MS1 IM frame.

`state.select_nearest_im_frame(rt)` (state.py:428) is kept (now a thin helper that returns the nearest MS1 frame index) and used by the TIC-click handler to compute `best_idx`.

## IM Panel Display

`pyopenms_viewer/panels/im_peak_map_panel.py`:

- When `state.selected_im_frame_idx is None`, render a placeholder: clear the image element to a blank canvas, set `info_label` to "No ion mobility data for this spectrum".
- When `state.selected_im_frame_idx is not None`, render via the existing `rasterizeIMFrame` path. Update `info_label` to include MS level and (for MS2+) precursor window:
  - MS1: `"MS1 frame #{idx} | RT={rt:.2f}s"`
  - MS2+: `"MS{level} frame #{idx} | RT={rt:.2f}s | precursor {lo:.2f}–{hi:.2f} m/z"`
- Helper methods on `ViewerState` (`get_im_frame_ms_level(idx)`, `get_im_frame_precursor_lower(idx)`, `get_im_frame_precursor_upper(idx)`) encapsulate the parallel-array layout so the panel doesn't index arrays directly.
- Panel title ("Ion Mobility Frame") stays as-is. The MS level lives in the info label.

`pyopenms_viewer/rendering/peak_map_renderer.py`: the existing early-return in `IMPeakMapRenderer.render` when `selected_im_frame_idx is None` (line 783) is reviewed to ensure it produces a clean empty canvas rather than stale pixels.

## Error Handling

- **Selected spectrum has no IM array:** IM panel clears, placeholder shown. See above.
- **MS2 spectrum has no precursor metadata:** isolation-window fields are `np.nan`. Info-label rendering falls back to `"MS{level} frame #{idx} | RT={rt:.2f}s"` (no precursor segment). No crash.
- **`rasterizeIMFrame` raises on a malformed MS2 spectrum:** propagates the same way it does today for malformed MS1 frames. Out of scope to add new try/except.

## Testing

**Fixture:** synthetic mzML built in a `conftest.py` fixture using `pyopenms.MSExperiment` + `MzMLFile().store()`. Contents: 2 MS1 frames with IM arrays, 4 MS2 frames with IM arrays and precursor isolation-window offsets, 1 MS2 frame *without* an IM array (for the placeholder path).

**Loader tests** (`tests/test_mzml_loader.py` or new file):
1. After load, `state.im_frame_indices` has length 6 (MS2-without-IM is excluded).
2. `state.im_frame_ms_levels` matches expected per-frame MS levels.
3. `state.im_frame_precursor_lower` / `_upper` are `nan` for MS1, match synthesized offsets for MS2.
4. `im_frame_indices` and `im_frame_rts` are sorted by RT and parallel to the new arrays.
5. `set(state.im_frame_position_by_index.keys()) == set(state.im_frame_indices)` and each `position_by_index[spec_idx]` correctly indexes back to `spec_idx` in `im_frame_indices`.

**State tests:**
6. `select_nearest_im_frame(rt)` only picks MS1 frames even when MS2 frames are closer in RT.
7. `get_im_frame_ms_level`, `get_im_frame_precursor_lower`, `get_im_frame_precursor_upper` return correct values for both MS1 and MS2 frames.

**Spectrum-panel integration tests:**
8. `show_spectrum(ms2_with_im_idx)` sets `state.selected_im_frame_idx == ms2_with_im_idx`.
9. `show_spectrum(ms2_no_im_idx)` sets `state.selected_im_frame_idx is None`.
10. `show_spectrum(ms1_im_idx)` sets `state.selected_im_frame_idx == ms1_im_idx`.

**TIC-click sync test:**
10a. Simulating a TIC click at an RT where an MS2 frame is *closer* than any MS1 IM frame ends with both `state.selected_spectrum_idx` and `state.selected_im_frame_idx` pointing at the nearest MS1 IM frame, not the MS2 frame.

**Rendering smoke tests:**
11. After selecting an MS2 frame, `state.get_im_frame_spectrum().getMSLevel() == 2` and a render call returns non-empty image bytes. (Asserts on inputs/outputs rather than mocking `rasterizeIMFrame`.)
12. With `selected_im_frame_idx = None`, the renderer produces a placeholder canvas and the info label contains "No ion mobility data".

**Regression coverage:**
13. Existing MS1-only IM tests in `tests/test_rendering.py` / `tests/test_ion_mobility.py` (if present) continue to pass unchanged.

## Out of Scope

- Aggregated "per precursor window" view across cycles.
- MS2 frame navigation UI inside the IM panel.
- Real-file performance testing of diaPASEF datasets (manual after merge).
- Cleanup of the dead `extract_ion_mobility_data` standalone loader.
- Mobilogram behavior changes for MS2 (the mobilogram code currently reads `im_df`, which is `None` on the live path; this is a pre-existing limitation that affects MS1 too and is out of scope here).
