"""End-to-end tests for RT-unit (seconds/minutes) signaling.

Covers the fixes on this branch:
- Toggling the peak-map RT unit re-renders the TIC x-axis (it previously stayed
  stale until an unrelated view_changed fired).
- The 3D peak-map view honors the RT unit instead of a hardcoded "RT (s)".
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def _tic_xtitle(page) -> str:
    """Current TIC Plotly x-axis title text ('' if not rendered)."""
    return page.evaluate(
        """() => {
            const t = document.querySelector('.js-plotly-plot .xtitle');
            return t ? t.textContent.trim() : '';
        }"""
    )


def _scene_x_title(page):
    """x-axis title of the first Plotly plot that has a 3D scene, or None."""
    return page.evaluate(
        """() => {
            for (const el of document.querySelectorAll('.js-plotly-plot')) {
                const s = el.layout && el.layout.scene;
                if (s && s.xaxis && s.xaxis.title) return (s.xaxis.title.text ?? s.xaxis.title);
            }
            return null;
        }"""
    )


def _set_rt_unit(page, unit: str) -> None:
    """Click the peak-map RT-unit toggle and wait for the TIC to reflect it.

    Idempotent: clicking the already-active option leaves the state unchanged.
    """
    assert unit in ("sec", "min")
    page.get_by_role("button", name=unit, exact=True).first.click()
    expected = "RT (min)" if unit == "min" else "RT (s)"
    page.wait_for_function(
        f"""() => {{
            const t = document.querySelector('.js-plotly-plot .xtitle');
            return t && t.textContent.trim() === {expected!r};
        }}""",
        timeout=15_000,
    )


def test_tic_axis_follows_rt_unit_toggle(page):
    """Toggling the peak-map RT unit drives the TIC x-axis, both directions."""
    # Normalize to seconds regardless of any persisted server-side state.
    _set_rt_unit(page, "sec")
    assert _tic_xtitle(page) == "RT (s)"

    # Seconds -> minutes: this is the behavior the fix restores.
    _set_rt_unit(page, "min")
    assert _tic_xtitle(page) == "RT (min)"

    # And back again.
    _set_rt_unit(page, "sec")
    assert _tic_xtitle(page) == "RT (s)"


def test_3d_view_axis_follows_rt_unit(page):
    """The 3D scene x-axis title reflects the RT unit (was hardcoded 'RT (s)')."""
    _set_rt_unit(page, "min")

    # Zoom to a small, peak-rich window so the 3D view actually renders
    # (it refuses to render regions larger than the RT/mz thresholds).
    page.keyboard.press("g")
    dialog = page.locator(".q-dialog")
    expect(dialog).to_be_visible(timeout=10_000)
    inputs = dialog.locator("input")
    expect(inputs).to_have_count(4, timeout=10_000)  # rt_min, rt_max, mz_min, mz_max (RT shown in minutes)
    for i, value in enumerate(["30.6", "31.0", "400", "445"]):
        inputs.nth(i).click()
        inputs.nth(i).fill(value)
        inputs.nth(i).press("Tab")
    dialog.get_by_role("button", name="Apply", exact=True).click()

    # Enable the 3D view and wait for the scene to appear.
    page.get_by_role("button", name="3D", exact=True).first.click()
    page.wait_for_function(
        """() => {
            for (const el of document.querySelectorAll('.js-plotly-plot')) {
                const s = el.layout && el.layout.scene;
                if (s && s.xaxis && s.xaxis.title) return true;
            }
            return false;
        }""",
        timeout=30_000,
    )

    assert _scene_x_title(page) == "RT (min)"
