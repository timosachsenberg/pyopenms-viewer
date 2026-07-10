"""End-to-end (browser) tests for the Ion Image panel (imzML / MSI).

Covers the parts of the PR #42 review fixes that unit tests cannot reach —
the NiceGUI lifecycle: the panel auto-opens and renders on load, the browse
loop (Extract → ion image) updates the heatmap, and the panel rehydrates
after a full page reload (Fix 3: token-deduped, deferred, race-safe render).

Self-contained: launches the real app as a subprocess with an imzML fixture
and drives it with a headless Chromium. Skips cleanly when the ``e2e`` extra
or the browser binary is not installed, so a plain ``uv run pytest`` (which
also excludes ``-m e2e``) is unaffected.

Run with:
    uv sync --extra e2e
    uv run playwright install chromium
    uv run pytest tests/e2e -m e2e
"""

import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="install the 'e2e' extra to run browser tests")

from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.e2e

# Example_Processed.imzML has a matching .ibd alongside (required by the loader).
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Example_Processed.imzML"
STARTUP_TIMEOUT_S = 90.0

# Plotly figure titles set by ImagingPanel — used as robust render signals
# instead of brittle DOM nesting.
TIC_TITLE = "TIC Image"
AGG_TITLE = "Mean spectrum"          # default aggregate mode
ION_TITLE = "Ion Image  m/z"         # after Extract / peak click
EMPTY_IMAGE_TITLE = "load an imzML file to display"

# Collect every Plotly plot's layout title text (heatmaps, spectra, overlay).
_TITLES_JS = """() => Array.from(document.querySelectorAll('.js-plotly-plot')).map(el => {
    const t = el.layout && el.layout.title;
    return (t && (t.text !== undefined ? t.text : t)) || '';
})"""

_HAS_TITLE_JS = """(sub) => Array.from(document.querySelectorAll('.js-plotly-plot')).some(el => {
    const t = el.layout && el.layout.title;
    const txt = (t && (t.text !== undefined ? t.text : t)) || '';
    return txt.includes(sub);
})"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"viewer process exited early with code {proc.returncode}")
        with contextlib.suppress(Exception):
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        time.sleep(0.5)
    raise TimeoutError(f"viewer did not come up at {url} within {timeout}s")


def _wait_for_title(page, substring: str, timeout: int = 60_000) -> None:
    page.wait_for_function(_HAS_TITLE_JS, arg=substring, timeout=timeout)


@pytest.fixture(scope="module")
def imzml_url():
    """Launch the app with an imzML file in browser mode; yield its base URL."""
    if not DATA_FILE.exists():
        pytest.skip(f"missing test data: {DATA_FILE}")
    if not DATA_FILE.with_suffix(".ibd").exists():
        pytest.skip(f"missing .ibd alongside {DATA_FILE}")

    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    log = tempfile.NamedTemporaryFile(
        "w+", suffix=".log", prefix="pyopenms_viewer_e2e_imaging_", delete=False
    )

    # NiceGUI switches ui.run() into an internal test mode when PYTEST_CURRENT_TEST
    # is set, then reads NICEGUI_SCREEN_TEST_PORT. The child must look like a
    # normal process, so strip both from its environment.
    child_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("NICEGUI_") and k != "PYTEST_CURRENT_TEST"
    }

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "pyopenms_viewer",
            str(DATA_FILE), "--browser", "--no-open", "--port", str(port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=child_env,
    )
    try:
        try:
            _wait_until_up(url, proc, STARTUP_TIMEOUT_S)
        except Exception as exc:
            log.flush()
            output = Path(log.name).read_text(errors="replace")[-4000:]
            raise RuntimeError(f"{exc}\n--- viewer output ---\n{output}") from exc
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
        log.close()
        with contextlib.suppress(OSError):
            Path(log.name).unlink()


@pytest.fixture
def page(imzml_url):
    """A fresh headless page on the running imzML viewer."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # browser binary not installed
            pytest.skip(f"chromium unavailable (run 'playwright install chromium'): {exc}")
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        pg.goto(imzml_url, wait_until="networkidle")
        try:
            yield pg
        finally:
            ctx.close()
            browser.close()


def test_ion_image_renders_and_extract_updates(page):
    """On load the Ion Image panel shows the TIC image + aggregate spectrum,
    and the Extract browse loop swaps the heatmap to an ion image."""
    # Panel auto-opens and renders the TIC image + aggregate spectrum.
    _wait_for_title(page, TIC_TITLE)
    _wait_for_title(page, AGG_TITLE)

    # Browse loop: Extract at the default m/z must flip the heatmap title.
    page.get_by_role("button", name="Extract", exact=True).first.click()
    _wait_for_title(page, ION_TITLE, timeout=30_000)

    titles = page.evaluate(_TITLES_JS)
    assert any(ION_TITLE in t for t in titles), titles
    # The image is no longer the empty placeholder.
    assert not any(EMPTY_IMAGE_TITLE in t for t in titles), titles


def test_panel_rehydrates_after_reload(page):
    """A full page reload must rehydrate the Ion Image panel (Fix 3), not leave
    it stuck on the empty placeholder."""
    _wait_for_title(page, TIC_TITLE)

    page.reload(wait_until="networkidle")

    # After reload the panel must render the TIC image again from preserved
    # server-side state — exactly the rehydration path the fix targets.
    _wait_for_title(page, TIC_TITLE)
    _wait_for_title(page, AGG_TITLE)
    titles = page.evaluate(_TITLES_JS)
    assert not any(EMPTY_IMAGE_TITLE in t for t in titles), titles
