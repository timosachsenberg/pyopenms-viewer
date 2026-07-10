"""Fixtures for the browser-driven end-to-end suite.

These tests launch the real application as a subprocess and drive it with
Playwright. They are skipped automatically when Playwright (the ``e2e`` extra)
or its browser binary is not installed, so a plain ``uv run pytest`` on the
default dev environment is unaffected.

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

# Skip the whole package cleanly if the extra isn't installed.
pytest.importorskip("playwright", reason="install the 'e2e' extra to run browser tests")

from playwright.sync_api import sync_playwright  # noqa: E402

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "BSA1_F1.mzML"
STARTUP_TIMEOUT_S = 90.0


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


@pytest.fixture(scope="session")
def viewer_url():
    """Launch the app with BSA1_F1.mzML in browser mode; yield its base URL."""
    if not DATA_FILE.exists():
        pytest.skip(f"missing test data: {DATA_FILE}")

    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    log = tempfile.NamedTemporaryFile("w+", suffix=".log", prefix="pyopenms_viewer_e2e_", delete=False)

    # NiceGUI switches ui.run() into an internal test mode when PYTEST_CURRENT_TEST
    # is set (helpers.is_pytest()), then reads NICEGUI_SCREEN_TEST_PORT. The child
    # must look like a normal process, so strip both from its environment.
    child_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("NICEGUI_") and k != "PYTEST_CURRENT_TEST"
    }

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pyopenms_viewer",
            str(DATA_FILE),
            "--browser",
            "--no-open",
            "--port",
            str(port),
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


@pytest.fixture(scope="session")
def _browser():
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:  # browser binary not installed
                pytest.skip(f"chromium unavailable (run 'playwright install chromium'): {exc}")
            yield browser
            browser.close()
    except Exception as exc:
        pytest.skip(f"playwright could not start: {exc}")


@pytest.fixture
def page(_browser, viewer_url):
    """A fresh page loaded on the running viewer, with data already loaded."""
    ctx = _browser.new_context(viewport={"width": 1600, "height": 1200})
    pg = ctx.new_page()
    pg.goto(viewer_url, wait_until="networkidle")
    # Wait for the CLI-loaded mzML to finish parsing (info label shows peak count).
    pg.wait_for_function(
        "() => document.body.innerText.includes('Peaks:')",
        timeout=60_000,
    )
    yield pg
    ctx.close()
