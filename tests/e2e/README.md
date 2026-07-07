# End-to-end (browser) tests

These tests launch the real app as a subprocess and drive it with a headless
browser via Playwright. They live behind the `e2e` optional extra and the
`e2e` pytest marker so a normal `uv run pytest` is unaffected when the extra
isn't installed.

## Setup

```bash
uv sync --extra e2e
uv run playwright install chromium   # one-time browser download
```

## Run

```bash
uv run pytest tests/e2e -m e2e
```

If Playwright or its browser binary is missing, the suite skips itself rather
than failing. The tests use the committed `tests/data/BSA1_F1.mzML` fixture and
pick a free port automatically, so multiple runs won't collide.
