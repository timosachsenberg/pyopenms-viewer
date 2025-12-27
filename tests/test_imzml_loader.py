"""Tests for the ImzMLLoader (imzML support)."""

from pathlib import Path
from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.loaders import ImzMLLoader

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_IMZML = TEST_DATA_DIR / "test.imzML"

import pytest

@pytest.mark.skipif(not TEST_IMZML.exists(), reason="No test imzML file available")
def test_parse_imzml_file_success():
    """Test that a valid imzML file can be parsed."""
    state = ViewerState()
    loader = ImzMLLoader(state)
    result = loader.parse(str(TEST_IMZML))
    assert result is True
    assert loader.coordinates is not None
    assert len(loader.coordinates) > 0

@pytest.mark.skipif(not TEST_IMZML.exists(), reason="No test imzML file available")
def test_process_imzml_file():
    """Test that imzML file can be processed and DataFrame is created."""
    state = ViewerState()
    loader = ImzMLLoader(state)
    assert loader.parse(str(TEST_IMZML))
    result = loader.process(str(TEST_IMZML))
    assert result is True
    assert state.df is not None
    assert len(state.df) > 0
    assert all(col in state.df.columns for col in ["mz", "intensity", "x", "y", "z"])
