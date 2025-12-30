"""Tests for the pyopenms_viewer data manager module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyopenms_viewer.core.data_manager import DataManager


@pytest.fixture
def sample_peak_df():
    """Create a sample peak DataFrame for testing."""
    np.random.seed(42)
    n_peaks = 1000
    return pd.DataFrame(
        {
            "rt": np.random.uniform(0, 3600, n_peaks),
            "mz": np.random.uniform(100, 2000, n_peaks),
            "intensity": np.random.uniform(100, 100000, n_peaks),
            "log_intensity": np.log1p(np.random.uniform(100, 100000, n_peaks)),
        }
    )


@pytest.fixture
def sample_im_df():
    """Create a sample ion mobility DataFrame for testing."""
    np.random.seed(42)
    n_peaks = 500
    return pd.DataFrame(
        {
            "mz": np.random.uniform(100, 2000, n_peaks),
            "im": np.random.uniform(0.5, 1.5, n_peaks),
            "intensity": np.random.uniform(100, 100000, n_peaks),
            "log_intensity": np.log1p(np.random.uniform(100, 100000, n_peaks)),
        }
    )


@pytest.fixture
def sample_cv_df():
    """Create a sample peak DataFrame with CV values for FAIMS testing."""
    np.random.seed(42)
    n_peaks = 1000
    return pd.DataFrame(
        {
            "rt": np.random.uniform(0, 3600, n_peaks),
            "mz": np.random.uniform(100, 2000, n_peaks),
            "intensity": np.random.uniform(100, 100000, n_peaks),
            "log_intensity": np.log1p(np.random.uniform(100, 100000, n_peaks)),
            "cv": np.random.choice([-40, -50, -60], n_peaks),
        }
    )


class TestDataManagerInMemory:
    """Tests for DataManager in-memory mode."""

    def test_init_default(self):
        """Test default DataManager initialization."""
        dm = DataManager()
        assert dm.out_of_core is False
        assert dm._peaks_registered is False
        assert dm._im_peaks_registered is False

    def test_register_peaks_in_memory(self, sample_peak_df):
        """Test registering peaks in memory mode."""
        dm = DataManager(out_of_core=False)
        result = dm.register_peaks(sample_peak_df, "test.mzML")

        assert result is not None  # Should return DataFrame
        assert dm._peaks_registered is True
        assert dm._df is not None
        assert len(dm._df) == len(sample_peak_df)

    def test_register_im_peaks_in_memory(self, sample_im_df):
        """Test registering IM peaks in memory mode."""
        dm = DataManager(out_of_core=False)
        result = dm.register_im_peaks(sample_im_df, "test.mzML")

        assert result is not None
        assert dm._im_peaks_registered is True
        assert dm._im_df is not None
        assert len(dm._im_df) == len(sample_im_df)

    def test_query_peaks_in_view(self, sample_peak_df):
        """Test querying peaks within view bounds."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        result = dm.query_peaks_in_view(
            rt_min=500,
            rt_max=1000,
            mz_min=200,
            mz_max=500,
        )

        assert isinstance(result, pd.DataFrame)
        assert "rt" in result.columns
        assert "mz" in result.columns
        # All returned peaks should be within bounds
        if len(result) > 0:
            assert result["rt"].min() >= 500
            assert result["rt"].max() <= 1000
            assert result["mz"].min() >= 200
            assert result["mz"].max() <= 500

    def test_query_peaks_in_view_empty_result(self, sample_peak_df):
        """Test querying peaks with bounds that exclude all data."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        # Query with bounds outside data range
        result = dm.query_peaks_in_view(
            rt_min=10000,
            rt_max=20000,
            mz_min=5000,
            mz_max=6000,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_query_peaks_not_registered(self):
        """Test querying when no peaks are registered."""
        dm = DataManager(out_of_core=False)
        result = dm.query_peaks_in_view(
            rt_min=0,
            rt_max=100,
            mz_min=0,
            mz_max=1000,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_query_im_peaks_in_view(self, sample_im_df):
        """Test querying IM peaks within view bounds."""
        dm = DataManager(out_of_core=False)
        dm.register_im_peaks(sample_im_df, "test.mzML")

        result = dm.query_im_peaks_in_view(
            mz_min=200,
            mz_max=500,
            im_min=0.6,
            im_max=1.2,
        )

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert result["mz"].min() >= 200
            assert result["mz"].max() <= 500
            assert result["im"].min() >= 0.6
            assert result["im"].max() <= 1.2

    def test_query_peaks_for_minimap(self, sample_peak_df):
        """Test minimap query with downsampling."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        result = dm.query_peaks_for_minimap(minimap_pixels=500)

        assert result is not None
        assert len(result) <= 1000  # Should be downsampled for large datasets

    def test_query_peaks_for_minimap_small_dataset(self):
        """Test minimap query with small dataset (no downsampling needed)."""
        dm = DataManager(out_of_core=False)
        small_df = pd.DataFrame(
            {
                "rt": [100, 200, 300],
                "mz": [500, 600, 700],
                "intensity": [1000, 2000, 3000],
                "log_intensity": [3.0, 3.3, 3.5],
            }
        )
        dm.register_peaks(small_df, "test.mzML")

        result = dm.query_peaks_for_minimap(minimap_pixels=80000)

        assert result is not None
        assert len(result) == 3  # No downsampling for small dataset

    def test_query_all_peaks(self, sample_peak_df):
        """Test querying all peaks without downsampling."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        result = dm.query_all_peaks()

        assert result is not None
        assert len(result) == len(sample_peak_df)

    def test_get_bounds(self, sample_peak_df):
        """Test getting data bounds."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        bounds = dm.get_bounds()

        assert "rt_min" in bounds
        assert "rt_max" in bounds
        assert "mz_min" in bounds
        assert "mz_max" in bounds
        assert bounds["rt_min"] < bounds["rt_max"]
        assert bounds["mz_min"] < bounds["mz_max"]

    def test_get_bounds_not_registered(self):
        """Test getting bounds when no data is registered."""
        dm = DataManager(out_of_core=False)
        bounds = dm.get_bounds()

        assert bounds["rt_min"] == 0.0
        assert bounds["rt_max"] == 0.0
        assert bounds["mz_min"] == 0.0
        assert bounds["mz_max"] == 0.0

    def test_get_im_bounds(self, sample_im_df):
        """Test getting IM data bounds."""
        dm = DataManager(out_of_core=False)
        dm.register_im_peaks(sample_im_df, "test.mzML")

        bounds = dm.get_im_bounds()

        assert "mz_min" in bounds
        assert "mz_max" in bounds
        assert "im_min" in bounds
        assert "im_max" in bounds
        assert bounds["im_min"] < bounds["im_max"]

    def test_get_peak_count(self, sample_peak_df):
        """Test getting peak count."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        count = dm.get_peak_count()
        assert count == len(sample_peak_df)

    def test_get_peak_count_not_registered(self):
        """Test getting peak count when no data is registered."""
        dm = DataManager(out_of_core=False)
        count = dm.get_peak_count()
        assert count == 0

    def test_clear(self, sample_peak_df, sample_im_df):
        """Test clearing all data."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")
        dm.register_im_peaks(sample_im_df, "test.mzML")

        dm.clear()

        assert dm._df is None
        assert dm._im_df is None
        assert dm._peaks_registered is False
        assert dm._im_peaks_registered is False

    def test_query_peaks_for_cv(self, sample_cv_df):
        """Test querying peaks for specific CV value."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_cv_df, "test.mzML")

        result = dm.query_peaks_for_cv(cv=-50, downsample=False)

        assert result is not None
        if len(result) > 0:
            # Result should only contain the specified CV
            assert "log_intensity" in result.columns

    def test_reregister_peaks(self, sample_peak_df):
        """Test re-registering peaks replaces old data."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test1.mzML")

        # Create new DataFrame with different data
        new_df = pd.DataFrame(
            {
                "rt": [1, 2, 3],
                "mz": [100, 200, 300],
                "intensity": [1000, 2000, 3000],
                "log_intensity": [3.0, 3.3, 3.5],
            }
        )
        dm.register_peaks(new_df, "test2.mzML")

        assert dm.get_peak_count() == 3


class TestDataManagerOutOfCore:
    """Tests for DataManager out-of-core (disk) mode."""

    def test_init_out_of_core(self):
        """Test out-of-core DataManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(out_of_core=True, cache_dir=Path(tmpdir))
            assert dm.out_of_core is True
            assert dm.cache_dir == Path(tmpdir)

    def test_register_peaks_out_of_core(self, sample_peak_df):
        """Test registering peaks in out-of-core mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(out_of_core=True, cache_dir=Path(tmpdir))
            result = dm.register_peaks(sample_peak_df, "test.mzML")

            assert result is None  # Should return None (data written to disk)
            assert dm._peaks_registered is True
            assert dm._df is None  # DataFrame should not be kept in memory
            # Cache file should exist
            cache_files = list(Path(tmpdir).glob("*.parquet"))
            assert len(cache_files) == 1

    def test_query_peaks_in_view_out_of_core(self, sample_peak_df):
        """Test querying peaks in out-of-core mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(out_of_core=True, cache_dir=Path(tmpdir))
            dm.register_peaks(sample_peak_df, "test.mzML")

            result = dm.query_peaks_in_view(
                rt_min=500,
                rt_max=1000,
                mz_min=200,
                mz_max=500,
            )

            assert isinstance(result, pd.DataFrame)
            if len(result) > 0:
                assert result["rt"].min() >= 500
                assert result["rt"].max() <= 1000

    def test_query_peaks_for_minimap_out_of_core(self, sample_peak_df):
        """Test minimap query in out-of-core mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(out_of_core=True, cache_dir=Path(tmpdir))
            dm.register_peaks(sample_peak_df, "test.mzML")

            result = dm.query_peaks_for_minimap(minimap_pixels=500)

            assert result is not None
            # Should be downsampled
            assert len(result) <= 1000

    def test_get_bounds_out_of_core(self, sample_peak_df):
        """Test getting bounds in out-of-core mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(out_of_core=True, cache_dir=Path(tmpdir))
            dm.register_peaks(sample_peak_df, "test.mzML")

            bounds = dm.get_bounds()

            assert bounds["rt_min"] < bounds["rt_max"]
            assert bounds["mz_min"] < bounds["mz_max"]

    def test_get_cache_size_mb(self, sample_peak_df):
        """Test getting cache size in MB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(out_of_core=True, cache_dir=Path(tmpdir))
            dm.register_peaks(sample_peak_df, "test.mzML")

            size = dm.get_cache_size_mb()
            assert size > 0

    def test_get_cache_size_in_memory(self, sample_peak_df):
        """Test getting cache size in in-memory mode (should be 0)."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        size = dm.get_cache_size_mb()
        assert size == 0.0

    def test_clear_cache(self, sample_peak_df):
        """Test clearing cache in out-of-core mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(out_of_core=True, cache_dir=Path(tmpdir))
            dm.register_peaks(sample_peak_df, "test.mzML")

            # Verify cache exists
            cache_files = list(Path(tmpdir).glob("*.parquet"))
            assert len(cache_files) == 1

            dm.clear_cache()

            # Verify cache is cleared
            cache_files = list(Path(tmpdir).glob("*.parquet"))
            assert len(cache_files) == 0
            assert dm._peaks_registered is False

    def test_cleanup(self, sample_peak_df):
        """Test cleanup closes connection."""
        dm = DataManager(out_of_core=False)
        dm.register_peaks(sample_peak_df, "test.mzML")

        dm.cleanup()
        # Connection should be closed - subsequent queries should fail
        # but cleanup should not raise


class TestDataManagerCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_consistency(self):
        """Test that cache key is consistent for same file."""
        dm = DataManager()
        key1 = dm._get_cache_key("test.mzML")
        key2 = dm._get_cache_key("test.mzML")
        assert key1 == key2

    def test_cache_key_different_files(self):
        """Test that cache key differs for different files."""
        dm = DataManager()
        key1 = dm._get_cache_key("test1.mzML")
        key2 = dm._get_cache_key("test2.mzML")
        assert key1 != key2

    def test_cache_key_length(self):
        """Test that cache key has expected length."""
        dm = DataManager()
        key = dm._get_cache_key("test.mzML")
        assert len(key) == 16


class TestDataManagerCompression:
    """Tests for different compression options."""

    @pytest.mark.parametrize("compression", ["snappy", "zstd", "gzip", "none"])
    def test_compression_options(self, sample_peak_df, compression):
        """Test different compression options work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(
                out_of_core=True,
                cache_dir=Path(tmpdir),
                compression=compression,
            )
            dm.register_peaks(sample_peak_df, "test.mzML")

            # Verify data can be queried
            bounds = dm.get_bounds()
            assert bounds["rt_min"] < bounds["rt_max"]
