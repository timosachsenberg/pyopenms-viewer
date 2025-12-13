"""Unified data manager using DuckDB for both in-memory and out-of-core modes.

This module provides a single interface for querying peak data regardless of whether
the data is stored in memory (pandas DataFrame) or on disk (Parquet files).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    pass


class DataManager:
    """Manages peak data access via DuckDB for both in-memory and out-of-core modes.

    In-memory mode: Registers pandas DataFrames directly (zero-copy)
    Out-of-core mode: Writes Parquet files, queries via DuckDB

    Attributes:
        out_of_core: Whether to use disk-based caching
        cache_dir: Directory for cache files
        compression: Parquet compression algorithm
    """

    def __init__(
        self,
        out_of_core: bool = False,
        cache_dir: Path | None = None,
        compression: str = "snappy",
    ):
        """Initialize the data manager.

        Args:
            out_of_core: If True, write data to Parquet and query from disk.
                        If False, register DataFrames directly with DuckDB.
            cache_dir: Directory for cache files. If None, uses a temp directory.
            compression: Parquet compression algorithm (zstd, snappy, gzip, none).
        """
        self.out_of_core = out_of_core
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="pyopenms_viewer_"))
        self.compression = compression
        self.conn = duckdb.connect(":memory:")

        self._peaks_registered = False
        self._im_peaks_registered = False
        self._source_file: str | None = None
        self._peak_cache_path: Path | None = None
        self._im_cache_path: Path | None = None

        # Cached counts to avoid repeated SQL queries
        self._peak_count: int = 0
        self._im_peak_count: int = 0

        # Keep DataFrame references for in-memory mode
        self._df: pd.DataFrame | None = None
        self._im_df: pd.DataFrame | None = None

    def _get_cache_key(self, filepath: str) -> str:
        """Generate cache key from file path and mtime.

        Args:
            filepath: Path to the source file

        Returns:
            16-character hash string
        """
        path = Path(filepath)
        if path.exists():
            stat = path.stat()
            key_str = f"{filepath}:{stat.st_mtime}:{stat.st_size}"
        else:
            key_str = filepath
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def register_peaks(self, df: pd.DataFrame, source_file: str) -> pd.DataFrame | None:
        """Register peak DataFrame with DuckDB.

        In-memory mode: Registers DataFrame directly (zero-copy)
        Out-of-core mode: Writes to Parquet, registers as view, returns None

        Args:
            df: Peak DataFrame with columns: rt, mz, intensity, log_intensity, [cv]
            source_file: Path to the source mzML file (used for cache key)

        Returns:
            DataFrame to keep in memory, or None if data was written to disk
        """
        self._source_file = source_file

        # Drop existing registrations if present
        if self._peaks_registered:
            self.conn.execute("DROP VIEW IF EXISTS peaks")
            # Unregister the table (works for both registered DataFrames and actual tables)
            try:
                self.conn.unregister("peaks_table")
            except Exception:
                pass  # May not exist or already unregistered

        if self.out_of_core:
            # Write to Parquet
            cache_key = self._get_cache_key(source_file)
            cache_path = self.cache_dir / f"peaks_{cache_key}.parquet"

            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False),
                cache_path,
                compression=self.compression,
                row_group_size=1_000_000,
            )

            self._peak_cache_path = cache_path

            # Register Parquet as view
            self.conn.execute(f"""
                CREATE VIEW peaks AS
                SELECT * FROM read_parquet('{cache_path}')
            """)
            self._peaks_registered = True
            self._peak_count = len(df)  # Cache count before freeing df
            self._df = None
            return None  # Signal to free DataFrame
        else:
            # Register DataFrame directly (zero-copy)
            self.conn.register("peaks_table", df)
            self.conn.execute("CREATE VIEW peaks AS SELECT * FROM peaks_table")
            self._peaks_registered = True
            self._peak_count = len(df)  # Cache count
            self._df = df
            return df  # Keep in memory

    def register_im_peaks(self, df: pd.DataFrame, source_file: str) -> pd.DataFrame | None:
        """Register ion mobility DataFrame with DuckDB.

        Args:
            df: Ion mobility DataFrame with columns: mz, im, intensity, log_intensity
            source_file: Path to the source mzML file

        Returns:
            DataFrame to keep in memory, or None if data was written to disk
        """
        # Drop existing registrations if present
        if self._im_peaks_registered:
            self.conn.execute("DROP VIEW IF EXISTS im_peaks")
            try:
                self.conn.unregister("im_peaks_table")
            except Exception:
                pass

        if self.out_of_core:
            cache_key = self._get_cache_key(source_file)
            cache_path = self.cache_dir / f"im_peaks_{cache_key}.parquet"

            self.cache_dir.mkdir(parents=True, exist_ok=True)

            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False),
                cache_path,
                compression=self.compression,
                row_group_size=500_000,
            )

            self._im_cache_path = cache_path

            self.conn.execute(f"""
                CREATE VIEW im_peaks AS
                SELECT * FROM read_parquet('{cache_path}')
            """)
            self._im_peaks_registered = True
            self._im_peak_count = len(df)  # Cache count before freeing df
            self._im_df = None
            return None
        else:
            self.conn.register("im_peaks_table", df)
            self.conn.execute("CREATE VIEW im_peaks AS SELECT * FROM im_peaks_table")
            self._im_peaks_registered = True
            self._im_peak_count = len(df)  # Cache count
            self._im_df = df
            return df

    def query_peaks_in_view(
        self,
        rt_min: float,
        rt_max: float,
        mz_min: float,
        mz_max: float,
        cv: float | None = None,
    ) -> pd.DataFrame:
        """Query peaks within view bounds via DuckDB.

        This is the unified query path for both in-memory and out-of-core modes.
        In-memory mode uses DuckDB's zero-copy DataFrame registration for good
        performance (~30ms for 5M peaks). Out-of-core mode queries from Parquet.

        Args:
            rt_min: Minimum retention time
            rt_max: Maximum retention time
            mz_min: Minimum m/z
            mz_max: Maximum m/z
            cv: Compensation voltage for FAIMS filtering (optional)

        Returns:
            DataFrame with peaks in the specified view
        """
        if not self._peaks_registered:
            return pd.DataFrame()

        query = """
            SELECT rt, mz, intensity, log_intensity
            FROM peaks
            WHERE rt >= ? AND rt <= ?
              AND mz >= ? AND mz <= ?
        """
        params: list = [rt_min, rt_max, mz_min, mz_max]

        if cv is not None:
            query += " AND cv = ?"
            params.append(cv)

        return self.conn.execute(query, params).fetchdf()

    def query_im_peaks_in_view(
        self,
        mz_min: float,
        mz_max: float,
        im_min: float,
        im_max: float,
    ) -> pd.DataFrame:
        """Query ion mobility peaks within view bounds via DuckDB.

        This is the unified query path for both in-memory and out-of-core modes.

        Args:
            mz_min: Minimum m/z
            mz_max: Maximum m/z
            im_min: Minimum ion mobility value
            im_max: Maximum ion mobility value

        Returns:
            DataFrame with IM peaks in the specified view
        """
        if not self._im_peaks_registered:
            return pd.DataFrame()

        query = """
            SELECT mz, im, intensity, log_intensity
            FROM im_peaks
            WHERE mz >= ? AND mz <= ?
              AND im >= ? AND im <= ?
        """
        return self.conn.execute(query, [mz_min, mz_max, im_min, im_max]).fetchdf()

    def query_peaks_top_n(
        self,
        rt_min: float,
        rt_max: float,
        mz_min: float,
        mz_max: float,
        limit: int,
        cv: float | None = None,
    ) -> pd.DataFrame:
        """Query top N peaks by intensity within view bounds via DuckDB.

        Uses ORDER BY + LIMIT pushdown for efficient top-k selection without
        fetching all peaks. DuckDB uses a heap-based algorithm (O(n log k))
        which is much faster than fetching all rows and sorting in Python.

        Args:
            rt_min: Minimum retention time
            rt_max: Maximum retention time
            mz_min: Minimum m/z
            mz_max: Maximum m/z
            limit: Maximum number of peaks to return (top N by intensity)
            cv: Compensation voltage for FAIMS filtering (optional)

        Returns:
            DataFrame with top N peaks by intensity in the specified view
        """
        if not self._peaks_registered:
            return pd.DataFrame()

        query = """
            SELECT rt, mz, intensity, log_intensity
            FROM peaks
            WHERE rt >= ? AND rt <= ?
              AND mz >= ? AND mz <= ?
        """
        params: list = [rt_min, rt_max, mz_min, mz_max]

        if cv is not None:
            query += " AND cv = ?"
            params.append(cv)

        query += " ORDER BY intensity DESC LIMIT ?"
        params.append(limit)

        return self.conn.execute(query, params).fetchdf()

    def query_mobilogram(
        self,
        mz_min: float,
        mz_max: float,
        im_min: float,
        im_max: float,
        n_bins: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query mobilogram histogram via DuckDB aggregation.

        Computes a histogram of summed intensities across ion mobility bins
        entirely in DuckDB, returning only the aggregated bin data instead
        of millions of individual peaks.

        Args:
            mz_min: Minimum m/z
            mz_max: Maximum m/z
            im_min: Minimum ion mobility value
            im_max: Maximum ion mobility value
            n_bins: Number of histogram bins (default 200)

        Returns:
            Tuple of (bin_centers, intensities) numpy arrays
        """
        if not self._im_peaks_registered:
            return np.array([]), np.array([])

        im_range = im_max - im_min
        if im_range <= 0:
            return np.array([]), np.array([])

        bin_width = im_range / n_bins

        # Use DuckDB to compute histogram aggregation
        # FLOOR((im - im_min) / bin_width) gives bin index 0 to n_bins-1
        # We clamp to valid range with LEAST/GREATEST
        query = """
            SELECT
                LEAST(GREATEST(FLOOR((im - ?) / ?), 0), ? - 1)::INT as bin_idx,
                SUM(intensity) as total_intensity
            FROM im_peaks
            WHERE mz >= ? AND mz <= ?
              AND im >= ? AND im <= ?
            GROUP BY bin_idx
            ORDER BY bin_idx
        """
        result = self.conn.execute(
            query, [im_min, bin_width, n_bins, mz_min, mz_max, im_min, im_max]
        ).fetchdf()

        if len(result) == 0:
            return np.array([]), np.array([])

        # Build full arrays with zeros for empty bins
        bin_centers = np.linspace(im_min + bin_width / 2, im_max - bin_width / 2, n_bins)
        intensities = np.zeros(n_bins, dtype=np.float64)

        # Fill in the bins that have data
        for _, row in result.iterrows():
            bin_idx = int(row["bin_idx"])
            if 0 <= bin_idx < n_bins:
                intensities[bin_idx] = row["total_intensity"]

        return bin_centers, intensities

    def query_peaks_for_minimap(self, minimap_pixels: int = 80000) -> pd.DataFrame | None:
        """Query peaks for minimap rendering with adaptive downsampling via DuckDB.

        Downsampling is adaptive based on data size vs minimap resolution.
        For a 400x200 minimap (80k pixels), we target roughly that many points
        for good visual quality. This significantly speeds up rendering for
        large datasets (e.g., 10M peaks: 71ms -> 9ms).

        This is the unified query path for both in-memory and out-of-core modes.

        Args:
            minimap_pixels: Target number of points (default: 400*200 = 80000)

        Returns:
            DataFrame with peaks for minimap, or None if no data
        """
        if not self._peaks_registered:
            return None

        # Get total count to determine sample rate
        total = self.conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0]

        if total <= minimap_pixels:
            # Small dataset - return all
            return self.conn.execute("""
                SELECT rt, mz, log_intensity FROM peaks
            """).fetchdf()
        else:
            # Adaptive downsampling: sample_rate = total / target_points
            sample_rate = max(1, total // minimap_pixels)
            return self.conn.execute(f"""
                SELECT rt, mz, log_intensity
                FROM (
                    SELECT rt, mz, log_intensity,
                           ROW_NUMBER() OVER () as rn
                    FROM peaks
                )
                WHERE rn % {sample_rate} = 0
            """).fetchdf()

    def query_all_peaks(self) -> pd.DataFrame | None:
        """Query all peaks without downsampling via DuckDB.

        Use this when downsampling is disabled and full data accuracy is needed.
        This is the unified query path for both in-memory and out-of-core modes.

        Returns:
            DataFrame with all peaks, or None if no data
        """
        if not self._peaks_registered:
            return None

        return self.conn.execute("""
            SELECT rt, mz, log_intensity FROM peaks
        """).fetchdf()

    def query_peaks_for_cv(
        self, cv: float, downsample: bool = True, minimap_pixels: int = 80000
    ) -> pd.DataFrame | None:
        """Query peaks for a specific FAIMS CV value via DuckDB.

        This is the unified query path for both in-memory and out-of-core modes.

        Args:
            cv: Compensation voltage value to filter by
            downsample: Whether to apply adaptive downsampling
            minimap_pixels: Target number of points for downsampling (if enabled)

        Returns:
            DataFrame with peaks for the specified CV, or None if no data
        """
        if not self._peaks_registered:
            return None

        if not downsample:
            # No downsampling - return all peaks for this CV
            return self.conn.execute("""
                SELECT rt, mz, log_intensity FROM peaks WHERE cv = ?
            """, [cv]).fetchdf()

        # Get count for this CV
        total = self.conn.execute(
            "SELECT COUNT(*) FROM peaks WHERE cv = ?", [cv]
        ).fetchone()[0]

        if total == 0:
            return pd.DataFrame()

        if total <= minimap_pixels:
            return self.conn.execute("""
                SELECT rt, mz, log_intensity FROM peaks WHERE cv = ?
            """, [cv]).fetchdf()
        else:
            sample_rate = max(1, total // minimap_pixels)
            return self.conn.execute(f"""
                SELECT rt, mz, log_intensity
                FROM (
                    SELECT rt, mz, log_intensity,
                           ROW_NUMBER() OVER () as rn
                    FROM peaks
                    WHERE cv = ?
                )
                WHERE rn % {sample_rate} = 0
            """, [cv]).fetchdf()

    def get_bounds(self) -> dict[str, float]:
        """Get data bounds without loading all data.

        Returns:
            Dictionary with rt_min, rt_max, mz_min, mz_max
        """
        if not self._peaks_registered:
            return {"rt_min": 0.0, "rt_max": 0.0, "mz_min": 0.0, "mz_max": 0.0}

        result = self.conn.execute("""
            SELECT
                MIN(rt) as rt_min, MAX(rt) as rt_max,
                MIN(mz) as mz_min, MAX(mz) as mz_max
            FROM peaks
        """).fetchone()

        return {
            "rt_min": float(result[0]) if result[0] is not None else 0.0,
            "rt_max": float(result[1]) if result[1] is not None else 0.0,
            "mz_min": float(result[2]) if result[2] is not None else 0.0,
            "mz_max": float(result[3]) if result[3] is not None else 0.0,
        }

    def get_im_bounds(self) -> dict[str, float]:
        """Get ion mobility data bounds.

        Returns:
            Dictionary with mz_min, mz_max, im_min, im_max
        """
        if not self._im_peaks_registered:
            return {"mz_min": 0.0, "mz_max": 0.0, "im_min": 0.0, "im_max": 0.0}

        result = self.conn.execute("""
            SELECT
                MIN(mz) as mz_min, MAX(mz) as mz_max,
                MIN(im) as im_min, MAX(im) as im_max
            FROM im_peaks
        """).fetchone()

        return {
            "mz_min": float(result[0]) if result[0] is not None else 0.0,
            "mz_max": float(result[1]) if result[1] is not None else 0.0,
            "im_min": float(result[2]) if result[2] is not None else 0.0,
            "im_max": float(result[3]) if result[3] is not None else 0.0,
        }

    def get_peak_count(self) -> int:
        """Get total peak count (cached, no SQL query).

        Returns:
            Number of peaks in the dataset
        """
        return self._peak_count

    def get_im_peak_count(self) -> int:
        """Get total ion mobility peak count (cached, no SQL query).

        Returns:
            Number of IM peaks in the dataset
        """
        return self._im_peak_count

    def get_cache_size_mb(self) -> float:
        """Get total cache size in MB.

        Returns:
            Cache size in megabytes (0 if in-memory mode)
        """
        if not self.out_of_core:
            return 0.0

        total = 0
        for f in self.cache_dir.glob("*.parquet"):
            total += f.stat().st_size
        return total / (1024 * 1024)

    def clear_cache(self):
        """Clear all cached Parquet files."""
        for f in self.cache_dir.glob("*.parquet"):
            try:
                f.unlink()
            except Exception:
                pass

        # Reset registration state
        if self._peaks_registered:
            try:
                self.conn.execute("DROP VIEW IF EXISTS peaks")
            except Exception:
                pass
            self._peaks_registered = False
        if self._im_peaks_registered:
            try:
                self.conn.execute("DROP VIEW IF EXISTS im_peaks")
            except Exception:
                pass
            self._im_peaks_registered = False

        self._peak_cache_path = None
        self._im_cache_path = None

    def clear(self):
        """Clear all registered data (both in-memory and cached)."""
        self.clear_cache()
        self._df = None
        self._im_df = None
        self._source_file = None
        self._peak_count = 0
        self._im_peak_count = 0

    def cleanup(self):
        """Clean up resources and close connection."""
        try:
            self.conn.close()
        except Exception:
            pass
