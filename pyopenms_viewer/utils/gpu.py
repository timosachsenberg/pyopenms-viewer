"""GPU and parallel acceleration support for datashader rendering.

This module provides optional performance optimizations for datashader:
1. GPU acceleration via NVIDIA RAPIDS cuDF (Linux x86_64 only)
2. Multi-threaded computation via Dask DataFrames

When available and enabled, pandas DataFrames are automatically converted
to cuDF (GPU) or Dask (multi-threaded CPU) DataFrames. Datashader detects
these types and uses optimized kernels.

All optimizations are transparent with graceful fallback to pandas.
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import cudf
    import dask.dataframe as dd

logger = logging.getLogger(__name__)

# Global flags to enable/disable optimizations
_gpu_enabled: bool = True
_dask_enabled: bool = True


@lru_cache(maxsize=1)
def is_cudf_available() -> bool:
    """Check if cuDF is available for GPU acceleration.

    Returns:
        True if cuDF can be imported and initialized, False otherwise.
    """
    try:
        import cudf  # noqa: F401

        # Try a simple operation to verify GPU is accessible
        test_df = cudf.DataFrame({"a": [1, 2, 3]})
        _ = len(test_df)
        logger.info("cuDF GPU acceleration is available")
        return True
    except ImportError:
        logger.debug("cuDF not installed - GPU acceleration disabled")
        return False
    except Exception as e:
        logger.warning(f"cuDF available but GPU initialization failed: {e}")
        return False


@lru_cache(maxsize=1)
def is_dask_available() -> bool:
    """Check if Dask is available for parallel computation.

    Returns:
        True if Dask can be imported, False otherwise.
    """
    try:
        import dask.dataframe  # noqa: F401

        logger.info("Dask parallel computation is available")
        return True
    except ImportError:
        logger.debug("Dask not installed - parallel computation disabled")
        return False


def set_gpu_enabled(enabled: bool) -> None:
    """Enable or disable GPU acceleration globally.

    Args:
        enabled: True to enable GPU acceleration (if available), False to force CPU.
    """
    global _gpu_enabled
    _gpu_enabled = enabled
    if enabled and is_cudf_available():
        logger.info("GPU acceleration enabled")
    elif enabled:
        logger.info("GPU acceleration requested but cuDF not available")
    else:
        logger.info("GPU acceleration disabled")


def set_dask_enabled(enabled: bool) -> None:
    """Enable or disable Dask parallel computation globally.

    Args:
        enabled: True to enable Dask (if available), False to force pandas.
    """
    global _dask_enabled
    _dask_enabled = enabled
    if enabled and is_dask_available():
        logger.info("Dask parallel computation enabled")
    elif enabled:
        logger.info("Dask requested but not available")
    else:
        logger.info("Dask parallel computation disabled")


def is_gpu_enabled() -> bool:
    """Check if GPU acceleration is currently enabled and available.

    Returns:
        True if GPU acceleration is both enabled and cuDF is available.
    """
    return _gpu_enabled and is_cudf_available()


def is_dask_enabled() -> bool:
    """Check if Dask parallel computation is currently enabled and available.

    Returns:
        True if Dask is both enabled and available.
    """
    return _dask_enabled and is_dask_available()


def to_accelerated_dataframe(df: pd.DataFrame, n_partitions: int = 4) -> "pd.DataFrame | cudf.DataFrame | dd.DataFrame":
    """Convert a pandas DataFrame to the fastest available format.

    Priority order:
    1. cuDF (GPU) - fastest for large datasets with NVIDIA GPU
    2. Dask (multi-threaded CPU) - good for large datasets on multi-core CPUs
    3. pandas (single-threaded CPU) - fallback

    Args:
        df: Pandas DataFrame to potentially convert.
        n_partitions: Number of partitions for Dask (ignored for GPU/pandas).
                     Default 4 works well for 4-8 core systems.

    Returns:
        Optimized DataFrame (cuDF > Dask > pandas).
    """
    # Priority 1: Try GPU acceleration first
    if is_gpu_enabled():
        try:
            import cudf

            gpu_df = cudf.DataFrame.from_pandas(df)
            logger.debug(f"Converted DataFrame ({len(df)} rows) to cuDF (GPU)")
            return gpu_df
        except Exception as e:
            logger.warning(f"Failed to convert DataFrame to GPU: {e}")

    # Priority 2: Try Dask for multi-threaded CPU
    if is_dask_enabled() and len(df) > 100000:  # Only worth it for larger datasets
        try:
            import dask.dataframe as dd

            # Convert to Dask DataFrame with multiple partitions
            dask_df = dd.from_pandas(df, npartitions=n_partitions)
            logger.debug(f"Converted DataFrame ({len(df)} rows) to Dask ({n_partitions} partitions)")
            return dask_df
        except Exception as e:
            logger.warning(f"Failed to convert DataFrame to Dask: {e}")

    # Priority 3: Fallback to pandas
    return df


def to_gpu_dataframe(df: pd.DataFrame) -> "pd.DataFrame | cudf.DataFrame":
    """Convert a pandas DataFrame to cuDF if GPU acceleration is available.

    This function is the main entry point for GPU acceleration. Pass your
    pandas DataFrame through this function before sending to datashader.
    Datashader automatically detects cuDF DataFrames and uses GPU kernels.

    Args:
        df: Pandas DataFrame to potentially convert.

    Returns:
        cuDF DataFrame if GPU is available and enabled, otherwise the
        original pandas DataFrame unchanged.
    """
    if not is_gpu_enabled():
        return df

    try:
        import cudf

        # cudf.DataFrame.from_pandas handles the conversion
        # This copies data to GPU memory
        gpu_df = cudf.DataFrame.from_pandas(df)
        logger.debug(f"Converted DataFrame ({len(df)} rows) to GPU")
        return gpu_df
    except Exception as e:
        logger.warning(f"Failed to convert DataFrame to GPU: {e}")
        return df


def is_gpu_dataframe(df) -> bool:
    """Check if a DataFrame is a cuDF GPU DataFrame.

    Args:
        df: DataFrame to check.

    Returns:
        True if df is a cuDF DataFrame, False otherwise.
    """
    type_name = type(df).__name__
    module_name = type(df).__module__
    return "cudf" in module_name or type_name == "DataFrame" and hasattr(df, "to_arrow")


def is_dask_dataframe(df) -> bool:
    """Check if a DataFrame is a Dask DataFrame.

    Args:
        df: DataFrame to check.

    Returns:
        True if df is a Dask DataFrame, False otherwise.
    """
    type_name = type(df).__name__
    module_name = type(df).__module__
    return "dask" in module_name and type_name == "DataFrame"


def get_dataframe_type(df) -> str:
    """Get a human-readable string describing the DataFrame type.

    Args:
        df: DataFrame to identify.

    Returns:
        One of: "cudf" (GPU), "dask" (parallel CPU), "pandas" (single-threaded CPU).
    """
    if is_gpu_dataframe(df):
        return "cudf"
    elif is_dask_dataframe(df):
        return "dask"
    else:
        return "pandas"
