"""Utility modules for coordinate transforms, filtering, GPU support, etc."""

from pyopenms_viewer.utils.coordinate_transform import CoordinateTransform
from pyopenms_viewer.utils.gpu import (
    get_dataframe_type,
    is_cudf_available,
    is_dask_available,
    is_dask_dataframe,
    is_dask_enabled,
    is_gpu_dataframe,
    is_gpu_enabled,
    set_dask_enabled,
    set_gpu_enabled,
    to_accelerated_dataframe,
    to_gpu_dataframe,
)

# Compatibility aliases
is_cudf_dataframe = is_gpu_dataframe

__all__ = [
    "CoordinateTransform",
    "get_dataframe_type",
    "is_cudf_available",
    "is_cudf_dataframe",
    "is_dask_available",
    "is_dask_dataframe",
    "is_dask_enabled",
    "is_gpu_dataframe",
    "is_gpu_enabled",
    "set_dask_enabled",
    "set_gpu_enabled",
    "to_accelerated_dataframe",
    "to_gpu_dataframe",
]
