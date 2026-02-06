from pyopenms_viewer.utils.gpu import (
    to_accelerated_dataframe,
    get_dataframe_type,
    is_gpu_dataframe,
    is_dask_dataframe,
    is_dask_enabled,
    is_gpu_enabled,
)
import pandas as pd
import numpy as np

# Create a large DataFrame (200k rows)
n = 200000
df = pd.DataFrame(
    {
        "rt": np.random.uniform(0, 3600, n),
        "mz": np.random.uniform(100, 2000, n),
        "intensity": np.random.uniform(1e4, 1e7, n),
    }
)

result = to_accelerated_dataframe(df, n_partitions=4)

print(f"GPU enabled: {is_gpu_enabled()}")
print(f"Dask enabled: {is_dask_enabled()}")
print(f"Input size: {len(df)} rows")
print(f"Result type: {type(result).__name__} from {type(result).__module__}")
print(f"Is GPU DataFrame: {is_gpu_dataframe(result)}")
print(f"Is Dask DataFrame: {is_dask_dataframe(result)}")
print(f"DataFrame type string: {get_dataframe_type(result)}")
if is_dask_dataframe(result):
    print(f"Dask partitions: {result.npartitions}")
