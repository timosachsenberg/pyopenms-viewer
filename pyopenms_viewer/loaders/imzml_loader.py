

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser

from pyopenms_viewer.core.state import ViewerState

class ImzMLLoader:
    """Loader for imzML files using pyimzML."""

    def __init__(self, state: ViewerState):
        """Initialize loader with state reference.

        Args:
            state: ViewerState instance to populate with data
        """
        self.state = state
        self.parser: Optional[ImzMLParser] = None
        self.coordinates = None

    def parse(self, filepath: str) -> bool:
        """Parse imzML file using pyimzML (blocking call).

        Args:
            filepath: Path to the imzML file

        Returns:
            True if successful and file has spectra
        """
        try:
            filename = Path(filepath).name
            print(f"Reading {filename} with ImzMLParser (this may take a while)...")
            self.parser = ImzMLParser(filepath)
            self.coordinates = self.parser.coordinates
            print(f"Loaded {len(self.coordinates)} spectra from {filename}")
            return len(self.coordinates) > 0
        except Exception as e:
            print(f"Error parsing imzML: {e}")
            return False

    def process(
        self,
        filepath: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        
        if self.parser is None or self.coordinates is None:
            return False

        total_peaks = sum(len(self.parser.getspectrum(i)[0]) for i in range(len(self.coordinates)))
        if total_peaks == 0:
            return False

        rts = np.zeros(total_peaks, dtype=np.float32)  
        mzs = np.empty(total_peaks, dtype=np.float32)
        intensities = np.empty(total_peaks, dtype=np.float32)
        xs = np.empty(total_peaks, dtype=np.int32)
        ys = np.empty(total_peaks, dtype=np.int32)
        zs = np.empty(total_peaks, dtype=np.int32)

        idx = 0
        for i, (x, y, z) in enumerate(self.coordinates):
            mz_array, intensity_array = self.parser.getspectrum(i)
            n = len(mz_array)
            mzs[idx:idx+n] = mz_array
            intensities[idx:idx+n] = intensity_array
            xs[idx:idx+n] = x
            ys[idx:idx+n] = y
            zs[idx:idx+n] = z
            idx += n
            if progress_callback and i % 100 == 0:
                progress_callback(f"Processed {i+1}/{len(self.coordinates)} spectra", (i+1)/len(self.coordinates))

        df = pd.DataFrame({
            "rt": rts,
            "mz": mzs,
            "intensity": intensities,
            "x": xs,
            "y": ys,
            "z": zs,
        })
        self.state.df = df
        self.state.filetype = "imzML"
        self.state.filepath = filepath
        self.state.data_bounds = None  
        return True
