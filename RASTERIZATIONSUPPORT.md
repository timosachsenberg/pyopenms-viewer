# Implement rasterization support

Use the newest wheel from https://pypi.cs.uni-tuebingen.de/packages/pyopenms-3.6.0.dev20260205-cp313-cp313-macosx_14_0_arm64.whl#sha256=e2c06415de0743dbdb270a1a5b1a013211531834a1e53412acf169d89ab99032
to utilize the newest pyopenms functionality to extract raster 2d numpy matrixes directly from an in memory MSExperiment object. Install from pypi.cs.uni-tuebingen.de as extra source.
The API is:

```
    def rasterizeRTMZ(MSExperiment self,
                    np.ndarray[np.float32_t, ndim=2, mode="c"] output not None,
                    double min_rt, double max_rt,
                    double min_mz, double max_mz,
                    unsigned int ms_level,
                    str aggregation="sum"):
        """
        Rasterize peak data from spectra into a 2D intensity matrix for visualization.

        This method creates a 2D heatmap/image representation of the MS data by binning peak
        intensities into a regular grid of pixels. It is optimized for high performance with
        multithreading (OpenMP) and SIMD vectorization.

        The output array has shape [mz_bins, rt_bins] where:
        - Rows correspond to m/z bins (y-axis in visualization)
        - Columns correspond to RT bins (x-axis in visualization)
        - Values are aggregated intensities (sum or max)

        Examples
        --------
        Quick example (recommended placement):

        >>> import numpy as np
        >>> from pyopenms import MSExperiment, MzMLFile
        >>> exp = MSExperiment()
        >>> MzMLFile().load("data.mzML", exp)
        >>> exp.updateRanges()
        >>> rt_bins, mz_bins = 800, 600
        >>> # Prefer np.empty + C-contiguous array to avoid double zero-fill
        >>> output = np.empty((mz_bins, rt_bins), dtype=np.float32)
        >>> exp.rasterizeRTMZ(output, exp.getMinRT(), exp.getMaxRT(),
        ...                 exp.getMinMZ(), exp.getMaxMZ(), 1, "sum")
        >>> # output now contains the rasterized intensity matrix

        Parameters
        ----------
        output : numpy.ndarray
            Pre-allocated 2D float32 array with shape (mz_bins, rt_bins) to store the
            aggregated intensity values. Will be filled in-place and zero-initialized.
        min_rt : float
            Minimum RT value for the output range.
        max_rt : float
            Maximum RT value for the output range.
        min_mz : float
            Minimum m/z value for the output range.
        max_mz : float
            Maximum m/z value for the output range.
        ms_level : int
            MS level of spectra to include (e.g., 1 for MS1, 2 for MS2).
        aggregation : str, optional
            Aggregation mode: "sum" (default) or "max".

        Returns
        -------
        None
            The output array is modified in-place.

        Notes
        -----
        - The experiment should be sorted by RT and m/z (call sortSpectra(True) if needed)
          for optimal performance and correct results.
        - This method uses per-thread accumulation buffers to avoid contention.
        """
        # Validate array is C-contiguous (required for correct pointer arithmetic)
        if not output.flags['C_CONTIGUOUS']:
            raise ValueError("Output array must be C-contiguous. Use np.ascontiguousarray() to convert.")

        # Validate array is writeable (required for safe C-level writes)
        if not output.flags['WRITEABLE']:
            raise ValueError("Output array must be writeable.")

        cdef _MSExperiment * exp_ = self.inst.get()
        cdef Size rt_bins = output.shape[1]
        cdef Size mz_bins = output.shape[0]

        # Get pointer to numpy array data using .data attribute which is safer
        cdef float* output_ptr = <float*>np.PyArray_DATA(output)

        cdef _MSExperimentRasterAggregation agg_mode
        if aggregation.lower() == "sum":
            agg_mode = _MSExperimentRasterAggregation.SUM
        elif aggregation.lower() == "max":
            agg_mode = _MSExperimentRasterAggregation.MAX
        else:
            raise ValueError(f"Invalid aggregation mode '{aggregation}'. Must be 'sum' or 'max'.")

        exp_.rasterizeRTMZ(output_ptr, rt_bins, mz_bins, min_rt, max_rt, min_mz, max_mz, ms_level, agg_mode)



    def get2DPeakDataLong(MSExperiment self, float min_rt, float max_rt, float min_mz, float max_mz, unsigned int ms_level):
        """Cython signature: tuple[np.array[float] rt, np.array[float] mz, np.array[float] inty] get2DPeakDataLong(float min_rt, float max_rt, float min_mz, float max_mz, unsigned int ms_level)"""
        cdef _MSExperiment * exp_ = self.inst.get()
        cdef libcpp_vector[float] rt
        cdef libcpp_vector[float] mz
        cdef libcpp_vector[float] inty
        exp_.get2DPeakData(min_rt, max_rt, min_mz, max_mz, ms_level, rt, mz, inty)

        cdef ArrayWrapperFloat rt_wrap = ArrayWrapperFloat()
        cdef ArrayWrapperFloat mz_wrap = ArrayWrapperFloat()
        cdef ArrayWrapperFloat inty_wrap = ArrayWrapperFloat()
        rt_wrap.set_data(rt)
        mz_wrap.set_data(mz)
        inty_wrap.set_data(inty)

        return (np.asarray(rt_wrap), np.asarray(mz_wrap), np.asarray(inty_wrap))
```

Use this to render the PeakMap and the MiniMap more efficiently.
On high-up zoom levels use the raster functionality to raster the matrix, convert to xarray and then shade with datashader.
There might be problems in deep zoom levels, when the pixels with a single point will become super small.
First of all we should probably increase the size of the pixels (i.e. increase the pixel width on the screen to fill the same space)
And I would really like a round shape for the points when in deep zoom levels (similar to how it is RIGHT NOW). I think we have two options that we could make configurable in the global app settings:
 - Check if we can make dynspread or spread from datashader work on that rasterized data
 - Use the above get2DPeakDataLong and actually use the "old" datashader Point Canvas like approach and create a temporary dataframe from which it gets rendered. We could start using this approach whenever we would also allow the 3D plot (certain number of points or small enough region). This cutoff should also be configurable.
No global dataframe should be stored anymore in in-memory mode.
The minimap should be cached and only re-shaded if colorscheme changes. The drawing of the excerpt box should be drawn on top without touching the data. This data only changes when a new file is loaded.
If the old Point Canvas is used in deeper zoom levels, 3D plot needs to reuse the exact same temporary dataframe. There should be a auto-update tickbox for 3D plot. If not ticked, the user needs to click update everytime and a warning must show next to the plot, that the 2D view is not in sync anymore (if 2D peak map was moved meanwhile). In that case 3D can hold a second dataframe for a while.