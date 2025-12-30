"""Performance benchmarks for pyopenms-viewer rendering.

This module measures rendering performance for peak maps in both
in-memory and out-of-core modes, producing timing tables.

Run with: pytest tests/test_performance.py -v -s
"""

import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.rendering.peak_map_renderer import PeakMapRenderer

# Fixed seed for reproducibility
RANDOM_SEED = 42

# Peak counts to benchmark
PEAK_COUNTS = [100_000, 1_000_000, 10_000_000, 100_000_000]

# Number of iterations for timing (more = more accurate but slower)
TIMING_ITERATIONS = 3


def generate_peak_df(n_peaks: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic peak DataFrame with realistic distribution.

    Args:
        n_peaks: Number of peaks to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with rt, mz, intensity, log_intensity columns
    """
    rng = np.random.default_rng(seed)

    # Generate realistic distributions
    rt = rng.uniform(0, 3600, n_peaks)  # 0-60 minutes in seconds
    mz = rng.uniform(100, 2000, n_peaks)  # Typical m/z range
    intensity = rng.lognormal(10, 2, n_peaks)  # Log-normal intensity
    log_intensity = np.log1p(intensity)

    return pd.DataFrame(
        {
            "rt": rt.astype(np.float32),
            "mz": mz.astype(np.float32),
            "intensity": intensity.astype(np.float32),
            "log_intensity": log_intensity.astype(np.float32),
        }
    )


def setup_state_in_memory(df: pd.DataFrame) -> ViewerState:
    """Configure ViewerState with data in in-memory mode.

    Args:
        df: Peak DataFrame

    Returns:
        Configured ViewerState
    """
    state = ViewerState()
    state.init_data_manager(out_of_core=False)

    # Set data bounds
    state.rt_min = float(df["rt"].min())
    state.rt_max = float(df["rt"].max())
    state.mz_min = float(df["mz"].min())
    state.mz_max = float(df["mz"].max())

    # Set view to full extent
    state.view_rt_min = state.rt_min
    state.view_rt_max = state.rt_max
    state.view_mz_min = state.mz_min
    state.view_mz_max = state.mz_max

    # Register data
    state.df = state.data_manager.register_peaks(df, "synthetic.mzML")

    return state


def setup_state_out_of_core(df: pd.DataFrame, cache_dir: Path) -> ViewerState:
    """Configure ViewerState with data in out-of-core mode.

    Args:
        df: Peak DataFrame
        cache_dir: Directory for Parquet cache

    Returns:
        Configured ViewerState
    """
    state = ViewerState()
    state.init_data_manager(out_of_core=True, cache_dir=cache_dir)

    # Set data bounds
    state.rt_min = float(df["rt"].min())
    state.rt_max = float(df["rt"].max())
    state.mz_min = float(df["mz"].min())
    state.mz_max = float(df["mz"].max())

    # Set view to full extent
    state.view_rt_min = state.rt_min
    state.view_rt_max = state.rt_max
    state.view_mz_min = state.mz_min
    state.view_mz_max = state.mz_max

    # Register data (writes to Parquet)
    state.data_manager.register_peaks(df, "synthetic.mzML")
    state.df = None  # Out-of-core mode doesn't keep df in memory

    return state


def time_render(renderer: PeakMapRenderer, state: ViewerState, iterations: int = 3) -> dict:
    """Time a render operation.

    Args:
        renderer: PeakMapRenderer instance
        state: Configured ViewerState
        iterations: Number of iterations to run

    Returns:
        Dictionary with timing statistics
    """
    times = []

    # Warmup run
    renderer.render(state, fast=False, draw_overlays=False, draw_axes=True)

    # Timed runs
    for _ in range(iterations):
        gc.collect()  # Clean up before timing
        start = time.perf_counter()
        renderer.render(state, fast=False, draw_overlays=False, draw_axes=True)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "min_ms": min(times),
        "max_ms": max(times),
        "avg_ms": sum(times) / len(times),
    }


def format_peak_count(n: int) -> str:
    """Format peak count for display."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    elif n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def format_time(ms: float) -> str:
    """Format time in milliseconds."""
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


class TestRenderingPerformance:
    """Performance benchmarks for peak map rendering."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup renderer for each test."""
        self.renderer = PeakMapRenderer(
            plot_width=1100,
            plot_height=550,
            margin_left=80,
            margin_top=20,
        )

    def test_benchmark_in_memory(self):
        """Benchmark rendering in in-memory mode."""
        print("\n" + "=" * 70)
        print("IN-MEMORY MODE BENCHMARKS")
        print("=" * 70)

        results = []

        for n_peaks in PEAK_COUNTS:
            peak_str = format_peak_count(n_peaks)
            print(f"\nGenerating {peak_str} peaks...", end=" ", flush=True)

            try:
                # Generate data
                gen_start = time.perf_counter()
                df = generate_peak_df(n_peaks)
                gen_time = (time.perf_counter() - gen_start) * 1000
                print(f"done ({format_time(gen_time)})")

                # Setup state
                print("Setting up state...", end=" ", flush=True)
                setup_start = time.perf_counter()
                state = setup_state_in_memory(df)
                setup_time = (time.perf_counter() - setup_start) * 1000
                print(f"done ({format_time(setup_time)})")

                # Benchmark render
                print(f"Benchmarking render ({TIMING_ITERATIONS} iterations)...", end=" ", flush=True)
                timing = time_render(self.renderer, state, TIMING_ITERATIONS)
                print(f"done (avg: {format_time(timing['avg_ms'])})")

                results.append(
                    {
                        "peaks": peak_str,
                        "mode": "in-memory",
                        "gen_ms": gen_time,
                        "setup_ms": setup_time,
                        "render_avg_ms": timing["avg_ms"],
                        "render_min_ms": timing["min_ms"],
                        "render_max_ms": timing["max_ms"],
                    }
                )

                # Cleanup
                del df
                del state
                gc.collect()

            except MemoryError:
                print("SKIPPED (out of memory)")
                results.append(
                    {
                        "peaks": peak_str,
                        "mode": "in-memory",
                        "gen_ms": None,
                        "setup_ms": None,
                        "render_avg_ms": None,
                        "render_min_ms": None,
                        "render_max_ms": None,
                        "error": "MemoryError",
                    }
                )

        # Print results table
        self._print_results_table(results, "In-Memory")

    def test_benchmark_out_of_core(self):
        """Benchmark rendering in out-of-core mode."""
        print("\n" + "=" * 70)
        print("OUT-OF-CORE MODE BENCHMARKS")
        print("=" * 70)

        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            for n_peaks in PEAK_COUNTS:
                peak_str = format_peak_count(n_peaks)
                print(f"\nGenerating {peak_str} peaks...", end=" ", flush=True)

                try:
                    # Generate data
                    gen_start = time.perf_counter()
                    df = generate_peak_df(n_peaks)
                    gen_time = (time.perf_counter() - gen_start) * 1000
                    print(f"done ({format_time(gen_time)})")

                    # Setup state (includes Parquet write)
                    print("Setting up state (writing Parquet)...", end=" ", flush=True)
                    setup_start = time.perf_counter()
                    state = setup_state_out_of_core(df, cache_dir)
                    setup_time = (time.perf_counter() - setup_start) * 1000
                    print(f"done ({format_time(setup_time)})")

                    # Free DataFrame memory - out-of-core doesn't need it
                    del df
                    gc.collect()

                    # Benchmark render
                    print(f"Benchmarking render ({TIMING_ITERATIONS} iterations)...", end=" ", flush=True)
                    timing = time_render(self.renderer, state, TIMING_ITERATIONS)
                    print(f"done (avg: {format_time(timing['avg_ms'])})")

                    # Get cache size
                    cache_size_mb = state.data_manager.get_cache_size_mb()

                    results.append(
                        {
                            "peaks": peak_str,
                            "mode": "out-of-core",
                            "gen_ms": gen_time,
                            "setup_ms": setup_time,
                            "render_avg_ms": timing["avg_ms"],
                            "render_min_ms": timing["min_ms"],
                            "render_max_ms": timing["max_ms"],
                            "cache_mb": cache_size_mb,
                        }
                    )

                    # Cleanup
                    state.data_manager.clear_cache()
                    del state
                    gc.collect()

                except MemoryError:
                    print("SKIPPED (out of memory)")
                    results.append(
                        {
                            "peaks": peak_str,
                            "mode": "out-of-core",
                            "gen_ms": None,
                            "setup_ms": None,
                            "render_avg_ms": None,
                            "render_min_ms": None,
                            "render_max_ms": None,
                            "error": "MemoryError",
                        }
                    )

        # Print results table
        self._print_results_table(results, "Out-of-Core")

    def test_benchmark_comparison(self):
        """Run comparison benchmark and print combined table."""
        print("\n" + "=" * 70)
        print("RENDERING PERFORMANCE COMPARISON")
        print("=" * 70)

        all_results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            for n_peaks in PEAK_COUNTS:
                peak_str = format_peak_count(n_peaks)
                print(f"\n--- {peak_str} peaks ---")

                try:
                    # Generate data once
                    print("Generating data...", end=" ", flush=True)
                    df = generate_peak_df(n_peaks)
                    print("done")

                    # In-memory benchmark
                    print("In-memory mode...", end=" ", flush=True)
                    state_mem = setup_state_in_memory(df)
                    timing_mem = time_render(self.renderer, state_mem, TIMING_ITERATIONS)
                    print(f"{format_time(timing_mem['avg_ms'])}")

                    all_results.append(
                        {
                            "peaks": peak_str,
                            "mode": "in-memory",
                            "render_avg_ms": timing_mem["avg_ms"],
                        }
                    )

                    del state_mem
                    gc.collect()

                    # Out-of-core benchmark
                    print("Out-of-core mode...", end=" ", flush=True)
                    state_ooc = setup_state_out_of_core(df, cache_dir)
                    del df  # Free memory for out-of-core test
                    gc.collect()

                    timing_ooc = time_render(self.renderer, state_ooc, TIMING_ITERATIONS)
                    cache_mb = state_ooc.data_manager.get_cache_size_mb()
                    print(f"{format_time(timing_ooc['avg_ms'])} (cache: {cache_mb:.1f}MB)")

                    all_results.append(
                        {
                            "peaks": peak_str,
                            "mode": "out-of-core",
                            "render_avg_ms": timing_ooc["avg_ms"],
                            "cache_mb": cache_mb,
                        }
                    )

                    state_ooc.data_manager.clear_cache()
                    del state_ooc
                    gc.collect()

                except MemoryError:
                    print("SKIPPED (out of memory)")
                    all_results.append(
                        {
                            "peaks": peak_str,
                            "mode": "in-memory",
                            "render_avg_ms": None,
                            "error": "MemoryError",
                        }
                    )
                    all_results.append(
                        {
                            "peaks": peak_str,
                            "mode": "out-of-core",
                            "render_avg_ms": None,
                            "error": "MemoryError",
                        }
                    )

        # Print combined comparison table
        self._print_comparison_table(all_results)

    def _print_results_table(self, results: list, title: str):
        """Print results as markdown table."""
        print(f"\n### {title} Results\n")
        print("| Peaks | Setup | Render (avg) | Render (min) | Render (max) |")
        print("|-------|-------|--------------|--------------|--------------|")

        for r in results:
            if r.get("error"):
                print(f"| {r['peaks']} | {r.get('error', 'N/A')} | - | - | - |")
            else:
                setup = format_time(r["setup_ms"]) if r["setup_ms"] else "-"
                avg = format_time(r["render_avg_ms"]) if r["render_avg_ms"] else "-"
                min_t = format_time(r["render_min_ms"]) if r["render_min_ms"] else "-"
                max_t = format_time(r["render_max_ms"]) if r["render_max_ms"] else "-"
                print(f"| {r['peaks']} | {setup} | {avg} | {min_t} | {max_t} |")

    def _print_comparison_table(self, results: list):
        """Print comparison table between modes."""
        print("\n### Performance Comparison\n")
        print("| Peaks | In-Memory | Out-of-Core | Cache Size |")
        print("|-------|-----------|-------------|------------|")

        # Group by peak count
        by_peaks = {}
        for r in results:
            peaks = r["peaks"]
            if peaks not in by_peaks:
                by_peaks[peaks] = {}
            by_peaks[peaks][r["mode"]] = r

        for peaks in by_peaks:
            mem = by_peaks[peaks].get("in-memory", {})
            ooc = by_peaks[peaks].get("out-of-core", {})

            mem_time = format_time(mem["render_avg_ms"]) if mem.get("render_avg_ms") else mem.get("error", "-")
            ooc_time = format_time(ooc["render_avg_ms"]) if ooc.get("render_avg_ms") else ooc.get("error", "-")
            cache = f"{ooc.get('cache_mb', 0):.1f}MB" if ooc.get("cache_mb") else "-"

            print(f"| {peaks} | {mem_time} | {ooc_time} | {cache} |")
