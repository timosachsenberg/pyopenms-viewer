# Changelog

All notable changes to pyopenms-viewer are documented in this file.

## [0.1.2] - 2024-12-03

### Fixed
- **PyInstaller windowed mode crash**: Fixed `AttributeError: 'NoneType' object has no attribute 'isatty'` when running the standalone executable in windowed mode. Redirects `sys.stdout`/`sys.stderr` to `devnull` when `None` to prevent uvicorn's logging formatter from crashing.

## [0.1.1] - 2024-12-02

### Fixed
- **PyInstaller build**: Collect all required DLLs for numpy, pandas, datashader, pyopenms, nicegui, and plotly
- **Windows build**: Use bash shell for PyInstaller step to fix PowerShell parsing errors with `--collect-all` flags
- **Windows runners**: Wrap PyInstaller command in `bash -lc` for cross-platform compatibility

## [0.1.0] - 2024-12-02

Initial release with comprehensive mass spectrometry visualization features.

### Features

#### Core Visualization
- **2D Peak Map**: Datashader-based rendering capable of handling 50+ million peaks with smooth zooming and panning
- **1D Spectrum Viewer**: Interactive Plotly-based spectrum display with zoom/pan
- **3D Peak View**: Optional 3D surface visualization of peak data
- **Total Ion Chromatogram (TIC)**: Clickable TIC plot for spectrum navigation
- **Minimap**: Overview navigation with viewport indicator

#### File Support
- **mzML files**: Full support for mass spectrometry data files
- **FeatureXML overlay**: Display feature centroids, bounding boxes, and convex hulls
- **idXML overlay**: Show peptide identification precursor positions on peak map
- **Drag & drop upload**: Easy file loading with automatic type detection

#### Spectrum Analysis
- **Peak annotation**: Automatic MS2 peak annotation using SpectrumAnnotator when identification is selected
- **Spectrum measurement tool**: Click two peaks to measure Δm/z with peak snapping and persistent annotations
- **Hover peak highlighting**: Live preview with nearest peak highlight (cyan/yellow indicator)
- **Auto-scale Y-axis**: Optional automatic Y-axis scaling so highest visible peak is at 95%
- **Intensity toggle**: Switch between percentage (%) and absolute intensity display

#### Peak Map Interactions
- **Drag to zoom**: Visual selection rectangle with cyan dashed border
- **Shift+drag to measure**: Yellow measurement line showing ΔRT and Δm/z
- **Ctrl+drag to pan**: Grab and move the view with orange crosshair indicator
- **Scroll wheel zoom**: Zoom in/out centered on cursor position
- **Spectrum marker**: Optional crosshair showing current spectrum position

#### Unified Spectra Panel
- **View filters**: Toggle between All/MS2/Identified spectra
- **Linked identifications**: Automatic RT and precursor m/z matching between spectra and IDs
- **Advanced columns**: Optional Peaks, TIC, BPI, m/z Range columns
- **Meta values display**: Show PeptideIdentification and PeptideHit meta values
- **All hits option**: Display all peptide hits per spectrum with rank column
- **Sequence display**: Optional peptide sequence overlay on 2D peak map
- **Filters**: RT range, sequence pattern, and minimum score filtering

#### User Interface
- **Dark/light mode**: Toggle with automatic theme-aware colors for all plots
- **Fullscreen mode**: Maximize viewer for detailed analysis
- **Panel reordering**: Customizable panel order via settings menu
- **Collapsible panels**: Clean UI with expandable sections
- **RT unit toggle**: Switch between seconds and minutes display
- **Swap axes**: Toggle RT/m/z axis orientation

#### Performance Optimizations
- **Fast render mode**: 1/4 resolution during panning for smooth interaction
- **Render throttling**: 50ms throttle between renders during continuous panning
- **Minimap skip**: Skip minimap redraw during panning to avoid flicker
- **Max aggregation**: Use maximum (not mean) for peak intensity at each pixel

#### Build & Distribution
- **GitHub Actions workflow**: Automated standalone binary builds
- **Cross-platform support**: Windows, Linux, macOS (x64 and ARM64)
- **Automatic releases**: Triggered on version tags (v*)

### Technical Details
- Built with NiceGUI 3.x web framework
- Datashader for server-side rendering of massive datasets
- pyOpenMS for mzML, FeatureXML, idXML file handling
- Plotly for interactive spectrum visualization
- OpenMP multi-threading for pyOpenMS operations

---

## Development History

### Project Setup
- Initial commit and dependency configuration
- Migration to NiceGUI 3.x API
- Renamed from mzml-viewer to pyopenms-viewer

### UI Evolution
- Added drag & drop file upload, removing manual path inputs
- Implemented collapsible panels with consistent widths
- Added loading progress indicators for large files
- Reorganized 3D view layout below 2D peak map
- Added panel reordering settings menu

### Visualization Improvements
- Switched to max aggregation for peak map rendering
- Improved light/dark mode compatibility across all components
- Made peakmap canvas and containers transparent for theme support
- Fixed 3D plot sizing with explicit dimensions
- Used black peaks in light mode, cyan in dark mode

### Measurement Tools
- Added peakmap measurement tool (Shift+drag)
- Added spectrum measurement tool with peak snapping
- Improved measurement UX with visual feedback and live preview
- Added Delete/Backspace to remove selected measurements
- Made measurement label backgrounds transparent

### Navigation & Interaction
- Added visual selection rectangle for zoom
- Implemented Ctrl+drag panning with clamping
- Added spectrum marker toggle for 2D peakmap
- Fixed zoom coordinate mapping when axes are swapped
- Disabled panning when fully zoomed out

### Data Display
- Merged Spectrum Table and Identifications into unified Spectra panel
- Added peak annotation feature for identified spectra
- Added sequence display, meta values, and all hits options
- Fixed automatic zoom for spectrum/ID selection

### Bug Fixes
- Fixed drag & drop upload for NiceGUI 3.x API changes
- Fixed spectrum plot y-axis to prevent negative values
- Fixed PIL rectangle drawing errors for feature bounding boxes
- Fixed minimap viewport rectangle rendering
- Added missing set_loading and update_loading_progress methods
- Cleared hover highlight when mouse leaves spectrum plot
- Preserved zoom state during spectrum measurement workflow
