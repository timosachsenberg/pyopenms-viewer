# pyopenms-viewer Feature Overview

A fast, interactive mzML peak map viewer built with NiceGUI, Datashader, and pyOpenMS. Designed to handle 50+ million peaks with smooth zooming and panning.

---

## File Format Support

### Supported Formats
| Format | Extension | Description |
|--------|-----------|-------------|
| **mzML** | `.mzML`, `.mzml` | Mass spectrometry peak map data |
| **FeatureXML** | `.featureXML`, `.featurexml` | Detected feature maps with centroids and hulls |
| **idXML** | `.idXML`, `.idxml` | Peptide identification results |

### Loading Methods
- **Command-line arguments**: Load multiple files at startup
- **Drag-and-drop**: Drop files directly into the viewer
- **Native file dialog**: Browse filesystem (native window mode)
- **Auto-detection**: Case-insensitive extension matching with XML fallback

---

## Peak Map Visualization (2D View)

### High-Performance Rendering
- **Datashader-based**: Server-side aggregation handles 50+ million peaks
- **Adaptive resolution**: Full resolution (1100×550) or fast mode (4× reduced) during interaction
- **Dynamic spreading**: `dynspread` improves visibility of sparse data
- **View filtering**: Only renders peaks within current viewport

### Colormaps
Seven colormap options: **Jet**, **Hot**, **Fire**, **Viridis**, **Plasma**, **Inferno**, **Magma**

### Navigation & Zoom
| Action | Description |
|--------|-------------|
| **Scroll wheel** | Zoom in/out at cursor position |
| **Drag** | Select rectangular region to zoom |
| **Alt+Drag** | Pan the view when zoomed in |
| **Shift+Drag** | Measure ΔRT and Δm/z between points |
| **Double-click** | Reset to full view |
| **Minimap click** | Center view at clicked location |

### Zoom History
- Tracks up to 10 zoom states
- "Back" button navigates through history
- Descriptive labels show range for each state

### Display Options
- **Swap axes**: Toggle between RT/m/z axis orientations
- **RT units**: Display in seconds or minutes
- **Spectrum marker**: Crosshair showing current spectrum position
- **Go-to-Range dialog**: Enter exact RT/m/z coordinates (press `G`)

---

## Feature Overlay (FeatureXML)

### Visualization Modes
| Mode | Default | Description |
|------|---------|-------------|
| **Centroids** | ON | Green dots at feature centers |
| **Bounding Boxes** | OFF | Yellow rectangles showing RT/m/z extent |
| **Convex Hulls** | OFF | Cyan filled polygons showing feature shape |

### Feature Interaction
- **Hover-snap**: Cursor snaps to nearest centroid within 15 pixels
- **Click selection**: Select features with visual feedback (magenta highlight)
- **Hover highlight**: Orange glow ring on hovered features
- **Notification**: Shows RT, m/z, intensity, and charge on selection

### Features Table Panel
- **Columns**: Index, RT, m/z, Intensity, Charge, Quality
- **Sorting**: Click column headers to sort
- **Filtering**: By minimum intensity, minimum quality, charge state
- **Zoom-to-feature**: Click row to zoom peak map to feature location
- **Export**: Download filtered data as TSV

### Selection Synchronization
- Feature selection syncs between peak map and table
- Table auto-navigates to show selected feature

---

## Peptide Identification Overlay (idXML)

### Visualization
- **Diamond markers**: Orange markers at precursor positions
- **Selection state**: Red highlight for selected IDs
- **Optional sequences**: Display peptide sequences on map

### ID-to-Spectrum Linking
- Automatic matching by RT (±5s) and m/z (±0.5 Da)
- Links identifications to MS2 spectra
- Preserves all meta values from PeptideIdentification and PeptideHit

---

## Spectrum Viewer (1D View)

### Navigation
| Button | Action |
|--------|--------|
| `< >` | Previous/Next spectrum |
| `<< >>` | First/Last spectrum |
| `< MS1` / `MS1 >` | Previous/Next MS1 only |
| `< MS2` / `MS2 >` | Previous/Next MS2 only |

### Display Options
- **Intensity mode**: Absolute or Relative (%)
- **Auto Y-scale**: Adjusts to 95% of visible peak maximum
- **m/z labels**: Toggle automatic peak labeling
- **Precursor marker**: Dashed orange line for MS2 precursor

### Interactive Tools
| Tool | Description |
|------|-------------|
| **📏 Measure** | Click two peaks to measure Δm/z |
| **🏷️ Label** | Click peaks to add custom annotations |
| **Clear Δ** | Remove all measurements |
| **Clear 🏷️** | Remove all labels |

### Peak Handling
- **Smart downsampling**: 70% uniform m/z coverage + 30% top intensity peaks
- **Maximum 5,000 peaks** displayed for performance
- **Peak snapping**: Click detection with 2D distance calculation

---

## MS/MS Spectrum Annotation

### Theoretical Spectrum Generation
- Uses pyOpenMS `TheoreticalSpectrumGenerator`
- Generates **b**, **y**, and **a** ions (c, x, z optional)
- Charge-state aware (up to min(precursor charge, 2))

### Ion Matching
- Configurable m/z tolerance (default: 0.05 Da)
- Tracks matched and unmatched theoretical ions
- Calculates coverage statistics

### Ion Visualization
| Ion Type | Color |
|----------|-------|
| b-ions | Blue (#1f77b4) |
| y-ions | Red (#d62728) |
| a-ions | Green (#2ca02c) |
| c-ions | Purple (#9467bd) |
| x-ions | Brown (#8c564b) |
| z-ions | Pink (#e377c2) |
| Precursor | Orange (#ff7f0e) |

### Label Formatting
- **Subscript indices**: y₅ instead of y5
- **Superscript charges**: y₅²⁺ for doubly charged
- **Neutral loss preservation**: y₇+H₂O²⁺

### Mirror View
- Experimental peaks displayed upward
- Theoretical peaks displayed downward
- Optional unmatched theoretical ions (dashed lines)
- Coverage percentage display

---

## Total Ion Chromatogram (TIC)

### Display
- Line plot with fill-to-zero styling
- Cyan trace on dark theme
- MS1 TIC or BPC source selection

### Interactions
- **Click**: Jump to spectrum at clicked RT
- **Drag**: Zoom RT range
- **Double-click**: Reset zoom

### Indicators
- **Yellow rectangle**: Current peak map RT view range
- **Red dashed line**: Currently selected spectrum position

---

## Spectra Table

### Columns
Index, RT, MS Level, CV, Peak Count, TIC, BPI, m/z Range, Precursor m/z, Precursor Charge, Sequence, Score

### View Modes
- **All**: Show all spectra
- **MS2**: Show only MS2 spectra
- **Identified**: Show only identified MS2 spectra

### Filtering
- RT range (min/max)
- Sequence pattern (substring match)
- Minimum score
- Show all hits toggle

### Export
- Download filtered table as TSV

---

## Ion Mobility Support

### Detection
- Automatic detection of TIMS (1/K₀) and drift time data
- Creates separate IM peak map panel

### IM Peak Map
- 2D view: m/z vs. ion mobility
- Mobilogram: Summed intensity vs. ion mobility profile
- Adaptive binning (50-200 bins)

---

## FAIMS Support

### Multi-CV Handling
- Automatic CV (compensation voltage) detection
- Per-CV data filtering and display
- Individual minimaps for each CV value

### CV Selection
- Click CV label to filter peak map
- CV indicator in toolbar

---

## 3D Peak View

### Activation
- "3D" toggle button in minimap area
- Requires zoomed region (RT ≤ 120s, m/z ≤ 50)

### Features
- 3D surface visualization via pyopenms-viz
- Maximum 5,000 peaks (subsampled by intensity)
- Feature bounding box overlays

---

## User Interface

### Panel System
| Panel | Auto-Visibility |
|-------|-----------------|
| TIC | Always |
| Chromatograms | When chromatograms present |
| Peak Map (2D) | Always |
| Ion Mobility Map | When IM data present |
| Spectrum (1D) | Always |
| Spectra Table | Always |
| Features Table | When features loaded |

### Panel Configuration
- Three-state visibility: Hide / Auto / Show
- Drag-to-reorder panels
- Settings accessible via gear icon

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `←` `→` | Pan left/right (RT) |
| `↑` `↓` | Pan up/down (m/z) |
| `Home` | Reset to full view |
| `G` | Open Go-to-Range dialog |
| `F11` | Toggle fullscreen |
| `Delete` | Remove selected measurement |

### Theming
- **Dark mode** (default) / **Light mode** toggle
- Persistent via environment variable
- CLI flags: `--dark` / `--light`

---

## Command-Line Interface

```
pyopenms-viewer [FILES] [OPTIONS]

Arguments:
  FILES                    mzML, featureXML, idXML files to load

Options:
  -p, --port INT          Port (default: 8080)
  -H, --host TEXT         Host (default: 0.0.0.0)
  -o, --open              Open browser automatically (default)
  -n, --no-open           Don't open browser
  --native                Run in native window (PyQt6)
  --dark                  Dark mode (default)
  --light                 Light mode
  --help                  Show help
```

### Examples
```bash
pyopenms-viewer                                    # Empty viewer
pyopenms-viewer sample.mzML                        # Load mzML
pyopenms-viewer sample.mzML features.featureXML   # With features
pyopenms-viewer sample.mzML ids.idXML             # With IDs
pyopenms-viewer --native sample.mzML              # Native window
pyopenms-viewer --port 9000 --no-open data.mzML   # Custom port
```

---

## Native Window Mode

### Requirements
```bash
uv sync --extra native
```

### Features
- Native OS window (1400×900 default)
- Native file dialog with multi-select
- PyQt6 + PyWebView backend

---

## Export Capabilities

### Table Export (TSV)
- Spectra table metadata
- Features table data
- Chromatogram metadata

---

## Performance Optimizations

### Rendering
- **Fast mode**: 4× reduced resolution during pan/drag
- **Throttling**: 50ms minimum between pan renders
- **View culling**: Only renders visible features/IDs
- **Limits**: Max 10,000 features, 5,000 IDs per frame

### Data Handling
- DataFrame filtering by view bounds
- Lazy rendering with debounced updates
- Zoom state preservation during interactions

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI Framework | NiceGUI |
| Peak Rendering | Datashader |
| MS Data | pyOpenMS |
| Plotting | Plotly |
| Data Processing | pandas, numpy |
| Native Window | PyQt6, pywebview |

---

*pyopenms-viewer is designed for high-performance visualization of mass spectrometry data, combining the flexibility of web-based interfaces with the rendering power needed for large-scale proteomics datasets.*
