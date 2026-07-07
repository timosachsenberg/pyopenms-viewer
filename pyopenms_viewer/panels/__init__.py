"""UI panel components for NiceGUI interface.

Each panel is a self-contained UI component that:
1. Receives a reference to ViewerState (shared, never copied)
2. Subscribes to relevant events
3. Updates its display when data changes

Available panels:
- TICPanel: Total Ion Chromatogram display
- PeakMapPanel: 2D RT vs m/z peak map (includes go-to range dialog)
- SpectrumPanel: 1D spectrum viewer
- ChromatogramPanel: Extracted ion chromatograms
- IMPeakMapPanel: Ion mobility vs m/z map
- SpectraTablePanel: Spectra metadata table
- FeaturesTablePanel: Features metadata table
"""

from pyopenms_viewer.panels.base_panel import BasePanel, PanelManager
from pyopenms_viewer.panels.chromatogram_panel import ChromatogramPanel
from pyopenms_viewer.panels.export_panel import ExportPanel
from pyopenms_viewer.panels.faims_panel import FAIMSPanel
from pyopenms_viewer.panels.features_table_panel import FeaturesTablePanel
from pyopenms_viewer.panels.im_peak_map_panel import IMPeakMapPanel
from pyopenms_viewer.panels.imaging_panel import ImagingPanel
from pyopenms_viewer.panels.log_panel import LogPanel
from pyopenms_viewer.panels.peak_map_panel import PeakMapPanel
from pyopenms_viewer.panels.scripting_panel import ScriptingPanel
from pyopenms_viewer.panels.spectra_table_panel import SpectraTablePanel
from pyopenms_viewer.panels.spectrum_panel import SpectrumPanel
from pyopenms_viewer.panels.tic_panel import TICPanel

__all__ = [
    "BasePanel",
    "PanelManager",
    "TICPanel",
    "ImagingPanel",
    "PeakMapPanel",
    "SpectrumPanel",
    "ChromatogramPanel",
    "IMPeakMapPanel",
    "SpectraTablePanel",
    "FeaturesTablePanel",
    "ExportPanel",
    "LogPanel",
    "ScriptingPanel",
    "FAIMSPanel",
]
